from __future__ import annotations

"""DINOv2分类与特征向量推理服务。

本模块不创建HTTP路由；它负责读取容器环境变量、加载训练配置和增量checkpoint、
构建完整DINOv2模型、执行图片预处理、分类及embedding提取。``server.py``在API
容器启动时创建一个全局 :class:`ModelService` 实例，使权重只加载一次。
"""

import io
import os
import threading
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

from src.config_utils import load_config
from src.data import build_transform
from src.model import DinoV2Classifier
from .lora_runtime import load_binmap_adapter, set_binmap_adapter


class ModelService:
    """加载一个DINOv2分类器，并串行保护GPU推理。

    一个API进程只创建一个实例。线程锁避免多个HTTP请求同时操作同一GPU模型，
    降低显存峰值和底层CUDA并发导致的不确定问题。
    """

    def __init__(self) -> None:
        """读取配置和checkpoint，恢复完整模型并移动到可用计算设备。

        环境变量 ``MODEL_CONFIG`` 和 ``MODEL_CHECKPOINT`` 由docker run传入；未设置
        时使用 ``/models/current`` 下的默认文件。模型先在CPU恢复，再整体移到GPU。
        """
        # API容器把宿主机deployed_model只读挂载到/models/current。
        self.config_path = Path(
            os.environ.get("MODEL_CONFIG", "/models/current/resolved_config.json")
        )
        self.checkpoint_path = Path(
            os.environ.get("MODEL_CHECKPOINT", "/models/current/best_model.pt")
        )
        # 在加载大型权重前先给出明确文件错误，便于排查挂载路径。
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Model config not found: {self.config_path}")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {self.checkpoint_path}")

        # resolved_config.json保存训练时实际模型结构和预处理参数。
        self.cfg = load_config(self.config_path)
        # checkpoint包含分类头/可选微调block增量、类别名称和训练元数据。
        checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        # checkpoint类别数必须与配置一致，否则分类logits无法可靠解释。
        self.class_names = list(checkpoint["class_names"])
        if len(self.class_names) != int(self.cfg["num_classes"]):
            raise ValueError(
                "Checkpoint class count does not match resolved_config.json"
            )

        # 先载入镜像内官方主干权重，再叠加训练得到的增量参数。
        self.model = DinoV2Classifier(self.cfg)
        self.model.set_trainable_blocks(
            int(checkpoint.get("unfreeze_last_blocks", 0))
        )
        self.model.load_delta(checkpoint["model_delta"])
        adapter_path = Path(os.environ.get("BINMAP_ADAPTER", "/models/binmap_adapter/best_adapter.pt"))
        self.binmap_adapter = load_binmap_adapter(self.model.backbone, adapter_path) if adapter_path.is_file() else None
        # CUDA可见时使用GPU；eval关闭Dropout并固定推理行为。
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device).eval()
        # API必须使用无随机增强的推理预处理，与离线测试保持一致。
        self.transform = build_transform(self.cfg, training=False)
        self.tta_modes = self.cfg.get("inference", {}).get("tta", ["identity"])
        # FastAPI可并发处理请求，锁保证同一时刻只有一个请求进入GPU关键区。
        self.lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        """返回健康检查信息，不重新加载模型也不执行图片推理。"""
        return {
            "status": "ok",
            "model_loaded": True,
            "device": str(self.device),
            "gpu": (
                torch.cuda.get_device_name(0)
                if self.device.type == "cuda"
                else None
            ),
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "binmap_adapter": self.binmap_adapter,
        }

    def _image_tensor(self, payload: bytes) -> torch.Tensor:
        """把上传的图片字节解码成形状 ``[1, 3, H, W]`` 的模型输入。

        PIL负责识别格式；所有图片统一转RGB，灰度图会复制为三通道。无法解码时
        转换为ValueError，交由HTTP层返回400，而不是泄露内部堆栈。
        """
        try:
            with Image.open(io.BytesIO(payload)) as image:
                return self.transform(image.convert("RGB")).unsqueeze(0)
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Uploaded file is not a supported image") from exc

    def analyze(self, payload: bytes, topk: int = 3, modality: str = "sem") -> dict[str, Any]:
        """对单张图片同时返回分类结果和L2归一化embedding。

        分类概率来自所有TTA视图logits的平均值；embedding先对每个视图归一化，
        再平均并二次归一化，可直接用于余弦相似度。返回embedding为普通Python
        list，便于JSON序列化。
        """
        # 预处理后才移动到GPU，避免解码阶段占用GPU资源。
        tensor = self._image_tensor(payload).to(self.device)
        logits = []
        features = []
        # lock串行化请求；inference_mode关闭梯度，显著降低推理显存。
        with self.lock, torch.inference_mode():
            set_binmap_adapter(self.model.backbone, modality == "binmap" and self.binmap_adapter is not None)
            for mode in self.tta_modes:
                # TTA只允许训练配置明确支持的原图、水平翻转和垂直翻转。
                if mode == "identity":
                    view = tensor
                elif mode == "hflip":
                    view = torch.flip(tensor, dims=(-1,))
                elif mode == "vflip":
                    view = torch.flip(tensor, dims=(-2,))
                else:
                    raise ValueError(f"Unsupported TTA mode: {mode}")
                # GPU上使用FP16自动混合精度；CPU上自动关闭。
                with torch.cuda.amp.autocast(
                    enabled=self.device.type == "cuda", dtype=torch.float16
                ):
                    # 同一次主干前向同时用于embedding和分类，避免重复计算。
                    feature = self.model.extract_features(view)
                    features.append(feature)
                    logits.append(self.model.forward_features(feature))

        # 先平均不同TTA视图的logits，再softmax得到类别概率。
        probabilities = torch.stack(logits).mean(0).softmax(dim=1)[0]
        # 两次L2归一化保证最终向量适合点积形式的余弦相似度检索。
        embedding = F.normalize(
            torch.stack([F.normalize(item.float(), dim=1) for item in features])
            .mean(0),
            dim=1,
        )[0]
        # topk强制限制在1到实际类别数之间，避免torch.topk越界。
        k = max(1, min(int(topk), len(self.class_names)))
        scores, indices = probabilities.topk(k)
        ranked = [
            {
                "class_index": int(index),
                "class_name": self.class_names[int(index)],
                "probability": float(score),
            }
            for score, index in zip(scores.cpu(), indices.cpu())
        ]
        # probabilities保留所有类别概率；topk提供按概率降序的紧凑结果。
        prediction = {
            "predicted_class": ranked[0]["class_name"],
            "class_name": ranked[0]["class_name"],
            "class_index": ranked[0]["class_index"],
            "confidence": ranked[0]["probability"],
            "probabilities": {
                name: float(probabilities[index])
                for index, name in enumerate(self.class_names)
            },
            "topk": ranked,
            "input_shape": list(tensor.shape),
        }
        return {"prediction": prediction, "embedding": embedding.cpu().tolist(), "embedding_model": "binmap-lora" if modality == "binmap" and self.binmap_adapter else "base-dinov2"}

    def classify(self, payload: bytes, topk: int = 3) -> dict[str, Any]:
        """只返回单图分类结果；内部复用analyze以保证分类逻辑唯一。"""
        return self.analyze(payload, topk=topk, modality="sem")["prediction"]

    def analyze_many(self, payloads: list[bytes], topk: int = 3, modality: str = "sem") -> dict[str, Any]:
        """按输入顺序处理多张图片，返回等长embeddings和predictions列表。

        当前实现逐张调用analyze，以控制显存峰值并兼容不同尺寸/格式上传；最大
        图片数由HTTP层的 ``MAX_BATCH_IMAGES`` 提前限制。
        """
        analyzed = [self.analyze(payload, topk=topk, modality=modality) for payload in payloads]
        return {
            "embeddings": [item["embedding"] for item in analyzed],
            "predictions": [item["prediction"] for item in analyzed],
            "embedding_model": analyzed[0]["embedding_model"] if analyzed else "base-dinov2",
        }
