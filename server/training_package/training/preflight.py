"""在正式训练前检查 Python、CUDA、模型权重、数据划分和前向计算链路。"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch

from src.config_utils import load_config
from src.data import (
    SurfaceDefectDataset,
    build_transform,
    load_records,
    resolve_three_way_split,
    split_distribution,
)
from src.model import DinoV2Classifier


def parse_args() -> argparse.Namespace:
    """读取待检查的 YAML 配置文件路径。"""
    parser = argparse.ArgumentParser(description="Validate the company DINOv2 training environment")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    """执行不修改训练结果的环境与数据预检，任一异常都会终止。"""
    args = parse_args()
    cfg = load_config(args.config)
    print(f"[python] {sys.version.split()[0]} | {platform.platform()}")
    print(f"[torch] {torch.__version__} | built_cuda={torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("容器内 torch.cuda.is_available() 为 False，请检查驱动和 NVIDIA Runtime")
    print(f"[gpu] count={torch.cuda.device_count()} name={torch.cuda.get_device_name(0)}")

    repo_dir = Path(cfg["model"]["repo_dir"])
    weights = Path(cfg["model"]["weights"])
    print(f"[model] repo_dir={repo_dir}")
    print(f"[model] weights={weights} size={weights.stat().st_size if weights.is_file() else 'MISSING'}")

    # 完整读取 CSV 并验证图片存在，然后确认每类覆盖 train/val/test。
    frame, raw_to_index = load_records(cfg)
    split_frame = resolve_three_way_split(frame, cfg)
    distribution = split_distribution(split_frame, list(cfg["class_names"]))
    print(f"[data] rows={len(frame)} raw_to_index={json.dumps(raw_to_index, ensure_ascii=False)}")
    print(f"[split] {json.dumps(distribution, ensure_ascii=False)}")

    # 使用一张真实图片完成特征提取和分类头前向，验证 GPU 链路与张量形状。
    model = DinoV2Classifier(cfg).cuda().eval()
    sample = SurfaceDefectDataset(frame.head(1), build_transform(cfg, training=False))[0]
    image = sample[0].unsqueeze(0).cuda(non_blocking=True)
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        features = model.extract_features(image)
        logits = model(image)
    print(f"[forward] image={tuple(image.shape)} features={tuple(features.shape)} logits={tuple(logits.shape)}")
    if not torch.isfinite(features).all() or not torch.isfinite(logits).all():
        raise RuntimeError("前向结果包含 NaN/Inf")
    print("[PASS] 服务器、GPU、公司镜像、DINOv2 权重、数据和类别配置均通过检查")


if __name__ == "__main__":
    main()
