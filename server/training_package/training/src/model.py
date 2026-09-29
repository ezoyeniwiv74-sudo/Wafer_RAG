"""构建 DINOv2 ViT-L/14 特征主干、分类头及增量检查点接口。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class DinoV2Classifier(nn.Module):
    """以本地官方 DINOv2 为主干、以轻量线性头为输出的缺陷分类器。"""

    def __init__(self, cfg: dict[str, Any]):
        """加载官方主干权重，并按配置创建 LayerNorm、Dropout 和线性分类层。"""
        super().__init__()
        model_cfg = cfg["model"]
        repo_dir = Path(model_cfg["repo_dir"]).expanduser().resolve()
        weights = Path(model_cfg["weights"]).expanduser().resolve()
        if not (repo_dir / "hubconf.py").is_file():
            raise FileNotFoundError(f"找不到 DINOv2 官方源码: {repo_dir}")
        if not weights.is_file():
            raise FileNotFoundError(f"找不到 DINOv2 官方权重: {weights}")

        self.feature_layers = int(model_cfg.get("feature_layers", 4))
        self.include_patch_mean = bool(model_cfg.get("include_patch_mean", True))
        # source="local" 保证服务器断网时只使用交付包中的官方源码。
        self.backbone = torch.hub.load(
            str(repo_dir),
            model_cfg.get("name", "dinov2_vitl14"),
            source="local",
            pretrained=False,
        )
        state_dict = torch.load(weights, map_location="cpu")
        self.backbone.load_state_dict(state_dict, strict=True)
        embed_dim = int(self.backbone.embed_dim)
        feature_dim = embed_dim * (self.feature_layers + int(self.include_patch_mean))
        self.feature_dim = feature_dim
        # 多层 CLS token 与最后一层 patch 均值拼接后，进入轻量分类头。
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(float(model_cfg.get("dropout", 0.2))),
            nn.Linear(feature_dim, int(cfg["num_classes"])),
        )
        self.set_trainable_blocks(0)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """提取最后若干 Transformer 层的 CLS token，并可拼接 patch 均值。"""
        outputs = self.backbone.get_intermediate_layers(
            images,
            n=self.feature_layers,
            reshape=False,
            return_class_token=True,
            norm=True,
        )
        patch_tokens = [item[0] for item in outputs]
        class_tokens = [item[1] for item in outputs]
        pieces = list(class_tokens)
        if self.include_patch_mean:
            pieces.append(patch_tokens[-1].mean(dim=1))
        return torch.cat(pieces, dim=-1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """对原始图片执行主干特征提取和分类，返回未归一化 logits。"""
        return self.head(self.extract_features(images))

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        """跳过冻结主干，直接用缓存特征训练或评估分类头。"""
        return self.head(features)

    def set_trainable_blocks(self, count: int) -> None:
        """冻结主干，仅按需解冻最后 count 个 block；分类头始终可训练。"""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if count > 0:
            blocks = list(self.backbone.blocks)
            if count > len(blocks):
                raise ValueError(f"unfreeze_last_blocks={count}，但模型只有 {len(blocks)} 个 block")
            for block in blocks[-count:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            for parameter in self.backbone.norm.parameters():
                parameter.requires_grad = True
        for parameter in self.head.parameters():
            parameter.requires_grad = True

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """仅导出当前可训练参数，形成依赖官方主干权重的增量检查点。"""
        trainable = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if name in trainable
        }

    def load_delta(self, state: dict[str, torch.Tensor]) -> None:
        """非严格加载增量参数，同时拒绝检查点中不存在于当前模型的键。"""
        incompatible = self.load_state_dict(state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        if unexpected:
            raise RuntimeError(f"checkpoint 包含模型中不存在的参数: {unexpected}")


def parameter_groups(
    model: DinoV2Classifier,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """为解冻主干和分类头建立不同学习率的 AdamW 参数组。"""
    backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    head = [p for p in model.head.parameters() if p.requires_grad]
    groups: list[dict[str, Any]] = []
    if backbone:
        groups.append({"params": backbone, "lr": backbone_lr, "weight_decay": weight_decay})
    groups.append({"params": head, "lr": head_lr, "weight_decay": weight_decay})
    return groups
