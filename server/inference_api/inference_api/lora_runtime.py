"""Minimal LoRA runtime used by the Bin Map retrieval adapter."""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for parameter in base.parameters():
            parameter.requires_grad = False
        self.rank = int(rank)
        self.scaling = float(alpha) / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Parameter(torch.empty(self.rank, base.in_features), requires_grad=False)
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.rank), requires_grad=False)
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.active = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.base(inputs)
        if not self.active:
            return output
        residual = F.linear(F.linear(self.dropout(inputs), self.lora_a), self.lora_b)
        return output + residual * self.scaling


def load_binmap_adapter(backbone: nn.Module, path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "dinov2_vitl14_binmap_lora_v1":
        raise ValueError("Unsupported Bin Map adapter format")
    cfg = checkpoint["config"]
    blocks = list(backbone.blocks)
    start = len(blocks) - int(cfg["lora_last_blocks"])
    for block_index in range(start, len(blocks)):
        for attribute in ("qkv", "proj"):
            original = getattr(blocks[block_index].attn, attribute)
            setattr(blocks[block_index].attn, attribute, LoRALinear(
                original, cfg["lora_rank"], cfg["lora_alpha"], 0.0,
            ))
    incompatible = backbone.load_state_dict(checkpoint["adapter_state"], strict=False)
    unexpected = [name for name in incompatible.unexpected_keys if "lora_" in name]
    if unexpected:
        raise ValueError(f"Unexpected adapter keys: {unexpected[:3]}")
    return {"path": str(path), "epoch": checkpoint.get("epoch"), "metrics": checkpoint.get("metrics", {})}


def set_binmap_adapter(backbone: nn.Module, active: bool) -> None:
    for module in backbone.modules():
        if isinstance(module, LoRALinear):
            module.active = bool(active)
