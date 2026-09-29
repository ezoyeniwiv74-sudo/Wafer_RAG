"""训练配置加载、合法性检查、随机种子和 JSON 序列化辅助函数。"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置，记录其绝对路径，并在返回前执行完整校验。"""
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    """校验类别、输入尺寸和三集合比例等会影响训练正确性的关键字段。"""
    names = cfg.get("class_names", [])
    count = int(cfg.get("num_classes", 0))
    if count <= 1:
        raise ValueError("num_classes 必须大于 1")
    if len(names) != count:
        raise ValueError(f"class_names 有 {len(names)} 项，但 num_classes={count}")
    size = int(cfg["model"]["image_size"])
    if size % 14:
        raise ValueError(f"DINOv2 ViT-L/14 的 image_size 必须能被 14 整除，当前为 {size}")
    val = float(cfg["split"]["val_ratio"])
    test = float(cfg["split"]["test_ratio"])
    if val <= 0 or test <= 0 or val + test >= 1:
        raise ValueError("val_ratio 和 test_ratio 必须大于 0，且两者之和小于 1")


def seed_everything(seed: int) -> None:
    """同步设置 Python、NumPy、CPU 与全部 CUDA 设备的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def save_json(data: Any, path: str | Path) -> None:
    """以 UTF-8 和便于审阅的缩进格式保存 JSON，并自动创建父目录。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def to_jsonable(value: Any) -> Any:
    """递归将 Path 和 NumPy 标量转换为标准 JSON 可序列化对象。"""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value
