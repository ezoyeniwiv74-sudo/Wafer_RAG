"""CSV 数据读取、路径解析、数据划分、图像增强和类别重采样。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _label_sort_key(value: str) -> tuple[int, float | str]:
    """数字标签按数值排序，非数字标签按字符串排序，确保映射稳定。"""
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def load_records(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    """加载 CSV、建立原始标签到整数索引的映射，并验证全部图片路径。"""
    data_cfg = cfg["data"]
    csv_path = Path(data_cfg["csv_path"]).expanduser().resolve()
    image_root = Path(data_cfg["image_root"]).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"找不到 CSV: {csv_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"找不到图片根目录: {image_root}")

    frame = pd.read_csv(csv_path, sep=data_cfg.get("csv_separator", ","), dtype=str)
    image_col = data_cfg["image_column"]
    label_col = data_cfg["label_column"]
    missing_columns = [c for c in (image_col, label_col) if c not in frame.columns]
    if missing_columns:
        raise ValueError(f"CSV 缺少字段: {missing_columns}; 实际字段: {list(frame.columns)}")

    # 丢弃缺少图片或标签的无效行，保留原始标签用于审计与结果回溯。
    frame = frame.dropna(subset=[image_col, label_col]).copy()
    frame["raw_label"] = frame[label_col].astype(str).str.strip()
    raw_labels = sorted(frame["raw_label"].unique().tolist(), key=_label_sort_key)
    if len(raw_labels) != int(cfg["num_classes"]):
        raise ValueError(
            f"CSV 中发现 {len(raw_labels)} 类 {raw_labels}，但 num_classes={cfg['num_classes']}"
        )
    raw_to_index = {label: i for i, label in enumerate(raw_labels)}
    frame["target"] = frame["raw_label"].map(raw_to_index).astype(int)

    def resolve_image(value: str) -> str:
        """将 CSV 中的相对路径统一解析为图片根目录下的绝对路径。"""
        candidate = Path(value.replace("\\", "/"))
        if not candidate.is_absolute():
            candidate = image_root / candidate
        return str(candidate.resolve())

    frame["resolved_path"] = frame[image_col].astype(str).map(resolve_image)
    missing = [p for p in frame["resolved_path"] if not Path(p).is_file()]
    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(f"有 {len(missing)} 张图片不存在，示例:\n{preview}")
    if frame.empty:
        raise ValueError("CSV 中没有可用记录")
    return frame.reset_index(drop=True), raw_to_index


def stratified_three_way_split(
    frame: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """逐类别随机划分，并保证样本数不少于3时三集合各至少有一张。"""
    rng = np.random.default_rng(seed)
    split = np.full(len(frame), "train", dtype=object)
    for _, group in frame.groupby("target", sort=True):
        indices = group.index.to_numpy(copy=True)
        rng.shuffle(indices)
        n = len(indices)
        if n < 3:
            raise ValueError(f"每类至少需要 3 张图片才能划分 train/val/test，发现某类只有 {n} 张")
        n_val = max(1, int(round(n * val_ratio)))
        n_test = max(1, int(round(n * test_ratio)))
        # 小类别优先缩减验证/测试数量，始终为训练集保留至少一张图片。
        while n_val + n_test > n - 1:
            if n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break
        split[indices[:n_val]] = "val"
        split[indices[n_val : n_val + n_test]] = "test"
    result = frame.copy()
    result["split"] = split
    return result


def resolve_three_way_split(
    frame: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """优先使用 CSV 中已审计的 split 字段，否则执行逐类别随机划分。"""
    data_cfg = cfg["data"]
    split_column = data_cfg.get("split_column")
    if split_column in (None, "", False):
        return stratified_three_way_split(
            frame,
            float(cfg["split"]["val_ratio"]),
            float(cfg["split"]["test_ratio"]),
            int(cfg["split"]["seed"]),
        )
    if split_column not in frame.columns:
        raise ValueError(
            f"配置了 data.split_column={split_column!r}，但 CSV 中没有该字段；"
            f"实际字段: {list(frame.columns)}"
        )

    result = frame.copy()
    result["split"] = result[split_column].astype(str).str.strip().str.lower()
    allowed = {"train", "val", "test"}
    unexpected = sorted(set(result["split"].unique()) - allowed)
    if unexpected:
        raise ValueError(f"{split_column} 只能包含 train/val/test，发现: {unexpected}")

    # 显式划分必须让每个类别都出现在三集合中，否则指标可能失真。
    missing_pairs: list[str] = []
    for target, group in result.groupby("target", sort=True):
        present = set(group["split"].unique())
        missing = sorted(allowed - present)
        if missing:
            missing_pairs.append(f"target={target} 缺少 {missing}")
    if missing_pairs:
        raise ValueError("CSV 显式划分要求每类都出现在 train/val/test：" + "; ".join(missing_pairs))
    return result.reset_index(drop=True)


def build_transform(cfg: dict[str, Any], training: bool) -> transforms.Compose:
    """构建 DINOv2 输入预处理；训练态额外启用配置中的灰度友好增强。"""
    size = int(cfg["model"]["image_size"])
    ops: list[Any] = [
        transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC, antialias=True),
    ]
    if training:
        aug = cfg["augmentation"]
        if aug.get("horizontal_flip", True):
            ops.append(transforms.RandomHorizontalFlip())
        if aug.get("vertical_flip", True):
            ops.append(transforms.RandomVerticalFlip())
        degrees = float(aug.get("rotation_degrees", 0))
        if degrees:
            ops.append(
                transforms.RandomRotation(
                    degrees=degrees,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=128,
                )
            )
        p_contrast = float(aug.get("autocontrast_probability", 0))
        if p_contrast:
            ops.append(transforms.RandomAutocontrast(p=p_contrast))
        p_blur = float(aug.get("blur_probability", 0))
        if p_blur:
            ops.append(transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 0.8))], p=p_blur))
    # 官方 DINOv2 使用 ImageNet 归一化参数；输入最终形状为 [3, H, W]。
    ops.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(ops)


class SurfaceDefectDataset(Dataset):
    """从 DataFrame 按需读取原始图片并返回张量、类别索引和源路径。"""

    def __init__(self, frame: pd.DataFrame, transform: transforms.Compose):
        """保存重置索引后的记录表、预处理流水线和采样器所需标签。"""
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.targets = self.frame["target"].astype(int).tolist()

    def __len__(self) -> int:
        """返回当前数据集中的图片数量。"""
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        """读取一张图片，转换成 RGB 后应用预处理并返回其标签。"""
        row = self.frame.iloc[index]
        path = str(row["resolved_path"])
        with Image.open(path) as image:
            # DINOv2 was pretrained on RGB. Grayscale is repeated into three channels.
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, int(row["target"]), path


class FeatureDataset(Dataset):
    """保存离线提取的 DINOv2 特征，用于只训练分类头的快速阶段。"""

    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        """统一特征和标签数据类型，并提供列表标签给加权采样器。"""
        self.features = features.float()
        self.targets_tensor = targets.long()
        self.targets = self.targets_tensor.tolist()

    def __len__(self) -> int:
        """返回缓存特征的样本数量。"""
        return len(self.targets_tensor)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """返回指定样本的特征向量和整数标签。"""
        return self.features[index], int(self.targets_tensor[index])


def make_weighted_sampler(targets: list[int], power: float) -> WeightedRandomSampler | None:
    """按类别频次的负幂生成采样权重；power<=0 时关闭重采样。"""
    if power <= 0:
        return None
    counts = Counter(targets)
    sample_weights = [counts[y] ** (-power) for y in targets]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(targets),
        replacement=True,
    )


def split_distribution(frame: pd.DataFrame, class_names: list[str]) -> dict[str, dict[str, int]]:
    """统计 train/val/test 中各业务类别的样本数，供日志和审计报告使用。"""
    result: dict[str, dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        counts = frame.loc[frame["split"] == split_name, "target"].value_counts().to_dict()
        result[split_name] = {class_names[i]: int(counts.get(i, 0)) for i in range(len(class_names))}
    return result
