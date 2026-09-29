"""训练循环、评估指标、学习率调度和混淆矩阵输出。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from tqdm import tqdm


def make_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int = 2,
) -> torch.optim.lr_scheduler.LambdaLR:
    """创建带线性预热的余弦退火学习率调度器。"""

    def multiplier(epoch: int) -> float:
        """返回当前 epoch 相对初始学习率的倍率。"""
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def train_epoch(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    accumulation: int,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
) -> float:
    """训练一个 epoch，支持 AMP、梯度累积和梯度范数裁剪。"""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    total_loss = 0.0
    total_items = 0
    batches = len(loader)  # type: ignore[arg-type]
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        inputs, targets = batch[0].to(device), batch[1].to(device)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda", dtype=torch.float16):
            logits = forward_fn(inputs)
            raw_loss = criterion(logits, targets)
            loss = raw_loss / accumulation
        # 先按累积步数缩放损失；达到边界后才更新一次参数。
        scaler.scale(loss).backward()
        if (step + 1) % accumulation == 0 or step + 1 == batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(raw_loss.detach()) * len(targets)
        total_items += len(targets)
    return total_loss / max(1, total_items)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """在无梯度模式下评估模型，返回指标以及真实/预测标签数组。"""
    model.eval()
    losses: list[float] = []
    all_targets: list[int] = []
    all_predictions: list[int] = []
    for batch in tqdm(loader, desc="eval", leave=False):
        inputs, targets = batch[0].to(device), batch[1].to(device)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda", dtype=torch.float16):
            logits = forward_fn(inputs)
            loss = criterion(logits, targets)
        losses.extend([float(loss)] * len(targets))
        all_targets.extend(targets.cpu().tolist())
        all_predictions.extend(logits.argmax(dim=1).cpu().tolist())
    y_true = np.asarray(all_targets, dtype=np.int64)
    y_pred = np.asarray(all_predictions, dtype=np.int64)
    # 类别极不平衡时，balanced_accuracy 与 macro_f1 比普通准确率更有参考价值。
    metrics = {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    return metrics, y_true, y_pred


def detailed_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    """生成包含各类别 precision、recall 和 F1 的结构化报告。"""
    labels = list(range(len(class_names)))
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    path: str | Path,
) -> None:
    """按真实类别归一化混淆矩阵并保存为 PNG，便于定位类别混淆。"""
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(class_names)), normalize="true")
    fig_size = max(7, len(class_names) * 0.9)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title="Normalized confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.5 else "black")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
