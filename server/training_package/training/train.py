"""DINOv2 缺陷分类主训练入口：数据划分、分类头训练、可选微调与测试。"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config_utils import load_config, save_json, seed_everything, to_jsonable
from src.data import (
    FeatureDataset,
    SurfaceDefectDataset,
    build_transform,
    load_records,
    make_weighted_sampler,
    resolve_three_way_split,
    split_distribution,
)
from src.engine import (
    detailed_report,
    evaluate,
    make_cosine_scheduler,
    save_confusion_matrix,
    train_epoch,
)
from src.model import DinoV2Classifier, parameter_groups


def parse_args() -> argparse.Namespace:
    """解析训练配置、特征缓存重建和冒烟测试开关。"""
    parser = argparse.ArgumentParser(description="DINOv2 Large SEM defect classifier")
    parser.add_argument("--config", default="config.yaml", help="YAML configuration")
    parser.add_argument("--rebuild-cache", action="store_true", help="Re-extract backbone features")
    parser.add_argument("--smoke-test", action="store_true", help="Run two short epochs to verify the pipeline")
    return parser.parse_args()


def dataloader(
    dataset,
    batch_size: int,
    workers: int,
    shuffle: bool = False,
    sampler=None,
) -> DataLoader:
    """按统一策略创建 DataLoader，并在有采样器时自动关闭 shuffle。"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=False,
    )


def cache_fingerprint(cfg: dict[str, Any], frame: pd.DataFrame) -> str:
    """根据数据、权重和特征配置生成缓存指纹，避免误用旧特征。"""
    csv_path = Path(cfg["data"]["csv_path"])
    weight_path = Path(cfg["model"]["weights"])
    payload = {
        "csv": str(csv_path.resolve()),
        "csv_size": csv_path.stat().st_size,
        "weights_size": weight_path.stat().st_size,
        "image_size": cfg["model"]["image_size"],
        "feature_layers": cfg["model"]["feature_layers"],
        "patch_mean": cfg["model"]["include_patch_mean"],
        "rows": len(frame),
        "split": frame["split"].tolist(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


@torch.inference_mode()
def extract_features(
    model: DinoV2Classifier,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    views: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """用冻结主干批量提取一个或多个增强视图的 CPU 特征缓存。"""
    model.eval()
    feature_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    # 多视图会重复遍历训练集，目标标签也同步重复，作为特征级数据增强。
    for view in range(views):
        for batch in tqdm(loader, desc=f"extract view {view + 1}/{views}", leave=False):
            images, targets = batch[0].to(device, non_blocking=True), batch[1]
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda", dtype=torch.float16):
                features = model.extract_features(images)
            feature_parts.append(features.float().cpu())
            target_parts.append(targets.long().cpu())
    return torch.cat(feature_parts), torch.cat(target_parts)


def get_or_build_feature_cache(
    cfg: dict[str, Any],
    model: DinoV2Classifier,
    split_frame: pd.DataFrame,
    device: torch.device,
    output_dir: Path,
    rebuild: bool,
) -> dict[str, FeatureDataset]:
    """加载匹配指纹的缓存，或为 train/val/test 重新提取并保存特征。"""
    train_cfg = cfg["training"]
    batch_size = int(train_cfg["batch_size"])
    workers = int(train_cfg["num_workers"])
    amp = bool(train_cfg["amp"])
    fingerprint = cache_fingerprint(cfg, split_frame)
    cache_path = output_dir / f"feature_cache_{fingerprint}.pt"
    if cache_path.is_file() and not rebuild:
        print(f"[cache] Loading {cache_path}")
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
    else:
        cached: dict[str, dict[str, torch.Tensor]] = {}
        # 训练集可生成多个随机增强视图；验证和测试只保留确定性视图。
        for split_name in ("train", "val", "test"):
            subset = split_frame[split_frame["split"] == split_name]
            is_train = split_name == "train"
            dataset = SurfaceDefectDataset(subset, build_transform(cfg, training=is_train))
            loader = dataloader(dataset, batch_size, workers, shuffle=False)
            views = int(train_cfg.get("feature_cache_train_views", 1)) if is_train else 1
            features, targets = extract_features(model, loader, device, amp, views)
            cached[split_name] = {"features": features, "targets": targets}
        torch.save(cached, cache_path)
        print(f"[cache] Saved {cache_path}")
    return {
        name: FeatureDataset(payload["features"], payload["targets"])
        for name, payload in cached.items()
    }


def save_checkpoint(
    model: DinoV2Classifier,
    cfg: dict[str, Any],
    path: Path,
    raw_to_index: dict[str, int],
    metrics: dict[str, float],
    epoch: int,
    phase: str,
    unfreeze_last_blocks: int,
) -> None:
    """保存最优可训练增量参数、类别映射、指标和模型结构元数据。"""
    checkpoint = {
        "format_version": 1,
        "model_delta": model.trainable_state_dict(),
        "class_names": cfg["class_names"],
        "raw_to_index": raw_to_index,
        "metrics": metrics,
        "epoch": epoch,
        "phase": phase,
        "unfreeze_last_blocks": unfreeze_last_blocks,
        "model_config": cfg["model"],
    }
    torch.save(checkpoint, path)


def run_training_phase(
    *,
    model: DinoV2Classifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    patience: int,
    metric_name: str,
    device: torch.device,
    amp: bool,
    accumulation: int,
    forward_train,
    forward_val,
    cfg: dict[str, Any],
    checkpoint_path: Path,
    raw_to_index: dict[str, int],
    history: list[dict[str, Any]],
    phase: str,
    unfreeze_last_blocks: int,
    best_score: float,
) -> float:
    """执行一个训练阶段，并按验证指标保存最优模型及执行早停。"""
    scheduler = make_cosine_scheduler(optimizer, epochs)
    stale = 0
    for epoch in range(1, epochs + 1):
        started = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            amp,
            accumulation,
            forward_train,
        )
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device, amp, forward_val)
        scheduler.step()
        row = {
            "phase": phase,
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "seconds": time.time() - started,
        }
        history.append(row)
        print(
            f"[{phase}] epoch {epoch:03d}/{epochs} "
            f"loss={train_loss:.4f} macro_f1={val_metrics['macro_f1']:.4f} "
            f"bal_acc={val_metrics['balanced_accuracy']:.4f} acc={val_metrics['accuracy']:.4f}"
        )
        # 只有验证指标严格提升时覆盖 best_model.pt，避免测试集参与模型选择。
        score = float(val_metrics[metric_name])
        if score > best_score:
            best_score = score
            stale = 0
            save_checkpoint(
                model,
                cfg,
                checkpoint_path,
                raw_to_index,
                val_metrics,
                epoch,
                phase,
                unfreeze_last_blocks,
            )
            print(f"[best] {metric_name}={best_score:.4f} -> {checkpoint_path}")
        else:
            stale += 1
            if stale >= patience:
                print(f"[early-stop] {phase}: {patience} epochs without improvement")
                break
    return best_score


def main() -> None:
    """组织完整训练流程并输出可复现配置、指标、报告和最优检查点。"""
    args = parse_args()
    cfg = load_config(args.config)
    # 冒烟模式仅验证数据到模型输出的完整链路，不用于评估业务精度。
    if args.smoke_test:
        cfg["training"]["head_epochs"] = 2
        cfg["training"]["finetune_epochs"] = 0
        cfg["training"]["feature_cache_train_views"] = 1
        cfg["training"]["early_stopping_patience"] = 2
        normal_output = Path(cfg["output_dir"])
        cfg["output_dir"] = str(normal_output.parent / f"{normal_output.name}_smoke_test")

    seed = int(cfg["split"]["seed"])
    seed_everything(seed)
    output_dir = Path(cfg["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(to_jsonable(cfg), output_dir / "resolved_config.json")

    # 先固定并落盘数据划分，保证后续复现实验时能追溯每张图片的归属。
    frame, raw_to_index = load_records(cfg)
    split_frame = resolve_three_way_split(frame, cfg)
    if args.smoke_test:
        # 每个集合、每个类别最多保留两张，使冒烟训练足够短且仍覆盖全部类别。
        split_frame = (
            split_frame.groupby(["split", "target"], group_keys=False, sort=False)
            .head(2)
            .reset_index(drop=True)
        )
    split_frame[["resolved_path", "raw_label", "target", "split"]].to_csv(
        output_dir / "splits.csv", index=False, encoding="utf-8-sig"
    )
    distribution = split_distribution(split_frame, cfg["class_names"])
    save_json(distribution, output_dir / "split_distribution.json")
    print("[data]", json.dumps(distribution, ensure_ascii=False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name(0)} | torch={torch.__version__}")
    else:
        print("[warning] CUDA unavailable; DINOv2 Large will be very slow on CPU")

    model = DinoV2Classifier(cfg).to(device)
    train_cfg = cfg["training"]
    criterion = nn.CrossEntropyLoss(label_smoothing=float(train_cfg["label_smoothing"]))
    checkpoint_path = output_dir / "best_model.pt"
    history: list[dict[str, Any]] = []
    best_score = -1.0

    # 阶段一：冻结 DINOv2 主干，用缓存特征快速训练轻量分类头。
    if bool(train_cfg.get("cache_backbone_features", True)):
        feature_sets = get_or_build_feature_cache(
            cfg, model, split_frame, device, output_dir, args.rebuild_cache
        )
        sampler = make_weighted_sampler(
            feature_sets["train"].targets, float(train_cfg.get("sampler_power", 0))
        )
        train_loader = dataloader(
            feature_sets["train"], int(train_cfg["feature_batch_size"]), 0, sampler=sampler
        )
        val_loader = dataloader(feature_sets["val"], int(train_cfg["feature_batch_size"]), 0)
        optimizer = torch.optim.AdamW(
            model.head.parameters(),
            lr=float(train_cfg["head_lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        head_epochs = int(train_cfg["head_epochs"])
        best_score = run_training_phase(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=head_epochs,
            patience=int(train_cfg["early_stopping_patience"]),
            metric_name=train_cfg["selection_metric"],
            device=device,
            amp=bool(train_cfg["amp"]),
            accumulation=1,
            forward_train=model.forward_features,
            forward_val=model.forward_features,
            cfg=cfg,
            checkpoint_path=checkpoint_path,
            raw_to_index=raw_to_index,
            history=history,
            phase="head",
            unfreeze_last_blocks=0,
            best_score=best_score,
        )

    unfreeze = int(train_cfg.get("unfreeze_last_blocks", 0))
    finetune_epochs = int(train_cfg.get("finetune_epochs", 0))
    # 阶段二（可选）：解冻末尾 Transformer blocks，用较小学习率端到端微调。
    if unfreeze > 0 and finetune_epochs > 0:
        model.set_trainable_blocks(unfreeze)
        train_data = SurfaceDefectDataset(
            split_frame[split_frame["split"] == "train"], build_transform(cfg, training=True)
        )
        val_data = SurfaceDefectDataset(
            split_frame[split_frame["split"] == "val"], build_transform(cfg, training=False)
        )
        sampler = make_weighted_sampler(train_data.targets, float(train_cfg.get("sampler_power", 0)))
        train_loader = dataloader(
            train_data,
            int(train_cfg["batch_size"]),
            int(train_cfg["num_workers"]),
            sampler=sampler,
        )
        val_loader = dataloader(
            val_data,
            int(train_cfg["batch_size"]),
            int(train_cfg["num_workers"]),
        )
        optimizer = torch.optim.AdamW(
            parameter_groups(
                model,
                float(train_cfg["backbone_lr"]),
                float(train_cfg["finetune_head_lr"]),
                float(train_cfg["weight_decay"]),
            )
        )
        best_score = run_training_phase(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=finetune_epochs,
            patience=int(train_cfg["early_stopping_patience"]),
            metric_name=train_cfg["selection_metric"],
            device=device,
            amp=bool(train_cfg["amp"]),
            accumulation=int(train_cfg["gradient_accumulation"]),
            forward_train=model,
            forward_val=model,
            cfg=cfg,
            checkpoint_path=checkpoint_path,
            raw_to_index=raw_to_index,
            history=history,
            phase="finetune",
            unfreeze_last_blocks=unfreeze,
            best_score=best_score,
        )

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    if not checkpoint_path.is_file():
        raise RuntimeError("训练未生成 checkpoint；请检查 head_epochs/finetune_epochs")

    # 从官方权重重建，再加载最优增量参数，防止后续阶段污染先前保存的最优模型。
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    best_model = DinoV2Classifier(cfg)
    best_model.set_trainable_blocks(int(checkpoint["unfreeze_last_blocks"]))
    best_model.load_delta(checkpoint["model_delta"])
    best_model = best_model.to(device)

    test_data = SurfaceDefectDataset(
        split_frame[split_frame["split"] == "test"], build_transform(cfg, training=False)
    )
    test_loader = dataloader(
        test_data,
        int(train_cfg["batch_size"]),
        int(train_cfg["num_workers"]),
    )
    tta_modes = cfg.get("inference", {}).get("tta", ["identity"])

    def tta_forward(images: torch.Tensor) -> torch.Tensor:
        """对配置指定的原图/翻转视图取平均 logits，用于最终测试评估。"""
        logits = []
        for mode in tta_modes:
            if mode == "identity":
                view = images
            elif mode == "hflip":
                view = torch.flip(images, dims=(-1,))
            elif mode == "vflip":
                view = torch.flip(images, dims=(-2,))
            else:
                raise ValueError(f"未知 TTA 模式: {mode}")
            logits.append(best_model(view))
        return torch.stack(logits).mean(dim=0)

    # 测试集仅在所有训练阶段结束后评估一次，不参与早停和模型选择。
    test_metrics, y_true, y_pred = evaluate(
        best_model,
        test_loader,
        criterion,
        device,
        bool(train_cfg["amp"]),
        tta_forward,
    )
    report = detailed_report(y_true, y_pred, cfg["class_names"])
    save_json(test_metrics, output_dir / "test_metrics.json")
    save_json(report, output_dir / "classification_report.json")
    save_confusion_matrix(y_true, y_pred, cfg["class_names"], output_dir / "confusion_matrix.png")
    print("[test]", json.dumps(test_metrics, ensure_ascii=False, indent=2))
    print(f"[done] Outputs: {output_dir}")


if __name__ == "__main__":
    # Windows/WSL 下多进程 DataLoader 需要在入口处启用 freeze_support。
    torch.multiprocessing.freeze_support()
    main()
