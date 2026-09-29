"""Label-free LoRA adaptation of DINOv2-Large for wafer Bin Map retrieval.

The script samples defect-centred regions from every ``DefectMapPost`` image,
creates two geometry-preserving views of the same region and optimises an
InfoNCE objective with an embedding queue.  No class label, DN field or
train/validation split label is read.  Only LoRA matrices inserted into the
last Transformer attention blocks are trainable; the original DINOv2 weights
remain unchanged and can still be used for SEM images.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass
class TrainConfig:
    data_root: str
    output_dir: str
    repo_dir: str
    weights: str
    image_size: int = 224
    epochs: int = 12
    batch_size: int = 2
    workers: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    temperature: float = 0.09
    queue_size: int = 512
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_last_blocks: int = 6
    seed: int = 20260826
    validation_ratio: float = 0.12


class LoRALinear(nn.Module):
    """Frozen linear layer plus a low-rank residual branch."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.rank))
        self.active = True
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.base(inputs)
        if not self.active:
            return output
        residual = F.linear(F.linear(self.dropout(inputs), self.lora_a), self.lora_b)
        return output + residual * self.scaling


def inject_lora(model: nn.Module, last_blocks: int, rank: int, alpha: float, dropout: float) -> list[str]:
    """Insert adapters into qkv and output projections of the last blocks."""
    blocks = list(model.blocks)
    start = max(0, len(blocks) - int(last_blocks))
    targets = []
    for block_index in range(start, len(blocks)):
        block = blocks[block_index]
        for attribute in ("qkv", "proj"):
            original = getattr(block.attn, attribute)
            if not isinstance(original, nn.Linear):
                raise TypeError(f"blocks.{block_index}.attn.{attribute} is not Linear")
            setattr(block.attn, attribute, LoRALinear(original, rank, alpha, dropout))
            targets.append(f"blocks.{block_index}.attn.{attribute}")
    return targets


def adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if ".lora_a" in name or ".lora_b" in name
    }


def set_lora_active(model: nn.Module, active: bool) -> None:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.active = bool(active)


def defect_mask(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    vivid = (red >= 120) & (red >= green + 22) & (red >= blue + 22)
    muted = (red >= 72) & (red >= green + 9) & (red >= blue + 7)
    dark = (red + green + blue < 120) & (np.max(rgb, axis=2) - np.min(rgb, axis=2) > 12)
    return vivid | muted | dark


def defect_crop(image: Image.Image, rng: random.Random) -> Image.Image:
    """Sample all wafer positions uniformly over defect pixels, not the centre."""
    image = image.convert("RGB")
    width, height = image.size
    mask = defect_mask(image)
    ys, xs = np.nonzero(mask)
    if len(xs):
        selected = rng.randrange(len(xs))
        centre_x, centre_y = int(xs[selected]), int(ys[selected])
    else:
        centre_x, centre_y = rng.randrange(width), rng.randrange(height)
    side = int(min(width, height) * rng.uniform(0.38, 0.72))
    side = max(48, min(side, width, height))
    jitter = int(side * 0.12)
    centre_x += rng.randint(-jitter, jitter)
    centre_y += rng.randint(-jitter, jitter)
    left = max(0, min(width - side, centre_x - side // 2))
    top = max(0, min(height - side, centre_y - side // 2))
    return image.crop((left, top, left + side, top + side))


def transformed_view(crop: Image.Image, rng: random.Random, size: int) -> torch.Tensor:
    image = crop.copy()
    # Right-angle rotations and flips express wafer orientation invariance.
    image = image.rotate(rng.choice((0, 90, 180, 270)), expand=False)
    if rng.random() < 0.5:
        image = ImageOps.mirror(image)
    if rng.random() < 0.5:
        image = ImageOps.flip(image)
    # Small translations are implemented by padding and recropping.
    pad = max(2, int(min(image.size) * 0.08))
    image = ImageOps.expand(image, border=pad, fill=(0, 0, 0))
    x = rng.randint(0, 2 * pad)
    y = rng.randint(0, 2 * pad)
    image = image.crop((x, y, x + crop.width, y + crop.height))
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.88, 1.12))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.88, 1.14))
    if rng.random() < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 0.7)))
    image = ImageOps.fit(image, (size, size), method=Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


class BinMapPairDataset(Dataset):
    def __init__(self, paths: list[Path], image_size: int, seed: int):
        self.paths = paths
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        rng = random.Random(self.seed + self.epoch * 1_000_003 + index * 97)
        with Image.open(self.paths[index]) as opened:
            crop = defect_crop(opened, rng)
        return transformed_view(crop, rng, self.image_size), transformed_view(crop, rng, self.image_size), index


def collect_images(root: Path) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    preferred = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
        and path.stem.lower() == "defectmappost"
    ]
    if preferred:
        return sorted(preferred)
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions)


def load_backbone(cfg: TrainConfig) -> nn.Module:
    repo = Path(cfg.repo_dir)
    sys.path.insert(0, str(repo))
    model = torch.hub.load(str(repo), "dinov2_vitl14", source="local", pretrained=False)
    state = torch.load(cfg.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    layers = model.get_intermediate_layers(images, n=4, return_class_token=True)
    cls = torch.cat([entry[1] for entry in layers], dim=-1)
    pooled = layers[-1][0].mean(dim=1)
    return F.normalize(torch.cat([cls, pooled], dim=-1).float(), dim=-1)


class FeatureQueue:
    def __init__(self, size: int, dimension: int, device: torch.device):
        self.values = torch.empty((0, dimension), device=device)
        self.size = int(size)

    def negatives(self) -> torch.Tensor:
        return self.values

    def update(self, values: torch.Tensor) -> None:
        self.values = torch.cat([values.detach(), self.values], dim=0)[: self.size]


def contrastive_loss(left: torch.Tensor, right: torch.Tensor, queue: FeatureQueue, temperature: float) -> torch.Tensor:
    candidates = torch.cat([right, queue.negatives()], dim=0)
    logits = left @ candidates.T / temperature
    targets = torch.arange(left.shape[0], device=left.device)
    first = F.cross_entropy(logits, targets)
    reverse = F.cross_entropy(right @ left.T / temperature, targets)
    return 0.5 * (first + reverse)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> dict:
    model.eval()
    positives, negatives, correct, total = [], [], 0, 0
    for view_a, view_b, _ in loader:
        view_a, view_b = view_a.to(device), view_b.to(device)
        with torch.cuda.amp.autocast(enabled=amp, dtype=torch.float16):
            left, right = features(model, view_a), features(model, view_b)
        matrix = left @ right.T
        positives.extend(matrix.diag().cpu().tolist())
        if matrix.numel() > matrix.shape[0]:
            mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
            negatives.extend(matrix[mask].cpu().tolist())
        correct += int((matrix.argmax(dim=1) == torch.arange(matrix.shape[0], device=device)).sum())
        total += matrix.shape[0]
    pos = float(np.mean(positives)) if positives else 0.0
    neg = float(np.mean(negatives)) if negatives else 0.0
    return {"positive_similarity": pos, "negative_similarity": neg, "retrieval_gap": pos - neg, "same_source_top1": correct / max(total, 1)}


def save_checkpoint(model: nn.Module, cfg: TrainConfig, targets: list[str], path: Path, epoch: int, metrics: dict) -> None:
    torch.save({
        "format": "dinov2_vitl14_binmap_lora_v1",
        "epoch": epoch,
        "config": asdict(cfg),
        "targets": targets,
        "adapter_state": adapter_state(model),
        "metrics": metrics,
    }, path)


def train(cfg: TrainConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = collect_images(Path(cfg.data_root))
    if len(paths) < 8:
        raise RuntimeError(f"need at least 8 Bin Map images, found {len(paths)}")
    generator = torch.Generator().manual_seed(cfg.seed)
    val_count = max(4, int(len(paths) * cfg.validation_ratio))
    train_count = len(paths) - val_count
    full = BinMapPairDataset(paths, cfg.image_size, cfg.seed)
    train_set, val_set = random_split(full, [train_count, val_count], generator=generator)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("training is configured for the existing CUDA DINOv2 container")
    model = load_backbone(cfg)
    targets = inject_lora(model, cfg.lora_last_blocks, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout)
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    queue = FeatureQueue(cfg.queue_size, 5120, device)

    print(json.dumps({
        "event": "start", "images": len(paths), "train": train_count, "val": val_count,
        "device": torch.cuda.get_device_name(0), "targets": targets,
        "trainable_parameters": sum(item.numel() for item in trainable),
    }, ensure_ascii=False), flush=True)
    baseline = evaluate(model, val_loader, device, True)
    history, best_gap = [], -1e9
    print(json.dumps({"event": "baseline", **baseline}, ensure_ascii=False), flush=True)
    started = time.time()
    for epoch in range(1, cfg.epochs + 1):
        full.set_epoch(epoch)
        model.train()
        running = 0.0
        for step, (view_a, view_b, _) in enumerate(train_loader, 1):
            view_a, view_b = view_a.to(device, non_blocking=True), view_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            # The same frozen backbone (LoRA disabled) acts as a semantic
            # anchor.  This does not need a second DINOv2 copy in GPU memory.
            set_lora_active(model, False)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                reference_left = features(model, view_a)
                reference_right = features(model, view_b)
            set_lora_active(model, True)
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                left = features(model, view_a)
                right = features(model, view_b)
                retrieval_loss = contrastive_loss(left, right, queue, cfg.temperature)
                # Preserve each sample's original identity and the complete
                # pairwise geometry while learning stronger augmentation
                # invariance.  Both targets are label-free DINOv2 features.
                identity_loss = 0.5 * (
                    (1.0 - (left * reference_left).sum(dim=1)).mean()
                    + (1.0 - (right * reference_right).sum(dim=1)).mean()
                )
                structure_loss = F.mse_loss(left @ right.T, reference_left @ reference_right.T)
                loss = retrieval_loss + 0.75 * identity_loss + 2.0 * structure_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            queue.update(right)
            running += float(loss.detach())
            if step == 1 or step % max(1, len(train_loader) // 20) == 0 or step == len(train_loader):
                percent = 100.0 * step / len(train_loader)
                print(f"epoch {epoch:02d}/{cfg.epochs} | {percent:6.2f}% | step {step}/{len(train_loader)} | loss {running/step:.5f}", flush=True)
        metrics = evaluate(model, val_loader, device, True)
        metrics.update({"epoch": epoch, "train_loss": running / max(len(train_loader), 1), "elapsed_seconds": round(time.time() - started, 1)})
        history.append(metrics)
        save_checkpoint(model, cfg, targets, output / "last_adapter.pt", epoch, metrics)
        if metrics["retrieval_gap"] > best_gap:
            best_gap = metrics["retrieval_gap"]
            save_checkpoint(model, cfg, targets, output / "best_adapter.pt", epoch, metrics)
        (output / "training_history.json").write_text(json.dumps({"baseline": baseline, "epochs": history}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"event": "epoch_end", **metrics, "best_gap": best_gap}, ensure_ascii=False), flush=True)
    (output / "resolved_config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(output), "best_gap": best_gap}, ensure_ascii=False), flush=True)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--weights", default="/app/weights/dinov2_vitl14_pretrain.pth")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-last-blocks", type=int, default=6)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train(parse_args())
