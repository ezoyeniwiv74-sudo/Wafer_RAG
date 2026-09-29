"""使用训练得到的增量检查点，对单张图片或目录中的图片执行缺陷分类。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from src.config_utils import load_config
from src.data import build_transform
from src.model import DinoV2Classifier


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    """解析推理配置、输入路径、输出文件和 Top-K 数量。"""
    parser = argparse.ArgumentParser(description="Predict SEM surface-defect classes")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--topk", type=int, default=3)
    return parser.parse_args()


def collect_images(path: Path) -> list[Path]:
    """收集一个图片文件，或递归收集目录下所有受支持的图片。"""
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    raise FileNotFoundError(path)


def main() -> None:
    """加载模型与检查点，逐图推理并将分类结果写入 CSV。"""
    args = parse_args()
    cfg = load_config(args.config)
    checkpoint_path = (
        Path(args.checkpoint).resolve()
        if args.checkpoint
        else Path(cfg["output_dir"]).resolve() / "best_model.pt"
    )
    # 检查点只保存分类头及可选解冻 block；官方 DINOv2 主干仍从配置指定的权重加载。
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_names = list(checkpoint["class_names"])
    model = DinoV2Classifier(cfg)
    model.set_trainable_blocks(int(checkpoint.get("unfreeze_last_blocks", 0)))
    model.load_delta(checkpoint["model_delta"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    transform = build_transform(cfg, training=False)
    tta_modes = cfg.get("inference", {}).get("tta", ["identity"])
    images = collect_images(Path(args.input).expanduser().resolve())
    if not images:
        raise ValueError("输入目录中没有支持的图片")

    rows = []
    # inference_mode 关闭梯度记录，降低批量预测时的显存和内存开销。
    with torch.inference_mode():
        for path in images:
            with Image.open(path) as image:
                tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
            # 对配置中的各个 TTA 视图分别前向，再平均 logits 后计算概率。
            logits = []
            for mode in tta_modes:
                if mode == "identity":
                    view = tensor
                elif mode == "hflip":
                    view = torch.flip(tensor, dims=(-1,))
                elif mode == "vflip":
                    view = torch.flip(tensor, dims=(-2,))
                else:
                    raise ValueError(f"未知 TTA 模式: {mode}")
                with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=torch.float16):
                    logits.append(model(view))
            probabilities = torch.stack(logits).mean(0).softmax(dim=1)[0]
            k = min(args.topk, len(class_names))
            scores, indices = probabilities.topk(k)
            record = {
                "image": str(path),
                "predicted_class": class_names[int(indices[0])],
                "confidence": float(scores[0]),
                "topk": json.dumps(
                    [
                        {"class": class_names[int(i)], "probability": float(s)}
                        for s, i in zip(scores.cpu(), indices.cpu())
                    ],
                    ensure_ascii=False,
                ),
            }
            rows.append(record)
            print(f"{path.name}: {record['predicted_class']} ({record['confidence']:.4f})")
    output_path = Path(args.output).expanduser().resolve()
    # utf-8-sig 便于结果文件直接使用中文 Windows Excel 打开。
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
