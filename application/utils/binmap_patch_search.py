"""Bin Map 局部缺陷块检索。

本模块不训练或下载任何模型。它先使用颜色和连通区域从 Bin Map 中找出
缺陷聚集区域，再调用项目现有 DINOv2 API 提取整图/局部块向量。查询图会
生成 0/90/180/270 度旋转版本，历史图只建一次索引，从而在控制 API 调用量
的同时提高旋转后的检索稳定性。
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import sys
from collections import deque
from pathlib import Path
from urllib.parse import quote

import numpy as np
try:
    from PIL import Image, ImageDraw, ImageFilter
except ModuleNotFoundError:
    # 交付目录复用原项目图像分类模块已经使用的 Pillow；无需联网或 pip 安装。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_vendor"))
    from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
PATCH_INDEX_PATH = Path(os.getenv(
    "WAFER_BINMAP_PATCH_INDEX", ROOT / ".cache" / "yedn_binmap_patch_index.npz"
))
MAX_BLOCKS_PER_IMAGE = int(os.getenv("WAFER_BINMAP_MAX_BLOCKS", "3"))
ROTATIONS = (0, 90, 180, 270)


def _open_rgb(image_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(image_bytes)) as opened:
        return opened.convert("RGB")


def _encode_png(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _data_url(image: Image.Image) -> str:
    return "data:image/png;base64," + base64.b64encode(_encode_png(image)).decode("ascii")


def standardized_region_crop(
    image: Image.Image,
    box: tuple[int, int, int, int],
    selection_mask: Image.Image | None = None,
) -> Image.Image:
    """Return the actual white-background image used for local retrieval.

    Bin Map backgrounds (black canvas, grey wafer and photographed grid) carry
    more pixels than the defect marks and can dominate an embedding.  Keep the
    red/brown defect pixels inside the selected region, remove unrelated
    background, then centre the result on a square white canvas without
    stretching its aspect ratio.  If no defect-coloured pixels are found, keep
    the selected content so a manually selected non-standard map still works.
    """
    crop = image.crop(box).convert("RGB")
    pixels = np.asarray(crop, dtype=np.uint8)
    keep = defect_mask(crop)
    if selection_mask is not None:
        selected = np.asarray(selection_mask.convert("L"), dtype=np.uint8) > 0
        keep &= selected
    if int(keep.sum()) == 0:
        keep = np.ones(keep.shape, dtype=bool)
        if selection_mask is not None:
            keep &= np.asarray(selection_mask.convert("L"), dtype=np.uint8) > 0
    isolated = np.full_like(pixels, 255)
    isolated[keep] = pixels[keep]
    isolated_image = Image.fromarray(isolated, mode="RGB")
    side = max(isolated_image.size)
    canvas = Image.new("RGB", (side, side), "white")
    canvas.paste(isolated_image, ((side - crop.width) // 2, (side - crop.height) // 2))
    if side > 512:
        canvas.thumbnail((512, 512), Image.Resampling.LANCZOS)
    return canvas


def defect_mask(image: Image.Image) -> np.ndarray:
    """提取模拟图和真实 PPT 截图中的红/褐色缺陷标记。"""
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    vivid_red = (red >= 125) & (red >= green + 25) & (red >= blue + 25)
    # 公司 PPT 中的晶圆截图经过压缩和拍摄，缺陷点常呈低饱和褐红色。
    muted_red = (red >= 75) & (red >= green + 10) & (red >= blue + 8) & (green <= red * 0.93)
    return vivid_red | muted_red


def is_binmap_image(image_bytes: bytes) -> bool:
    """兼容黑底模拟图和浅色网格晶圆截图，排除灰度 SEM 图。"""
    try:
        image = _open_rgb(image_bytes)
    except Exception:
        return False
    thumb = image.copy()
    thumb.thumbnail((256, 256))
    rgb = np.asarray(thumb, dtype=np.uint8)
    mask = defect_mask(thumb)
    dark_ratio = float(np.mean(np.max(rgb, axis=2) < 35))
    quantized = (rgb // 24).reshape(-1, 3)
    color_count = len(np.unique(quantized, axis=0))
    gray_ratio = float(np.mean(np.max(rgb, axis=2) - np.min(rgb, axis=2) < 14))
    defect_ratio = float(mask.mean())
    simulated_map = defect_ratio >= 0.001 and dark_ratio >= 0.03 and color_count <= 100
    photographed_map = defect_ratio >= 0.0007 and gray_ratio >= 0.58 and color_count <= 150
    return simulated_map or photographed_map


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """返回二值图中的 8 邻域连通区域：x0, y0, x1, y1, 像素数。"""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for y, x in np.argwhere(mask):
        if seen[y, x]:
            continue
        queue = deque([(int(y), int(x))])
        seen[y, x] = True
        min_x = max_x = int(x)
        min_y = max_y = int(y)
        count = 0
        while queue:
            cy, cx = queue.popleft()
            count += 1
            min_x, max_x = min(min_x, cx), max(max_x, cx)
            min_y, max_y = min(min_y, cy), max(max_y, cy)
            for ny in range(max(0, cy - 1), min(height, cy + 2)):
                for nx in range(max(0, cx - 1), min(width, cx + 2)):
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
        result.append((min_x, min_y, max_x + 1, max_y + 1, count))
    return result


def _square_box(box, width: int, height: int, minimum: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    size = max(x1 - x0, y1 - y0, minimum)
    size = min(size, width, height)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = int(round(cx - size / 2))
    top = int(round(cy - size / 2))
    left = max(0, min(left, width - size))
    top = max(0, min(top, height - size))
    return left, top, left + size, top + size


def _overlap(box_a, box_b) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    intersection = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    area = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
    return intersection / max(area, 1)


def _avoid_outer_border(box, width: int, height: int) -> tuple[int, int, int, int]:
    """Shift an automatic local window away from the canvas/black frame."""
    x0, y0, x1, y1 = box
    margin = max(2, int(round(min(width, height) * 0.025)))
    box_width, box_height = x1 - x0, y1 - y0
    if box_width <= width - 2 * margin:
        x0 = max(margin, min(x0, width - margin - box_width))
        x1 = x0 + box_width
    if box_height <= height - 2 * margin:
        y0 = max(margin, min(y0, height - margin - box_height))
        y1 = y0 + box_height
    return int(x0), int(y0), int(x1), int(y1)


def _geometry(mask: np.ndarray) -> np.ndarray:
    """生成与旋转无关的径向缺陷分布和缺陷占比描述子。"""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return np.zeros(9, dtype=np.float32)
    height, width = mask.shape
    cx, cy = (width - 1) / 2, (height - 1) / 2
    radius = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    radius /= max(np.sqrt(cx ** 2 + cy ** 2), 1.0)
    hist, _ = np.histogram(radius, bins=np.linspace(0, 1, 9))
    hist = hist.astype(np.float32)
    hist /= max(float(hist.sum()), 1.0)
    return np.concatenate([hist, np.asarray([float(mask.mean())], dtype=np.float32)])


def _covered_defect_ratio(mask: np.ndarray, boxes) -> float:
    """Return the share of defect pixels already explained by local boxes."""
    total = int(mask.sum())
    if total <= 0 or not boxes:
        return 0.0
    covered = np.zeros_like(mask, dtype=bool)
    for box in boxes:
        x0, y0, x1, y1 = box
        covered[y0:y1, x0:x1] = True
    return float((mask & covered).sum()) / float(total)


def _global_pattern_metrics(mask: np.ndarray) -> dict:
    """Measure whether defects form a genuine wafer-wide pattern.

    Raw min/max coordinates are too sensitive to isolated pixels.  Use robust
    percentiles and require several *meaningfully occupied* grid cells so that
    a few distant noise points cannot turn every Bin Map into a full-map ROI.
    """
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return {"is_global": False, "x_span": 0.0, "y_span": 0.0,
                "active_cells": 0, "dispersion": 0.0}

    x_low, x_high = np.percentile(xs, [5, 95])
    y_low, y_high = np.percentile(ys, [5, 95])
    x_span = float(x_high - x_low + 1) / max(width, 1)
    y_span = float(y_high - y_low + 1) / max(height, 1)

    cell_counts = np.zeros((3, 3), dtype=np.int64)
    grid_y = np.minimum(2, ys * 3 // max(height, 1))
    grid_x = np.minimum(2, xs * 3 // max(width, 1))
    np.add.at(cell_counts, (grid_y, grid_x), 1)
    total = int(len(xs))
    meaningful_count = max(4, int(np.ceil(total * 0.06)))
    active_cells = int((cell_counts >= meaningful_count).sum())
    probabilities = cell_counts[cell_counts > 0].astype(np.float64) / max(total, 1)
    entropy = float(-(probabilities * np.log(probabilities)).sum()) if len(probabilities) else 0.0
    dispersion = entropy / np.log(9.0) if entropy > 0 else 0.0

    # The full-image embedding already participates in every retrieval, so a
    # visible full-map ROI should be exceptional.  Require meaningful defects
    # in all nine cells and a near-uniform spatial distribution.  This avoids
    # turning a few distant clusters (or wafer-edge noise) into a global box.
    is_global = active_cells == 9 and dispersion >= 0.93
    return {
        "is_global": bool(is_global),
        "x_span": x_span,
        "y_span": y_span,
        "active_cells": active_cells,
        "dispersion": dispersion,
    }


def _segment_binmap_legacy(image_bytes: bytes, max_blocks: int = MAX_BLOCKS_PER_IMAGE) -> dict:
    """定位缺陷聚集区域，返回整图、局部块、框坐标和几何描述子。"""
    image = _open_rgb(image_bytes)
    width, height = image.size
    mask = defect_mask(image)
    rgb_array = np.asarray(image, dtype=np.uint8)
    dark_background = np.max(rgb_array, axis=2) < 35
    if not mask.any():
        return {"image": image, "blocks": [], "geometry": _geometry(mask)}

    # 适度膨胀把相邻坏 Die 合为一个模式区域，仍保留整图位置上下文。
    kernel = max(3, min(11, (min(width, height) // 40) | 1))
    dilated = np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(kernel))
    ) > 0
    minimum_box = max(48, int(min(width, height) * 0.24))
    candidates = []
    for x0, y0, x1, y1, _ in _components(dilated):
        original_count = int(mask[y0:y1, x0:x1].sum())
        if original_count <= 0:
            continue
        # A component spanning most of the canvas is usually the wafer rim or
        # a compressed red band, not one useful local defect.  Let the local
        # density windows below split it into smaller evidence regions instead
        # of drawing a nearly full-image box that includes the black border.
        component_width, component_height = x1 - x0, y1 - y0
        component_area_ratio = (component_width * component_height) / max(width * height, 1)
        component_span_ratio = max(component_width / max(width, 1), component_height / max(height, 1))
        if component_area_ratio > 0.34 or component_span_ratio > 0.62:
            continue
        pad = max(6, int(max(x1 - x0, y1 - y0) * 0.2))
        box = _avoid_outer_border(
            _square_box((x0 - pad, y0 - pad, x1 + pad, y1 + pad), width, height, minimum_box),
            width, height,
        )
        bx0, by0, bx1, by1 = box
        if max((bx1 - bx0) / max(width, 1), (by1 - by0) / max(height, 1)) > 0.48:
            continue
        if float(dark_background[by0:by1, bx0:bx1].mean()) > 0.40:
            continue
        density = original_count / max((bx1 - bx0) * (by1 - by0), 1)
        candidates.append((original_count * (1.0 + density * 4.0), box, original_count, False))

    # 随机散点图可能没有明显连通簇：补充滑动热点窗口，但不把整图机械切满。
    window = min(width, height, max(minimum_box, int(min(width, height) * 0.30)))
    step = max(1, window // 2)
    for top in range(0, max(height - window + 1, 1), step):
        for left in range(0, max(width - window + 1, 1), step):
            right, bottom = min(width, left + window), min(height, top + window)
            count = int(mask[top:bottom, left:right].sum())
            if count and float(dark_background[top:bottom, left:right].mean()) <= 0.40:
                candidates.append((count * 0.9, _avoid_outer_border((left, top, right, bottom), width, height), count, False))

    # Select local evidence first.  The whole-map feature is already computed
    # elsewhere for ranking, so a visible full-map ROI must earn its place by
    # explaining defects that the local boxes cannot cover.
    selected = []
    for score, box, count, _ in sorted(candidates, key=lambda item: item[0], reverse=True):
        if any(_overlap(box, old["bbox_px"]) > 0.58 for old in selected):
            continue
        x0, y0, x1, y1 = box
        crop = standardized_region_crop(image, box)
        selected.append({
            "bbox_px": box,
            "bbox_norm": [round(x0 / width, 6), round(y0 / height, 6),
                          round((x1 - x0) / width, 6), round((y1 - y0) / height, 6)],
            "defect_pixels": count,
            "crop": crop,
            "score": round(float(score), 4),
            "global": False,
            "selection_reason": "local_defect_region",
        })
        if len(selected) >= max(1, max_blocks):
            break

    metrics = _global_pattern_metrics(mask)
    local_coverage = _covered_defect_ratio(mask, [item["bbox_px"] for item in selected])
    unexplained_ratio = 1.0 - local_coverage

    # A global ROI is a rare visual fallback, never a reserved slot.  The full
    # image remains part of the hidden retrieval score even when no global box
    # is drawn.  Only an extremely dispersed pattern that local boxes explain
    # poorly may receive a visible full-map region.
    add_global = False
    if not selected:
        add_global = True
    elif metrics["is_global"]:
        if len(selected) < max(1, max_blocks):
            add_global = local_coverage < 0.40 and unexplained_ratio >= 0.60
        else:
            add_global = (
                local_coverage < 0.35
                and unexplained_ratio >= 0.65
            )

    if add_global:
        active = ~dark_background
        active_y, active_x = np.nonzero(active)
        if len(active_x):
            global_box = (
                int(active_x.min()), int(active_y.min()),
                int(active_x.max()) + 1, int(active_y.max()) + 1,
            )
        else:
            global_box = (0, 0, width, height)
        if len(selected) >= max(1, max_blocks):
            selected.pop(-1)
        gx0, gy0, gx1, gy1 = global_box
        selected.append({
            "bbox_px": global_box,
            "bbox_norm": [
                round(gx0 / width, 6), round(gy0 / height, 6),
                round((gx1 - gx0) / width, 6), round((gy1 - gy0) / height, 6),
            ],
            "defect_pixels": int(mask.sum()),
            "crop": standardized_region_crop(image, global_box),
            "score": round(float(mask.sum()) * (0.2 + 0.8 * unexplained_ratio), 4),
            "global": True,
            "selection_reason": "global_pattern_fallback",
        })
    for index, block in enumerate(selected, 1):
        block["block_id"] = index
    return {"image": image, "blocks": selected, "geometry": _geometry(mask)}


def segment_binmap(image_bytes: bytes, max_blocks: int = MAX_BLOCKS_PER_IMAGE) -> dict:
    """Locate at most three defect regions using adaptive, diverse scales.

    The full-image embedding is computed by the normal retrieval pipeline, so
    there is no reserved visible global box.  A large box is still a valid
    candidate when a real connected defect group spans a large part of the
    wafer.  Candidate sizes come from multi-strength connected components and
    multi-scale density windows; selection rewards new defect coverage and
    rejects redundant same-place boxes.
    """
    image = _open_rgb(image_bytes)
    width, height = image.size
    mask = defect_mask(image)
    rgb_array = np.asarray(image, dtype=np.uint8)
    dark_background = np.max(rgb_array, axis=2) < 35
    if not mask.any():
        return {"image": image, "blocks": [], "geometry": _geometry(mask)}

    total_defects = int(mask.sum())
    min_side = min(width, height)
    minimum_box = max(24, int(round(min_side * 0.14)))
    minimum_pixels = max(2, int(total_defects * 0.002))
    global_metrics = _global_pattern_metrics(mask)
    candidate_by_box = {}

    def add_candidate(box, source, source_scale=0.0):
        box = _avoid_outer_border(tuple(int(value) for value in box), width, height)
        x0, y0, x1, y1 = box
        area = max((x1 - x0) * (y1 - y0), 1)
        count = int(mask[y0:y1, x0:x1].sum())
        if count < minimum_pixels:
            return
        if float(dark_background[y0:y1, x0:x1].mean()) > 0.55:
            return
        density = count / area
        area_ratio = area / max(width * height, 1)
        # Do not ban broad real defects.  Only a near-canvas rectangle with no
        # genuinely dispersed pattern is rejected as frame/background noise.
        if area_ratio > 0.88 and not global_metrics["is_global"]:
            return
        coverage = count / max(total_defects, 1)
        source_bonus = 0.12 if source.startswith("component") else 0.0
        if source == "global_defect_extent":
            # This is not a reserved full-image slot.  It is generated only
            # when the measured defect distribution is genuinely wafer-wide.
            source_bonus = 0.10
        relative_density = min(density / max(float(mask.mean()), 1e-6), 3.0) / 3.0
        quality = 0.62 * coverage + 0.26 * relative_density + source_bonus
        key = tuple(int(value) for value in box)
        candidate = {
            "bbox_px": key,
            "defect_pixels": count,
            "density": density,
            "area_ratio": area_ratio,
            "quality": quality,
            "source": source,
            "source_scale": source_scale,
        }
        old = candidate_by_box.get(key)
        if old is None or quality > old["quality"]:
            candidate_by_box[key] = candidate

    # Multiple dilation strengths form naturally small, medium and large
    # connected groups.  This restores the useful behaviour of the early
    # multi-scale implementation while keeping later border safeguards.
    kernels = {3, 5}
    for ratio in (0.025, 0.05, 0.09):
        kernels.add(max(3, min(31, int(round(min_side * ratio)) | 1)))
    for kernel in sorted(kernels):
        dilated = np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(kernel))
        ) > 0
        for x0, y0, x1, y1, _ in _components(dilated):
            count = int(mask[y0:y1, x0:x1].sum())
            if count < minimum_pixels:
                continue
            component_side = max(x1 - x0, y1 - y0)
            pad = max(3, int(round(component_side * (0.10 + min(kernel / max(min_side, 1), 0.08)))))
            box = _square_box(
                (x0 - pad, y0 - pad, x1 + pad, y1 + pad),
                width,
                height,
                max(minimum_box, component_side + 2 * pad),
            )
            add_candidate(box, f"component_k{kernel}", component_side / max(min_side, 1))

    # A wafer-wide candidate is data-driven and competes with every local
    # candidate.  It is never inserted unconditionally, which avoids the old
    # behaviour where every result displayed one almost-full-image rectangle.
    if global_metrics["is_global"]:
        ys, xs = np.nonzero(mask)
        span = max(int(xs.max()) - int(xs.min()) + 1, int(ys.max()) - int(ys.min()) + 1)
        pad = max(4, int(round(span * 0.05)))
        broad_box = _square_box(
            (
                int(xs.min()) - pad,
                int(ys.min()) - pad,
                int(xs.max()) + 1 + pad,
                int(ys.max()) + 1 + pad,
            ),
            width,
            height,
            max(minimum_box, span + 2 * pad),
        )
        add_candidate(broad_box, "global_defect_extent", span / max(min_side, 1))

    # Density windows are deliberately multi-scale and are a fallback for
    # scattered defects, not a fixed 30% grid that cuts every pattern alike.
    window_scales = [0.18, 0.28, 0.40, 0.56]
    # A 72% density window is allowed only for a genuinely wafer-wide defect
    # distribution.  Otherwise it would repeatedly dominate merely because it
    # contains more pixels, recreating the unwanted same-size large boxes.
    if global_metrics["is_global"]:
        window_scales.append(0.72)
    for scale in window_scales:
        window = min(min_side, max(minimum_box, int(round(min_side * scale))))
        step = max(4, window // 3)
        tops = list(range(0, max(height - window + 1, 1), step))
        lefts = list(range(0, max(width - window + 1, 1), step))
        final_top, final_left = max(0, height - window), max(0, width - window)
        if final_top not in tops:
            tops.append(final_top)
        if final_left not in lefts:
            lefts.append(final_left)
        for top in tops:
            for left in lefts:
                add_candidate((left, top, left + window, top + window), "density_window", scale)

    candidates = list(candidate_by_box.values())
    selected = []
    covered = np.zeros_like(mask, dtype=bool)
    max_blocks = max(1, min(int(max_blocks), 3))
    while candidates and len(selected) < max_blocks:
        best = None
        best_utility = -1.0
        for candidate in candidates:
            x0, y0, x1, y1 = candidate["bbox_px"]
            new_count = int((mask[y0:y1, x0:x1] & ~covered[y0:y1, x0:x1]).sum())
            new_ratio = new_count / max(total_defects, 1)
            novel_fraction = new_count / max(candidate["defect_pixels"], 1)
            overlaps = [_overlap(candidate["bbox_px"], old["bbox_px"]) for old in selected]
            if overlaps and max(overlaps) > 0.88:
                continue
            nested_bonus = 0.0
            if selected and new_ratio < 0.01:
                parent = max(selected, key=lambda old: _overlap(candidate["bbox_px"], old["bbox_px"]))
                scale_ratio = candidate["area_ratio"] / max(parent["area_ratio"], 1e-8)
                if not (scale_ratio <= 0.55 and candidate["density"] >= parent["density"] * 1.20):
                    continue
                nested_bonus = 0.06
            scale_diversity = 0.0
            if selected:
                nearest = min(abs(candidate["area_ratio"] - old["area_ratio"]) for old in selected)
                scale_diversity = min(nearest * 0.30, 0.08)
            window_penalty = (
                0.18 * candidate["area_ratio"]
                if candidate["source"] == "density_window" else 0.0
            )
            # Once a broad region has been selected, do not keep rewarding a
            # second similarly sized window for the same already-covered red
            # pixels.  A separate small cluster keeps a high novel_fraction
            # and can therefore outrank repetitive, same-scale windows.
            residual_quality = candidate["quality"] * (0.20 + 0.80 * novel_fraction)
            utility = (
                0.58 * new_ratio + 0.34 * residual_quality
                + scale_diversity + nested_bonus - window_penalty
            )
            if utility > best_utility:
                best, best_utility = candidate, utility
        if best is None:
            break
        candidates.remove(best)
        x0, y0, x1, y1 = best["bbox_px"]
        covered[y0:y1, x0:x1] |= mask[y0:y1, x0:x1]
        chosen = dict(best)
        chosen["score"] = round(float(best_utility), 6)
        selected.append(chosen)

    # Last-resort box follows the actual defect extent.  It may be large, but
    # it is never inserted as a fixed global slot.
    if not selected:
        ys, xs = np.nonzero(mask)
        box = _square_box(
            (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
            width,
            height,
            minimum_box,
        )
        add_candidate(box, "defect_extent_fallback", 1.0)
        selected = list(candidate_by_box.values())[:1]

    for index, block in enumerate(selected, 1):
        x0, y0, x1, y1 = block["bbox_px"]
        block.update({
            "block_id": index,
            "bbox_norm": [
                round(x0 / width, 6), round(y0 / height, 6),
                round((x1 - x0) / width, 6), round((y1 - y0) / height, 6),
            ],
            "crop": standardized_region_crop(image, block["bbox_px"]),
            "global": bool(
                block["source"] == "global_defect_extent"
                or block["area_ratio"] >= 0.78
            ),
            "selection_reason": block.pop("source"),
        })
    return {"image": image, "blocks": selected, "geometry": _geometry(mask)}


def segment_manual_region(image_bytes: bytes, bbox_norm) -> dict:
    """Create one searchable block from a user-drawn normalized rectangle.

    ``bbox_norm`` uses ``[left, top, width, height]`` in the 0..1 image
    coordinate system.  The region is intentionally not passed through the
    automatic defect detector: drawing the box is an explicit user choice.
    """
    # New clients send a list of rectangles/polygons for one image.  Merge the
    # individually cropped blocks into one segmented result while keeping the
    # legacy four-number rectangle format backward compatible.
    is_rectangle = (
        isinstance(bbox_norm, (list, tuple)) and len(bbox_norm) == 4
        and all(isinstance(item, (int, float)) for item in bbox_norm)
    )
    if isinstance(bbox_norm, (list, tuple)) and not is_rectangle:
        image = _open_rgb(image_bytes)
        blocks = []
        for region in bbox_norm:
            segmented = segment_manual_region(image_bytes, region)
            for block in segmented["blocks"]:
                copied = dict(block)
                copied["block_id"] = len(blocks) + 1
                blocks.append(copied)
        if not blocks:
            raise ValueError("manual regions are empty")
        return {"image": image, "blocks": blocks, "geometry": _geometry(defect_mask(image))}

    image = _open_rgb(image_bytes)
    width, height = image.size
    polygon_norm = None
    if isinstance(bbox_norm, dict) and str(bbox_norm.get("type", "")).lower() == "polygon":
        try:
            polygon_norm = [[float(x), float(y)] for x, y in bbox_norm.get("points", [])]
        except (TypeError, ValueError):
            raise ValueError("manual polygon points are invalid")
        if len(polygon_norm) < 3:
            raise ValueError("manual polygon needs at least three points")
        left = min(point[0] for point in polygon_norm)
        top = min(point[1] for point in polygon_norm)
        right = max(point[0] for point in polygon_norm)
        bottom = max(point[1] for point in polygon_norm)
        box_width, box_height = right - left, bottom - top
    else:
        try:
            left, top, box_width, box_height = [float(value) for value in bbox_norm]
        except (TypeError, ValueError):
            raise ValueError("manual region must contain left, top, width and height")
    try:
        values = [left, top, box_width, box_height]
    except (TypeError, ValueError):
        raise ValueError("manual region must contain left, top, width and height")
    if not all(np.isfinite(values)):
        raise ValueError("manual region contains a non-finite value")
    left = max(0.0, min(1.0, left))
    top = max(0.0, min(1.0, top))
    box_width = max(0.0, min(1.0 - left, box_width))
    box_height = max(0.0, min(1.0 - top, box_height))
    if box_width < 0.02 or box_height < 0.02:
        raise ValueError("manual region is too small")

    x0 = max(0, min(width - 1, int(round(left * width))))
    y0 = max(0, min(height - 1, int(round(top * height))))
    x1 = max(x0 + 1, min(width, int(round((left + box_width) * width))))
    y1 = max(y0 + 1, min(height, int(round((top + box_height) * height))))
    crop_box = (x0, y0, x1, y1)
    selection_mask = None
    if polygon_norm is not None:
        polygon_px = [
            ((x * width) - x0, (y * height) - y0)
            for x, y in polygon_norm
        ]
        raw_crop = image.crop(crop_box)
        selection_mask = Image.new("L", raw_crop.size, 0)
        ImageDraw.Draw(selection_mask).polygon(polygon_px, fill=255)
    crop = standardized_region_crop(image, crop_box, selection_mask)
    mask = defect_mask(image)
    block = {
        "bbox_px": (x0, y0, x1, y1),
        "bbox_norm": [round(x0 / width, 6), round(y0 / height, 6),
                      round((x1 - x0) / width, 6), round((y1 - y0) / height, 6)],
        "defect_pixels": int(mask[y0:y1, x0:x1].sum()),
        "crop": crop,
        "score": 1.0,
        "block_id": 1,
        "manual": True,
        "shape": "polygon" if polygon_norm is not None else "rectangle",
        "polygon_norm": polygon_norm,
    }
    return {"image": image, "blocks": [block], "geometry": _geometry(mask)}


def _rotated_bytes(image: Image.Image) -> list[bytes]:
    return [_encode_png(image.rotate(angle, expand=False)) for angle in ROTATIONS]


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    return left @ right.T


class BinMapPatchSearcher:
    """维护历史局部块缓存，并把局部证据融合到整图检索结果。"""

    def __init__(self, owner):
        self.owner = owner
        self._index = None

    def _signature(self, images) -> str:
        # Keep the patch cache reusable after the project is copied to another
        # drive or computer.  The DN/file name and byte size identify the
        # ordered inputs without embedding the old machine's absolute path.
        values = [
            f"{item['dn_no']}/{item['file_name']}:{Path(item['path']).stat().st_size}"
            for item in images
        ]
        values += [self.owner.provider.__class__.__name__, "binmap-patches-v10-adaptive-multiscale-defectmappost", str(MAX_BLOCKS_PER_IMAGE)]
        return hashlib.sha1("|".join(values).encode("utf-8")).hexdigest()

    def ensure_index(self):
        if self._index is not None:
            return self._index
        # The local patch index follows the same business rule as the global
        # Bin Map index: only DefectMapPost participates in similarity.
        images = [item for item in self.owner.catalog() if item["image_type"] == "DefectMapPost"]
        signature = self._signature(images)
        if PATCH_INDEX_PATH.exists():
            try:
                cached = np.load(PATCH_INDEX_PATH, allow_pickle=False)
                if str(cached["signature"].item()) == signature:
                    self._index = {
                        "items": json.loads(str(cached["items"].item())),
                        "vectors": cached["vectors"].astype(np.float32),
                        "maps": json.loads(str(cached["maps"].item())),
                    }
                    return self._index
            except Exception:
                pass

        items, crops, maps = [], [], []
        for image_item in images:
            try:
                content = Path(image_item["path"]).read_bytes()
                segmented = segment_binmap(content)
            except Exception:
                continue
            maps.append({
                "dn_no": image_item["dn_no"], "file_name": image_item["file_name"],
                "url": image_item["url"], "geometry": segmented["geometry"].tolist(),
            })
            for block in segmented["blocks"]:
                items.append({
                    "dn_no": image_item["dn_no"], "file_name": image_item["file_name"],
                    "url": image_item["url"], "block_id": block["block_id"],
                    "bbox_px": list(block["bbox_px"]), "bbox_norm": block["bbox_norm"],
                    "defect_pixels": block["defect_pixels"],
                })
                crops.append(_encode_png(block["crop"]))
        if not crops:
            raise RuntimeError("没有从历史 Bin Map 中提取到可索引的局部缺陷块")
        batches = []
        batch_size = max(1, int(os.getenv("WAFER_IMAGE_BATCH_SIZE", "32")))
        for start in range(0, len(crops), batch_size):
            batches.append(self.owner.provider.embed_bytes(crops[start:start + batch_size], "binmap"))
        vectors = np.concatenate(batches).astype(np.float32)
        PATCH_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            PATCH_INDEX_PATH, signature=signature, vectors=vectors,
            items=json.dumps(items, ensure_ascii=False), maps=json.dumps(maps, ensure_ascii=False),
        )
        self._index = {"items": items, "vectors": vectors, "maps": maps}
        return self._index

    def rerank(
        self, image_bytes_list: list[bytes], baseline: dict, limit: int = 100,
        manual_regions: list | None = None,
    ) -> dict:
        manual_regions = list(manual_regions or [])
        query_maps = []
        for query_index, content in enumerate(image_bytes_list):
            region = manual_regions[query_index] if query_index < len(manual_regions) else None
            if region:
                query_maps.append((query_index, segment_manual_region(content, region)))
            elif is_binmap_image(content):
                query_maps.append((query_index, segment_binmap(content)))
        if not query_maps:
            baseline["local_block_search"] = {"enabled": False, "reason": "输入中未识别到 Bin Map"}
            return baseline

        index = self.ensure_index()
        history_vectors = index["vectors"]
        history_items = index["items"]
        query_block_meta, rotated_crops = [], []
        for query_index, segmented in query_maps:
            for block in segmented["blocks"]:
                start = len(rotated_crops)
                rotated_crops.extend(_rotated_bytes(block["crop"]))
                # ``segment_binmap`` numbers blocks inside each image, so two
                # query Bin Maps can both contain block ``1``.  Use one
                # monotonically increasing display id for the whole request;
                # the same id is copied into the matching history evidence.
                display_block_id = len(query_block_meta) + 1
                query_block_meta.append({
                    "query_index": query_index, "block_id": display_block_id,
                    "source_block_id": block["block_id"],
                    "bbox_norm": block["bbox_norm"], "rotation_start": start,
                    "preview_url": _data_url(block["crop"]),
                    "manual": bool(block.get("manual")),
                    "shape": block.get("shape", "rectangle"),
                    "polygon_norm": block.get("polygon_norm"),
                })
        if not query_block_meta:
            baseline["local_block_search"] = {"enabled": False, "reason": "未定位到稳定的缺陷聚集区域"}
            return baseline

        query_vectors = []
        batch_size = max(1, int(os.getenv("WAFER_IMAGE_BATCH_SIZE", "32")))
        for start in range(0, len(rotated_crops), batch_size):
            query_vectors.append(self.owner.provider.embed_bytes(rotated_crops[start:start + batch_size], "binmap"))
        query_vectors = np.concatenate(query_vectors).astype(np.float32)
        similarities = _cosine_rows(query_vectors, history_vectors)

        by_dn = {}
        for meta in query_block_meta:
            start = meta["rotation_start"]
            block_scores = similarities[start:start + len(ROTATIONS)]
            best_rotation = np.argmax(block_scores, axis=0)
            best_scores = np.max(block_scores, axis=0)
            for history_index, history in enumerate(history_items):
                evidence = {
                    "query_index": meta["query_index"], "query_block_id": meta["block_id"],
                    # 查询块预览只在顶层 query_blocks 返回一次。若把 Base64 预览
                    # 复制进每条 DN 证据，几十条结果会膨胀成上百 MB 的 JSON。
                    "query_bbox_norm": meta["bbox_norm"],
                    "manual": bool(meta.get("manual")),
                    "history_block_id": history["block_id"], "history_bbox_norm": history["bbox_norm"],
                    "history_bbox_px": history["bbox_px"], "history_url": history["url"],
                    "history_file_name": history["file_name"],
                    "history_patch_url": (
                        f"/api/yedn/patch/{quote(history['dn_no'], safe='')}/"
                        f"{quote(history['file_name'], safe='')}?"
                        f"x={history['bbox_px'][0]}&y={history['bbox_px'][1]}&"
                        f"w={history['bbox_px'][2] - history['bbox_px'][0]}&"
                        f"h={history['bbox_px'][3] - history['bbox_px'][1]}&normalized=1"
                    ),
                    "similarity": round(float(best_scores[history_index]), 6),
                    "rotation": ROTATIONS[int(best_rotation[history_index])],
                }
                key = (meta["query_index"], meta["block_id"])
                old = by_dn.setdefault(history["dn_no"], {}).get(key)
                if old is None or evidence["similarity"] > old["similarity"]:
                    by_dn[history["dn_no"]][key] = evidence

        # 使用径向分布描述子补充位置/规模关系；它本身对旋转不敏感。
        query_geometry = [entry[1]["geometry"] for entry in query_maps]
        map_by_dn = {}
        for item in index["maps"]:
            map_by_dn.setdefault(item["dn_no"], []).append(np.asarray(item["geometry"], dtype=np.float32))

        baseline_by_dn = {item["dn_no"]: item for item in baseline.get("results", [])}
        for dn_no, item in baseline_by_dn.items():
            evidences = sorted(by_dn.get(dn_no, {}).values(), key=lambda value: value["similarity"], reverse=True)
            if not evidences:
                continue
            block_scores = np.asarray([value["similarity"] for value in evidences], dtype=np.float32)
            peak_score = float(block_scores.max())
            ordered_scores = np.sort(block_scores)[::-1]
            multi_score = float(ordered_scores[:min(3, len(ordered_scores))].mean())
            # A block is considered independently convincing only above this
            # conservative cosine threshold.  Coverage rewards records that
            # explain several distinct query defects, while ``peak_score``
            # keeps one exceptionally strong local match near the top.
            strong_threshold = float(os.getenv("WAFER_BINMAP_STRONG_BLOCK", "0.84"))
            strong = [value for value in evidences if value["similarity"] >= strong_threshold]
            query_block_total = max(len(query_block_meta), 1)
            coverage = len(strong) / query_block_total
            multi_defect_score = 0.72 * multi_score + 0.28 * coverage
            local_score = max(0.94 * peak_score, multi_defect_score)
            geometry_scores = []
            for query_desc in query_geometry:
                q_hist, q_ratio = query_desc[:8], max(float(query_desc[8]), 1e-8)
                best = 0.0
                for history_desc in map_by_dn.get(dn_no, []):
                    h_hist, h_ratio = history_desc[:8], max(float(history_desc[8]), 1e-8)
                    radial = float(np.dot(q_hist, h_hist) / max(np.linalg.norm(q_hist) * np.linalg.norm(h_hist), 1e-8))
                    ratio = float(np.exp(-abs(np.log(q_ratio / h_ratio))))
                    best = max(best, 0.75 * radial + 0.25 * ratio)
                geometry_scores.append(best)
            geometry_score = float(np.mean(geometry_scores)) if geometry_scores else 0.0
            global_score = float(item.get("average_similarity", item.get("score", 0.0)))
            # DINO 余弦分数保持原尺度；局部证据优先，整图与几何负责稳定上下文。
            manual_evidence = any(value.get("manual") for value in evidences)
            if manual_evidence:
                # A drawn region is an explicit retrieval instruction, so the
                # local visual evidence dominates while the full image remains
                # a small context stabilizer.
                final_score = 0.10 * global_score + 0.85 * local_score + 0.05 * geometry_score
            else:
                # Multi-defect agreement receives an extra context bonus. A
                # single very high local match still ranks strongly because
                # local_score contains the peak-preserving branch above.
                context_weight = min(0.12, 0.12 * coverage)
                final_score = (
                    (0.22 - context_weight / 2) * global_score
                    + (0.70 + context_weight / 2) * local_score
                    + 0.08 * geometry_score
                )
            display_evidences = strong[:3] if len(strong) >= 2 and coverage >= 0.5 else evidences[:1]
            block_matches = sorted(evidences, key=lambda value: value["query_block_id"])
            item.update({
                "global_similarity": round(global_score, 6),
                "local_similarity": round(local_score, 6),
                "local_peak_similarity": round(peak_score, 6),
                "multi_defect_similarity": round(multi_defect_score, 6),
                "matched_block_count": len(strong),
                "query_block_count": query_block_total,
                "block_coverage": round(coverage, 6),
                "local_match_mode": "multi" if len(display_evidences) >= 2 else "single",
                "geometry_similarity": round(geometry_score, 6),
                "score": round(final_score, 6), "ranking_score": round(final_score, 6),
                "average_similarity": round(final_score, 6),
                "block_evidence": display_evidences[:3],
                # Complete per-query-region evidence is kept separately from
                # the small set of boxes drawn on the result thumbnail.  The
                # UI uses this list for the expandable No.1/No.2/... panel.
                "block_matches": block_matches,
            })
            best = evidences[0]
            item["best_match"] = {
                "image_type": "DefectMapPost", "file_name": best["history_file_name"],
                "url": best["history_url"], "similarity": best["similarity"],
                "bbox_norm": best["history_bbox_norm"],
            }

        ranked = sorted(baseline_by_dn.values(), key=lambda value: value.get("score", -1), reverse=True)
        for rank, item in enumerate(ranked, 1):
            item["rank"] = rank
        baseline["results"] = ranked[:min(max(int(limit), 1), 500)]
        baseline["count"] = len(baseline["results"])
        baseline["mode"] = "binmap_global_and_local_blocks"
        baseline["local_block_search"] = {
            "enabled": True, "rotation_degrees": list(ROTATIONS),
            "query_binmap_count": len(query_maps), "query_block_count": len(query_block_meta),
            "history_block_count": len(history_items),
            "manual_region_count": sum(1 for meta in query_block_meta if meta.get("manual")),
        }
        baseline["query_blocks"] = [{key: value for key, value in meta.items() if key != "rotation_start"}
                                     for meta in query_block_meta]
        return baseline
