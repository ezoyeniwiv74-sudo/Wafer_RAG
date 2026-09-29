"""YEDN 双模态、按 DN 聚合的纯图像相似度检索。"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

import numpy as np

from config import QDRANT_ONLY, UNIFIED_BINMAP_VECTOR, UNIFIED_SEM_VECTOR
from utils.payload_parser import canonicalize_payload, canonical_record_id, merge_payloads, describe_payload, is_missing
from utils.qdrant_client import get_qdrant_manager

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SERVER_URL = os.getenv("WAFER_IMAGE_SERVER_URL", "http://127.0.0.1:9911/analyze").strip()
IMAGE_SERVER_TOKEN = os.getenv("WAFER_IMAGE_SERVER_TOKEN", "").strip()
IMAGE_SERVER_MODEL = os.getenv("WAFER_IMAGE_SERVER_MODEL", "company-dinov2-vitl14-binmap-lora-v1").strip()
IMAGE_SERVER_TIMEOUT = float(os.getenv("WAFER_IMAGE_SERVER_TIMEOUT", "180"))
IMAGE_BATCH_SIZE = int(os.getenv("WAFER_IMAGE_BATCH_SIZE", "32"))
INDEX_PATHS = {
    "binmap": Path(os.getenv("WAFER_BINMAP_INDEX", ROOT / ".cache/yedn_binmap_index.npz")),
    "sem": Path(os.getenv("WAFER_SEM_INDEX", ROOT / ".cache/yedn_sem_index.npz")),
}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
# ``IMAGE_TYPES`` controls detail-page display order only.  It intentionally
# contains every standard image that may exist in a company DN directory.
# Retrieval eligibility is defined separately in ``MODALITY_IMAGE_TYPES`` so
# adding a display image can never silently change similarity ranking.
IMAGE_TYPES = (
    "DefectMapPost",
    "TypicalDefectImage1",
    "TypicalDefectImage2",
    "AddScanDefectMapGallary",
    "AddScanDefectMapGallary1",
    "DefectBarChart",
    "DefectMapPre",
    "DefectParetoChart",
    "DefectTrendChart",
    "DieStack",
    "EDXDefectImage",
    "MoreImage",
    "TypicalMap",
    "TypicalMap2",
)
MODALITY_IMAGE_TYPES = {
    # Business retrieval contract (2026-09): Bin Map queries compare only
    # with DefectMapPost; SEM queries compare only with the two typical SEM
    # images.  Every other image remains available in record details.
    "binmap": ("DefectMapPost",),
    "sem": ("TypicalDefectImage1", "TypicalDefectImage2"),
}
MODALITY_LABELS = {
    "binmap": "Wafer Bin Map（晶圆缺陷分布图）",
    "sem": "SEM 缺陷扫描图（扫描电子显微镜图）",
}

ANALYSIS_FIELDS = (
    ("defect_type", "缺陷类型"),
    ("platform", "平台"),
    ("product_id", "产品"),
    ("machine", "机台/腔体"),
    ("lot_id", "Lot"),
    ("station", "站点"),
    ("map_location", "Map位置"),
    ("dft_location", "DFT位置"),
)


def _statistics_table(ranked, evidence_limit=20):
    """统计相似记录中每个业务字段最常出现的值。

    频次相同时，以这些记录的图像相似度之和作为稳定的次级排序依据。
    对外只返回字段名与常见值，避免把实现细节暴露给最终用户。
    """
    evidence = ranked[:max(1, min(int(evidence_limit), 50))]
    rows = []
    for field, label in ANALYSIS_FIELDS:
        counts = {}
        weights = {}
        for item in evidence:
            value = item.get("fields", {}).get(field)
            if is_missing(value) or value == 0:
                continue
            value = str(value)
            counts[value] = counts.get(value, 0) + 1
            weights[value] = weights.get(value, 0.0) + max(float(item.get("average_similarity", 0.0)), 0.0)
        if not counts:
            continue
        dominant = max(counts, key=lambda value: (weights[value], counts[value], value))
        rows.append({"field": field, "label": label, "dominant_value": dominant})
    return rows


def resolve_yedn_root():
    configured = os.getenv("WAFER_YEDN_ROOT", "").strip()
    candidates = ([Path(configured)] if configured else []) + [
        ROOT / "data" / "YEDN",
        ROOT.parent / "data" / "YEDN",
        Path.cwd() / "data" / "YEDN",
    ]
    for path in candidates:
        if path.exists() and path.is_dir():
            return path.resolve()
    return candidates[0].resolve()


def classify_image(path):
    name = "".join(ch for ch in path.stem.lower() if ch.isalnum())
    aliases = (
        ("addscandefectmapgallary1", "AddScanDefectMapGallary1"),
        ("addscandefectmapgallery1", "AddScanDefectMapGallary1"),
        ("addscandefectmapgallary", "AddScanDefectMapGallary"),
        ("addscandefectmapgallery", "AddScanDefectMapGallary"),
        ("typicaldefectimage1", "TypicalDefectImage1"),
        ("typeicaldefectimage1", "TypicalDefectImage1"),
        ("typicaldefectimage2", "TypicalDefectImage2"),
        ("typeicaldefectimage2", "TypicalDefectImage2"),
        ("defectparetochart", "DefectParetoChart"),
        ("defecttrendchart", "DefectTrendChart"),
        ("defectbarchart", "DefectBarChart"),
        ("defectmappost", "DefectMapPost"),
        ("defectmappre", "DefectMapPre"),
        ("edxdefectimage", "EDXDefectImage"),
        ("diestack", "DieStack"),
        ("moreimage", "MoreImage"),
        ("typicalmap2", "TypicalMap2"),
        ("typeicalmap2", "TypicalMap2"),
        ("typicalmap", "TypicalMap"),
        ("typeicalmap", "TypicalMap"),
    )
    for alias, canonical in aliases:
        if alias in name:
            return canonical
    # Detail pages must show every image that is physically present in the DN
    # directory.  Unknown image names therefore keep their original stem, but
    # cannot enter an index because they are absent from MODALITY_IMAGE_TYPES.
    return path.stem.strip() or "Image"


class ServerDinoProvider:
    def analyze_bytes(self, images, modality):
        if modality not in MODALITY_IMAGE_TYPES:
            raise ValueError("modality must be 'binmap' or 'sem'")
        if not IMAGE_SERVER_URL:
            raise RuntimeError("尚未配置图像分类服务，请设置 WAFER_IMAGE_SERVER_URL")
        payload = {
            "model": IMAGE_SERVER_MODEL,
            "modality": modality,
            "images": [base64.b64encode(data).decode("ascii") for data in images],
        }
        headers = {"Content-Type": "application/json"}
        if IMAGE_SERVER_TOKEN:
            headers["Authorization"] = f"Bearer {IMAGE_SERVER_TOKEN}"
        request = urllib.request.Request(
            IMAGE_SERVER_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=IMAGE_SERVER_TIMEOUT) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"分类 API 返回 HTTP {exc.code}: {detail[:300]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"无法连接分类 API {IMAGE_SERVER_URL}: {exc.reason}") from exc
        vectors = result.get("embeddings") or result.get("data")
        if isinstance(vectors, list) and vectors and isinstance(vectors[0], dict):
            vectors = [item.get("embedding") for item in vectors]
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != len(images):
            raise RuntimeError("图像服务返回的 embeddings 数量或维度不正确")
        array /= np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)
        return array, list(result.get("predictions") or [])

    def embed_bytes(self, images, modality):
        return self.analyze_bytes(images, modality)[0]


class ImageSearcher:
    def __init__(self):
        self.root = resolve_yedn_root()
        self.provider = ServerDinoProvider()
        self.indexes = {}
        self._records = None
        self._catalog_cache = None
        self._catalog_by_dn = {}
        self._binmap_patch_searcher = None

    def _apply_binmap_local_search(self, image_bytes_list, result, limit, manual_regions=None):
        """按需加载 Bin Map 局部块检索器；SEM 查询不会触发额外 API 调用。"""
        from utils.binmap_patch_search import BinMapPatchSearcher
        if self._binmap_patch_searcher is None:
            self._binmap_patch_searcher = BinMapPatchSearcher(self)
        return self._binmap_patch_searcher.rerank(
            image_bytes_list, result, limit, manual_regions=manual_regions
        )

    def _catalog_folder(self, folder):
        """Scan one DN directory instead of walking the complete image library."""
        items = []
        dn_no = folder.name.strip()
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            image_type = classify_image(path)
            if image_type:
                items.append({
                    "dn_no": dn_no,
                    "image_type": image_type,
                    "file_name": path.name,
                    "path": str(path.resolve()),
                    "url": f"/api/yedn/image/{quote(dn_no, safe='')}/{quote(path.name, safe='')}",
                })
        return items

    def catalog_for_dn(self, dn_no):
        """Return images for one detail drawer without scanning every DN."""
        normalized = str(dn_no).strip().casefold()
        if normalized in self._catalog_by_dn:
            return self._catalog_by_dn[normalized]
        if not normalized or not self.root.exists():
            return []
        folder = (self.root / str(dn_no).strip()).resolve()
        if self.root.resolve() not in folder.parents or not folder.is_dir():
            return []
        items = self._catalog_folder(folder)
        self._catalog_by_dn[normalized] = items
        return items

    def catalog(self):
        if self._catalog_cache is not None:
            return self._catalog_cache
        items = []
        if not self.root.exists():
            self._catalog_cache = items
            return items
        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir() or not folder.name.upper().startswith("DN-"):
                continue
            folder_items = self._catalog_folder(folder)
            items.extend(folder_items)
            self._catalog_by_dn[folder.name.strip().casefold()] = folder_items
        self._catalog_cache = items
        return items

    def records_by_dn(self):
        if self._records is not None:
            return self._records
        result = {}
        for point in get_qdrant_manager().scroll_all_points():
            payload = canonicalize_payload(getattr(point, "payload", {}) or {})
            dn_no = canonical_record_id(payload, getattr(point, "id", None))
            if dn_no:
                result[dn_no] = merge_payloads(result.get(dn_no, {}), payload)
        for payload in result.values():
            payload["text_content"] = describe_payload(payload)
        self._records = result
        return result

    def status(self):
        items = self.catalog()
        dns = {item["dn_no"] for item in items}
        type_counts = {key: sum(item["image_type"] == key for item in items) for key in IMAGE_TYPES}
        return {
            "root": str(self.root),
            "exists": self.root.exists(),
            "dn_count": len(dns),
            "image_count": len(items),
            "type_counts": type_counts,
            "modality_image_counts": {
                modality: sum(item["image_type"] in types for item in items)
                for modality, types in MODALITY_IMAGE_TYPES.items()
            },
            "server_configured": bool(IMAGE_SERVER_URL),
            "model": IMAGE_SERVER_MODEL,
        }

    def filters(self):
        dns = {item["dn_no"] for item in self.catalog()}
        keys = ("platform", "map_location", "machine", "station", "product_id", "lot_id")
        values = {key: set() for key in keys}
        for dn_no, record in self.records_by_dn().items():
            if dn_no not in dns:
                continue
            for key in keys:
                value = record.get(key)
                if not is_missing(value):
                    values[key].add(str(value))
        return {**{key: sorted(found) for key, found in values.items()}, "_status": self.status()}

    def _signature(self, items, modality):
        # Keep the cache portable across computers and installation folders.
        # Absolute paths and nanosecond timestamps change after copying or
        # extracting a delivery ZIP even when every image is identical.
        text = "|".join(
            f"{item['dn_no']}/{item['file_name']}:{Path(item['path']).stat().st_size}"
            for item in items
        )
        return hashlib.sha1((text + "|" + IMAGE_SERVER_MODEL + "|" + modality).encode()).hexdigest()

    def ensure_index(self, modality):
        if modality in self.indexes:
            return self.indexes[modality]
        if modality not in MODALITY_IMAGE_TYPES:
            raise ValueError("不支持的图片类型")
        items = [item for item in self.catalog() if item["image_type"] in MODALITY_IMAGE_TYPES[modality]]
        if not items:
            raise RuntimeError(f"未找到 {MODALITY_LABELS[modality]} 数据，当前目录：{self.root}")
        signature = self._signature(items, modality)
        cache_path = INDEX_PATHS[modality]
        if cache_path.exists():
            try:
                cached = np.load(cache_path, allow_pickle=False)
                if str(cached["signature"].item()) == signature:
                    vectors = cached["vectors"].astype(np.float32)
                    if len(vectors) != len(items):
                        raise ValueError("缓存向量数量与当前图片数量不一致")
                    # Use freshly scanned catalog entries so every path points
                    # to the current installation, not the machine that built
                    # the cache originally.  Vector order is protected by the
                    # signature above.
                    result = (items, vectors)
                    self.indexes[modality] = result
                    return result
            except Exception:
                pass
        batches = []
        for start in range(0, len(items), IMAGE_BATCH_SIZE):
            batch = items[start : start + IMAGE_BATCH_SIZE]
            batches.append(self.provider.embed_bytes([Path(item["path"]).read_bytes() for item in batch], modality))
        vectors = np.concatenate(batches, axis=0)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            vectors=vectors,
            items=json.dumps(items, ensure_ascii=False),
            signature=signature,
        )
        result = (items, vectors)
        self.indexes[modality] = result
        return result

    @staticmethod
    def _matches(record, filters):
        """同一字段允许多个候选值（OR），不同字段之间按 AND 匹配。"""
        for key, requested in filters.items():
            values = requested if isinstance(requested, (list, tuple, set)) else [requested]
            normalized = {str(value).strip().casefold() for value in values if str(value).strip()}
            if normalized and str(record.get(key, "")).strip().casefold() not in normalized:
                return False
        return True

    def search(self, image_bytes, modality, filters=None, limit=50):
        """提取输入向量，并按每个 DN 下同模态图片的平均余弦相似度排序。"""
        started = time.perf_counter()
        items, vectors = self.ensure_index(modality)
        index_ms = (time.perf_counter() - started) * 1000
        query_vectors, _ = self.provider.analyze_bytes([image_bytes], modality)
        query = query_vectors[0]
        embed_ms = (time.perf_counter() - started) * 1000 - index_ms
        filters = {k: v for k, v in (filters or {}).items() if v}
        records = self.records_by_dn()
        scores = vectors @ query

        grouped = {}
        for index, item in enumerate(items):
            enriched = {
                "image_type": item["image_type"],
                "file_name": item["file_name"],
                "url": item["url"],
                "similarity": round(float(scores[index]), 6),
            }
            grouped.setdefault(item["dn_no"], []).append(enriched)

        ranked = []
        for dn_no, result_images in grouped.items():
            payload = records.get(dn_no, {"dn_no": dn_no, "record_id": dn_no})
            if not self._matches(payload, filters):
                continue
            result_images.sort(key=lambda image: image["similarity"], reverse=True)
            mean_score = float(np.mean([image["similarity"] for image in result_images]))
            visual_score = float(np.clip((mean_score + 1.0) / 2.0, 0.0, 1.0))
            ranked.append({
                "dn_no": dn_no,
                "record_id": dn_no,
                "score": round(mean_score, 6),
                "visual_score": round(visual_score, 6),
                "ranking_score": round(visual_score, 6),
                "average_similarity": round(mean_score, 6),
                "best_similarity": result_images[0]["similarity"],
                "best_match": result_images[0],
                "image_count": len(result_images),
                "result_images": result_images,
                "fields": payload,
                "has_qdrant_record": dn_no in records,
            })
        ranked.sort(key=lambda item: item["score"], reverse=True)
        for rank, item in enumerate(ranked, 1):
            item["rank"] = rank
        shown = ranked[:min(int(limit or 50), 100)]
        statistics_table = _statistics_table(ranked)
        total_ms = (time.perf_counter() - started) * 1000
        return {
            "provider": "server",
            "modality": modality,
            "modality_label": MODALITY_LABELS[modality],
            "count": len(shown),
            "candidate_count": len(ranked),
            "results": shown,
            "statistics_table": statistics_table,
            "timings_ms": {
                "index_load": round(index_ms, 2),
                "query_embedding": round(embed_ms, 2),
                "ranking": round(max(0, total_ms - index_ms - embed_ms), 2),
                "total": round(total_ms, 2),
            },
        }

    def _embed_query_batch(self, image_bytes_list, modality):
        """按 API 单批上限提取多张查询图向量。"""
        batches = []
        for start in range(0, len(image_bytes_list), IMAGE_BATCH_SIZE):
            batch = image_bytes_list[start:start + IMAGE_BATCH_SIZE]
            vectors, _ = self.provider.analyze_bytes(batch, modality)
            batches.append(vectors)
        return np.concatenate(batches, axis=0)

    def search_many(self, image_bytes_list, filters=None, limit=100, manual_regions=None):
        """统一检索一到多张图片，并自动融合 Bin Map 与 SEM 索引。

        每张输入图分别与两个图像索引比较；对同一个 DN，先取该输入图在 DN
        全部历史图片中的最高相似度，再对所有输入图的得分求平均。这样既允许
        输入多张同类图，也允许 PPT 中同时包含 Bin Map 和 SEM 图。
        """
        if not image_bytes_list:
            raise ValueError("至少需要一张可解析图片")
        manager = get_qdrant_manager()
        if getattr(manager, "using_unified_collection", False):
            result = self._search_many_unified(manager, image_bytes_list, filters, limit)
            if QDRANT_ONLY:
                # All retrieval/ranking is completed by Qdrant multivectors.
                # The legacy patch reranker opens historical files from disk.
                return result
            return self._apply_binmap_local_search(
                image_bytes_list, result, limit, manual_regions=manual_regions
            )
        started = time.perf_counter()
        available = {}
        index_errors = {}
        index_ms = 0.0
        for modality in MODALITY_IMAGE_TYPES:
            index_started = time.perf_counter()
            try:
                available[modality] = self.ensure_index(modality)
            except Exception as exc:
                index_errors[modality] = str(exc)
            index_ms += (time.perf_counter() - index_started) * 1000
        if not available:
            raise RuntimeError("没有可用的历史图片索引：" + "；".join(index_errors.values()))

        query_vectors = {}
        embed_started = time.perf_counter()
        for modality in available:
            query_vectors[modality] = self._embed_query_batch(image_bytes_list, modality)
        embed_ms = (time.perf_counter() - embed_started) * 1000

        input_count = len(image_bytes_list)
        grouped = {}
        for modality, (items, vectors) in available.items():
            matrix = query_vectors[modality] @ vectors.T
            for image_index, item in enumerate(items):
                group = grouped.setdefault(item["dn_no"], {
                    "query_best": np.full(input_count, -np.inf, dtype=np.float32),
                    "evidence": {},
                })
                scores = matrix[:, image_index]
                group["query_best"] = np.maximum(group["query_best"], scores)
                evidence_key = (item["image_type"], item["file_name"])
                best_query = int(np.argmax(scores))
                evidence = {
                    "image_type": item["image_type"],
                    "file_name": item["file_name"],
                    "url": item["url"],
                    "similarity": round(float(scores[best_query]), 6),
                    "matched_query_index": best_query,
                }
                old = group["evidence"].get(evidence_key)
                if old is None or evidence["similarity"] > old["similarity"]:
                    group["evidence"][evidence_key] = evidence

        filters = {key: value for key, value in (filters or {}).items() if value}
        records = self.records_by_dn()
        ranked = []
        for dn_no, group in grouped.items():
            payload = records.get(dn_no, {"dn_no": dn_no, "record_id": dn_no})
            if filters and not self._matches(payload, filters):
                continue
            best_per_query = group["query_best"]
            finite = best_per_query[np.isfinite(best_per_query)]
            if finite.size != input_count:
                continue
            mean_score = float(np.mean(finite))
            result_images = sorted(
                group["evidence"].values(),
                key=lambda image: image["similarity"],
                reverse=True,
            )
            visual_score = float(np.clip((mean_score + 1.0) / 2.0, 0.0, 1.0))
            ranked.append({
                "dn_no": dn_no,
                "record_id": dn_no,
                "score": round(mean_score, 6),
                "visual_score": round(visual_score, 6),
                "ranking_score": round(visual_score, 6),
                "average_similarity": round(mean_score, 6),
                "best_similarity": result_images[0]["similarity"],
                "best_match": result_images[0],
                "image_count": len(result_images),
                "query_image_count": input_count,
                "query_similarities": [round(float(value), 6) for value in best_per_query],
                "result_images": result_images[:8],
                "fields": payload,
                "has_qdrant_record": dn_no in records,
            })
        ranked.sort(key=lambda item: item["score"], reverse=True)
        for rank, item in enumerate(ranked, 1):
            item["rank"] = rank
        shown = ranked[:min(int(limit or 100), 500)]
        total_ms = (time.perf_counter() - started) * 1000
        result = {
            "provider": "server",
            "mode": "multi_image_fusion",
            "input_image_count": input_count,
            "indexed_modalities": list(available),
            "index_errors": index_errors,
            "count": len(shown),
            "candidate_count": len(ranked),
            "results": shown,
            "statistics_table": _statistics_table(ranked),
            "timings_ms": {
                "index_load": round(index_ms, 2),
                "query_embedding": round(embed_ms, 2),
                "ranking": round(max(0.0, total_ms - index_ms - embed_ms), 2),
                "total": round(total_ms, 2),
            },
        }
        return self._apply_binmap_local_search(
            image_bytes_list, result, limit, manual_regions=manual_regions
        )

    def _search_many_unified(self, manager, image_bytes_list, filters=None, limit=100):
        """Query image multivectors stored on complete DN points in Qdrant."""
        started = time.perf_counter()
        query_vectors = {}
        errors = {}
        for modality in MODALITY_IMAGE_TYPES:
            try:
                query_vectors[modality] = self._embed_query_batch(image_bytes_list, modality)
            except Exception as exc:
                errors[modality] = str(exc)
        if not query_vectors:
            raise RuntimeError("图像向量生成失败：" + "；".join(errors.values()))

        names = {"sem": UNIFIED_SEM_VECTOR, "binmap": UNIFIED_BINMAP_VECTOR}
        grouped = {}
        fetch_limit = min(max(int(limit) * 4, 100), 1000)
        for modality, vectors in query_vectors.items():
            try:
                points = manager.named_multivector_search(
                    names[modality], vectors.tolist(), limit=fetch_limit
                ) or []
            except Exception as exc:
                errors[modality] = str(exc)
                continue
            divisor = max(len(vectors), 1)
            for point in points:
                payload = canonicalize_payload(getattr(point, "payload", {}) or {})
                dn_no = canonical_record_id(payload, getattr(point, "id", None))
                if not dn_no:
                    continue
                score = float(getattr(point, "score", 0.0)) / divisor
                grouped.setdefault(dn_no, {"scores": [], "payload": payload})["scores"].append(score)

        cleaned_filters = {key: value for key, value in (filters or {}).items() if value}
        ranked = []
        for dn_no, item in grouped.items():
            payload = item["payload"]
            if cleaned_filters and not self._matches(payload, cleaned_filters):
                continue
            score = max(item["scores"])
            images = list(payload.get("images") or [])
            evidence = [{
                "image_type": image.get("image_type", "image"),
                "file_name": image.get("file_name", ""),
                "url": image.get("url", ""),
                "similarity": round(score, 6),
            } for image in images]
            best = evidence[0] if evidence else {
                "image_type": "image", "file_name": "", "url": "", "similarity": round(score, 6)
            }
            ranked.append({
                "dn_no": dn_no, "record_id": dn_no,
                "score": round(score, 6),
                "visual_score": round(float(np.clip((score + 1.0) / 2.0, 0.0, 1.0)), 6),
                "ranking_score": round(score, 6),
                "average_similarity": round(score, 6),
                "best_similarity": round(score, 6),
                "best_match": best,
                "image_count": len(images),
                "query_image_count": len(image_bytes_list),
                "result_images": evidence[:8],
                "fields": payload,
                "has_qdrant_record": True,
            })
        ranked.sort(key=lambda value: value["score"], reverse=True)
        for rank, item in enumerate(ranked, 1):
            item["rank"] = rank
        shown = ranked[:min(int(limit or 100), 500)]
        elapsed = (time.perf_counter() - started) * 1000
        return {
            "provider": "qdrant",
            "mode": "unified_dn_multivector",
            "input_image_count": len(image_bytes_list),
            "indexed_modalities": list(query_vectors),
            "index_errors": errors,
            "count": len(shown),
            "candidate_count": len(ranked),
            "results": shown,
            "statistics_table": _statistics_table(ranked),
            "timings_ms": {"total": round(elapsed, 2)},
        }


_searcher = None


def get_image_searcher():
    global _searcher
    if _searcher is None:
        _searcher = ImageSearcher()
    return _searcher
