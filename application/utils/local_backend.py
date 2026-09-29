"""与 QdrantManager 接口一致的本地 CSV/内存后端。"""

import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import LOCAL_DATA_PATH, LOCAL_EMBED_CACHE, LOCAL_BUILD_EMBEDDINGS
from utils.payload_parser import parse_payload_from_text
from utils.field_extractor import FieldExtractor


@dataclass
class LocalPoint:
    id: str
    payload: dict
    score: float = 0.0


FIELD_RULES = {
    "dn_no": r"DN编号为([^，。]+)",
    "class_type": r"类型是([^，。]+)",
    "lot_id": r"LotID为([^，。]+)",
    "product_id": r"属于产品([^，。]+)",
    "platform": r"处于([^，。]+?)平台",
    "station": r"平台的([^，。]+?)站点",
    "machine": r"扫描机器([^，。]+?)于",
    "scan_time": r"于([^，。]+?)完成扫描",
    "total_wafers": r"Wafer总数为(\d+)",
    "scanned_wafers": r"Wafer数量为(\d+)",
    "bad_wafers": r"坏的Wafer数量为(\d+)",
    "defect_type": r"核心缺陷类型为([^，。]+)",
    "edx_elements": r"元素为([^，。]+)",
    "edx_bg": r"EDX背景为([^，。；;]+)",
    "map_location": r"Map上面分布的位置为([^，。]+)",
    "dft_location": r"DFT上面分布的位置为([^，。]+)",
}


def _parse(text):
    payload = parse_payload_from_text(text)
    payload.update({"search_text": str(text).strip(), "grain_type": "full"})
    return payload


class LocalManager:
    def __init__(self):
        df = pd.read_csv(LOCAL_DATA_PATH)
        column = "TEXT" if "TEXT" in df.columns else df.columns[0]
        self.points = []
        for index, text in enumerate(df[column].dropna()):
            payload = _parse(text)
            rid = payload.get("record_id") or f"local-{index}"
            self.points.append(LocalPoint(str(rid), payload))
        # PPT 导入记录已是结构化 payload，直接追加到本地后端；同一 DN 以后导入的内容覆盖旧值。
        try:
            from utils.ppt_import import load_imported_records
            by_id = {point.id: point for point in self.points}
            for payload in load_imported_records():
                rid = str(payload.get("dn_no") or payload.get("record_id") or "").strip()
                if rid:
                    by_id[rid] = LocalPoint(rid, dict(payload))
            self.points = list(by_id.values())
        except Exception:
            # 历史导入文件损坏不应阻断原有 CSV 的启动；导入接口会返回更具体的错误。
            pass
        self._matrix = None
        self._last_query = ""
        self.requires_query_embedding = LOCAL_BUILD_EMBEDDINGS or LOCAL_EMBED_CACHE.exists()

    def set_query(self, query):
        self._last_query = str(query or "")

    def check_connection(self):
        return {"success": True, "points_count": len(self.points),
                "message": f"本地数据加载成功: {LOCAL_DATA_PATH}"}

    def scroll_all_points(self):
        return self.points

    def build_exact_filter(self, conditions):
        return dict(conditions or {})

    def exact_search(self, query_filter, limit=10):
        started = time.perf_counter()
        conditions = query_filter or {}
        result = [p for p in self.points if all(
            str(p.payload.get(k, "")).strip().casefold() == str(v).strip().casefold()
            for k, v in conditions.items())]
        return result[:limit], (time.perf_counter() - started) * 1000, None

    def normalized_exact_search(self, conditions, limit=10):
        results, elapsed, error = self.exact_search(conditions, limit)
        normalized = [LocalPoint(p.id, dict(p.payload), 1.0) for p in results]
        return normalized, elapsed, error

    def vector_search(self, query_vector, limit=10, score_threshold=0.5):
        started = time.perf_counter()
        if self._matrix is None:
            cache_ok = LOCAL_EMBED_CACHE.exists()
            if cache_ok:
                try:
                    cached = np.load(LOCAL_EMBED_CACHE)
                    cache_ok = cached.shape[0] == len(self.points)
                except Exception:
                    cache_ok = False
            if cache_ok:
                self._matrix = cached.astype(np.float32)
            elif LOCAL_BUILD_EMBEDDINGS:
                from utils.embedding11 import get_embedding_generator
                embedder = get_embedding_generator()
                texts = [p.payload.get("search_text") or p.payload.get("text_content", "")
                         for p in self.points]
                vectors = (embedder.get_embeddings(texts) if hasattr(embedder, "get_embeddings")
                           else [embedder.get_embedding(text) for text in texts])
                self._matrix = np.asarray(vectors, dtype=np.float32)
                LOCAL_EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
                np.save(LOCAL_EMBED_CACHE, self._matrix)
            else:
                # 快速本地模式：首次运行不阻塞在 500 条 Qwen 文档向量上，
                # 先用字符 bigram 近似语义召回；设置 WAFER_LOCAL_BUILD_EMBEDDINGS=1
                # 后可切换为严格 Qwen 向量检索。
                from collections import Counter
                def grams(value):
                    value = str(value).lower()
                    return Counter(value[i:i + 2] for i in range(max(0, len(value) - 1)))
                qgrams = grams(getattr(self, "_last_query", ""))
                def cosine(a, b):
                    dot = sum(a[k] * b.get(k, 0) for k in a)
                    na = sum(v * v for v in a.values()) ** 0.5
                    nb = sum(v * v for v in b.values()) ** 0.5
                    return dot / (na * nb) if na and nb else 0.0
                scored = [(cosine(qgrams, grams(p.payload.get("search_text")
                                                or p.payload.get("text_content", ""))), p)
                          for p in self.points]
                scored.sort(key=lambda x: x[0], reverse=True)
                return [LocalPoint(p.id, dict(p.payload), float(score))
                        for score, p in scored[:limit]], (time.perf_counter() - started) * 1000, None
            self._matrix /= np.maximum(np.linalg.norm(self._matrix, axis=1, keepdims=True), 1e-12)
        query = np.asarray(query_vector, dtype=np.float32)
        query /= max(float(np.linalg.norm(query)), 1e-12)
        scores = self._matrix @ query
        order = np.argsort(-scores)
        result = [LocalPoint(self.points[i].id, dict(self.points[i].payload), float(scores[i]))
                  for i in order if scores[i] >= score_threshold][:limit]
        return result, (time.perf_counter() - started) * 1000, None
