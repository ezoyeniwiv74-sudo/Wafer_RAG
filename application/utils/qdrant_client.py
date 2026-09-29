"""
Qdrant 客户端封装模块
提供精确查询和语义向量查询的统一接口
"""

import time
from types import SimpleNamespace
from qdrant_client import QdrantClient, models
from config import (
    QDRANT_URL, COLLECTION_NAME, RUNTIME_PROFILE, QDRANT_SCROLL_BATCH,
    UNIFIED_COLLECTION_NAME, UNIFIED_TEXT_VECTOR, QDRANT_LOCAL_PATH, QDRANT_ONLY,
)
from utils.field_extractor import FieldExtractor
from utils.payload_parser import canonicalize_payload, canonical_record_id


class QdrantManager:
    """Qdrant 管理器"""

    def __init__(self):
        self.client = (
            QdrantClient(path=QDRANT_LOCAL_PATH)
            if QDRANT_LOCAL_PATH else QdrantClient(url=QDRANT_URL)
        )
        self.collection_name = self._select_collection()
        self.using_unified_collection = self.collection_name == UNIFIED_COLLECTION_NAME
        self._points_cache = None

    def _select_collection(self):
        """Prefer the DN-centric collection after migration has created it."""
        try:
            if UNIFIED_COLLECTION_NAME and self.client.collection_exists(UNIFIED_COLLECTION_NAME):
                return UNIFIED_COLLECTION_NAME
        except Exception:
            # A temporary Qdrant outage must not change import-time behaviour.
            pass
        if QDRANT_ONLY:
            # Fail visibly later in check/search instead of reading a legacy or
            # local dataset when the unified production collection is missing.
            return UNIFIED_COLLECTION_NAME
        return COLLECTION_NAME

    def check_connection(self):
        """检查 Qdrant 连接和 Collection"""
        try:
            collection_info = self.client.get_collection(
                collection_name=self.collection_name
            )
            return {
                "success": True,
                "points_count": collection_info.points_count,
                "message": f"Qdrant 连接成功，Collection: {self.collection_name}"
            }
        except Exception as e:
            return {
                "success": False,
                "points_count": 0,
                "message": (
                    f"Qdrant 连接失败或 Collection 不存在\n"
                    f"1. Qdrant 服务是否启动\n"
                    f"2. 地址是否为 {QDRANT_URL}\n"
                    f"3. Collection 是否为 {self.collection_name}\n"
                    f"原始错误：{e}"
                )
            }

    def exact_search(self, query_filter, limit: int = 10):
        """
        精确查询（Payload 过滤）

        Args:
            query_filter: Qdrant Filter 对象
            limit: 返回结果数量上限

        Returns:
            (points列表, 耗时ms, 错误信息)
        """
        start_time = time.perf_counter()
        try:
            points = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return points[0], elapsed_ms, None
        except Exception as e:
            return None, 0, f"Qdrant 精确查询失败: {e}"

    def vector_search(self, query_vector: list, limit: int = 10,
                      score_threshold: float = 0.5):
        """
        向量语义检索

        Args:
            query_vector: 查询向量
            limit: 返回结果数量上限
            score_threshold: 最低相似度阈值

        Returns:
            (points列表, 耗时ms, 错误信息)
        """
        start_time = time.perf_counter()
        try:
            query_args = {
                "collection_name": self.collection_name,
                "query": [query_vector] if self.using_unified_collection else query_vector,
                "limit": limit,
                "score_threshold": score_threshold,
                "with_payload": True,
                "with_vectors": False,
            }
            if self.using_unified_collection:
                query_args["using"] = UNIFIED_TEXT_VECTOR
            result = self.client.query_points(**query_args)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return result.points, elapsed_ms, None
        except Exception as e:
            error_text = str(e).lower()
            if "dimension" in error_text or "vector" in error_text:
                error_type = "查询向量与 Collection 向量维度不一致"
            elif "collection" in error_text:
                error_type = f"Collection 不存在: {self.collection_name}"
            elif "connection" in error_text:
                error_type = "无法连接 Qdrant 服务"
            else:
                error_type = "Qdrant 向量查询异常"
            return None, 0, f"{error_type}，原始错误: {e}"

    def named_multivector_search(self, vector_name, query_vectors, limit=100,
                                 query_filter=None):
        """Search one named multivector and return complete DN points.

        MaxSim returns a sum over all query vectors. Dividing by the number of
        uploaded images keeps scores comparable for one-image and multi-image
        requests.
        """
        if not self.using_unified_collection:
            return None
        vectors = [list(map(float, vector)) for vector in query_vectors]
        if not vectors:
            return []
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=vectors,
            using=vector_name,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return result.points

    def build_exact_filter(self, conditions: dict) -> models.Filter:
        """
        构建精确匹配过滤条件

        Args:
            conditions: {字段名: 字段值} 的字典

        Returns:
            Qdrant Filter 对象
        """
        if not conditions:
            return None

        must_conditions = []
        for field_name, field_value in conditions.items():
            must_conditions.append(
                models.FieldCondition(
                    key=field_name,
                    match=models.MatchValue(value=field_value)
                )
            )

        return models.Filter(must=must_conditions)
    def scroll_all_points(self):
        """
        扫描整个collection
        用于BM25/TF-IDF构建语料
        """

        if self._points_cache is not None:
            return self._points_cache

        all_points=[]

        offset=None


        while True:


            points, next_offset = (
                self.client.scroll(
                    collection_name=self.collection_name,

                    limit=QDRANT_SCROLL_BATCH,

                    offset=offset,

                    with_payload=True,

                    with_vectors=False
                )
            )


            all_points.extend(points)


            if next_offset is None:

                break


            offset=next_offset


        self._points_cache = all_points
        return all_points

    def normalized_exact_search(self, conditions, limit=10):
        """兼容旧库缺字段/数字 record_id：从原始描述补齐后在客户端精确过滤。"""
        started = time.perf_counter()
        records = {}
        for point in self.scroll_all_points():
            payload = canonicalize_payload(point.payload)
            rid = canonical_record_id(payload, getattr(point, "id", None))
            if not rid or rid in records:
                continue
            if FieldExtractor.condition_match(payload, conditions):
                records[rid] = SimpleNamespace(id=rid, payload=payload, score=1.0)
                if len(records) >= limit:
                    break
        return list(records.values()), (time.perf_counter() - started) * 1000, None

# 单例模式
_qdrant_manager = None


def get_qdrant_manager() -> QdrantManager:
    """Return the Qdrant backend; local mode is opt-in for offline tests only."""
    global _qdrant_manager
    if _qdrant_manager is None:
        if RUNTIME_PROFILE == "local" and not QDRANT_ONLY:
            from utils.local_backend import LocalManager
            _qdrant_manager = LocalManager()
        else:
            _qdrant_manager = QdrantManager()
    return _qdrant_manager
