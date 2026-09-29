"""Qdrant-backed industrial wafer analysis knowledge retrieval."""
from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient

from config import (
    ANALYSIS_KNOWLEDGE_COLLECTION,
    ANALYSIS_KNOWLEDGE_TOP_K,
    QDRANT_LOCAL_PATH,
    QDRANT_URL,
)
from utils.embedding11 import get_embedding_generator


class AnalysisKnowledgeStore:
    """Retrieve methodology guidance from Qdrant, never from local files."""

    def __init__(self):
        self.client = (
            QdrantClient(path=QDRANT_LOCAL_PATH)
            if QDRANT_LOCAL_PATH else QdrantClient(url=QDRANT_URL)
        )

    def status(self) -> dict[str, Any]:
        try:
            info = self.client.get_collection(ANALYSIS_KNOWLEDGE_COLLECTION)
            return {
                "ok": True,
                "collection": ANALYSIS_KNOWLEDGE_COLLECTION,
                "points_count": int(info.points_count or 0),
                "source": "qdrant",
            }
        except Exception as exc:
            return {
                "ok": False,
                "collection": ANALYSIS_KNOWLEDGE_COLLECTION,
                "points_count": 0,
                "source": "qdrant",
                "detail": str(exc),
            }

    def retrieve(self, query: str, limit: int | None = None) -> list[dict[str, Any]]:
        if not query.strip():
            query = "晶圆异常空间分布 缺陷分类 设备批次 根因分析 验证方法"
        vector = get_embedding_generator().get_embedding(query, is_query=True)
        result = self.client.query_points(
            collection_name=ANALYSIS_KNOWLEDGE_COLLECTION,
            query=vector,
            limit=max(1, min(int(limit or ANALYSIS_KNOWLEDGE_TOP_K), 12)),
            with_payload=True,
            with_vectors=False,
        )
        output = []
        for point in result.points:
            payload = dict(getattr(point, "payload", {}) or {})
            payload["score"] = round(float(getattr(point, "score", 0.0)), 6)
            output.append(payload)
        return output


_knowledge_store = None


def get_analysis_knowledge_store() -> AnalysisKnowledgeStore:
    global _knowledge_store
    if _knowledge_store is None:
        _knowledge_store = AnalysisKnowledgeStore()
    return _knowledge_store
