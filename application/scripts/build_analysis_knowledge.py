"""Embed curated primary-source methodology notes into Qdrant."""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient, models

from config import ANALYSIS_KNOWLEDGE_COLLECTION, QDRANT_LOCAL_PATH, QDRANT_URL
from utils.embedding11 import get_embedding_generator


def document_text(item):
    return "\n".join([
        item["title"], item["organization"], " ".join(item.get("topics", [])),
        item["summary_zh"], "；".join(item.get("analysis_methods", [])),
        item.get("limitations", ""),
    ])


def main():
    parser = argparse.ArgumentParser(description="Build the wafer methodology Qdrant collection")
    parser.add_argument("--source", default=str(ROOT / "data" / "analysis_knowledge.json"))
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()
    records = json.loads(Path(args.source).read_text(encoding="utf-8"))
    if not records:
        raise SystemExit("knowledge source is empty")
    vectors = get_embedding_generator().get_embeddings([document_text(item) for item in records], batch_size=8)
    client = QdrantClient(path=QDRANT_LOCAL_PATH) if QDRANT_LOCAL_PATH else QdrantClient(url=QDRANT_URL)
    exists = client.collection_exists(ANALYSIS_KNOWLEDGE_COLLECTION)
    if exists and args.recreate:
        client.delete_collection(ANALYSIS_KNOWLEDGE_COLLECTION)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=ANALYSIS_KNOWLEDGE_COLLECTION,
            vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
        )
        for field in ("knowledge_id", "organization", "topics"):
            client.create_payload_index(
                collection_name=ANALYSIS_KNOWLEDGE_COLLECTION,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
    points = []
    for item, vector in zip(records, vectors):
        payload = dict(item)
        payload["document_text"] = document_text(item)
        payload["source_type"] = "official_industry_primary_source"
        points.append(models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, item["knowledge_id"])),
            vector=vector,
            payload=payload,
        ))
    client.upsert(ANALYSIS_KNOWLEDGE_COLLECTION, points=points, wait=True)
    print(json.dumps({
        "success": True,
        "collection": ANALYSIS_KNOWLEDGE_COLLECTION,
        "points_count": len(points),
        "vector_size": len(vectors[0]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
