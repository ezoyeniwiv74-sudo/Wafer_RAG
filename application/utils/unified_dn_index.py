"""Build and maintain one Qdrant point for each DN record.

The module intentionally keeps raw image files outside Qdrant.  A point stores
the complete business payload, a portable image manifest, text chunk vectors,
and modality-specific image multivectors.  Re-importing the same DN updates the
same deterministic UUID, so ingestion is idempotent.
"""
from __future__ import annotations

import hashlib
import json
import os
import shelve
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from config import (
    COLLECTION_NAME,
    QDRANT_SCROLL_BATCH,
    QDRANT_LOCAL_PATH,
    QDRANT_URL,
    RUNTIME_PROFILE,
    UNIFIED_BINMAP_VECTOR,
    UNIFIED_COLLECTION_NAME,
    UNIFIED_SEM_VECTOR,
    UNIFIED_TEXT_VECTOR,
)
from utils.payload_parser import (
    canonical_record_id,
    canonicalize_payload,
    describe_payload,
    is_missing,
    merge_payloads,
)

BUSINESS_FIELDS = (
    "dn_no", "class_type", "lot_id", "product_id", "platform", "station",
    "machine", "scan_time", "total_wafers", "scanned_wafers", "bad_wafers",
    "defect_type", "edx_elements", "edx_bg", "map_location", "dft_location",
)
PAYLOAD_INDEX_FIELDS = (
    "dn_no", "lot_id", "product_id", "platform", "station", "machine",
    "defect_type", "map_location", "dft_location", "source_type",
)


def point_id_for_dn(dn_no: str) -> str:
    """Return a stable Qdrant-compatible UUID for a DN number."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"wafer-dn:{normalize_dn(dn_no)}"))


def normalize_dn(value: str) -> str:
    return str(value or "").strip().upper()


def build_text_chunks(payload: dict) -> list[str]:
    """Create a small set of complementary chunks without duplicating a DN."""
    value = canonicalize_payload(payload)
    full_text = describe_payload(value).strip()
    identity = "；".join(
        f"{key}={value[key]}" for key in BUSINESS_FIELDS
        if not is_missing(value.get(key)) and value.get(key) != 0
    )
    defect = "；".join(
        f"{key}={value[key]}" for key in (
            "defect_type", "class_type", "edx_elements", "edx_bg",
            "map_location", "dft_location", "platform", "machine", "station",
        ) if not is_missing(value.get(key)) and value.get(key) != 0
    )
    result = []
    for text in (identity, defect, full_text):
        text = str(text or "").strip()
        if text and text.casefold() not in {old.casefold() for old in result}:
            result.append(text)
    return result or [normalize_dn(value.get("dn_no") or value.get("record_id"))]


def merge_records(points) -> dict[str, dict]:
    """Merge legacy multi-grain points into one canonical payload per DN."""
    merged: dict[str, dict] = {}
    for point in points:
        payload = canonicalize_payload(getattr(point, "payload", {}) or {})
        dn_no = normalize_dn(canonical_record_id(payload, getattr(point, "id", None)))
        if not dn_no or dn_no.startswith("UNKNOWN"):
            continue
        payload["dn_no"] = dn_no
        payload["record_id"] = dn_no
        merged[dn_no] = merge_payloads(merged.get(dn_no, {}), payload)
    return merged


def image_manifest(catalog: list[dict]) -> dict[str, list[dict]]:
    """Group catalog entries and replace machine-specific paths with keys."""
    grouped = defaultdict(list)
    for item in catalog:
        dn_no = normalize_dn(item.get("dn_no"))
        if not dn_no:
            continue
        grouped[dn_no].append({
            "image_type": item["image_type"],
            "file_name": item["file_name"],
            "storage_key": f"YEDN/{dn_no}/{item['file_name']}",
            "url": item.get("url") or f"/api/yedn/image/{dn_no}/{item['file_name']}",
        })
    return dict(grouped)


def content_hash(payload: dict, images: list[dict]) -> str:
    stable = {
        "payload": {key: payload.get(key) for key in sorted(payload) if key not in {"updated_at", "content_hash"}},
        "images": images,
    }
    encoded = json.dumps(stable, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def scroll_collection(client, collection_name: str):
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=QDRANT_SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        yield from points
        if offset is None:
            break


class UnifiedDnIndexer:
    """Read legacy data, generate vectors, and upsert complete DN points."""

    def __init__(self, client=None):
        from qdrant_client import QdrantClient
        self.client = client or (
            QdrantClient(path=QDRANT_LOCAL_PATH)
            if QDRANT_LOCAL_PATH else QdrantClient(url=QDRANT_URL)
        )

    def load_records(self):
        # A clean local Docker installation has no legacy Qdrant collection.
        # Bootstrap directly from the bundled CSV in that case; installations
        # migrated from an older release can still use wafer_collection.
        use_bundled_source = (
            RUNTIME_PROFILE == "local"
            or bool(QDRANT_LOCAL_PATH)
            or not self.client.collection_exists(COLLECTION_NAME)
        )
        if use_bundled_source:
            from utils.local_backend import LocalManager
            records = merge_records(LocalManager().scroll_all_points())
        else:
            records = merge_records(scroll_collection(self.client, COLLECTION_NAME))

        # PPT imports may not yet have been inserted into the legacy collection.
        try:
            from utils.ppt_import import load_imported_records
            for payload in load_imported_records():
                current = canonicalize_payload(payload)
                dn_no = normalize_dn(canonical_record_id(current))
                if dn_no:
                    current["dn_no"] = current["record_id"] = dn_no
                    records[dn_no] = merge_payloads(records.get(dn_no, {}), current)
        except Exception:
            pass
        return records

    @staticmethod
    def _embed_texts(records):
        from utils.embedding11 import get_embedding_generator
        embedder = get_embedding_generator()
        tasks = []
        result = {dn_no: [] for dn_no in records}
        for dn_no, payload in records.items():
            # A single complete business description is sufficient for the
            # DN-level collection and keeps initial CPU ingestion practical.
            tasks.append((dn_no, build_text_chunks(payload)[-1]))
        batch_size = max(1, int(os.getenv("WAFER_UNIFIED_TEXT_BATCH", "4")))
        cache_path = str((Path(__file__).resolve().parents[1] / ".cache" / "unified_text_vectors").resolve())
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with shelve.open(cache_path) as cache:
            for start in range(0, len(tasks), batch_size):
                batch = tasks[start:start + batch_size]
                keys = [hashlib.sha256(text.encode("utf-8")).hexdigest() for _, text in batch]
                vectors = [cache.get(key) for key in keys]
                missing = [index for index, vector in enumerate(vectors) if vector is None]
                if missing:
                    texts = [batch[index][1] for index in missing]
                    generated = (embedder.get_embeddings(texts, batch_size=batch_size)
                                 if hasattr(embedder, "get_embeddings")
                                 else [embedder.get_embedding(text) for text in texts])
                    for index, vector in zip(missing, generated):
                        vectors[index] = vector
                        cache[keys[index]] = vector
                    cache.sync()
                for (dn_no, _), vector in zip(batch, vectors):
                    result[dn_no].append(vector)
                print(f"[text] {min(start + len(batch), len(tasks))}/{len(tasks)}")
        return result

    @staticmethod
    def _embed_images(searcher, catalog):
        vectors = {"sem": {}, "binmap": {}}
        allowed_paths = {str(Path(item["path"]).resolve()) for item in catalog}
        for modality in vectors:
            items, values = searcher.ensure_index(modality)
            used = 0
            for item, vector in zip(items, values):
                if str(Path(item["path"]).resolve()) not in allowed_paths:
                    continue
                dn_no = normalize_dn(item["dn_no"])
                vectors[modality].setdefault(dn_no, []).append(vector.tolist())
                used += 1
            print(f"[{modality}] reused {used} cached vectors")
        return vectors

    def _create_collection(self, text_size, sem_size, binmap_size, recreate=False):
        from qdrant_client import models
        if self.client.collection_exists(UNIFIED_COLLECTION_NAME):
            if not recreate:
                return
            self.client.delete_collection(UNIFIED_COLLECTION_NAME)
        multi = models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM)
        vectors_config = {
            UNIFIED_TEXT_VECTOR: models.VectorParams(
                size=text_size, distance=models.Distance.COSINE, multivector_config=multi
            ),
        }
        if sem_size:
            vectors_config[UNIFIED_SEM_VECTOR] = models.VectorParams(
                size=sem_size, distance=models.Distance.COSINE, multivector_config=multi
            )
        if binmap_size:
            vectors_config[UNIFIED_BINMAP_VECTOR] = models.VectorParams(
                size=binmap_size, distance=models.Distance.COSINE, multivector_config=multi
            )
        self.client.create_collection(
            collection_name=UNIFIED_COLLECTION_NAME,
            vectors_config=vectors_config,
        )
        for field in PAYLOAD_INDEX_FIELDS:
            self.client.create_payload_index(
                collection_name=UNIFIED_COLLECTION_NAME,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )

    def run(self, execute=False, recreate=False, batch_size=16, dn_filter=None):
        from qdrant_client import models
        from utils.image_search import get_image_searcher

        records = self.load_records()
        selected_dns = {normalize_dn(value) for value in (dn_filter or []) if normalize_dn(value)}
        if selected_dns:
            records = {dn: payload for dn, payload in records.items() if dn in selected_dns}
        searcher = get_image_searcher()
        catalog = searcher.catalog()
        if selected_dns:
            catalog = [item for item in catalog if normalize_dn(item.get("dn_no")) in selected_dns]
        manifests = image_manifest(catalog)
        report = {
            "collection": UNIFIED_COLLECTION_NAME,
            "dn_count": len(records),
            "image_count": len(catalog),
            "missing_images": sorted(set(records) - set(manifests)),
            "orphan_image_dns": sorted(set(manifests) - set(records)),
            "execute": bool(execute),
        }
        if not execute:
            return report

        text_vectors = self._embed_texts(records)
        image_vectors = self._embed_images(searcher, catalog)
        text_size = len(next(iter(text_vectors.values()))[0])
        sem_size = len(next(iter(image_vectors["sem"].values()))[0]) if image_vectors["sem"] else 0
        binmap_size = len(next(iter(image_vectors["binmap"].values()))[0]) if image_vectors["binmap"] else 0
        self._create_collection(text_size, sem_size, binmap_size, recreate=recreate)

        points = []
        for dn_no, payload in records.items():
            images = manifests.get(dn_no, [])
            complete = canonicalize_payload(payload)
            complete.update({
                "dn_no": dn_no,
                "record_id": dn_no,
                "text_content": describe_payload(complete),
                "images": images,
                "image_count": len(images),
                "schema_version": 2,
                "content_hash": content_hash(complete, images),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            named_vectors = {UNIFIED_TEXT_VECTOR: text_vectors[dn_no]}
            if dn_no in image_vectors["sem"]:
                named_vectors[UNIFIED_SEM_VECTOR] = image_vectors["sem"][dn_no]
            if dn_no in image_vectors["binmap"]:
                named_vectors[UNIFIED_BINMAP_VECTOR] = image_vectors["binmap"][dn_no]
            points.append(models.PointStruct(
                id=point_id_for_dn(dn_no), vector=named_vectors, payload=complete
            ))
            if len(points) >= batch_size:
                self.client.upsert(UNIFIED_COLLECTION_NAME, points=points, wait=True)
                points = []
        if points:
            self.client.upsert(UNIFIED_COLLECTION_NAME, points=points, wait=True)
        report.update({
            "text_vector_size": text_size,
            "sem_vector_size": sem_size,
            "binmap_vector_size": binmap_size,
        })
        return report
