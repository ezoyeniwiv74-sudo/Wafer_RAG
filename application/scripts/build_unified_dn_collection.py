"""Command-line entry point for the DN-centric Qdrant migration."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.unified_dn_index import UnifiedDnIndexer
from config import UNIFIED_COLLECTION_NAME


def main():
    parser = argparse.ArgumentParser(
        description="Group text and images by DN and write one complete Qdrant point per DN."
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Generate embeddings and write Qdrant. Without this flag only validate inputs.",
    )
    parser.add_argument(
        "--recreate", action="store_true",
        help="Delete and recreate only wafer_dn_records_v2. The legacy collection is untouched.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--dn", action="append", default=[],
        help="Only update this DN (repeatable). The unified collection must already exist.",
    )
    parser.add_argument(
        "--report", default=str(ROOT / "logs" / "unified_dn_import_report.json")
    )
    args = parser.parse_args()
    if args.recreate and not args.execute:
        parser.error("--recreate requires --execute")

    indexer = UnifiedDnIndexer()
    if args.dn and not indexer.client.collection_exists(UNIFIED_COLLECTION_NAME):
        parser.error("--dn can only be used after the full unified collection has been created")
    report = indexer.run(
        execute=args.execute, recreate=args.recreate,
        batch_size=max(1, args.batch_size), dn_filter=args.dn,
    )
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[PASS] report: {target}")


if __name__ == "__main__":
    main()
