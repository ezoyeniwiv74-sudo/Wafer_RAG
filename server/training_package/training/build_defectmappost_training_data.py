"""Build an unlabeled Bin Map training set from YEDN DN directories.

This file is designed to be opened and run directly in PyCharm.

Production rule
---------------
For every direct ``DN-*`` directory under ``YEDN_ROOT``, only the file named
``DefectMapPost.JPG`` (case-insensitive) is accepted.  A DN without that file
is recorded as skipped and is never added to the generated training set.

Local demo rule
---------------
For the local demonstration, a separate helper can place public WM811K maps
into the local YEDN directories under the exact filename
``DefectMapPost.JPG``.  This builder then runs in the same ``company`` mode as
it will inside the company: only the exact target filename is extracted.

Generated layout
----------------
``OUTPUT_PARENT/build_YYYYMMDD_HHMMSS/DN-.../DefectMapPost.JPG``

No labels.csv is generated because the downstream Bin Map LoRA training is
self-supervised and reads images recursively without class labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tarfile
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# PyCharm direct-run settings
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
# 在项目 tools 目录内运行时使用项目根目录；作为独立交付脚本放在部署资料
# 根目录时，则使用脚本所在目录。这样 PyCharm 直接运行和单文件交付都能得到
# 可预测的 data/binmap_ssl_defectmappost 输出位置。
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.casefold() == "tools" else SCRIPT_DIR

# "demo": legacy isolated demo mode using public TypicalMap images.
# "company": extract only real DefectMapPost.JPG files from COMPANY_YEDN_ROOT.
RUN_MODE = "company"

COMPANY_YEDN_ROOT = PROJECT_ROOT / "data" / "YEDN"
PUBLIC_YEDN_ROOT = PROJECT_ROOT / "data" / "YEDN"
DEMO_YEDN_ROOT = PROJECT_ROOT / "data" / "YEDN_DefectMapPost_demo"
OUTPUT_PARENT = PROJECT_ROOT / "data" / "binmap_ssl_defectmappost"

# The fixed archive name makes the upload and server commands repeatable.
# Its top-level directory is ``binmap_ssl`` so extracting it with
# ``tar -C /data/RAG/dataset`` creates the exact training data path expected
# by the deployment document: /data/RAG/dataset/binmap_ssl.
TAR_OUTPUT_PATH = OUTPUT_PARENT / "company_training_data.tar"
TAR_ROOT_NAME = "binmap_ssl"

# 0 means all eligible DN directories.  Set a positive number only for a
# small smoke test; company formal construction should keep 0.
MAX_DN_COUNT = 0

TARGET_FILENAME = "DefectMapPost.JPG"
SUPPORTED_PUBLIC_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def dn_directories(root: Path) -> list[Path]:
    """Return direct ``DN-*`` children in a stable order."""
    if not root.is_dir():
        raise FileNotFoundError(f"YEDN directory does not exist: {root}")
    directories = sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.upper().startswith("DN-")),
        key=lambda path: path.name.casefold(),
    )
    if MAX_DN_COUNT > 0:
        directories = directories[:MAX_DN_COUNT]
    return directories


def find_direct_file_case_insensitive(directory: Path, filename: str) -> Path | None:
    """Find one direct child by filename without accepting substitute images."""
    exact = directory / filename
    if exact.is_file():
        return exact
    expected = filename.casefold()
    matches = sorted(
        path for path in directory.iterdir() if path.is_file() and path.name.casefold() == expected
    )
    return matches[0] if matches else None


def find_public_typical_map(directory: Path) -> Path | None:
    """Select one public TypicalMap image for demo data generation only."""
    candidates = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_PUBLIC_SUFFIXES
        and path.stem.casefold().startswith("typicalmap")
    )
    return candidates[0] if candidates else None


def create_public_demo_yedn(public_root: Path, demo_root: Path) -> dict[str, int]:
    """Create simulated DefectMapPost.JPG files without touching public_root."""
    # The production extractor below only copies existing JPG files and uses
    # the Python standard library.  Pillow is imported lazily because it is
    # needed solely to convert public PNG maps for the local demo.
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise RuntimeError(
            "Demo mode needs Pillow. Run this script with the existing "
            "D:\\Anaconda\\python.exe interpreter, or switch RUN_MODE to "
            "'company' when extracting real DefectMapPost.JPG files."
        ) from exc

    demo_root.mkdir(parents=True, exist_ok=True)
    created = 0
    skipped = 0
    invalid = 0

    for dn_dir in dn_directories(public_root):
        source = find_public_typical_map(dn_dir)
        if source is None:
            skipped += 1
            continue

        destination_dir = demo_root / dn_dir.name
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / TARGET_FILENAME
        try:
            with Image.open(source) as image:
                # JPEG cannot store alpha; a white canvas preserves transparent
                # or palette-based public maps in a deterministic RGB file.
                rgba = image.convert("RGBA")
                canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                canvas.alpha_composite(rgba)
                canvas.convert("RGB").save(
                    destination,
                    format="JPEG",
                    quality=95,
                    subsampling=0,
                )
            created += 1
        except (UnidentifiedImageError, OSError):
            invalid += 1

    return {"created": created, "missing_typicalmap": skipped, "invalid_image": invalid}


def extract_training_set(yedn_root: Path, output_parent: Path) -> tuple[Path, dict[str, int]]:
    """Copy only real DefectMapPost.JPG files into a fresh training build."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = output_parent / f"build_{timestamp}"
    suffix = 1
    while output_root.exists():
        output_root = output_parent / f"build_{timestamp}_{suffix:02d}"
        suffix += 1
    output_root.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, str]] = []
    copied = 0
    missing = 0

    for dn_dir in dn_directories(yedn_root):
        source = find_direct_file_case_insensitive(dn_dir, TARGET_FILENAME)
        if source is None:
            missing += 1
            rows.append(
                {
                    "dn": dn_dir.name,
                    "status": "skipped_missing_DefectMapPost.JPG",
                    "source_path": "",
                    "training_path": "",
                }
            )
            continue

        destination_dir = output_root / dn_dir.name
        destination_dir.mkdir(parents=True, exist_ok=False)
        destination = destination_dir / TARGET_FILENAME
        shutil.copy2(source, destination)
        copied += 1
        rows.append(
            {
                "dn": dn_dir.name,
                "status": "included",
                "source_path": str(source),
                "training_path": str(destination.relative_to(output_root)),
            }
        )

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["dn", "status", "source_path", "training_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source_yedn": str(yedn_root),
        "output_root": str(output_root),
        "included_dn_count": copied,
        "skipped_missing_count": missing,
        "labels_required": False,
        "target_filename": TARGET_FILENAME,
    }
    (output_root / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_root, {"included": copied, "skipped_missing": missing}


def package_training_set(output_root: Path, tar_path: Path) -> Path:
    """Package one completed build into a server-ready uncompressed TAR.

    A temporary TAR is written first and moved into place only after the
    archive closes successfully.  This prevents a failed run from leaving a
    half-written ``company_training_data.tar`` that could be uploaded by
    mistake.
    """
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = tar_path.with_name(f".{tar_path.name}.tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    try:
        with tarfile.open(temporary_path, mode="w") as archive:
            archive.add(output_root, arcname=TAR_ROOT_NAME, recursive=True)
        os.replace(temporary_path, tar_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return tar_path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an unlabeled DefectMapPost Bin Map training directory."
    )
    parser.add_argument(
        "--mode",
        choices=("demo", "company"),
        default=RUN_MODE,
        help="demo creates a public-data simulation; company reads real DefectMapPost.JPG files.",
    )
    parser.add_argument("--yedn-root", type=Path, default=None)
    parser.add_argument("--output-parent", type=Path, default=OUTPUT_PARENT)
    parser.add_argument("--demo-root", type=Path, default=DEMO_YEDN_ROOT)
    parser.add_argument(
        "--tar-output",
        type=Path,
        default=TAR_OUTPUT_PATH,
        help="Fixed path of the server-ready company_training_data.tar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    source_root: Path

    if args.mode == "demo":
        public_root = args.yedn_root or PUBLIC_YEDN_ROOT
        demo_result = create_public_demo_yedn(public_root, args.demo_root)
        print(f"[DEMO] Public source: {public_root}")
        print(f"[DEMO] Simulated YEDN: {args.demo_root}")
        print(f"[DEMO] Result: {json.dumps(demo_result, ensure_ascii=False)}")
        source_root = args.demo_root
    else:
        source_root = args.yedn_root or COMPANY_YEDN_ROOT

    output_root, result = extract_training_set(source_root, args.output_parent)
    print(f"[PASS] Training data: {output_root}")
    print(f"[PASS] Included DN: {result['included']}")
    print(f"[PASS] Skipped DN without {TARGET_FILENAME}: {result['skipped_missing']}")
    if result["included"] == 0:
        raise RuntimeError(f"No {TARGET_FILENAME} files were found under {source_root}")
    tar_path = package_training_set(output_root, args.tar_output)
    print(f"[PASS] Training data TAR: {tar_path}")


if __name__ == "__main__":
    main()
