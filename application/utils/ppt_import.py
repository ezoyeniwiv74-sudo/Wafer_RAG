"""使用 Python 标准库解析公司 DN 缺陷 PPTX，并转换为系统可直接检索的数据。

不依赖 python-pptx：PPTX 本身是 ZIP + XML，本模块直接读取其中的幻灯片文本、
图片关系和版面坐标。导入结果写到 ``data/ppt_import_records.jsonl``，图片写入
``data/YEDN/<DN>/``，因此会自动进入现有文字检索与图像检索数据链路。
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET

try:
    from PIL import Image, ImageOps
except ModuleNotFoundError:
    # 项目已离线携带 Pillow，避免公司环境额外联网安装依赖。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_vendor"))
    from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
YEDN_DIR = DATA_DIR / "YEDN"
UPLOAD_DIR = DATA_DIR / "ppt_uploads"
RECORDS_PATH = DATA_DIR / "ppt_import_records.jsonl"
HISTORY_PATH = DATA_DIR / "ppt_import_history.json"

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}

SEARCH_FILTER_FIELDS = (
    "lot_id", "machine", "station", "platform", "defect_type",
    "map_location", "dft_location", "product_id",
)

# DINOv2 最终会把输入缩放到模型尺寸。PPT 中常见的 2K/4K 原图若原样送入
# 分类 API，不会增加检索信息，却会显著放大上传、解码和 JSON 响应占用。
PPT_SEARCH_MAX_SIDE = 1024


def _prepare_picture_for_search(content, extension, max_side=PPT_SEARCH_MAX_SIDE):
    """生成轻量的检索副本；PPT 原文件及正式入库图片均不会被改写。"""
    try:
        with Image.open(io.BytesIO(content)) as opened:
            image = ImageOps.exif_transpose(opened)
            image.load()
            width, height = image.size
            limit = max(256, int(max_side))
            if max(width, height) <= limit:
                return content, extension
            scale = limit / max(width, height)
            target = (max(1, round(width * scale)), max(1, round(height * scale)))
            image = image.resize(target, Image.Resampling.LANCZOS)
            if image.mode not in ("RGB", "RGBA", "L"):
                image = image.convert("RGB")
            output = io.BytesIO()
            # PNG keeps Bin Map colours and hard boundaries stable for local segmentation.
            image.save(output, format="PNG", optimize=True, compress_level=6)
            return output.getvalue(), ".png"
    except (OSError, ValueError):
        # Unsupported embedded media is left untouched; the downstream validator
        # will produce the user-facing error if it is not a decodable image.
        return content, extension


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9._\u4e00-\u9fff-]+", "_", str(value)).strip("._") or "uploaded.pptx"


def _resolve_part(base_part, target):
    """把 OOXML 关系中的相对路径转换为 ZIP 内部的规范路径。"""
    parts = list(PurePosixPath(base_part).parent.parts)
    for part in PurePosixPath(target.replace("\\", "/")).parts:
        if part == "..":
            if parts:
                parts.pop()
        elif part not in ("", "."):
            parts.append(part)
    return "/".join(parts)


def _relationships(archive, rel_path, base_part):
    if rel_path not in archive.namelist():
        return {}
    root = ET.fromstring(archive.read(rel_path))
    return {
        rel.get("Id"): _resolve_part(base_part, rel.get("Target", ""))
        for rel in root.findall("pr:Relationship", NS)
    }


def _slide_parts(archive):
    presentation = ET.fromstring(archive.read("ppt/presentation.xml"))
    rels = _relationships(
        archive, "ppt/_rels/presentation.xml.rels", "ppt/presentation.xml"
    )
    parts = []
    for slide_id in presentation.findall(".//p:sldId", NS):
        part = rels.get(slide_id.get(f"{{{NS['r']}}}id"))
        if part and part in archive.namelist():
            parts.append(part)
    return parts


def _position(element):
    off = element.find(".//a:xfrm/a:off", NS)
    if off is None:
        return (10**18, 10**18)
    return (int(off.get("y", 0)), int(off.get("x", 0)))


def _slide_content(archive, slide_part):
    root = ET.fromstring(archive.read(slide_part))
    texts = []
    for element in list(root.findall(".//p:sp", NS)) + list(root.findall(".//p:graphicFrame", NS)):
        value = "".join((node.text or "") for node in element.findall(".//a:t", NS)).strip()
        if value:
            y, x = _position(element)
            texts.append({"text": re.sub(r"\s+", " ", value), "x": x, "y": y})
    texts.sort(key=lambda item: (item["y"], item["x"]))

    rel_name = f"{PurePosixPath(slide_part).parent}/_rels/{PurePosixPath(slide_part).name}.rels"
    rels = _relationships(archive, rel_name, slide_part)
    pictures = []
    for picture in root.findall(".//p:pic", NS):
        blip = picture.find(".//a:blip", NS)
        if blip is None:
            continue
        media_part = rels.get(blip.get(f"{{{NS['r']}}}embed"))
        if not media_part or media_part not in archive.namelist():
            continue
        y, x = _position(picture)
        pictures.append({
            "part": media_part,
            "bytes": archive.read(media_part),
            "extension": Path(media_part).suffix.lower() or ".png",
            "x": x,
            "y": y,
        })
    pictures.sort(key=lambda item: (item["y"], item["x"]))
    return texts, pictures


def _first_match(pattern, value, default="UNKNOWN", flags=re.I):
    match = re.search(pattern, value or "", flags)
    return match.group(1).strip() if match else default


def _parse_record(text_items, slide_number, source_file):
    lines = [item["text"].strip() for item in text_items if item["text"].strip()]
    title = next((line for line in lines if re.search(r"DN-[A-Za-z0-9-]+", line, re.I)), "")
    dn_no = _first_match(r"(DN-[A-Za-z0-9-]+)", title)
    if dn_no == "UNKNOWN":
        dn_no = f"PPT-{Path(source_file).stem}-{slide_number:03d}"
    class_type = _first_match(r"\b(Class\s*\d+)\b", title)
    product_id = _first_match(r"\bClass\s*\d+\s+([A-Za-z0-9_.-]+)", title)
    station_from_title = _first_match(
        r"\bClass\s*\d+\s+[A-Za-z0-9_.-]+\s+([A-Za-z0-9_.-]+)", title
    )
    defect_from_title = _first_match(r"\s([A-Za-z][A-Za-z0-9_-]+)\s*$", title)

    identity = next((line for line in lines if re.search(r"\bTotal\s*\d+", line, re.I)), "")
    identity_parts = [part.strip() for part in identity.split("/")]
    platform = _first_match(r"\b([A-Z]{2,}\d+[A-Z0-9]*)\b", identity)
    lot_id = identity_parts[1] if len(identity_parts) > 1 else "UNKNOWN"
    bad_wafers = int(_first_match(r"\bNG\s*(\d+)", identity, "0"))
    scanned_wafers = int(_first_match(r"AdhCANC\s*(\d+)", identity, "0"))
    total_wafers = int(_first_match(r"\bTotal\s*(\d+)", identity, "0"))

    station_line = next((line for line in lines if line == station_from_title), station_from_title)
    defect_line = next((line for line in lines if re.match(r"^#?\s*\d+\s+", line)), "")
    defect_type = _first_match(r"^#?\s*\d+\s+([A-Za-z][A-Za-z0-9_-]+)", defect_line, defect_from_title)
    distribution = next((line for line in lines if re.search(r"Cluster|Random|Center|Edge", line, re.I)), "")
    map_location = distribution.split("/")[-1].strip() if distribution else "UNKNOWN"

    timed_lines = [line for line in lines if re.search(r"\d{4}/\d{1,2}/\d{1,2}", line)]
    scan_line = timed_lines[0] if timed_lines else ""
    machine = _first_match(r"^([A-Za-z0-9_.-]+)", scan_line)
    scan_time = _first_match(r"(\d{4}/\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2}:\d{2})", scan_line)
    status = next((line for line in lines if line.casefold() in {"down", "up"}), "UNKNOWN")
    lot_status = next((line for line in lines if line.lower().startswith(("scrap", "hold", "release"))), "UNKNOWN")

    payload = {
        "dn_no": dn_no,
        "record_id": dn_no,
        "class_type": class_type,
        "lot_id": lot_id,
        "product_id": product_id,
        "platform": platform,
        "station": station_line or station_from_title,
        "machine": machine,
        "scan_time": scan_time,
        "total_wafers": total_wafers,
        "scanned_wafers": scanned_wafers,
        "bad_wafers": bad_wafers,
        "defect_type": defect_type,
        "edx_elements": "UNKNOWN",
        "edx_bg": "UNKNOWN",
        "map_location": map_location,
        "dft_location": "UNKNOWN",
        "equipment_status": status,
        "lot_status": lot_status,
        "source_type": "pptx",
        "source_file": source_file,
        "source_slide": slide_number,
        "source_text": "\n".join(lines),
    }
    payload["text_content"] = (
        f"当前记录的DN编号为{dn_no}，类型是{class_type}，LotID为{lot_id}。"
        f"该批次属于产品{product_id}，当前处于{platform}平台的{payload['station']}站点。"
        f"扫描机器{machine}于{scan_time}完成扫描。Wafer总数为{total_wafers}，"
        f"实际扫描数量为{scanned_wafers}，坏Wafer数量为{bad_wafers}，"
        f"核心缺陷类型为{defect_type}，Map位置为{map_location}。"
    )
    return payload


def _write_images(record, pictures):
    """Persist all extracted images while assigning strict retrieval roles.

    The number of extracted pictures is not fixed.  Only the first detected
    Bin Map (``DefectMapPost``) and the first two SEM images
    (``TypicalDefectImage1/2``) participate in similarity retrieval.  Extra
    pictures keep display-only names and remain visible in record details.
    """
    from utils.binmap_patch_search import is_binmap_image

    folder = YEDN_DIR / record["dn_no"]
    folder.mkdir(parents=True, exist_ok=True)
    saved = []
    binmap_count = 0
    sem_count = 0
    for index, picture in enumerate(pictures):
        if is_binmap_image(picture["bytes"]):
            binmap_count += 1
            stem = "DefectMapPost" if binmap_count == 1 else f"ImportedBinMap{binmap_count}"
        else:
            sem_count += 1
            stem = (
                f"TypicalDefectImage{sem_count}"
                if sem_count <= 2 else f"ImportedSEMImage{sem_count}"
            )
        path = folder / f"{stem}{picture['extension']}"
        path.write_bytes(picture["bytes"])
        saved.append({"image_type": stem, "file_name": path.name})
    return saved


def load_imported_records():
    if not RECORDS_PATH.exists():
        return []
    records = []
    for line in RECORDS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _save_records(new_records):
    merged = {str(item.get("dn_no")): item for item in load_imported_records() if item.get("dn_no")}
    for item in new_records:
        merged[str(item["dn_no"])] = item
    RECORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECORDS_PATH.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in merged.values()),
        encoding="utf-8",
    )


def list_imports():
    if not HISTORY_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save_history(items):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def is_pptx_bytes(content):
    """判断输入是否为可解析的 OOXML PowerPoint，不依赖文件扩展名。"""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            return "ppt/presentation.xml" in archive.namelist()
    except (zipfile.BadZipFile, OSError):
        return False


def extract_pptx_for_search(content, file_name="uploaded.pptx"):
    """解析 PPTX 供一次性图像检索使用，不写入数据目录或检索库。

    返回所有幻灯片中的图片、解析出的记录字段，以及可用于结果后二次筛选的
    候选条件。同一字段在多页中出现多个值时按 OR 条件返回。
    """
    parsed = []
    images = []
    safe_name = _safe_name(file_name)
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            if "ppt/presentation.xml" not in archive.namelist():
                raise ValueError("文件不是有效的 PPTX")
            slide_parts = _slide_parts(archive)
            for slide_number, slide_part in enumerate(slide_parts, 1):
                texts, pictures = _slide_content(archive, slide_part)
                record = _parse_record(texts, slide_number, safe_name)
                parsed.append(record)
                for picture_number, picture in enumerate(pictures, 1):
                    search_bytes, search_extension = _prepare_picture_for_search(
                        picture["bytes"], picture["extension"]
                    )
                    images.append({
                        "bytes": search_bytes,
                        "extension": search_extension,
                        "source_name": f"{safe_name} · 第{slide_number}页 · 图片{picture_number}",
                        "slide": slide_number,
                        "original_size": len(picture["bytes"]),
                    })
    except zipfile.BadZipFile as exc:
        raise ValueError("文件不是有效的 PPTX") from exc

    auto_filters = {}
    for key in SEARCH_FILTER_FIELDS:
        values = []
        for record in parsed:
            value = record.get(key)
            if value in (None, "", "UNKNOWN", "UNKNOWN_DN", 0, "0"):
                continue
            value = str(value).strip()
            if value and value.casefold() not in {item.casefold() for item in values}:
                values.append(value)
        if values:
            auto_filters[key] = values

    return {
        "file_name": safe_name,
        "slide_count": len(parsed),
        "record_count": len(parsed),
        "image_count": len(images),
        "auto_filters": auto_filters,
        "records": [
            {key: record.get(key) for key in ("dn_no",) + SEARCH_FILTER_FIELDS}
            for record in parsed
        ],
    }, images


def import_pptx_bytes(content, file_name="uploaded.pptx"):
    digest = hashlib.sha256(content).hexdigest()
    existing = next((item for item in list_imports() if item.get("sha256") == digest), None)
    if existing:
        return {**existing, "already_imported": True}

    safe_name = _safe_name(file_name)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stored_path = UPLOAD_DIR / f"{digest[:12]}_{safe_name}"
    stored_path.write_bytes(content)

    parsed = []
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        slide_parts = _slide_parts(archive)
        for slide_number, slide_part in enumerate(slide_parts, 1):
            texts, pictures = _slide_content(archive, slide_part)
            record = _parse_record(texts, slide_number, safe_name)
            record["images"] = _write_images(record, pictures)
            parsed.append(record)
    _save_records(parsed)

    imported_at = datetime.now().astimezone().isoformat(timespec="seconds")
    summary = {
        "file_name": safe_name,
        "stored_path": str(stored_path),
        "sha256": digest,
        "imported_at": imported_at,
        "slide_count": len(parsed),
        "record_count": len(parsed),
        "image_count": sum(len(item["images"]) for item in parsed),
        "records": [
            {"dn_no": item["dn_no"], "slide": item["source_slide"], "images": item["images"]}
            for item in parsed
        ],
        "already_imported": False,
    }
    history = list_imports()
    history.insert(0, summary)
    _save_history(history)
    return summary


def import_pptx_file(path):
    path = Path(path)
    return import_pptx_bytes(path.read_bytes(), path.name)
