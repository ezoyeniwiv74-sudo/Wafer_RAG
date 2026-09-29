"""晶圆缺陷检索系统的本地/服务器统一 FastAPI 入口。"""
from __future__ import annotations

import base64
import binascii
import os
import time
from pathlib import Path
from typing import Literal
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent
# The web application is Qdrant-first.  Local CSV/YEDN sources are migration
# inputs only and must never silently change the production retrieval backend.
os.environ.setdefault("WAFER_PROFILE", "server")
os.environ.setdefault("WAFER_QDRANT_ONLY", "1")

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from searchers.hybrid_searcher import get_hybrid_searcher
from utils.field_extractor import get_field_extractor
from utils.payload_parser import (
    canonical_record_id,
    canonicalize_payload,
    describe_payload,
    is_missing,
    merge_payloads,
)
from utils.qdrant_client import get_qdrant_manager

app = FastAPI(title="晶圆缺陷智能检索", version="2.0.0")
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")

FILTER_FIELDS = (
    "lot_id", "machine", "station", "platform", "defect_type",
    "map_location", "dft_location", "product_id",
)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"请求处理失败：{type(exc).__name__}: {exc}"},
    )


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(50, ge=1, le=500)
    offset: int = Field(0, ge=0)
    filters: dict[str, list[str]] = Field(default_factory=dict)


class ReportRequest(BaseModel):
    query: str = Field("", max_length=500)
    results: list[dict] = Field(..., min_length=1, max_length=100)


class ReportImageInput(BaseModel):
    name: str = Field("uploaded-image", max_length=255)
    data_base64: str


class MultimodalReportRequest(BaseModel):
    query: str = Field("", max_length=500)
    user_context: str = Field("", max_length=4000)
    results: list[dict] = Field(default_factory=list, max_length=100)
    images: list[ReportImageInput] = Field(default_factory=list, max_length=8)


def serialize(item):
    payload = canonicalize_payload(getattr(item, "payload", {}) or {})
    payload["text_content"] = describe_payload(payload)
    return {
        "record_id": canonical_record_id(payload, getattr(item, "id", None)) or "UNKNOWN",
        "score": float(getattr(item, "score", 1.0) or 1.0),
        "fields": payload,
    }


_record_cache = None


def records():
    global _record_cache
    if _record_cache is not None:
        return _record_cache
    out = {}
    for point in get_qdrant_manager().scroll_all_points():
        payload = canonicalize_payload(point.payload)
        rid = canonical_record_id(payload, getattr(point, "id", None))
        if rid:
            out[rid] = merge_payloads(out.get(rid, {}), payload)
    _record_cache = out
    return _record_cache


def _clean_filters(raw):
    cleaned = {}
    for key, values in (raw or {}).items():
        if key not in FILTER_FIELDS:
            continue
        values = values if isinstance(values, (list, tuple, set)) else [values]
        found = []
        for value in values:
            value = str(value).strip()
            if value and value.casefold() not in {item.casefold() for item in found}:
                found.append(value)
        if found:
            cleaned[key] = found
    return cleaned


def _matches(payload, filters):
    for key, choices in filters.items():
        value = str(payload.get(key, "")).strip().casefold()
        if value not in {choice.casefold() for choice in choices}:
            return False
    return True


def _reset_runtime_caches():
    """导入新记录后让下一次请求重新读取本地数据与图片目录。"""
    global _record_cache
    _record_cache = None
    modules_and_names = {
        "utils.qdrant_client": "_qdrant_manager",
        "utils.image_search": "_searcher",
        "searchers.hybrid_searcher": "_hybrid_searcher",
        "searchers.exact_searcher": "_exact_searcher",
        "searchers.bm25_searcher": "_bm25_searcher",
        "searchers.tfidf_searcher": "_tfidf_searcher",
        "searchers.semantic_searcher": "_semantic_searcher",
        "searchers.fusion_searcher": "_fusion_searcher",
        "searchers.rerank_searcher": "_rerank_searcher",
    }
    import importlib
    for module_name, attribute in modules_and_names.items():
        module = importlib.import_module(module_name)
        if hasattr(module, attribute):
            setattr(module, attribute, None)


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/health")
def health():
    result = get_qdrant_manager().check_connection()
    return {"ok": bool(result.get("success")), **result}


@app.post("/api/search")
def search(request: SearchRequest):
    query = request.query.strip().rstrip("?？")
    filters = _clean_filters(request.filters)
    searcher = get_hybrid_searcher()
    fetch_size = max(request.offset + request.limit, 500 if filters else 0)
    result, elapsed, mode = searcher.search(query, top_k=fetch_size)
    serialized = [serialize(item) for item in result]
    if filters:
        serialized = [item for item in serialized if _matches(item["fields"], filters)]
    selected = serialized[request.offset:request.offset + request.limit]
    return {
        "query": query,
        "conditions": get_field_extractor().extract_conditions(query),
        "filters": filters,
        "mode": mode,
        "elapsed_ms": round(float(elapsed), 2),
        "count": len(selected),
        "results": selected,
    }


@app.get("/api/report/status")
def report_status():
    from utils.report_generator import model_status
    from utils.analysis_knowledge import get_analysis_knowledge_store
    result = model_status()
    result["knowledge"] = get_analysis_knowledge_store().status()
    result["retrieval_source"] = "qdrant"
    return result


@app.post("/api/report")
def report(request: ReportRequest):
    from utils.report_generator import ReportServiceError, generate_report
    try:
        return generate_report(request.query.strip(), request.results)
    except ReportServiceError as exc:
        raise HTTPException(503, str(exc)) from exc


def _merge_report_records(groups, limit=20):
    """Deduplicate multimodal evidence while preserving explicitly selected rows."""
    merged = {}
    order = []
    for source, rows in groups:
        for item in rows:
            fields = canonicalize_payload(item.get("fields") or {})
            record_id = str(item.get("record_id") or item.get("dn_no") or fields.get("dn_no") or "").strip()
            if not record_id:
                continue
            score = float(item.get("score", item.get("average_similarity", 0.0)) or 0.0)
            if record_id not in merged:
                order.append(record_id)
                merged[record_id] = {
                    "record_id": record_id, "score": score, "fields": fields,
                    "evidence_sources": [source],
                }
            else:
                merged[record_id]["score"] = max(merged[record_id]["score"], score)
                merged[record_id]["fields"] = merge_payloads(merged[record_id]["fields"], fields)
                if source not in merged[record_id]["evidence_sources"]:
                    merged[record_id]["evidence_sources"].append(source)
    return [merged[record_id] for record_id in order[:limit]]


@app.post("/api/report/multimodal")
def multimodal_report(request: MultimodalReportRequest):
    """Combine selected records, user text and uploaded images into one report."""
    if not request.results and not request.user_context.strip() and not request.images:
        raise HTTPException(400, "请至少提供选中记录、现场文字或图片中的一项")

    text_rows = []
    if request.user_context.strip():
        try:
            hits, _, _ = get_hybrid_searcher().search(request.user_context.strip(), top_k=10)
            text_rows = [serialize(item) for item in hits]
        except Exception as exc:
            raise HTTPException(503, f"现场文字检索 Qdrant 失败：{exc}") from exc

    image_bytes = []
    image_names = []
    total_size = 0
    for image in request.images:
        encoded = image.data_base64
        if "," in encoded and encoded.lstrip().startswith("data:"):
            encoded = encoded.split(",", 1)[1]
        try:
            content = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(400, f"图片 {image.name} 内容无效") from exc
        if not content or len(content) > 20 * 1024 * 1024:
            raise HTTPException(413, f"图片 {image.name} 为空或超过 20MB")
        total_size += len(content)
        if total_size > 80 * 1024 * 1024:
            raise HTTPException(413, "报告图片合计不能超过 80MB")
        image_bytes.append(content)
        image_names.append(image.name)

    image_rows = []
    if image_bytes:
        try:
            from utils.image_search import get_image_searcher
            image_result = get_image_searcher().search_many(image_bytes, limit=10)
            image_rows = [{
                "record_id": item.get("dn_no"),
                "score": item.get("average_similarity", item.get("score", 0)),
                "fields": item.get("fields") or {},
            } for item in image_result.get("results", [])]
        except Exception as exc:
            raise HTTPException(503, f"上传图片检索 Qdrant 失败：{exc}") from exc

    combined = _merge_report_records([
        ("selected", request.results),
        ("user_text_qdrant", text_rows),
        ("user_image_qdrant", image_rows),
    ])
    if not combined:
        raise HTTPException(404, "用户输入未在 Qdrant 中检索到可用于分析的历史记录")
    metadata = {
        "selected_count": len(request.results),
        "text_retrieved_count": len(text_rows),
        "image_retrieved_count": len(image_rows),
        "uploaded_image_count": len(image_bytes),
        "uploaded_image_names": image_names,
        "merged_evidence_count": len(combined),
        "retrieval_source": "qdrant",
    }
    from utils.report_generator import ReportServiceError, generate_report
    try:
        result = generate_report(
            request.query.strip(), combined,
            additional_context=request.user_context.strip(),
            input_metadata=metadata,
        )
        result["evidence_breakdown"] = metadata
        return result
    except ReportServiceError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/api/filters")
def filters():
    data = records().values()
    return {
        key: sorted({str(item[key]) for item in data if not is_missing(item.get(key))})
        for key in FILTER_FIELDS
    }


@app.get("/api/dn-search")
def dn_search(dn_no: str = Query(..., min_length=1, max_length=100)):
    normalized = dn_no.strip().casefold()
    found = []
    for rid, payload in records().items():
        if rid.casefold() == normalized:
            point = type("Result", (), {"id": rid, "payload": payload, "score": 1.0})()
            found.append(serialize(point))
            break
    return {"dn_no": dn_no.strip(), "count": len(found), "results": found}


@app.get("/api/filter-search")
def filter_search(
    lot_id: list[str] = Query(default=[]), machine: list[str] = Query(default=[]),
    station: list[str] = Query(default=[]), platform: list[str] = Query(default=[]),
    defect_type: list[str] = Query(default=[]), map_location: list[str] = Query(default=[]),
    dft_location: list[str] = Query(default=[]), product_id: list[str] = Query(default=[]),
    limit: int = Query(100, ge=1, le=5000),
):
    conditions = _clean_filters(locals())
    if not conditions:
        raise HTTPException(400, "请至少选择一个筛选条件")
    started = time.perf_counter()
    matched = []
    for rid, payload in records().items():
        if _matches(payload, conditions):
            point = type("Result", (), {"id": rid, "payload": payload, "score": 1.0})()
            matched.append(serialize(point))
            if len(matched) >= limit:
                break
    return {
        "conditions": conditions,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "count": len(matched),
        "results": matched,
    }


@app.get("/api/image/filters")
def image_filters():
    from utils.image_search import get_image_searcher
    return get_image_searcher().filters()


@app.get("/api/image/status")
def image_status():
    from utils.image_search import get_image_searcher
    return get_image_searcher().status()


@app.get("/api/yedn/image/{dn_no}/{file_name}")
def yedn_image(dn_no: str, file_name: str):
    from utils.image_search import get_image_searcher
    root = get_image_searcher().root.resolve()
    path = (root / dn_no / file_name).resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(404, "未找到该 DN 图片")
    return FileResponse(path, headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/yedn/patch/{dn_no}/{file_name}")
def yedn_patch(
    dn_no: str, file_name: str,
    x: int = Query(..., ge=0), y: int = Query(..., ge=0),
    w: int = Query(..., ge=1), h: int = Query(..., ge=1),
    normalized: bool = Query(False),
):
    """按已验证的 DN 图片路径返回局部裁剪，供命中框点击放大。"""
    from utils.binmap_patch_search import _encode_png, _open_rgb, standardized_region_crop
    from utils.image_search import get_image_searcher
    root = get_image_searcher().root.resolve()
    path = (root / dn_no / file_name).resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(404, "未找到该 DN 图片")
    image = _open_rgb(path.read_bytes())
    right, bottom = min(image.width, x + w), min(image.height, y + h)
    if x >= right or y >= bottom:
        raise HTTPException(400, "局部区域超出图片范围")
    crop = standardized_region_crop(image, (x, y, right, bottom)) if normalized else image.crop((x, y, right, bottom))
    return Response(_encode_png(crop), media_type="image/png")


@app.get("/api/yedn/{dn_no}")
def yedn_detail(dn_no: str):
    from utils.image_search import IMAGE_TYPES, get_image_searcher
    normalized = dn_no.strip().casefold()
    images = [
        {key: item[key] for key in ("image_type", "file_name", "url")}
        for item in get_image_searcher().catalog_for_dn(dn_no)
        if item["dn_no"].casefold() == normalized
    ]
    if not images:
        raise HTTPException(404, f"未找到 {dn_no} 对应的图片")
    order = {name: index for index, name in enumerate(IMAGE_TYPES)}
    images.sort(key=lambda item: (order.get(item["image_type"], 99), item["file_name"]))
    record = next((payload for rid, payload in records().items() if rid.casefold() == normalized), {})
    fields = canonicalize_payload(dict(record))
    fields["text_content"] = describe_payload(fields)
    return {"dn_no": dn_no.strip(), "count": len(images), "images": images, "fields": fields}


@app.post("/api/image/ppt-extract")
async def image_ppt_extract(request: Request):
    """把一个 PPTX 展开成可直接参与多图检索的图片列表。"""
    from utils.binmap_patch_search import is_binmap_image
    from utils.ppt_import import extract_pptx_for_search, is_pptx_bytes

    content = await request.body()
    if not content:
        raise HTTPException(400, "请选择PPTX文件")
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(413, "PPTX文件不能超过100MB")
    if not is_pptx_bytes(content):
        raise HTTPException(400, "文件不是有效的PPTX")
    file_name = unquote(request.headers.get("X-File-Name", "uploaded.pptx"))
    try:
        ppt, pictures = extract_pptx_for_search(content, file_name=file_name)
    except ValueError as exc:
        raise HTTPException(400, f"{file_name}解析失败：{exc}") from exc
    if not pictures:
        raise HTTPException(400, f"{file_name}中没有可检索图片")

    stem = Path(file_name).stem
    modality_counts = {"binmap": 0, "sem": 0}
    output = []
    for index, picture in enumerate(pictures):
        modality = "binmap" if is_binmap_image(picture["bytes"]) else "sem"
        modality_counts[modality] += 1
        if modality == "binmap":
            role = "DefectMapPost" if modality_counts[modality] == 1 else f"BinMap{modality_counts[modality]}"
        else:
            sem_roles = ("TypicalDefectImage1", "TypicalDefectImage2")
            role_index = modality_counts[modality] - 1
            role = sem_roles[role_index] if role_index < len(sem_roles) else f"SEMImage{role_index + 1}"
        extension = str(picture.get("extension") or ".png").lower()
        mime = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".bmp": "image/bmp", ".webp": "image/webp", ".tif": "image/tiff", ".tiff": "image/tiff",
        }.get(extension, "image/png")
        output.append({
            "index": index, "slide": picture.get("slide"), "modality": modality,
            "name": f"{stem} · {role}{extension}", "content_type": mime,
            "data_base64": base64.b64encode(picture["bytes"]).decode("ascii"),
        })
    return {
        "file_name": file_name, "slide_count": ppt["slide_count"],
        "image_count": len(output), "images": output,
    }


@app.post("/api/image/detect-modality")
async def image_detect_modality(request: Request):
    """Classify one uploaded image as Bin Map or SEM for UI controls."""
    from utils.binmap_patch_search import is_binmap_image

    content = await request.body()
    if not content:
        raise HTTPException(400, "请选择图片")
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(413, "单张图片不能超过 20MB")
    try:
        modality = "binmap" if is_binmap_image(content) else "sem"
    except Exception as exc:
        raise HTTPException(400, "无法识别该图片") from exc
    return {"modality": modality}


@app.post("/api/image/search")
async def image_search(
    request: Request, modality: Literal["binmap", "sem"] | None = None,
    platform: list[str] = Query(default=[]), defect_type: list[str] = Query(default=[]),
    map_location: list[str] = Query(default=[]), dft_location: list[str] = Query(default=[]),
    machine: list[str] = Query(default=[]), station: list[str] = Query(default=[]),
    product_id: list[str] = Query(default=[]), lot_id: list[str] = Query(default=[]),
    limit: int = Query(100, ge=1, le=500),
):
    """统一接受多张图片或 PPTX，并执行跨模态多图融合检索。

    新页面使用 JSON+Base64 传输多个文件；旧页面直接发送单张图片的方式继续兼容。
    PPTX 只为本次查询解析，不会自动写入数据目录。
    """
    from utils.image_search import get_image_searcher
    from utils.ppt_import import extract_pptx_for_search, is_pptx_bytes

    content_type = request.headers.get("content-type", "").lower()
    source_files = []
    query_images = []

    def normalize_manual_region(value, file_name):
        if value in (None, "", []):
            return None
        # A file can contain several independently selected regions.  Keep the
        # old single rectangle/polygon payload valid for existing clients, and
        # normalize the new list form one item at a time.
        is_rectangle = (
            isinstance(value, (list, tuple)) and len(value) == 4
            and all(isinstance(item, (int, float)) for item in value)
        )
        if isinstance(value, (list, tuple)) and not is_rectangle:
            if not (1 <= len(value) <= 3):
                raise HTTPException(400, f"{file_name} 的手动选区数量必须在 1 到 3 个之间")
            return [normalize_manual_region(item, file_name) for item in value]
        if isinstance(value, dict) and str(value.get("type", "")).lower() == "polygon":
            points = value.get("points")
            if not isinstance(points, list) or not (3 <= len(points) <= 100):
                raise HTTPException(400, f"{file_name} 的不规则选区无效")
            normalized = []
            try:
                for point in points:
                    if not isinstance(point, (list, tuple)) or len(point) != 2:
                        raise ValueError
                    x, y = float(point[0]), float(point[1])
                    if x < 0 or y < 0 or x > 1.000001 or y > 1.000001:
                        raise ValueError
                    normalized.append([x, y])
            except (TypeError, ValueError) as exc:
                raise HTTPException(400, f"{file_name} 的不规则选区超出图片范围") from exc
            if max(point[0] for point in normalized) - min(point[0] for point in normalized) < 0.02:
                raise HTTPException(400, f"{file_name} 的不规则选区过小")
            if max(point[1] for point in normalized) - min(point[1] for point in normalized) < 0.02:
                raise HTTPException(400, f"{file_name} 的不规则选区过小")
            return {"type": "polygon", "points": normalized}
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise HTTPException(400, f"{file_name} 的手动画框坐标无效")
        try:
            left, top, width, height = [float(number) for number in value]
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, f"{file_name} 的手动画框坐标无效") from exc
        if left < 0 or top < 0 or width <= 0 or height <= 0:
            raise HTTPException(400, f"{file_name} 的手动画框超出图片范围")
        if left + width > 1.000001 or top + height > 1.000001:
            raise HTTPException(400, f"{file_name} 的手动画框超出图片范围")
        if width < 0.02 or height < 0.02:
            raise HTTPException(400, f"{file_name} 的手动画框过小")
        return [left, top, width, height]

    if "application/json" in content_type:
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(400, "上传内容不是有效的 JSON") from exc
        uploaded = payload.get("files") if isinstance(payload, dict) else None
        if not isinstance(uploaded, list) or not uploaded:
            raise HTTPException(400, "请至少选择一个图片或 PPTX 文件")
        if len(uploaded) > 30:
            raise HTTPException(400, "一次最多选择 30 个文件")
        try:
            requested_limit = int(payload.get("limit", limit))
        except (TypeError, ValueError):
            requested_limit = limit
        limit = max(1, min(requested_limit, 500))
        total_size = 0
        for index, item in enumerate(uploaded, 1):
            if not isinstance(item, dict):
                raise HTTPException(400, f"第 {index} 个文件信息不完整")
            name = unquote(str(item.get("name") or f"file-{index}"))
            encoded = str(item.get("data_base64") or "")
            if "," in encoded and encoded.lstrip().startswith("data:"):
                encoded = encoded.split(",", 1)[1]
            try:
                content = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise HTTPException(400, f"{name} 的文件内容无效") from exc
            total_size += len(content)
            if total_size > 200 * 1024 * 1024:
                raise HTTPException(413, "本次上传文件合计不能超过 200MB")
            source_files.append({"name": name, "size": len(content)})
            if is_pptx_bytes(content):
                if len(content) > 100 * 1024 * 1024:
                    raise HTTPException(413, f"{name} 不能超过 100MB")
                try:
                    ppt, extracted = extract_pptx_for_search(content, file_name=name)
                except ValueError as exc:
                    raise HTTPException(400, f"{name} 解析失败：{exc}") from exc
                source_files[-1].update({
                    "kind": "pptx", "slide_count": ppt["slide_count"],
                    "image_count": ppt["image_count"],
                })
                for picture in extracted:
                    if len(query_images) >= 100:
                        break
                    query_images.append({
                        "bytes": picture["bytes"],
                        "name": picture["source_name"],
                        "extension": picture["extension"],
                        "source_type": "pptx",
                    })
            else:
                if len(content) > 20 * 1024 * 1024:
                    raise HTTPException(413, f"图片 {name} 不能超过 20MB")
                lowered = name.lower()
                looks_like_image = (
                    content.startswith((b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"BM", b"II*\x00", b"MM\x00*"))
                    or (content.startswith(b"RIFF") and content[8:12] == b"WEBP")
                    or lowered.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"))
                )
                if not looks_like_image:
                    raise HTTPException(400, f"{name} 不是支持的图片或 PPTX 文件")
                source_files[-1].update({"kind": "image", "image_count": 1})
                manual_region = normalize_manual_region(item.get("manual_region"), name)
                query_images.append({
                    "bytes": content, "name": name,
                    "extension": Path(name).suffix.lower() or ".png",
                    "source_type": "image",
                    "manual_region": manual_region,
                })
    else:
        # 兼容旧版单图调用。
        content = await request.body()
        if not content:
            raise HTTPException(400, "请选择一张晶圆图片")
        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(413, "图片不能超过 20MB")
        name = unquote(request.headers.get("X-Image-Name", "uploaded-image"))
        query_images = [{"bytes": content, "name": name, "extension": Path(name).suffix or ".png", "source_type": "image", "manual_region": None}]
        source_files = [{"name": name, "size": len(content), "kind": "image", "image_count": 1}]

    if not query_images:
        raise HTTPException(400, "没有从上传内容中解析出可检索图片")
    try:
        result = get_image_searcher().search_many(
            [item["bytes"] for item in query_images], filters=None, limit=limit,
            manual_regions=[item.get("manual_region") for item in query_images],
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    mime_types = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".bmp": "image/bmp", ".webp": "image/webp", ".tif": "image/tiff", ".tiff": "image/tiff",
    }
    previews = []
    for index, item in enumerate(query_images[:30]):
        extension = str(item.get("extension") or ".png").lower()
        mime = mime_types.get(extension, "image/png")
        previews.append({
            "index": index,
            "name": item["name"],
            "source_type": item["source_type"],
            "url": f"data:{mime};base64,{base64.b64encode(item['bytes']).decode('ascii')}",
            "manual_region": item.get("manual_region"),
        })
    result.update({
        "source_files": source_files,
        "query_images": previews,
        # PPT 查询只使用解析出的图片，不再把文本字段自动转为筛选条件。
        "auto_filters": {},
        "auto_filter_source": None,
    })
    return result


@app.post("/api/ppt/import")
async def ppt_import(request: Request):
    content = await request.body()
    if not content:
        raise HTTPException(400, "请选择 PPTX 文件")
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(413, "PPTX 文件不能超过 100MB")
    file_name = unquote(request.headers.get("X-File-Name", "uploaded.pptx"))
    if not file_name.lower().endswith(".pptx"):
        raise HTTPException(400, "当前仅支持 .pptx 文件")
    from utils.ppt_import import import_pptx_bytes
    result = import_pptx_bytes(content, file_name=file_name)
    # When the DN-centric collection is already online, persist both newly
    # parsed text and images in the same Qdrant points automatically.  A sync
    # error is reported without losing the successfully imported source files.
    from config import AUTO_SYNC_UNIFIED_IMPORT, RUNTIME_PROFILE
    if AUTO_SYNC_UNIFIED_IMPORT and RUNTIME_PROFILE == "server":
        try:
            dn_nos = [item.get("dn_no") for item in result.get("records", []) if item.get("dn_no")]
            from utils.unified_dn_index import UnifiedDnIndexer
            indexer = UnifiedDnIndexer()
            from config import UNIFIED_COLLECTION_NAME
            if indexer.client.collection_exists(UNIFIED_COLLECTION_NAME):
                result["vector_sync"] = indexer.run(
                    execute=True, recreate=False, dn_filter=dn_nos
                )
            else:
                result["vector_sync"] = {
                    "success": False,
                    "pending": True,
                    "message": "统一集合尚未初始化；请先执行一次全量统一入库。",
                }
        except Exception as exc:
            result["vector_sync"] = {"success": False, "error": str(exc)}
    _reset_runtime_caches()
    return result


@app.get("/api/ppt/imports")
def ppt_imports():
    from utils.ppt_import import list_imports
    return {"files": list_imports()}


@app.get("/api/ppt/library")
def ppt_library(query: str = ""):
    """查询已导入 PPT，并按产品、平台和设备汇总动态字典。"""
    from utils.ppt_import import list_imports, load_imported_records
    imported = list_imports()
    parsed = load_imported_records()
    keyword = query.strip().casefold()
    if keyword:
        matching_dn = {
            str(item.get("dn_no", "")) for item in parsed
            if keyword in " ".join(str(item.get(key, "")) for key in (
                "dn_no", "product_id", "platform", "machine", "station", "defect_type"
            )).casefold()
        }
        imported = [item for item in imported if (
            keyword in str(item.get("file_name", "")).casefold()
            or any(str(record.get("dn_no", "")) in matching_dn for record in item.get("records", []))
        )]
    dictionaries = {}
    for key in ("product_id", "platform", "machine", "defect_type"):
        counts = {}
        for item in parsed:
            value = str(item.get(key, "")).strip()
            if not value or value.upper() == "UNKNOWN":
                continue
            counts[value] = counts.get(value, 0) + 1
        dictionaries[key] = [
            {"value": value, "count": count}
            for value, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
        ]
    return {"files": imported, "total": len(list_imports()), "dictionaries": dictionaries}


@app.get("/api/similar/{record_id}")
def similar(
    record_id: str,
    by: Literal["platform", "defect", "combined"] = "combined",
    limit: int = Query(10, ge=1, le=100),
):
    all_records = records()
    source = all_records.get(record_id)
    if not source:
        raise HTTPException(404, "未找到该 DN 记录")
    primary = "machine" if by == "platform" and not is_missing(source.get("machine")) else "platform"
    exact_value = str(source.get(primary, ""))
    prefix = exact_value[:-2] if len(exact_value) > 2 else exact_value
    result = []
    for rid, payload in all_records.items():
        if rid == record_id:
            continue
        value = str(payload.get(primary, ""))
        same_defect = payload.get("defect_type") == source.get("defect_type")
        if value == exact_value and (by != "combined" or same_defect):
            score = 1.0
        elif prefix and value.startswith(prefix):
            score = 0.8
        else:
            continue
        point = type("Result", (), {"id": rid, "payload": payload, "score": score})()
        result.append(serialize(point))
        if len(result) >= limit:
            break
    return {"source_id": record_id, "by": by, "field": primary, "results": result}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("WAFER_WEB_PORT", "8002"))
    uvicorn.run("api:app", host="127.0.0.1", port=port)
