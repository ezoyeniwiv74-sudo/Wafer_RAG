from __future__ import annotations

"""FastAPI HTTP入口：暴露健康检查、分类和特征分析接口。

API容器通过 ``python -m uvicorn inference_api.server:app`` 导入本模块。导入时会
创建全局ModelService并一次性加载模型到GPU；之后每个请求复用同一模型。接口层
只负责参数/大小校验、Base64解码和HTTP错误转换，实际模型计算位于model_service。
"""

import base64
import json
import logging
import os
import time
import uuid
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .model_service import ModelService


# 读取docker run传入的限制。MB转换为字节，便于与上传payload长度比较。
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "20")) * 1024 * 1024
MAX_BATCH_IMAGES = int(os.environ.get("MAX_BATCH_IMAGES", "32"))

# API_LOG_DIR应在docker run时挂载到宿主机，保证删除或重建容器后日志仍然存在。
# 未配置时使用/tmp临时目录，方便本地调试，但该目录中的日志不会持久保存。
LOG_DIR = Path(os.environ.get("API_LOG_DIR", "/tmp/dinov2_api_logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_RETENTION_DAYS = max(1, int(os.environ.get("API_LOG_RETENTION_DAYS", "30")))


def _daily_file_logger(name: str, filename: str, level: int) -> logging.Logger:
    """创建每天轮转、保留指定天数的UTF-8文件日志器。

    Uvicorn开发热重载或测试代码可能重复导入本模块，因此仅在日志器尚无handler时
    添加文件handler，避免同一条记录被重复写入。正式部署固定workers=1，可安全
    使用标准库TimedRotatingFileHandler进行日志轮转。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        handler = TimedRotatingFileHandler(
            LOG_DIR / filename,
            when="midnight",
            interval=1,
            backupCount=LOG_RETENTION_DAYS,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


# api_access.jsonl每行是一个独立JSON对象，便于人工查看或导入日志平台。
audit_logger = _daily_file_logger("dinov2.api.audit", "api_access.jsonl", logging.INFO)
# api_error.log保存未处理异常的完整Python堆栈，便于定位服务端故障。
error_logger = _daily_file_logger("dinov2.api.error", "api_error.log", logging.ERROR)

# 模块导入时加载一次模型；启动失败也写入持久化错误日志，然后继续抛出使启动失败。
try:
    service = ModelService()
except Exception:
    error_logger.exception("DINOv2 model initialization failed")
    raise


class AnalyzeRequest(BaseModel):
    """``/analyze`` JSON请求结构。

    ``model``和``modality``用于调用方标识及回传；``images``必须至少包含一个
    Base64字符串。批量上限在路由函数中根据环境变量检查。
    """
    model: str = ""
    modality: str = ""
    images: list[str] = Field(..., min_length=1)

# Uvicorn命令中的 ``inference_api.server:app`` 指向这个FastAPI对象。
app = FastAPI(
    title="DINOv2 Semiconductor Defect Classifier",
    version="1.0.0",
)

# API_ALLOWED_ORIGINS可用逗号分隔多个前端来源；默认*允许任意来源访问。
origins = [
    item.strip()
    for item in os.environ.get("API_ALLOWED_ORIGINS", "*").split(",")
    if item.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    # 不开放跨域凭证，避免浏览器自动携带Cookie等敏感认证信息。
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def write_api_audit_log(request: Request, call_next):
    """为每个HTTP请求记录身份、来源、结果、耗时以及服务端异常。

    X-Client-ID用于标识调用系统或用户；它本身不构成认证，生产环境应由可信网关
    校验身份并覆盖该请求头。日志不读取或保存请求体，因此不会落盘图片、Base64、
    模型向量或API密钥。X-Forwarded-For单独保存，避免误把可伪造值当作直连IP。
    """
    started = time.perf_counter()
    # 限制外部请求头长度，避免异常大字段污染日志；缺失时由服务生成UUID。
    request_id = (request.headers.get("X-Request-ID") or str(uuid.uuid4()))[:128]
    client_id = (request.headers.get("X-Client-ID") or "anonymous")[:128]
    direct_ip = request.client.host if request.client else "unknown"
    forwarded_for = (request.headers.get("X-Forwarded-For") or "")[:512]
    request.state.request_id = request_id
    status_code = 500
    error_type = None

    try:
        response = await call_next(request)
        status_code = response.status_code
        # 请求ID随响应返回，用户报错时可用它在日志中精确定位一次请求。
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as exc:
        error_type = type(exc).__name__
        error_logger.exception(
            "request_id=%s client_id=%s direct_ip=%s method=%s path=%s",
            request_id,
            client_id,
            direct_ip,
            request.method,
            request.url.path,
        )
        raise
    finally:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        record = {
            "time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "request_id": request_id,
            "client_id": client_id,
            "direct_ip": direct_ip,
            "forwarded_for": forwarded_for or None,
            "method": request.method,
            "path": request.url.path,
            "status_code": status_code,
            "elapsed_ms": elapsed_ms,
            "content_length": request.headers.get("Content-Length"),
            "user_agent": (request.headers.get("User-Agent") or "")[:512] or None,
            "error_type": error_type,
        }
        audit_logger.info(json.dumps(record, ensure_ascii=False, separators=(",", ":")))


@app.get("/health")
def health() -> dict:
    """返回模型、设备、GPU和类别信息，供部署健康检查调用。"""
    return service.health()


@app.post("/api/v1/classify")
async def classify(
    file: UploadFile = File(...),
    topk: int = Query(default=3, ge=1, le=20),
) -> dict:
    """接收multipart单图片并返回top-k分类结果。

    ``file``是表单字段；``topk``限制为1到20，模型服务还会进一步限制到实际
    类别数。函数多读1字节用于可靠判断是否超过上传上限。
    """
    # 上限+1可以区分“刚好等于限制”和“确实超出限制”。
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded image is too large")
    # 图片解码类错误转换为HTTP 400，避免返回服务器内部异常页面。
    try:
        result = service.classify(payload, topk=topk)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result["filename"] = file.filename
    return result


@app.post("/analyze")
@app.post("/api/v1/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    """接收Base64图片列表，返回每张图的embedding和完整分类概率。

    两个路径指向同一实现：``/analyze``兼容现有图搜图调用方，版本化路径
    ``/api/v1/analyze``供新系统使用。返回列表顺序与请求images顺序完全一致。
    """
    # 在Base64解码和GPU推理前先拒绝过大批次，避免不必要内存占用。
    if len(request.images) > MAX_BATCH_IMAGES:
        raise HTTPException(
            status_code=413,
            detail=f"A batch can contain at most {MAX_BATCH_IMAGES} images",
        )
    payloads = []
    try:
        for encoded in request.images:
            # validate=True拒绝含非法字符或错误填充的Base64文本。
            payload = base64.b64decode(encoded, validate=True)
            if not payload:
                raise ValueError("Empty image")
            if len(payload) > MAX_UPLOAD_BYTES:
                raise ValueError("Image is too large")
            payloads.append(payload)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # topk设为类别总数，使图搜图调用方获得所有类别概率而非仅前三名。
    try:
        modality = request.modality.strip().lower() or "sem"
        if modality not in {"binmap", "sem"}:
            raise ValueError("modality must be 'binmap' or 'sem'")
        result = service.analyze_many(payloads, topk=len(service.class_names), modality=modality)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # 原样回传调用方model/modality标识，并合并模型服务输出。
    return {
        "model": request.model or "dinov2-company-classifier",
        "modality": request.modality,
        **result,
    }
