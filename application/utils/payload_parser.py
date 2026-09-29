"""晶圆异常描述的统一解析与旧 payload 修复。"""

import re
import unicodedata


UNKNOWN_VALUES = {None, "", "UNKNOWN", "UNKNOWN_DN", "未知", "nan"}


PATTERNS = {
    "dn_no": [r"DN\s*(?:编号|号)?\s*(?:为|是|[:：])?\s*(DN-[A-Za-z0-9-]+)"],
    "class_type": [r"(?:类型|类别)\s*(?:为|是|[:：])?\s*(Class\s*\d+)"],
    "lot_id": [
        r"Lot\s*ID\s*(?:[（(][^）)]*?[为是:]?[）)])?\s*(?:为|是|[:：])?\s*([A-Za-z0-9][A-Za-z0-9_.-]*)",
        r"(?:批次编号|批次号|批号)\s*(?:为|是|[:：])?\s*([A-Za-z0-9][A-Za-z0-9_.-]*)",
    ],
    "product_id": [r"(?:属于)?产品(?:编号|ID)?\s*(?:为|是|[:：])?\s*([A-Za-z0-9][A-Za-z0-9_.-]*)"],
    "platform": [r"(?:处于|位于)?\s*([A-Za-z0-9][A-Za-z0-9_.-]*)\s*平台"],
    "station": [r"平台(?:的|下)?\s*([A-Za-z0-9][A-Za-z0-9_.-]*)\s*站点"],
    "machine": [r"(?:扫描机器|扫描机|机器|机台)\s*(?:为|是|[:：])?\s*([A-Za-z0-9][A-Za-z0-9_.-]*)"],
    "scan_time": [r"于\s*(\d{4}[/-]\d{1,2}[/-]\d{1,2}\s+\d{1,2}:\d{1,2}:\d{1,2})\s*完成(?:了)?扫描"],
    "total_wafers": [r"Wafer总数\s*(?:为|是|[:：])?\s*(\d+)"],
    "scanned_wafers": [r"(?:实际)?扫描过的Wafer数量\s*(?:为|是|[:：])?\s*(\d+)"],
    "bad_wafers": [r"坏的Wafer数量\s*(?:为|是|[:：])?\s*(\d+)"],
    "defect_type": [r"(?:核心)?缺陷类型\s*(?:为|是|[:：])?\s*([A-Za-z0-9][A-Za-z0-9_.-]*)"],
    "edx_elements": [r"EDX检测到的元素\s*(?:为|是|[:：])?\s*([^，,。；;]+)"],
    "edx_bg": [r"EDX背景\s*(?:为|是|[:：])?\s*([^，,。；;]+)"],
    "map_location": [r"Map(?:上面)?分布的位置\s*(?:为|是|[:：])?\s*([^，,。；;]+)"],
    "dft_location": [r"DFT(?:上面)?分布的位置\s*(?:为|是|[:：])?\s*([^，,。；;]+)"],
}

INTEGER_FIELDS = {"total_wafers", "scanned_wafers", "bad_wafers"}


def is_missing(value):
    if value is None:
        return True
    return str(value).strip() in {"", "UNKNOWN", "UNKNOWN_DN", "未知", "nan"}


def parse_payload_from_text(text):
    text = unicodedata.normalize("NFKC", str(text or "")).strip()
    payload = {}
    for field, patterns in PATTERNS.items():
        value = None
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                value = match.group(1).strip(" \t\r\n，。；;:：()（）")
                break
        if value is not None:
            payload[field] = int(value) if field in INTEGER_FIELDS else value
    # 缺失字段只标记 UNKNOWN/0，不因缺键导致整条记录生成失败；业务值“无”保留原值。
    for field in PATTERNS:
        if field not in payload:
            payload[field] = 0 if field in INTEGER_FIELDS else "UNKNOWN"
    payload["text_content"] = text
    if not is_missing(payload.get("dn_no")):
        payload["record_id"] = payload["dn_no"]
    return payload


def canonicalize_payload(payload):
    """用原始描述补齐缺失字段，并统一 DN 为跨环境主键。"""
    result = dict(payload or {})
    # 不同写入脚本曾使用过多种文本键；统一回收到 text_content，避免前端“暂无描述”。
    text = next((result.get(key) for key in (
        "text_content", "description", "TEXT", "text", "search_text", "content", "document"
    ) if not is_missing(result.get(key))), "")
    if isinstance(text, (list, dict)):
        text = str(text)
    if text:
        result["text_content"] = str(text).strip()
    parsed = parse_payload_from_text(text) if text else {}
    for field, value in parsed.items():
        if is_missing(result.get(field)):
            result[field] = value
    dn_no = result.get("dn_no")
    if is_missing(dn_no):
        dn_no = parsed.get("dn_no")
    if not is_missing(dn_no):
        result["dn_no"] = str(dn_no)
        result["record_id"] = str(dn_no)
    elif not is_missing(result.get("record_id")):
        result["record_id"] = str(result["record_id"])
    return result


def merge_payloads(*payloads):
    """合并同一 DN 的多粒度 payload，优先保留完整描述与非缺失字段。"""
    merged = {}
    for raw in payloads:
        current = canonicalize_payload(raw)
        for key, value in current.items():
            if is_missing(merged.get(key)) or (key == "text_content" and len(str(value)) > len(str(merged.get(key, "")))):
                merged[key] = value
    return canonicalize_payload(merged)


def describe_payload(payload):
    """旧库确实没有原文时生成可读详情，不再展示模糊的“暂无描述”。"""
    value = canonicalize_payload(payload)
    if not is_missing(value.get("text_content")):
        return value["text_content"]
    names = {
        "dn_no": "DN编号", "class_type": "类型", "lot_id": "LotID", "product_id": "产品",
        "platform": "平台", "station": "站点", "machine": "机器", "scan_time": "扫描时间",
        "total_wafers": "Wafer总数", "scanned_wafers": "已扫描Wafer", "bad_wafers": "坏Wafer",
        "defect_type": "缺陷类型", "edx_elements": "EDX元素", "edx_bg": "EDX背景",
        "map_location": "Map位置", "dft_location": "DFT位置",
    }
    parts = [f"{label}为{value.get(key)}" for key, label in names.items()
             if not is_missing(value.get(key)) and value.get(key) != 0]
    return "；".join(parts) + ("。" if parts else "原始记录未保存描述文本。")


def canonical_record_id(payload, fallback=None):
    normalized = canonicalize_payload(payload)
    value = normalized.get("dn_no")
    if is_missing(value):
        value = normalized.get("record_id")
    if is_missing(value):
        value = fallback
    return "" if is_missing(value) else str(value).strip()
