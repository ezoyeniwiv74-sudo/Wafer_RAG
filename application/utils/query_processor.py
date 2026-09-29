"""查询规范化、领域同义词扩展及中英混合分词。"""

import re
import unicodedata


SYNONYMS = {
    "右下角": "bottom right", "右下": "bottom right",
    "左下角": "bottom left", "左下": "bottom left",
    "右上角": "top right", "右上": "top right",
    "左上角": "top left", "左上": "top left",
    "中心": "center", "中央": "center", "中间": "center",
    "随机分布": "random", "随机": "random",
    "残留": "residue", "残渣": "residue",
    "划伤": "scratch", "刮伤": "scratch",
    "颗粒": "particle", "粒子": "particle",
    "机台": "machine", "扫描机": "machine",
    "批次": "lotid", "批号": "lotid",
}

STOPWORDS = {"查询", "查找", "帮我", "请", "一下", "相关", "记录", "信息", "异常", "哪些", "所有", "有没有", "对应"}


def normalize_text(text):
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    text = re.sub(r"[?？!！]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sanitize_query(text):
    """清理标点但保留工业字段值大小写，供 Payload 精确过滤使用。"""
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = re.sub(r"[?？!！]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def expand_query(query):
    normalized = normalize_text(query)
    additions = [canonical for alias, canonical in SYNONYMS.items() if alias in normalized]
    return normalized + (" " + " ".join(additions) if additions else "")


def tokenize(text):
    """保留编号整体，同时加入英文词、中文单字和中文 bigram。"""
    text = expand_query(text)
    tokens = re.findall(r"[a-z0-9]+(?:[_.-][a-z0-9]+)*|[\u4e00-\u9fff]+", text)
    output = []
    for token in tokens:
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            output.extend(ch for ch in token if ch not in STOPWORDS)
            output.extend(token[i:i + 2] for i in range(len(token) - 1)
                          if token[i:i + 2] not in STOPWORDS)
        elif token not in STOPWORDS:
            output.append(token)
    return output
