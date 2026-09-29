import os
from pathlib import Path

# local: CSV + 本地 Qwen；server: Qdrant + vLLM。其余检索参数完全共用。
RUNTIME_PROFILE = os.getenv("WAFER_PROFILE", "server").strip().lower()
if RUNTIME_PROFILE not in {"local", "server"}:
    raise ValueError("WAFER_PROFILE 只能是 local 或 server")
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_DATA_PATH = Path(os.getenv("WAFER_LOCAL_DATA", PROJECT_ROOT / "data/DN_Description_500.csv"))
LOCAL_MODEL_PATH = Path(os.getenv("WAFER_LOCAL_MODEL", PROJECT_ROOT / "models/Qwen3-Embedding-0.6B"))
LOCAL_EMBED_BATCH_SIZE = int(os.getenv("WAFER_LOCAL_EMBED_BATCH", "64"))
LOCAL_EMBED_MAX_LENGTH = int(os.getenv("WAFER_LOCAL_MAX_LENGTH", "512"))
LOCAL_EMBED_CACHE = Path(os.getenv("WAFER_LOCAL_CACHE", PROJECT_ROOT / ".cache/local_embeddings.npy"))
LOCAL_BUILD_EMBEDDINGS = os.getenv("WAFER_LOCAL_BUILD_EMBEDDINGS", "0").lower() in {"1", "true", "yes"}

# ====================
# Qdrant / Embedding 基础配置
# ====================


QDRANT_URL = os.getenv("WAFER_QDRANT_URL", "http://127.0.0.1:6333").strip()
# Optional embedded Qdrant storage for a single-machine demonstration. Leave
# empty in production, where WAFER_QDRANT_URL points to the Qdrant server.
QDRANT_LOCAL_PATH = os.getenv("WAFER_QDRANT_LOCAL_PATH", "").strip()


COLLECTION_NAME = (
    "wafer_collection"
)

# DN-centric collection. The application prefers it after migration and keeps
# COLLECTION_NAME as a reversible fallback until the new collection exists.
UNIFIED_COLLECTION_NAME = os.getenv(
    "WAFER_UNIFIED_COLLECTION", "wafer_dn_records_v2"
).strip()
UNIFIED_TEXT_VECTOR = os.getenv("WAFER_UNIFIED_TEXT_VECTOR", "text_chunks").strip()
UNIFIED_SEM_VECTOR = os.getenv("WAFER_UNIFIED_SEM_VECTOR", "sem_images").strip()
UNIFIED_BINMAP_VECTOR = os.getenv("WAFER_UNIFIED_BINMAP_VECTOR", "binmap_images").strip()
AUTO_SYNC_UNIFIED_IMPORT = os.getenv(
    "WAFER_AUTO_SYNC_UNIFIED_IMPORT", "1"
).strip().lower() in {"1", "true", "yes", "on"}
QDRANT_ONLY = os.getenv("WAFER_QDRANT_ONLY", "1").strip().lower() in {
    "1", "true", "yes", "on"
}
ANALYSIS_KNOWLEDGE_COLLECTION = os.getenv(
    "WAFER_ANALYSIS_KNOWLEDGE_COLLECTION", "wafer_analysis_knowledge_v1"
).strip()
ANALYSIS_KNOWLEDGE_TOP_K = int(os.getenv("WAFER_ANALYSIS_KNOWLEDGE_TOP_K", "4"))



VLLM_BASE_URL = os.getenv("WAFER_VLLM_BASE_URL", "http://127.0.0.1:8333/v1").strip()


VLLM_API_KEY = os.getenv(
    "WAFER_VLLM_API_KEY", "token-not-needed-for-local-vllm"
).strip()



MODEL_NAME = os.getenv("WAFER_EMBEDDING_MODEL", "qwen3-embedding:0.6b").strip()
OLLAMA_EMBED_KEEP_ALIVE = os.getenv("WAFER_OLLAMA_EMBED_KEEP_ALIVE", "0").strip()

# 独立的生成式报告模型。与 embedding 服务分端口，避免把检索向量模型
# 误用于文本生成，也便于单独控制显存占用。
REPORT_LLM_BASE_URL = os.getenv(
    "WAFER_REPORT_LLM_BASE_URL", "http://127.0.0.1:8333/v1"
).strip()
REPORT_LLM_API_KEY = os.getenv(
    "WAFER_REPORT_LLM_API_KEY", "token-not-needed-for-local-vllm"
).strip()
REPORT_LLM_MODEL = os.getenv("WAFER_REPORT_LLM_MODEL", "qwen3-wafer-report:latest").strip()
REPORT_MAX_RECORDS = int(os.getenv("WAFER_REPORT_MAX_RECORDS", "20"))



# ====================
# 检索基础配置
# ====================


# 最终返回数量

DEFAULT_TOP_K = 10



# 第一阶段召回数量

RECALL_TOP_K = 50

# evaluator 处理上千条相关记录时允许扩大候选集；交互搜索仍默认只召回 50。
MAX_RECALL_TOP_K = 5000



# 向量相似度阈值

SCORE_THRESHOLD = 0.5




# ====================
# Rerank配置
# ====================


ENABLE_RERANK = True



# 第一阶段向量分数

VECTOR_WEIGHT = 0.55



# 字段匹配分数

FIELD_WEIGHT = 0.25



# grain类型贡献

GRAIN_WEIGHT = 0.20





# ====================
# Multi Retrieval配置
# ====================


ENABLE_MULTI_RETRIEVAL = True



# 是否开启BM25

ENABLE_BM25 = True



# 是否开启字符TF-IDF

ENABLE_TFIDF = True



# 是否开启BGE-M3
# 当前先关闭
# 后续如果需要第二embedding再开启

ENABLE_BGE = False




# ====================
# Fusion 权重
# ====================


# Qwen3 embedding

QWEN_WEIGHT = 0.45



# BM25关键词

BM25_WEIGHT = 0.30



# 字符TF-IDF

TFIDF_WEIGHT = 0.15



# 字段匹配

FUSION_FIELD_WEIGHT = 0.10





# ====================
# Grain权重
# ====================


GRAIN_WEIGHTS = {


    # 缺陷聚焦
    "defect_focus":
        1.00,


    # 机器相关
    "machine_focus":
        0.90,


    # 结构化摘要
    "summary":
        0.80,


    # 关键词
    "keywords":
        0.80,


    # 自然语言
    "natural_language":
        0.70,


    # 完整文本
    "full":
        0.60

}





# ====================
# BM25 / TF-IDF配置
# ====================


# 每次从Qdrant读取

KEYWORD_INDEX_FROM_QDRANT = True



# Qdrant扫描批次

QDRANT_SCROLL_BATCH = 1000




# ====================
# 字段显示名称
# ====================


FIELD_DISPLAY_NAMES = {

    "dn_no":
        "DN编号",


    "class_type":
        "类型",


    "lot_id":
        "LotID",


    "product_id":
        "产品编号",


    "platform":
        "平台",


    "station":
        "站点",


    "machine":
        "扫描机器",


    "scan_time":
        "扫描时间",


    "total_wafers":
        "Wafer总数",


    "scanned_wafers":
        "已扫描Wafer数",


    "bad_wafers":
        "坏Wafer数",


    "defect_type":
        "缺陷类型",


    "edx_elements":
        "EDX元素",


    "edx_bg":
        "EDX背景",


    "map_location":
        "Map位置",


    "dft_location":
        "DFT位置"

}





# ====================
# 查询字段提取规则
# ====================


FIELD_PATTERNS = {


    "dn_no":

        r"\b(DN-[A-Za-z0-9-]+)\b",



    "class_type":

        r"(?:Class|类别|类型)"
        r"\s*[:：为是]?\s*"
        r"(Class\d+)",



    "lot_id":

        r"(?:LotID|Lot\s*ID|批次号|批次编号)"
        r"\s*[:：为是]?\s*"
        r"([A-Za-z0-9_.-]+)",



    "product_id":

        r"(?:ProductID|产品编号|产品ID|产品)"
        r"\s*[:：=为是]?\s*"
        r"([A-Za-z0-9_.-]+)",



    "platform":

        r"(?:平台|Tech|工艺平台)"
        r"\s*[:：为是]?\s*"
        r"([A-Za-z0-9]+)",



    "station":

        r"(?:站点|Station|工艺站点)"
        r"\s*[:：为是]?\s*"
        r"([A-Za-z0-9_]+)",



    "machine":

        r"(?:扫描机器|机器|Machine)"
        r"\s*[:：为是]?\s*"
        r"([A-Za-z0-9]+)",



    "defect_type":

        r"(?:缺陷类型|Defect|缺陷代码)"
        r"\s*[:：为是]?\s*"
        r"([A-Za-z0-9_]+)",



    "map_location":

        r"(?:Map位置|Map分布)"
        r"\s*[:：为是]?\s*"
        r"([A-Za-z\s]+)"

}
