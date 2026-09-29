from config import (
    DEFAULT_TOP_K,
    SCORE_THRESHOLD
)


from utils.field_extractor import (
    get_field_extractor
)


from searchers.exact_searcher import (
    get_exact_searcher
)


from searchers.semantic_searcher import (
    get_semantic_searcher
)


from searchers.bm25_searcher import (
    get_bm25_searcher
)


from searchers.tfidf_searcher import (
    get_tfidf_searcher
)


from searchers.fusion_searcher import (
    get_fusion_searcher
)


from searchers.rerank_searcher import (
    get_rerank_searcher
)
from utils.query_processor import expand_query, sanitize_query
import os
import time


def text_embedding_disabled():
    """Return whether the semantic text-vector retrieval branch is disabled."""
    return os.getenv("WAFER_DISABLE_TEXT_EMBEDDING", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }




class HybridSearcher:


    def __init__(self):


        self.extractor = (
            get_field_extractor()
        )


        self.exact_searcher = (
            get_exact_searcher()
        )


        # Exact 查询不加载模型、不扫描全库；混合检索组件首次需要时再初始化。
        self.semantic_searcher = None
        self.bm25_searcher = None
        self.tfidf_searcher = None
        self.fusion_searcher = None
        self.reranker = None
        self.last_timings = {}

    def _ensure_hybrid_components(self):
        started = time.perf_counter()
        if not text_embedding_disabled() and self.semantic_searcher is None:
            self.semantic_searcher = get_semantic_searcher()
        if self.bm25_searcher is None:
            self.bm25_searcher = get_bm25_searcher()
            self.tfidf_searcher = get_tfidf_searcher()
            self.fusion_searcher = get_fusion_searcher()
            self.reranker = get_rerank_searcher()
        return (time.perf_counter() - started) * 1000



    def search(
            self,
            query,
            top_k=DEFAULT_TOP_K,
            score_threshold=SCORE_THRESHOLD):


        total_started=time.perf_counter()
        query=sanitize_query(query)
        lexical_query=expand_query(query)


        if not query:

            raise ValueError(
                "查询不能为空"
            )



        # ======================
        # Exact
        # ======================


        if self.extractor.is_exact_query(query):
            results, elapsed, mode = self.exact_searcher.search(

                query,

                top_k

            )
            total=(time.perf_counter()-total_started)*1000
            self.last_timings={"route": round(total-elapsed,2), "payload_search": round(elapsed,2), "total": round(total,2)}
            return results, total, mode

        init_ms=self._ensure_hybrid_components()



        # ======================
        # Semantic
        # ======================


        if text_embedding_disabled():
            semantic_results = []
            semantic_wall = 0.0
        else:
            stage=time.perf_counter()
            semantic_results, elapsed, mode = (

                self.semantic_searcher.search(

                    query,

                    top_k=max(50, top_k),

                    score_threshold=
                        score_threshold

                )

            )
            semantic_wall=(time.perf_counter()-stage)*1000



        # ======================
        # BM25
        # ======================


        stage=time.perf_counter()
        bm25_results = (

            self.bm25_searcher.search(

                lexical_query,

                top_k=max(50, top_k)

            )

        )
        bm25_ms=(time.perf_counter()-stage)*1000



        # ======================
        # TFIDF
        # ======================


        stage=time.perf_counter()
        tfidf_results = (

            self.tfidf_searcher.search(

                lexical_query,

                top_k=max(50, top_k)

            )

        )
        tfidf_ms=(time.perf_counter()-stage)*1000

        # 单个结构化字段不直接结束检索，但把 payload 精确命中作为独立召回通道。
        stage=time.perf_counter()
        field_results = []
        if self.extractor.extract_conditions(query):
            try:
                field_results = self.exact_searcher.search(query, max(top_k, 50))[0]
            except Exception:
                field_results = []
        field_ms=(time.perf_counter()-stage)*1000



        # ======================
        # Fusion
        # ======================


        stage=time.perf_counter()
        fused_results=(

            self.fusion_searcher.fuse(

                semantic_results,

                bm25_results,

                tfidf_results,

                field_results

            )

        )
        fusion_ms=(time.perf_counter()-stage)*1000



        # 防止fusion为空

        if not fused_results:

            fused_results = semantic_results



        # ======================
        # Rerank
        # ======================


        stage=time.perf_counter()
        rerank_results=(

            self.reranker.rerank(

                fused_results,

                query

            )

        )
        rerank_ms=(time.perf_counter()-stage)*1000

        total=(time.perf_counter()-total_started)*1000
        semantic_detail=(
            {}
            if self.semantic_searcher is None
            else getattr(self.semantic_searcher,"last_timings",{})
        )
        self.last_timings={"initialize":round(init_ms,2), **semantic_detail,
                           "semantic_total":round(semantic_wall,2), "bm25":round(bm25_ms,2),
                           "tfidf":round(tfidf_ms,2), "field_recall":round(field_ms,2),
                           "fusion":round(fusion_ms,2), "rerank":round(rerank_ms,2), "total":round(total,2)}



        return (

            rerank_results[:top_k],

            total,

            (
                "BM25+TFIDF+Fusion+Rerank (text embedding disabled)"
                if text_embedding_disabled()
                else "Qwen3+BM25+TFIDF+Fusion+Rerank"
            )

        )



    def explain_query(self,query):


        conditions=(

            self.extractor
            .extract_conditions(query)

        )


        return {

            "query":
                query,

            "conditions":
                conditions,

            "strategy":
                (
                    "Exact Payload"
                    if conditions
                    else
                    "Multi Retrieval"
                )

        }




_hybrid_searcher=None



def get_hybrid_searcher():


    global _hybrid_searcher


    if _hybrid_searcher is None:


        _hybrid_searcher = HybridSearcher()



    return _hybrid_searcher
