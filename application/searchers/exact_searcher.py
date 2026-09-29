from utils.qdrant_client import get_qdrant_manager
from utils.field_extractor import get_field_extractor
from utils.payload_parser import canonicalize_payload, canonical_record_id, merge_payloads
from types import SimpleNamespace


class ExactSearcher:


    def __init__(self):

        self.qdrant = get_qdrant_manager()

        self.extractor = get_field_extractor()



    def merge_by_record_id(self, results):

        """
        multi-grain结果去重
        """

        records={}


        for item in results:


            payload=canonicalize_payload(item.payload)


            record_id=canonical_record_id(payload, item.id)


            if record_id not in records:
                records[record_id]=SimpleNamespace(id=record_id, payload=payload, score=1.0)
            else:
                records[record_id].payload=merge_payloads(records[record_id].payload, payload)



        return list(
            records.values()
        )



    def search(
            self,
            query:str,
            top_k:int=10):


        conditions=(
            self.extractor
            .extract_conditions(query)
        )


        if not conditions:

            raise ValueError(
                "未提取精确字段"
            )



        query_filter=(

            self.qdrant
            .build_exact_filter(
                conditions
            )

        )


        # 优先使用数据库 Payload 索引，通常毫秒级；仅零命中时扫描旧数据并从描述补字段。
        results,elapsed_ms,error=self.qdrant.exact_search(query_filter, limit=top_k*10)
        if not error and not results and hasattr(self.qdrant, "normalized_exact_search"):
            results,elapsed_ms,error=self.qdrant.normalized_exact_search(conditions, limit=top_k)


        if error:

            raise RuntimeError(error)



        # ⭐ multi grain去重

        results=self.merge_by_record_id(
            results
        )


        return (

            results[:top_k],

            elapsed_ms,

            "Exact Payload + record_id merge"

        )



_exact_searcher=None



def get_exact_searcher():


    global _exact_searcher


    if _exact_searcher is None:


        _exact_searcher=ExactSearcher()


    return _exact_searcher
