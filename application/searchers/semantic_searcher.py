from utils.qdrant_client import get_qdrant_manager
from utils.embedding11 import get_embedding_generator
from config import RECALL_TOP_K, MAX_RECALL_TOP_K
from types import SimpleNamespace
from utils.payload_parser import canonicalize_payload, canonical_record_id
import time


class SemanticSearcher:


    def __init__(self):

        self.qdrant = get_qdrant_manager()

        self.embedder = None
        self.last_timings = {}


    def merge_by_record_id(self, results):

        """
        多粒度结果合并

        一个record_id只保留一次

        """

        records={}


        for item in results:


            payload=canonicalize_payload(item.payload)


            record_id=canonical_record_id(payload, item.id)



            score=float(
                getattr(
                    item,
                    "score",
                    0
                )
            )


            grain_type=payload.get(
                "grain_type",
                ""
            )



            if record_id not in records:


                records[record_id]={

                    "item":item,

                    "score":score,

                    "grain_types":[
                        grain_type
                    ]

                }


            else:


                old=records[record_id]


                old["grain_types"].append(
                    grain_type
                )


                # 保留最高分grain

                if score > old["score"]:

                    old["score"]=score

                    old["item"]=item



        merged=[]


        for value in records.values():

            source=value["item"]
            payload=canonicalize_payload(source.payload)
            payload["matched_grains"]=[g for g in value["grain_types"] if g]
            item=SimpleNamespace(
                id=canonical_record_id(payload, getattr(source, "id", None)),
                payload=payload,
                score=float(value["score"]),
            )


            merged.append(item)



        merged.sort(

            key=lambda x:x.score,

            reverse=True

        )


        return merged



    def search(
            self,
            query,
            top_k=10,
            score_threshold=0.5):


        if hasattr(self.qdrant, "set_query"):
            self.qdrant.set_query(query)

        if getattr(self.qdrant, "requires_query_embedding", True):
            if self.embedder is None:
                self.embedder = get_embedding_generator()
            vector,embed_time=self.embedder.get_embedding_with_time(query, is_query=True)
        else:
            vector,embed_time=[],0.0


        results,search_time,error=(

            self.qdrant.vector_search(

                query_vector=vector,

                limit=min(max(RECALL_TOP_K, top_k), MAX_RECALL_TOP_K),

                score_threshold=
                    score_threshold

            )

        )
        self.last_timings = {
            "embedding": round(float(embed_time), 2),
            "vector_search": round(float(search_time), 2),
        }


        if error:

            raise RuntimeError(error)



        merged=self.merge_by_record_id(
            results
        )


        return (

            merged[:top_k],

            embed_time+search_time,

            "Qwen3 multi-grain"

        )



_semantic_searcher=None



def get_semantic_searcher():

    global _semantic_searcher


    if _semantic_searcher is None:

        _semantic_searcher=SemanticSearcher()


    return _semantic_searcher
