from config import (

    QWEN_WEIGHT,

    BM25_WEIGHT,

    TFIDF_WEIGHT,

    FUSION_FIELD_WEIGHT

)
from types import SimpleNamespace
from utils.payload_parser import canonicalize_payload, canonical_record_id



class FusionSearcher:

    RRF_K = 60



    def __init__(self):

        pass



    def get_record_id(
            self,
            item):


        if isinstance(
                item,
                dict):


            payload = item.get("payload") or item
            return canonical_record_id(payload, item.get("record_id"))


        return canonical_record_id(item.payload, getattr(item, "id", None))



    def fuse(
            self,
            semantic_results,
            bm25_results,
            tfidf_results,
            field_results=None):



        scores={}


        objects={}



        groups=[


            (
                semantic_results,
                QWEN_WEIGHT
            ),


            (
                bm25_results,
                BM25_WEIGHT
            ),


            (
                tfidf_results,
                TFIDF_WEIGHT
            ),

            (field_results or [], FUSION_FIELD_WEIGHT)

        ]



        for results,weight in groups:


            for rank,item in enumerate(results):


                rid=(

                    self.get_record_id(
                        item
                    )

                )


                if rid is None:

                    continue



                # Weighted Reciprocal Rank Fusion：只比较名次，不混用不同算法的原始分数尺度。
                scores[rid]=(

                    scores.get(
                        rid,
                        0
                    )

                    +

                    weight/(self.RRF_K+rank+1)

                )



                if rid not in objects or hasattr(item, "payload"):
                    objects[rid]=item




        final=[]



        for rid,score in scores.items():


            source=objects[rid]
            raw_payload = source.payload if hasattr(source, "payload") else source.get("payload", {})
            item = SimpleNamespace(id=rid, payload=canonicalize_payload(raw_payload), score=float(score))


            final.append(item)




        final.sort(

            key=lambda x:x.score,

            reverse=True

        )



        return final




_fusion_searcher=None



def get_fusion_searcher():


    global _fusion_searcher



    if _fusion_searcher is None:


        _fusion_searcher=(

            FusionSearcher()

        )



    return _fusion_searcher
