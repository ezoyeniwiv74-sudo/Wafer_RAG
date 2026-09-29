from config import (
    VECTOR_WEIGHT,
    FIELD_WEIGHT
)

from utils.field_extractor import (
    get_field_extractor
)
from utils.query_processor import normalize_text, tokenize



GRAIN_WEIGHT={

    "defect_focus":1.0,

    "machine_focus":0.9,

    "summary":0.8,

    "keywords":0.8,

    "natural_language":0.7,

    "full":0.6

}



class RerankSearcher:

    FIELD_IMPORTANCE = {
        "dn_no": 3.0, "lot_id": 2.5, "machine": 2.0,
        "product_id": 1.8, "platform": 1.6, "station": 1.5,
        "defect_type": 1.8, "map_location": 1.3, "dft_location": 1.3,
    }


    def __init__(self):

        self.extractor=(
            get_field_extractor()
        )



    def calculate_field_score(
            self,
            payload,
            conditions):


        if not conditions:

            return 0



        hit=0.0
        total=0.0



        for k,v in conditions.items():


            db=self.extractor.normalize(
                payload.get(
                    k,
                    ""
                )
            )
            expected = self.extractor.normalize(v)
            weight = self.FIELD_IMPORTANCE.get(k, 1.0)
            total += weight
            if db == expected:
                hit += weight
            elif expected and (expected in db or db in expected):
                hit += weight * 0.7



        return hit/max(total, 1.0)

    def calculate_text_score(self, payload, query):
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return 0.0
        text = payload.get("search_text", payload.get("text_content", ""))
        document_tokens = set(tokenize(text))
        return len(query_tokens & document_tokens) / len(query_tokens)



    def calculate_grain_score(
            self,
            payload):


        grains=payload.get(
            "matched_grains",
            []
        )


        if not grains:

            return 0



        scores=[

            GRAIN_WEIGHT.get(
                g,
                0.5
            )

            for g in grains

        ]


        return max(scores)



    def rerank(
            self,
            results,
            query):


        conditions=(

            self.extractor
            .extract_conditions(
                query
            )

        )


        scored=[]
        max_base = max((float(getattr(item, "score", 0)) for item in results), default=1.0) or 1.0


        for item in results:


            payload=dict(
                item.payload or {}
            )


            vector_score=float(
                getattr(
                    item,
                    "score",
                    0
                )
            ) / max_base


            field_score=(

                self.calculate_field_score(
                    payload,
                    conditions
                )

            )


            grain_score=(

                self.calculate_grain_score(
                    payload
                )

            )

            text_score = self.calculate_text_score(payload, query)



            item.score=(

                0.50*
                vector_score

                +

                0.35*
                field_score

                +

                0.10*
                text_score

                +

                0.05*
                grain_score

            )


            scored.append(item)



        scored.sort(

            key=lambda x:x.score,

            reverse=True

        )


        return scored

_rerank_searcher = None

def get_rerank_searcher():

    global _rerank_searcher

    if _rerank_searcher is None:
        _rerank_searcher = RerankSearcher()

    return _rerank_searcher
