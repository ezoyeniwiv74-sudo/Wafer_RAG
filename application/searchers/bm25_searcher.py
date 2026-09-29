import math
from collections import Counter

from utils.qdrant_client import get_qdrant_manager
from utils.query_processor import tokenize
from utils.payload_parser import canonicalize_payload, canonical_record_id


class BM25Searcher:


    def __init__(self):

        self.qdrant = (
            get_qdrant_manager()
        )

        self.documents=[]
        self.doc_freq=Counter()
        self.avg_length=0.0

        self.load_documents()



    def load_documents(self):


        points = (
            self.qdrant
            .scroll_all_points()
        )


        seen=set()


        for p in points:


            payload=canonicalize_payload(p.payload)


            record_id=canonical_record_id(payload, getattr(p, "id", None))


            # 一个异常只保存一次

            if record_id in seen:

                continue


            seen.add(record_id)


            text = payload.get("search_text", payload.get("text_content", ""))
            tokens = tokenize(text)
            self.documents.append({

                "record_id":
                    record_id,

                "text": text,
                "tokens": Counter(tokens),
                "length": len(tokens),

                "payload":
                    payload

            })

            self.doc_freq.update(set(tokens))

        if self.documents:
            self.avg_length = sum(d["length"] for d in self.documents) / len(self.documents)



    def search(
            self,
            query,
            top_k=50):


        results=[]


        query_tokens = list(dict.fromkeys(tokenize(query)))
        total_docs = len(self.documents)


        for doc in self.documents:


            score=0.0
            k1, b = 1.5, 0.75
            for word in query_tokens:
                tf = doc["tokens"].get(word, 0)
                if not tf:
                    continue
                df = self.doc_freq.get(word, 0)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                norm = tf + k1 * (1 - b + b * doc["length"] / max(self.avg_length, 1))
                score += idf * tf * (k1 + 1) / norm



            if score>0:

                results.append({

                    "record_id":
                        doc["record_id"],

                    "score":
                        score,

                    "payload":
                        doc["payload"]

                })



        results.sort(

            key=lambda x:x["score"],

            reverse=True

        )


        return results[:top_k]



_bm25_searcher=None



def get_bm25_searcher():

    global _bm25_searcher


    if _bm25_searcher is None:

        _bm25_searcher=BM25Searcher()


    return _bm25_searcher
