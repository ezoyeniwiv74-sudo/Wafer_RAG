import math

from collections import Counter

from utils.qdrant_client import (
    get_qdrant_manager
)
from utils.payload_parser import canonicalize_payload, canonical_record_id



class TFIDFSearcher:


    def __init__(self):


        self.qdrant=(

            get_qdrant_manager()

        )


        self.documents=[]


        self.idf={}


        self.load()



    # ====================
    # 字符级切分
    # ====================


    def char_ngram(
            self,
            text,
            n=2):


        text=text.lower()


        return [

            text[i:i+n]

            for i in range(

                len(text)-n+1

            )

        ]



    # ====================
    # 加载
    # ====================


    def load(self):


        points=(

            self.qdrant
            .scroll_all_points()

        )


        records={}



        for p in points:


            payload=canonicalize_payload(p.payload)


            rid=canonical_record_id(payload, getattr(p, "id", None))


            if rid is None:

                continue



            if rid in records:

                continue



            text=payload.get(

                "search_text",

                payload.get(
                    "text_content",
                    ""
                )

            )



            records[rid]={

                "record_id":
                    rid,


                "tokens":
                    set(
                        self.char_ngram(
                            text
                        )
                    ),


                "payload":
                    payload

            }



        self.documents=list(
            records.values()
        )



        # IDF计算

        df=Counter()



        for doc in self.documents:


            for token in doc["tokens"]:

                df[token]+=1



        total=len(
            self.documents
        )



        for token,count in df.items():


            self.idf[token]=math.log(

                total /

                (1+count)

            )



        print(

            f"TFIDF加载完成:"
            f"{len(self.documents)}条"

        )




    def vectorize(self,tokens):


        counter=Counter(tokens)



        result={}



        for token,tf in counter.items():


            result[token]=(

                tf*

                self.idf.get(
                    token,
                    0
                )

            )


        return result



    def cosine(
            self,
            a,
            b):


        if not a or not b:

            return 0



        dot=sum(

            a[k]*b.get(k,0)

            for k in a

        )


        na=math.sqrt(

            sum(

                x*x

                for x in a.values()

            )

        )


        nb=math.sqrt(

            sum(

                x*x

                for x in b.values()

            )

        )


        if na==0 or nb==0:

            return 0



        return dot/(na*nb)




    def search(
            self,
            query,
            top_k=50):


        qtokens=(

            self.char_ngram(
                query
            )

        )


        qvec=self.vectorize(
            qtokens
        )


        results=[]



        for doc in self.documents:


            dvec=self.vectorize(

                doc["tokens"]

            )


            score=self.cosine(

                qvec,

                dvec

            )



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




_tfidf_searcher=None



def get_tfidf_searcher():


    global _tfidf_searcher



    if _tfidf_searcher is None:


        _tfidf_searcher=(

            TFIDFSearcher()

        )



    return _tfidf_searcher
