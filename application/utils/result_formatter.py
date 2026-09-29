"""
结果格式化模块

支持:
- multi-grain结果展示
- record_id去重
- grain命中展示
"""


from collections import OrderedDict

from config import FIELD_DISPLAY_NAMES
from utils.payload_parser import canonicalize_payload, canonical_record_id




class ResultFormatter:



    @staticmethod
    def format_results(
            results,
            elapsed_ms: float,
            search_mode: str):


        print("\n" + "="*80)

        print(
            "检索结果"
        )

        print("="*80)


        print(
            f"检索方式: {search_mode}"
        )


        print(
            f"检索耗时: {elapsed_ms:.2f} ms"
        )



        # ==========================
        # record_id去重
        # ==========================


        deduped = OrderedDict()



        for item in results:


            payload=canonicalize_payload(item.payload)



            record_id = canonical_record_id(payload, getattr(item, "id", None)) or "UNKNOWN"



            score=getattr(
                item,
                "score",
                None
            )


            if score is None:

                score=1.0



            if (

                record_id not in deduped

                or

                score >
                deduped[record_id]["score"]

            ):


                deduped[record_id]={


                    "item":
                        item,


                    "score":
                        score,


                    "payload":
                        payload

                }



        print(
            f"结果数量: {len(deduped)}"
        )



        if not deduped:


            print(
                "\n未找到满足条件的记录。"
            )


            return



        # ==========================
        # 排序
        # ==========================


        sorted_results=sorted(

            deduped.values(),

            key=lambda x:x["score"],

            reverse=True

        )



        for index,data in enumerate(
                sorted_results,
                start=1):


            item=data["item"]

            payload=data["payload"]

            score=data["score"]



            print("\n"+"-"*80)


            print(
                f"Top {index}"
            )


            print(

                "Record ID:",

                canonical_record_id(payload, getattr(item, "id", None)) or "UNKNOWN"

            )



            if score==1.0 and "精确" in search_mode:


                print(
                    "匹配方式: Payload精确匹配"
                )


            else:


                print(

                    f"综合得分: {float(score):.6f}"

                )



            # ======================
            # grain信息
            # ======================


            grains=payload.get(
                "matched_grains",
                []
            )


            if grains:


                print(
                    "命中粒度:"
                )


                print(
                    " , ".join(grains)
                )



            print(
                "\n字段信息:"
            )



            for field_key,display_name in FIELD_DISPLAY_NAMES.items():


                value=payload.get(

                    field_key,

                    "未知"

                )


                print(

                    f"{display_name}: {value}"

                )



            # 原始文本

            if payload.get(
                    "text_content"):


                print(
                    "\n异常描述:"
                )


                print(
                    payload["text_content"]
                )





    @staticmethod
    def format_single_result(
            item,
            index:int=1):


        payload=canonicalize_payload(item.payload)


        lines=[]


        lines.append(

            f"Top {index}"

        )


        lines.append(

            "Record ID: "

            +

            str(

                canonical_record_id(payload, getattr(item, "id", None)) or "UNKNOWN"

            )

        )


        score=getattr(
            item,
            "score",
            None
        )


        if score is not None:


            lines.append(

                f"综合得分: {float(score):.6f}"

            )



        grains=payload.get(
            "matched_grains",
            []
        )


        if grains:


            lines.append(

                "命中粒度:"

                +

                ",".join(grains)

            )



        for field_key,display_name in FIELD_DISPLAY_NAMES.items():


            value=payload.get(
                field_key,
                "未知"
            )


            lines.append(

                f"{display_name}: {value}"

            )


        return "\n".join(lines)





# ==========================
# Singleton
# ==========================


_result_formatter=None



def get_result_formatter():


    global _result_formatter



    if _result_formatter is None:


        _result_formatter=ResultFormatter()



    return _result_formatter
