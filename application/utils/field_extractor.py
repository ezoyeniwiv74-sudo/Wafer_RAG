"""
字段提取模块

功能:

1. 从用户问题提取结构化字段
2. 判断是否应该走Exact Search
3. 给Rerank提供字段匹配依据
"""


import re
import unicodedata

from config import FIELD_PATTERNS
from utils.payload_parser import parse_payload_from_text, is_missing




class FieldExtractor:

    ALIASES = {"dn_no": ["dn", "dn编号", "dn号"], "lot_id": ["lotid", "lot id", "lot", "批次", "批次号"], "platform": ["平台", "工艺平台", "tech"], "station": ["站点", "工艺站点", "station"], "machine": ["扫描机器", "机器", "机台", "machine"], "defect_type": ["缺陷类型", "缺陷", "defect"]}

    @staticmethod
    def normalize(value):
        value = unicodedata.normalize("NFKC", str(value or ""))
        return re.sub(r"[\s:：=,，。；;（）()\[\]{}]", "", value).lower()

    @staticmethod
    def condition_match(payload, conditions):
        return all(
            FieldExtractor.normalize(payload.get(field, "")) == FieldExtractor.normalize(value)
            for field, value in conditions.items()
        )



    @staticmethod
    def extract_conditions(query:str)->dict:


        conditions={}


        query=unicodedata.normalize("NFKC", query.strip())



        if not query:

            return conditions



        for field_name,pattern in FIELD_PATTERNS.items():


            match=re.search(

                pattern,

                query,

                flags=re.IGNORECASE

            )


            if match:


                value=match.group(1).strip()


                if value:


                    conditions[field_name]=value.strip(" \t\n，。；;:：,.")

        # 支持“平台为CL040LP”“在CL040LP平台”等自然表达
        for field, aliases in FieldExtractor.ALIASES.items():
            if field in conditions:
                continue
            alias = "(?:" + "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True)) + r")(?![A-Za-z])"
            match = re.search(alias + r"\s*(?:为|是|叫|等于|=|:|：)?\s*([A-Za-z0-9][A-Za-z0-9_.-]*)", query, re.I)
            if match:
                conditions[field] = match.group(1).strip("，。；;:：")

        # 中文常见的“值在字段名前”表达：CL040LP平台、Residue_36缺陷。
        suffix_rules = {
            "platform": r"([A-Za-z0-9][A-Za-z0-9_.-]*)\s*(?:工艺)?平台",
            "machine": r"([A-Za-z0-9][A-Za-z0-9_.-]*)\s*(?:扫描机器|机器|机台)",
            "defect_type": r"([A-Za-z0-9][A-Za-z0-9_.-]*)\s*(?:缺陷类型|缺陷)",
            "lot_id": r"([A-Za-z0-9][A-Za-z0-9_.-]*)\s*(?:LotID|批次)",
        }
        for field, pattern in suffix_rules.items():
            if field not in conditions:
                match = re.search(pattern, query, re.I)
                if match:
                    conditions[field] = match.group(1)

        # 兼容“哪些 Buried_Particle_5 发生在 CL028HK”这类省略字段名的表达。
        relation = re.search(r"哪些\s*([A-Za-z0-9][A-Za-z0-9_.-]*)\s*发生在\s*([A-Za-z0-9][A-Za-z0-9_.-]*)", query, re.I)
        if relation:
            conditions.setdefault("defect_type", relation.group(1))
            conditions.setdefault("platform", relation.group(2))

        # 复用入库解析器，保证查询端和数据端对 LotID 等复杂格式采用同一规则。
        parsed = parse_payload_from_text(query)
        for field in FIELD_PATTERNS:
            if field in parsed and not is_missing(parsed[field]) and field not in conditions:
                conditions[field] = parsed[field]



        return conditions




    @staticmethod
    def is_exact_query(query:str)->bool:


        """
        判断是否进入精确查询


        精确查询:

        1. DN编号

        2. LotID

        3. Machine

        4. 多字段组合


        单个平台/缺陷描述
        不认为精确

        """



        conditions=(

            FieldExtractor
            .extract_conditions(
                query
            )

        )



        if not conditions:

            return False



        exact_fields={

            "dn_no",

            "lot_id",

            "machine",
            "platform",
            "station",
            "map_location",
            "dft_location"

        }



        # 强唯一字段

        if any(

            f in conditions

            for f in exact_fields

        ):

            return True

        # 完整缺陷代码或显式“缺陷类型”走精确过滤；“Residue缺陷”仍保留语义检索。
        defect = conditions.get("defect_type")
        if defect and ("缺陷类型" in query or re.fullmatch(r"[A-Za-z][A-Za-z0-9.-]*_\d+", str(defect))):
            return True



        # 多字段组合

        # 两个及以上结构化条件通常表达集合过滤（如平台+缺陷类型）。
        if len(conditions)>=2:

            return True



        return False





    @staticmethod
    def extract_keywords(query:str):


        """
        提供给BM25/TFIDF使用
        """

        return [

            x.strip()

            for x in re.split(

                r"\s+|，|,|。",

                query

            )

            if x.strip()

        ]






_field_extractor=None




def get_field_extractor()->FieldExtractor:


    global _field_extractor



    if _field_extractor is None:


        _field_extractor=FieldExtractor()



    return _field_extractor
