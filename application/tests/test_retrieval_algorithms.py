from types import SimpleNamespace

from searchers.fusion_searcher import FusionSearcher
from searchers.rerank_searcher import RerankSearcher
from utils.field_extractor import FieldExtractor
from utils.query_processor import expand_query, tokenize
from utils.payload_parser import parse_payload_from_text, canonicalize_payload, is_missing


def point(rid, score, **payload):
    payload.setdefault("record_id", rid)
    payload.setdefault("search_text", "")
    return SimpleNamespace(id=rid, score=score, payload=payload)


def test_synonym_expansion_and_mixed_tokenization():
    expanded = expand_query("右下角有残留")
    assert "bottom right" in expanded
    assert "residue" in expanded
    assert "cl040lp" in tokenize("CL040LP平台")


def test_natural_field_extraction():
    query = "请找CL040LP平台的Residue_36缺陷"
    assert FieldExtractor.extract_conditions(query) == {
        "platform": "CL040LP", "defect_type": "Residue_36"
    }


def test_rrf_keeps_lexical_only_candidates():
    semantic = [point("A", 0.9)]
    lexical = [{"record_id": "B", "score": 4, "payload": {"record_id": "B"}}]
    result = FusionSearcher().fuse(semantic, lexical, [])
    assert {item.payload["record_id"] for item in result} == {"A", "B"}


def test_field_aware_rerank_promotes_exact_field_match():
    wrong = point("A", 1.0, platform="CL028HK", defect_type="Particle_12")
    right = point("B", 0.8, platform="CL040LP", defect_type="Residue_36")
    result = RerankSearcher().rerank(
        [wrong, right], "CL040LP平台的Residue_36缺陷"
    )
    assert result[0].payload["record_id"] == "B"


def test_irregular_lot_and_all_fields_are_recovered():
    text = ("当前记录的的DN编号为DN-20250723-09，类型是Class2，"
            "LotID（批次编号为）FP00874。该批次属于产品1002D01-000-F，"
            "当前处于CL040LP平台的YE_BACKSIDEONLY_WEI_DF站点。"
            "该批次由扫描机器FLWEIF01于2025/7/22 20:51:33完成了扫描。"
            "所有Wafer总数为25，其中实际扫描过的Wafer数量为19。"
            "发现坏的Wafer数量为4，核心缺陷类型为Backside_Abnormal_128。"
            "EDX检测到的元素为无，对应的EDX背景为无；"
            "缺陷在Map上面分布的位置为Random，缺陷在DFT上面分布的位置为Random。")
    payload = parse_payload_from_text(text)
    assert payload["record_id"] == "DN-20250723-09"
    assert payload["lot_id"] == "FP00874"
    assert payload["station"] == "YE_BACKSIDEONLY_WEI_DF"
    assert payload["edx_elements"] == "无"
    assert payload["map_location"] == "Random"


def test_old_numeric_record_id_is_replaced_by_dn():
    payload = canonicalize_payload({
        "record_id": 123,
        "dn_no": "DN-20250723-09",
        "lot_id": "UNKNOWN",
        "text_content": "DN编号为DN-20250723-09，LotID为FE00429.0064。",
    })
    assert payload["record_id"] == "DN-20250723-09"
    assert payload["lot_id"] == "FE00429.0064"


def test_business_value_none_is_kept_but_missing_fields_are_defaulted():
    payload = parse_payload_from_text(
        "DN编号为DN-20250723-09，EDX检测到的元素为无，对应的EDX背景为无。"
    )
    assert payload["edx_elements"] == "无"
    assert payload["edx_bg"] == "无"
    assert not is_missing("无")
    assert payload["lot_id"] == "UNKNOWN"
    assert set(payload) >= {"platform", "station", "machine", "defect_type"}


def test_single_platform_and_single_defect_are_exact_queries():
    assert FieldExtractor.is_exact_query("EF032LP平台")
    assert FieldExtractor.is_exact_query("缺陷类型为Si_Damage_428的记录有哪些")
    assert FieldExtractor.extract_conditions("哪些Buried_Particle_5发生在CL028HK？") == {
        "defect_type": "Buried_Particle_5", "platform": "CL028HK"
    }
