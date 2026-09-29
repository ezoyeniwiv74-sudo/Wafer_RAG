from types import SimpleNamespace

from utils.unified_dn_index import (
    build_text_chunks,
    image_manifest,
    merge_records,
    point_id_for_dn,
)


def test_same_dn_always_has_same_point_id():
    assert point_id_for_dn("DN-20241021-01") == point_id_for_dn(" dn-20241021-01 ")
    assert point_id_for_dn("DN-20241021-01") != point_id_for_dn("DN-20241021-02")


def test_legacy_grains_merge_to_one_dn_payload():
    points = [
        SimpleNamespace(id=101, payload={
            "dn_no": "DN-1", "lot_id": "LOT-A", "text_content": "short"
        }),
        SimpleNamespace(id=102, payload={
            "record_id": "DN-1", "platform": "P01", "text_content": "a longer record"
        }),
    ]
    merged = merge_records(points)
    assert list(merged) == ["DN-1"]
    assert merged["DN-1"]["lot_id"] == "LOT-A"
    assert merged["DN-1"]["platform"] == "P01"
    assert merged["DN-1"]["text_content"] == "a longer record"


def test_manifest_contains_portable_keys_not_absolute_paths():
    manifest = image_manifest([{
        "dn_no": "DN-1", "image_type": "TypicalMap",
        "file_name": "TypicalMap.png", "path": r"E:\\data\\YEDN\\DN-1\\TypicalMap.png",
        "url": "/api/yedn/image/DN-1/TypicalMap.png",
    }])
    item = manifest["DN-1"][0]
    assert item["storage_key"] == "YEDN/DN-1/TypicalMap.png"
    assert "E:\\" not in str(item)


def test_text_chunks_keep_fields_and_full_description():
    chunks = build_text_chunks({
        "dn_no": "DN-1", "platform": "P01", "defect_type": "Scratch",
        "text_content": "完整异常记录",
    })
    assert any("platform=P01" in value for value in chunks)
    assert any("defect_type=Scratch" in value for value in chunks)
    assert "完整异常记录" in chunks
