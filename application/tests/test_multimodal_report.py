from api import _merge_report_records


def test_multimodal_evidence_is_deduplicated_and_sources_are_preserved():
    selected = [{"record_id": "DN-1", "score": 0.7, "fields": {"dn_no": "DN-1", "machine": "M1"}}]
    image = [{"record_id": "DN-1", "score": 0.9, "fields": {"dn_no": "DN-1", "defect_type": "Particle"}}]
    merged = _merge_report_records([("selected", selected), ("user_image_qdrant", image)])
    assert len(merged) == 1
    assert merged[0]["score"] == 0.9
    assert merged[0]["fields"]["machine"] == "M1"
    assert merged[0]["fields"]["defect_type"] == "Particle"
    assert merged[0]["evidence_sources"] == ["selected", "user_image_qdrant"]
