from utils.report_generator import _extract_json, build_visualizations, compact_records


def test_report_prompt_records_are_capped_and_auditable():
    rows = [
        {
            "record_id": f"DN-{index}",
            "score": 0.9,
            "fields": {
                "dn_no": f"DN-{index}",
                "machine": "M1",
                "secret": "must-not-enter-prompt",
                "text_content": "edge particle",
            },
        }
        for index in range(40)
    ]
    compact = compact_records(rows)
    assert len(compact) == 20
    assert compact[0]["record_id"] == "DN-0"
    assert "secret" not in compact[0]


def test_visualization_counts_come_from_records():
    charts = build_visualizations([
        {"machine": "M1", "defect_type": "Particle"},
        {"machine": "M1", "defect_type": "Scratch"},
        {"machine": "M2", "defect_type": "Particle"},
    ])
    machine = next(chart for chart in charts if chart["field"] == "machine")
    assert machine["items"] == [
        {"label": "M1", "count": 2},
        {"label": "M2", "count": 1},
    ]


def test_extract_json_removes_qwen_thinking_wrapper():
    parsed = _extract_json('<think>internal</think>\n```json\n{"title":"ok"}\n```')
    assert parsed == {"title": "ok"}


def test_compact_record_description_is_bounded_for_local_context_window():
    compact = compact_records([{
        "record_id": "DN-1", "score": 1,
        "fields": {"dn_no": "DN-1", "text_content": "x" * 1000},
    }])
    assert len(compact[0]["description"]) == 180
