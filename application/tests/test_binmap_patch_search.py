import io

import numpy as np

from utils.binmap_patch_search import (
    Image, defect_mask, is_binmap_image, segment_binmap, segment_manual_region,
)


def make_binmap(rotation=0):
    array = np.zeros((240, 240, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:240, :240]
    wafer = (xx - 120) ** 2 + (yy - 120) ** 2 <= 105 ** 2
    array[wafer] = (214, 219, 226)
    for x in range(78, 142, 10):
        for y in range(88, 142, 10):
            array[y:y + 7, x:x + 7] = (238, 47, 52)
    image = Image.fromarray(array, mode="RGB")
    image = image.rotate(rotation)
    output = io.BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def test_binmap_detection_and_segmentation_locate_defect_cluster():
    content = make_binmap()
    result = segment_binmap(content)

    assert is_binmap_image(content)
    assert result["blocks"]
    assert int(defect_mask(result["image"]).sum()) > 0
    x, y, width, height = result["blocks"][0]["bbox_norm"]
    assert x < 0.5 < x + width
    assert y < 0.5 < y + height


def test_rotation_keeps_radial_geometry_descriptor_stable():
    first = segment_binmap(make_binmap(0))["geometry"]
    rotated = segment_binmap(make_binmap(90))["geometry"]

    assert np.allclose(first, rotated, atol=0.02)


def test_manual_region_uses_exact_user_box_instead_of_auto_segmentation():
    result = segment_manual_region(make_binmap(), [0.1, 0.2, 0.3, 0.4])

    assert len(result["blocks"]) == 1
    block = result["blocks"][0]
    assert block["manual"] is True
    assert np.allclose(block["bbox_norm"], [0.1, 0.2, 0.3, 0.4], atol=0.005)
    assert block["crop"].size == (96, 96)
    assert np.asarray(block["crop"])[0, 0].tolist() == [255, 255, 255]


def test_manual_full_image_region_is_allowed():
    result = segment_manual_region(make_binmap(), [0.0, 0.0, 1.0, 1.0])

    assert np.allclose(result["blocks"][0]["bbox_norm"], [0.0, 0.0, 1.0, 1.0])


def test_genuinely_wafer_wide_defects_may_produce_a_global_region():
    array = np.zeros((240, 240, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:240, :240]
    wafer = (xx - 120) ** 2 + (yy - 120) ** 2 <= 105 ** 2
    array[wafer] = (214, 219, 226)
    # Dense defects cover the whole wafer.  Three local windows cannot explain
    # most of them, so this is the exceptional case where a visible global ROI
    # is useful in addition to the always-on full-image embedding.
    for x, y in (
        (55, 55), (120, 35), (185, 55),
        (35, 120), (120, 120), (200, 120),
        (55, 185), (120, 200), (185, 185),
    ):
        array[y:y + 7, x:x + 7] = (238, 47, 52)
    output = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(output, "PNG")

    result = segment_binmap(output.getvalue())

    broad = [block for block in result["blocks"] if block.get("global")]
    assert broad
    assert any(block["selection_reason"] == "global_defect_extent" for block in broad)


def test_a_few_distant_clusters_do_not_force_a_global_region():
    array = np.zeros((240, 240, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:240, :240]
    wafer = (xx - 120) ** 2 + (yy - 120) ** 2 <= 105 ** 2
    array[wafer] = (214, 219, 226)
    for x, y in ((48, 92), (176, 98), (72, 164), (164, 166)):
        array[y:y + 9, x:x + 9] = (238, 47, 52)
    output = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(output, "PNG")

    result = segment_binmap(output.getvalue())

    assert result["blocks"]
    assert not any(block.get("global") for block in result["blocks"])


def test_compact_automatic_regions_stay_local_and_avoid_canvas_border():
    result = segment_binmap(make_binmap())

    assert 1 <= len(result["blocks"]) <= 3
    for block in result["blocks"]:
        x, y, width, height = block["bbox_norm"]
        assert x > 0 and y > 0
        assert x + width < 1 and y + height < 1
        assert width <= 0.48 and height <= 0.48


def test_large_and_small_patterns_create_different_box_scales():
    array = np.zeros((240, 240, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:240, :240]
    wafer = (xx - 120) ** 2 + (yy - 120) ** 2 <= 105 ** 2
    array[wafer] = (214, 219, 226)
    # One broad connected pattern plus one genuinely separate small pattern.
    array[72:158, 66:151] = (238, 47, 52)
    array[174:184, 178:188] = (238, 47, 52)
    output = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(output, "PNG")

    blocks = segment_binmap(output.getvalue())["blocks"]
    sides = sorted(round(block["bbox_norm"][2], 3) for block in blocks)

    assert len(sides) >= 2
    assert sides[-1] - sides[0] >= 0.15


def test_multiple_manual_rectangles_and_polygon_are_preserved():
    regions = [
        [0.08, 0.12, 0.24, 0.28],
        {"type": "polygon", "points": [[0.55, 0.2], [0.82, 0.24], [0.72, 0.58], [0.52, 0.48]]},
    ]
    result = segment_manual_region(make_binmap(), regions)

    assert len(result["blocks"]) == 2
    assert [block["block_id"] for block in result["blocks"]] == [1, 2]
    assert [block["shape"] for block in result["blocks"]] == ["rectangle", "polygon"]
