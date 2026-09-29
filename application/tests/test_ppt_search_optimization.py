import io

from utils.ppt_import import Image, _prepare_picture_for_search


def test_large_ppt_picture_is_reduced_for_search_without_changing_aspect_ratio():
    image = Image.new("RGB", (2048, 1024), (220, 225, 230))
    for x in range(0, 2048, 32):
        image.paste((235, 45, 50), (x, 300, min(x + 12, 2048), 720))
    source = io.BytesIO()
    image.save(source, "PNG")

    optimized, extension = _prepare_picture_for_search(source.getvalue(), ".png")

    with Image.open(io.BytesIO(optimized)) as result:
        assert result.size == (1024, 512)
    assert extension == ".png"


def test_small_ppt_picture_is_not_reencoded():
    image = Image.new("RGB", (640, 480), "white")
    source = io.BytesIO()
    image.save(source, "PNG")
    content = source.getvalue()

    optimized, extension = _prepare_picture_for_search(content, ".png")

    assert optimized == content
    assert extension == ".png"
