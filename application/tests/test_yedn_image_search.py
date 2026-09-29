from pathlib import Path

import numpy as np

import utils.image_search as image_module
from utils.image_search import ImageSearcher, classify_image


def test_company_image_names_are_recognized():
    assert classify_image(Path("MoreImage.JPG")) == "MoreImage"
    assert classify_image(Path("TypicalDefectImage1.png")) == "TypicalDefectImage1"
    assert classify_image(Path("TypeicalDefectImage2.TIF")) == "TypicalDefectImage2"
    assert classify_image(Path("TypicalMap.jpeg")) == "TypicalMap"
    assert classify_image(Path("TypicalMap2.jpeg")) == "TypicalMap2"
    assert classify_image(Path("DefectMapPost.JPG")) == "DefectMapPost"
    assert classify_image(Path("DefectBarChart.jpg")) == "DefectBarChart"
    assert classify_image(Path("AddScanDefectMapGallary1.JPG")) == "AddScanDefectMapGallary1"


def make_searcher(tmp_path, monkeypatch):
    root = tmp_path / "data" / "YEDN"
    dns = ["DN-20241024-08", "DN-20241024-09", "DN-20241024-10"]
    names = [
        "MoreImage.JPG", "TypicalDefectImage1.JPG", "TypicalDefectImage2.JPG",
        "TypicalMap.JPG", "DefectMapPost.JPG", "DefectTrendChart.JPG",
        "AddScanDefectMapGallary.JPG", "AddScanDefectMapGallary1.JPG",
        "DefectBarChart.JPG", "DefectMapPre.JPG", "DefectParetoChart.JPG",
        "DieStack.JPG", "EDXDefectImage.JPG", "OperatorSnapshot.JPG",
    ]
    for index, dn_no in enumerate(dns):
        folder = root / dn_no
        folder.mkdir(parents=True)
        for name in names:
            (folder / name).write_bytes(bytes([65 + index]))

    class FakeProvider:
        def embed_bytes(self, images, modality):
            mapping = {65: [1.0, 0.0], 66: [0.9, 0.1], 67: [0.8, 0.2], 81: [1.0, 0.0]}
            rows = np.asarray([mapping[data[0]] for data in images], dtype=np.float32)
            return rows / np.linalg.norm(rows, axis=1, keepdims=True)

        def analyze_bytes(self, images, modality):
            rows = self.embed_bytes(images, modality)
            predictions = [{"class_index": 0, "class_name": "Defect_1", "confidence": 0.9,
                            "probabilities": {"Defect_1": 0.9, "Defect_2": 0.1}} for _ in images]
            return rows, predictions

    searcher = ImageSearcher()
    searcher.root = root
    searcher.provider = FakeProvider()
    searcher._records = {
        dns[0]: {"dn_no": dns[0], "class_type": "Class1", "defect_type": "Particle", "text_content": "记录A"},
        dns[1]: {"dn_no": dns[1], "class_type": "Class1", "defect_type": "Particle", "text_content": "记录B"},
        dns[2]: {"dn_no": dns[2], "class_type": "Class2", "defect_type": "Scratch", "text_content": "记录C"},
    }
    monkeypatch.setattr(image_module, "INDEX_PATHS", {
        "binmap": tmp_path / "binmap_index.npz",
        "sem": tmp_path / "sem_index.npz",
    })
    # Keep unit tests independent from a Qdrant service that may be running on
    # the developer machine. These tests exercise the local fusion path.
    class LocalOnlyManager:
        using_unified_collection = False

    monkeypatch.setattr(image_module, "get_qdrant_manager", lambda: LocalOnlyManager())
    return searcher, dns


def assert_common_results(result, dns, expected_images):
    assert [item["dn_no"] for item in result["results"]] == dns
    assert all(item["image_count"] == expected_images for item in result["results"])
    assert all(len(item["result_images"]) == expected_images for item in result["results"])
    assert result["results"][0]["score"] >= result["results"][1]["score"]
    assert [item["rank"] for item in result["results"]] == [1, 2, 3]
    assert result["candidate_count"] == 3
    assert all(set(row) == {"field", "label", "dominant_value"}
               for row in result["statistics_table"])
    assert next(row for row in result["statistics_table"]
                if row["field"] == "defect_type")["dominant_value"] == "Particle"


def test_binmap_search_uses_only_defect_map_post(tmp_path, monkeypatch):
    searcher, dns = make_searcher(tmp_path, monkeypatch)
    result = searcher.search(b"Query", modality="binmap", limit=500)
    assert_common_results(result, dns, 1)
    assert all(item["result_images"][0]["image_type"] == "DefectMapPost" for item in result["results"])


def test_sem_search_averages_only_two_typical_scanning_images(tmp_path, monkeypatch):
    searcher, dns = make_searcher(tmp_path, monkeypatch)
    result = searcher.search(b"Query", modality="sem", limit=500)
    assert_common_results(result, dns, 2)
    assert all({image["image_type"] for image in item["result_images"]}
               == {"TypicalDefectImage1", "TypicalDefectImage2"}
               for item in result["results"])


def test_multi_value_filter_is_or_within_field(tmp_path, monkeypatch):
    searcher, dns = make_searcher(tmp_path, monkeypatch)
    result = searcher.search(
        b"Query", modality="sem", filters={"defect_type": ["Particle", "Scratch"]}, limit=500
    )
    assert [item["dn_no"] for item in result["results"]] == dns


def test_multi_image_search_fuses_every_query_against_all_indexes(tmp_path, monkeypatch):
    searcher, dns = make_searcher(tmp_path, monkeypatch)

    result = searcher.search_many([b"Query", b"Query"], limit=500)

    assert result["mode"] == "multi_image_fusion"
    assert result["input_image_count"] == 2
    assert [item["dn_no"] for item in result["results"]] == dns
    assert result["results"][0]["query_image_count"] == 2
    assert len(result["results"][0]["query_similarities"]) == 2
    assert {
        item["image_type"] for item in result["results"][0]["result_images"]
    } == {"DefectMapPost", "TypicalDefectImage1", "TypicalDefectImage2"}


def test_catalog_keeps_display_only_images_out_of_retrieval(tmp_path, monkeypatch):
    searcher, _ = make_searcher(tmp_path, monkeypatch)
    catalog_types = {item["image_type"] for item in searcher.catalog()}
    assert {
        "MoreImage", "TypicalMap", "DefectTrendChart",
        "AddScanDefectMapGallary", "AddScanDefectMapGallary1",
        "DefectBarChart", "DefectMapPre", "DefectParetoChart",
        "DieStack", "EDXDefectImage", "OperatorSnapshot",
    } <= catalog_types
    assert {item["image_type"] for item in searcher.ensure_index("binmap")[0]} == {"DefectMapPost"}
    assert {item["image_type"] for item in searcher.ensure_index("sem")[0]} == {
        "TypicalDefectImage1", "TypicalDefectImage2"
    }


def test_catalog_for_dn_scans_only_requested_folder_and_caches_it(tmp_path, monkeypatch):
    searcher, dns = make_searcher(tmp_path, monkeypatch)
    original = searcher._catalog_folder
    scanned = []

    def tracked(folder):
        scanned.append(folder.name)
        return original(folder)

    monkeypatch.setattr(searcher, "_catalog_folder", tracked)
    first = searcher.catalog_for_dn(dns[1])
    second = searcher.catalog_for_dn(dns[1].lower())

    assert len(first) == 14
    assert second is first
    assert scanned == [dns[1]]
