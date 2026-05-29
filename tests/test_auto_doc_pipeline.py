import os
import pytest
from fleet.auto_doc_pipeline import DocFile, AutoDocPipeline


class TestDocFile:
    def test_is_stale(self):
        df = DocFile(path="README.md", content="hello")
        df.last_hash = "different"
        assert df.is_stale() is True

    def test_is_fresh(self):
        df = DocFile(path="README.md", content="hello")
        df.last_hash = df.compute_hash()
        assert df.is_stale() is False


class TestAutoDocPipeline:
    def test_init(self):
        p = AutoDocPipeline()
        assert p.project_root == "."
        assert p.changes_detected == []

    def test_scan_all(self):
        p = AutoDocPipeline()
        p.scan_all(code_dirs=["fleet/"])
        assert len(p.code_hashes) > 0

    def test_detect_changes_none(self):
        p = AutoDocPipeline()
        p.scan_all(code_dirs=["fleet/"])
        changed = p.detect_changes(code_dirs=["fleet/"])
        assert changed == []

    def test_regenerate(self):
        p = AutoDocPipeline()
        p.scan_all(code_dirs=["fleet/"])
        docs = p.regenerate()
        assert "README.md" in docs
        assert "API_REFERENCE.md" in docs
        assert "MODULE_INDEX.md" in docs
        assert len(docs["README.md"]) > 0

    def test_write_to_disk(self, tmp_path):
        p = AutoDocPipeline(project_root=str(tmp_path))
        p.scan_all(code_dirs=["fleet/"])
        p.regenerate()
        written = p.write_to_disk(docs_dir=str(tmp_path / "docs"))
        assert len(written) > 0
        for path in written:
            assert os.path.exists(path)

    def test_get_status(self):
        p = AutoDocPipeline()
        p.scan_all(code_dirs=["fleet/"])
        p.regenerate()
        s = p.get_status()
        assert "modules" in s
        assert "stale_docs" in s
        assert s["doc_files"] == 3

    def test_to_dict(self):
        p = AutoDocPipeline()
        p.scan_all(code_dirs=["fleet/"])
        d = p.to_dict()
        assert "status" in d
        assert "code_files" in d
