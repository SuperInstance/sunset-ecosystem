import pytest
import numpy as np
from fleet.model_registry import ModelArtifact, ModelRegistry


class TestModelArtifact:
    def test_to_dict(self):
        a = ModelArtifact(name="m", version="v1", data=[1, 2], checksum="abc")
        d = a.to_dict()
        assert d["name"] == "m"
        assert d["checksum"] == "abc"


class TestModelRegistry:
    def test_init(self):
        r = ModelRegistry()
        assert r.artifacts == {}

    def test_publish(self):
        r = ModelRegistry()
        a = r.publish("test", {"x": 1})
        assert a.name == "test"
        assert "test" in r.artifacts

    def test_publish_with_version(self):
        r = ModelRegistry()
        a = r.publish("test", {"x": 1}, version="v1")
        assert a.version == "v1"

    def test_publish_auto_version(self):
        r = ModelRegistry()
        r.publish("test", {"x": 1})
        a2 = r.publish("test", {"x": 2})
        assert a2.version == "v2"

    def test_publish_with_tags(self):
        r = ModelRegistry()
        r.publish("test", {"x": 1}, tags={"type": "breeder"})
        assert len(r.tags_index) == 1

    def test_load(self):
        r = ModelRegistry()
        r.publish("test", {"x": 1})
        data = r.load("test")
        assert data == {"x": 1}

    def test_load_latest(self):
        r = ModelRegistry()
        r.publish("test", {"x": 1}, version="v1")
        r.publish("test", {"x": 2}, version="v2")
        data = r.load("test", version="latest")
        assert data == {"x": 2}

    def test_load_missing(self):
        r = ModelRegistry()
        assert r.load("missing") is None

    def test_get_metadata(self):
        r = ModelRegistry()
        r.publish("test", {"x": 1}, metadata={"accuracy": 0.9})
        meta = r.get_metadata("test")
        assert meta == {"accuracy": 0.9}

    def test_list_versions(self):
        r = ModelRegistry()
        r.publish("test", {"x": 1}, version="v1")
        r.publish("test", {"x": 2}, version="v2")
        versions = r.list_versions("test")
        assert versions == ["v1", "v2"]

    def test_search_by_tag(self):
        r = ModelRegistry()
        r.publish("a", {"x": 1}, tags={"type": "breeder"})
        r.publish("b", {"x": 2}, tags={"type": "other"})
        results = r.search_by_tag("type", "breeder")
        assert len(results) == 1
        assert results[0].name == "a"

    def test_verify_checksum(self):
        r = ModelRegistry()
        r.publish("test", {"x": 1})
        assert r.verify_checksum("test", "v1") is True

    def test_verify_checksum_bad(self):
        r = ModelRegistry()
        r.publish("test", {"x": 1})
        assert r.verify_checksum("test", "nonexistent") is False

    def test_get_stats(self):
        r = ModelRegistry()
        r.publish("a", {"x": 1})
        r.publish("a", {"x": 2}, version="v2")
        r.publish("b", {"x": 3})
        stats = r.get_stats()
        assert stats["models"] == 2
        assert stats["total_artifacts"] == 3

    def test_export_manifest(self):
        r = ModelRegistry()
        r.publish("a", {"x": 1})
        manifest = r.export_manifest()
        assert "a" in manifest
        assert "node" in manifest

    def test_to_dict(self):
        r = ModelRegistry()
        r.publish("a", {"x": 1})
        d = r.to_dict()
        assert d["stats"]["models"] == 1
