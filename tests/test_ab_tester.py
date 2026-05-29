import pytest
import numpy as np
from fleet.ab_tester import ABTester, ABTestVariant, ABTest


class TestABTestVariant:
    def test_to_dict(self):
        v = ABTestVariant("control", {"x": 1}, 50.0)
        d = v.to_dict()
        assert d["name"] == "control"
        assert d["traffic_percentage"] == 50.0


class TestABTest:
    def test_to_dict(self):
        t = ABTest("t1", "test", [ABTestVariant("a", {}, 50.0)])
        d = t.to_dict()
        assert d["name"] == "test"
        assert d["status"] == "running"


class TestABTester:
    def test_init(self):
        ab = ABTester()
        assert ab.fleet_node_id == "default"

    def test_create(self):
        ab = ABTester()
        test = ab.create("my_test", [
            {"name": "control", "config": {"a": 1}},
            {"name": "variant", "config": {"a": 2}},
        ])
        assert test.name == "my_test"
        assert len(test.variants) == 2
        assert ab.get_stats()["total_tests"] == 1

    def test_record_metric(self):
        ab = ABTester()
        test = ab.create("test", [{"name": "a"}, {"name": "b"}])
        assert ab.record_metric(test.test_id, "a", "fitness", 1.0) is True
        assert ab.record_metric(test.test_id, "a", "fitness", 2.0) is True
        assert ab.record_metric(test.test_id, "c", "fitness", 1.0) is False

    def test_get_winner(self):
        ab = ABTester()
        test = ab.create("test", [{"name": "a"}, {"name": "b"}])
        ab.record_metric(test.test_id, "a", "fitness", 1.0)
        ab.record_metric(test.test_id, "b", "fitness", 2.0)
        winner = ab.get_winner(test.test_id, "fitness")
        assert winner == "b"

    def test_get_winner_no_data(self):
        ab = ABTester()
        test = ab.create("test", [{"name": "a"}])
        assert ab.get_winner(test.test_id, "fitness") is None

    def test_end(self):
        ab = ABTester()
        test = ab.create("test", [{"name": "a"}])
        assert ab.end(test.test_id) is True
        assert test.status == "ended"
        assert test.ended_at is not None

    def test_end_missing(self):
        ab = ABTester()
        assert ab.end("missing") is False

    def test_get_all_tests(self):
        ab = ABTester()
        ab.create("t1", [{"name": "a"}])
        ab.create("t2", [{"name": "b"}])
        assert len(ab.get_all_tests()) == 2

    def test_get_stats(self):
        ab = ABTester()
        ab.create("t1", [{"name": "a"}])
        t2 = ab.create("t2", [{"name": "b"}])
        ab.end(t2.test_id)
        stats = ab.get_stats()
        assert stats["running"] == 1
        assert stats["ended"] == 1

    def test_export_json(self):
        ab = ABTester()
        ab.create("test", [{"name": "a"}])
        j = ab.export_json()
        assert "test" in j

    def test_to_dict(self):
        ab = ABTester()
        ab.create("test", [{"name": "a"}])
        d = ab.to_dict()
        assert d["stats"]["total_tests"] == 1
