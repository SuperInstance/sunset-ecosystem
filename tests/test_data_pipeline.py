import pytest
from fleet.data_pipeline import DataPipeline, DataRecord


class TestDataRecord:
    def test_to_dict(self):
        r = DataRecord("r1", {"x": 1}, 0.0, "src")
        d = r.to_dict()
        assert d["record_id"] == "r1"
        assert d["data"]["x"] == 1


class TestDataPipeline:
    def test_init(self):
        dp = DataPipeline()
        assert dp.fleet_node_id == "default"

    def test_ingest(self):
        dp = DataPipeline()
        r = dp.ingest({"x": 1}, source="test")
        assert r.source == "test"
        assert r.data["x"] == 1

    def test_add_transform(self):
        dp = DataPipeline()
        dp.add_transform(lambda d: {**d, "y": 2})
        r = dp.ingest({"x": 1})
        processed = dp.process(r)
        assert processed.data["y"] == 2

    def test_add_destination(self):
        dp = DataPipeline()
        received = []
        dp.add_destination(lambda r: received.append(r))
        r = dp.ingest({"x": 1})
        dp.route(r)
        assert len(received) == 1

    def test_process_all(self):
        dp = DataPipeline()
        dp.add_transform(lambda d: {**d, "processed": True})
        dp.ingest({"x": 1})
        dp.ingest({"x": 2})
        results = dp.process_all()
        assert len(results) == 2
        assert all(r.data["processed"] for r in results)

    def test_get_records(self):
        dp = DataPipeline()
        dp.ingest({"x": 1}, source="a")
        dp.ingest({"x": 2}, source="b")
        records = dp.get_records(source="a")
        assert len(records) == 1
        assert records[0].source == "a"

    def test_get_stats(self):
        dp = DataPipeline()
        dp.ingest({"x": 1})
        dp.process_all()
        stats = dp.get_stats()
        assert stats["records"] == 1
        assert stats["stats"]["ingested"] == 1

    def test_export_json(self):
        dp = DataPipeline()
        dp.ingest({"x": 1})
        j = dp.export_json()
        assert "x" in j
        assert "stats" in j

    def test_to_dict(self):
        dp = DataPipeline()
        dp.ingest({"x": 1})
        d = dp.to_dict()
        assert d["stats"]["records"] == 1
