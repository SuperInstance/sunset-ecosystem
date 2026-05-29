import pytest
from fleet.backup_manager import BackupManager, Snapshot


class TestSnapshot:
    def test_to_dict(self):
        s = Snapshot("s1", 0.0, {"x": 1})
        d = s.to_dict()
        assert d["snapshot_id"] == "s1"
        assert d["data"]["x"] == 1


class TestBackupManager:
    def test_init(self):
        bm = BackupManager()
        assert bm.fleet_node_id == "default"
        assert bm.get_stats()["total_snapshots"] == 0

    def test_snapshot(self):
        bm = BackupManager()
        s = bm.snapshot({"state": "active"})
        assert s.snapshot_id.startswith("snap_")
        assert s.data["state"] == "active"
        assert bm.get_stats()["total_snapshots"] == 1

    def test_restore(self):
        bm = BackupManager()
        s = bm.snapshot({"state": "active"})
        restored = bm.restore(s.snapshot_id)
        assert restored["state"] == "active"

    def test_restore_missing(self):
        bm = BackupManager()
        assert bm.restore("missing") is None

    def test_get_snapshots_with_tag(self):
        bm = BackupManager()
        bm.snapshot({"a": 1}, tags={"env": "prod"})
        bm.snapshot({"b": 2}, tags={"env": "dev"})
        snaps = bm.get_snapshots(tag_key="env", tag_value="prod")
        assert len(snaps) == 1
        assert snaps[0].data["a"] == 1

    def test_get_latest(self):
        bm = BackupManager()
        bm.snapshot({"a": 1})
        bm.snapshot({"b": 2})
        latest = bm.get_latest()
        assert latest.data["b"] == 2

    def test_get_latest_empty(self):
        bm = BackupManager()
        assert bm.get_latest() is None

    def test_delete(self):
        bm = BackupManager()
        s = bm.snapshot({"a": 1})
        assert bm.delete(s.snapshot_id) is True
        assert bm.get_stats()["total_snapshots"] == 0

    def test_delete_missing(self):
        bm = BackupManager()
        assert bm.delete("missing") is False

    def test_max_snapshots(self):
        bm = BackupManager(max_snapshots=3)
        for i in range(5):
            bm.snapshot({"i": i})
        assert bm.get_stats()["total_snapshots"] == 3

    def test_export_json(self):
        bm = BackupManager()
        bm.snapshot({"a": 1})
        j = bm.export_json()
        assert "a" in j
        assert "snapshots" in j

    def test_to_dict(self):
        bm = BackupManager()
        bm.snapshot({"a": 1})
        d = bm.to_dict()
        assert d["stats"]["total_snapshots"] == 1
