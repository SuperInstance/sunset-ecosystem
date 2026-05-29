"""Tests for backup_manager.py — Backup scheduling and rotation.

Run: python3 -m pytest tests/test_backup_manager.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.backup_manager import BackupManager


class TestBackupManager:
    def test_create(self):
        mgr = BackupManager()
        assert mgr.stats()["schedules"] == 0

    def test_add_schedule(self):
        mgr = BackupManager()
        mgr.add_schedule("daily", interval_sec=86400, retention=7)
        assert mgr.stats()["schedules"] == 1

    def test_remove_schedule(self):
        mgr = BackupManager()
        mgr.add_schedule("daily", interval_sec=86400, retention=7)
        assert mgr.remove_schedule("daily") is True
        assert mgr.remove_schedule("missing") is False

    def test_record_backup(self):
        mgr = BackupManager()
        mgr.add_schedule("daily", interval_sec=86400, retention=7)
        mgr.record_backup("daily", success=True)
        assert mgr.last_backup("daily")["success"] is True

    def test_rotation(self):
        mgr = BackupManager()
        mgr.add_schedule("daily", interval_sec=86400, retention=2)
        mgr.record_backup("daily", success=True)
        mgr.record_backup("daily", success=True)
        mgr.record_backup("daily", success=True)
        assert len(mgr.backup_history("daily")) == 2

    def test_due_schedules(self):
        mgr = BackupManager()
        mgr.add_schedule("daily", interval_sec=0.05, retention=2)
        assert mgr.due_schedules() == ["daily"]
        mgr.record_backup("daily", success=True)
        assert mgr.due_schedules() == []
        time.sleep(0.06)
        assert mgr.due_schedules() == ["daily"]

    def test_success_rate(self):
        mgr = BackupManager()
        mgr.add_schedule("daily", interval_sec=86400, retention=7)
        mgr.record_backup("daily", success=True)
        mgr.record_backup("daily", success=False)
        assert mgr.success_rate("daily") == 0.5

    def test_invalid_schedule(self):
        mgr = BackupManager()
        with pytest.raises(ValueError):
            mgr.record_backup("missing", success=True)

    def test_repr(self):
        mgr = BackupManager()
        assert "BackupManager" in repr(mgr)
