"""Tests for FleetWeatherReport.

Phase 5.4 — Fleet Weather Report system.

All tests use mocked conductors/stats so they run without heavy
sunset-ecosystem dependencies.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from fleet.fleet_weather_report import (
    BreedingSummary,
    FleetStats,
    FleetWeatherReport,
    NodeHealth,
    _append_history,
    _maybe_load_history,
)


# ── fixtures ────────────────────────────────────────────────

@pytest.fixture
def mock_conductor():
    """Return a MagicMock that looks enough like FleetConductorV2."""
    conductor = MagicMock()
    conductor._node_id = "test-node"
    conductor._status_log = []
    conductor._subsystems = {}

    def _get_status():
        return {
            "node_id": "test-node",
            "uptime_seconds": 3600.0,
            "beat_count": 42,
            "subsystems": {
                "traps": {
                    "state": "healthy",
                    "consecutive_failures": 0,
                    "last_error": "",
                    "detail": {
                        "thermal": {"temperature": 65.0},
                    },
                },
                "mesh": {
                    "state": "healthy",
                    "consecutive_failures": 0,
                    "detail": {"stats": {"total_entries": 23}},
                },
                "breeder": {
                    "state": "healthy",
                    "consecutive_failures": 0,
                },
            },
            "nodes": ["peer-alpha", "peer-beta"],
            "agents": ["agent-1", "agent-2"],
            "drift_ms": 3.5,
            "diversity": 23,
            "health": "healthy",
            "queued_tasks": 0,
            "timestamp": 1234567890.0,
        }

    conductor.get_status = _get_status
    return conductor


@pytest.fixture
def mock_conductor_with_breeder():
    """Conductor with an active breeder subsystem instance."""
    conductor = MagicMock()
    conductor._node_id = "breeder-node"
    conductor._status_log = [
        {"beat_number": 1, "subsystem_ticks": {"breeder": "ok"}},
        {"beat_number": 2, "subsystem_ticks": {"breeder": "ok"}},
        {"beat_number": 3, "subsystem_ticks": {"breeder": "ok", "mesh": "err"}},
    ]

    breeder_instance = MagicMock()
    breeder_instance.get_status.return_value = {
        "breeds_attempted": 10,
        "flux_passes": 8,
        "flux_fails": 1,
        "thermal_throttled": 1,
        "queue_depth": 3,
    }

    mesh_instance = MagicMock()
    mesh_instance.stats = {"total_entries": 47}

    breeder_wrapper = MagicMock()
    breeder_wrapper.instance = breeder_instance
    breeder_wrapper.health = MagicMock()
    breeder_wrapper.health.consecutive_failures = 0
    breeder_wrapper.health.last_error = ""

    mesh_wrapper = MagicMock()
    mesh_wrapper.instance = mesh_instance
    mesh_wrapper.health = MagicMock()
    mesh_wrapper.health.consecutive_failures = 0
    mesh_wrapper.health.last_error = ""

    trap_wrapper = MagicMock()
    trap_wrapper.instance = None
    trap_wrapper.health = MagicMock()
    trap_wrapper.health.consecutive_failures = 4
    trap_wrapper.health.last_error = "thermal overload"

    conductor._subsystems = {
        "breeder": breeder_wrapper,
        "mesh": mesh_wrapper,
        "traps": trap_wrapper,
    }

    def _get_status():
        return {
            "node_id": "breeder-node",
            "nodes": ["peer-1"],
            "drift_ms": 12.5,
            "diversity": 47,
            "subsystems": {
                "traps": {
                    "state": "degraded",
                    "consecutive_failures": 4,
                    "last_error": "thermal overload",
                }
            },
        }

    conductor.get_status = _get_status
    return conductor


@pytest.fixture
def clean_history(monkeypatch):
    """Redirect history file to a temp path and clean up after test."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    fake_path = Path(path)
    monkeypatch.setattr(
        "fleet.fleet_weather_report._HISTORY_PATH", fake_path
    )
    yield fake_path
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


# ── unit tests for data classes ─────────────────────────────

class TestBreedingSummary:
    def test_success_rate_zero_attempts(self):
        b = BreedingSummary()
        assert b.success_rate == 0.0

    def test_success_rate_calculation(self):
        b = BreedingSummary(attempted=10, flux_passes=8)
        assert b.success_rate == 80.0

    def test_success_rate_all_pass(self):
        b = BreedingSummary(attempted=5, flux_passes=5)
        assert b.success_rate == 100.0

    def test_success_rate_all_fail(self):
        b = BreedingSummary(attempted=5, flux_passes=0)
        assert b.success_rate == 0.0


# ── FleetWeatherReport core tests ───────────────────────────

class TestWeatherReportFromMockConductor:
    def test_basic_conductor_parsing(self, mock_conductor):
        report = FleetWeatherReport.from_conductor(mock_conductor)
        s = report.stats

        assert s.fleet_name == "test-node"
        assert s.node_count == 3  # self + 2 peers
        assert s.diversity_score == 23
        assert len(s.node_health) == 3  # self + 2 peers

        # Self node
        self_node = s.node_health[0]
        assert self_node.node_id == "test-node"
        assert self_node.drift_ms == 3.5
        assert self_node.beat_sync == "synced"
        assert self_node.thermal_status == "warm"

    def test_breeder_stats_extraction(self, mock_conductor_with_breeder):
        report = FleetWeatherReport.from_conductor(mock_conductor_with_breeder)
        s = report.stats

        assert s.breeding.attempted == 10
        assert s.breeding.flux_passes == 8
        assert s.breeding.flux_fails == 1
        assert s.breeding.thermal_throttled == 1
        assert s.breeding.success_rate == 80.0

    def test_notable_events_from_conductor(self, mock_conductor_with_breeder):
        report = FleetWeatherReport.from_conductor(mock_conductor_with_breeder)
        s = report.stats

        # Mesh error in status log
        assert any("mesh" in ev.lower() for ev in s.errors)
        # Anomaly from trap consecutive failures
        assert any("traps" in ev.lower() for ev in s.anomalies)
        # High diversity discovery
        # Diversity score is 47 (<50) so no high-diversity discovery
        # but errors + anomalies should populate notable_events
        assert len(s.notable_events) > 0

    def test_drift_status_classification(self):
        """Nodes with drift >= 10 ms should be 'drifting'."""
        conductor = MagicMock()
        conductor._node_id = "drifty"
        conductor._status_log = []
        conductor._subsystems = {}

        conductor.get_status.return_value = {
            "nodes": [],
            "drift_ms": 15.0,
            "subsystems": {},
        }

        report = FleetWeatherReport.from_conductor(conductor)
        self_node = report.stats.node_health[0]
        assert self_node.beat_sync == "drifting"

    def test_thermal_status_tiers(self):
        """Thermal status maps correctly to temperature ranges."""
        def make_conductor(temp):
            c = MagicMock()
            c._node_id = "thermal-test"
            c._status_log = []
            c._subsystems = {}
            c.get_status.return_value = {
                "nodes": [],
                "subsystems": {
                    "traps": {
                        "detail": {"thermal": {"temperature": temp}},
                        "consecutive_failures": 0,
                        "last_error": "",
                    }
                },
            }
            return c

        for temp, expected in [
            (40.0, "cool"),
            (65.0, "warm"),
            (80.0, "hot"),
            (95.0, "throttled"),
        ]:
            report = FleetWeatherReport.from_conductor(make_conductor(temp))
            assert report.stats.node_health[0].thermal_status == expected


class TestWeatherReportMarkdownFormat:
    def test_all_sections_present(self, mock_conductor):
        report = FleetWeatherReport.from_conductor(mock_conductor)
        md = report.to_markdown()

        assert "# Fleet Weather Report" in md
        assert "## Breeding Summary" in md
        assert "## Node Health" in md
        assert "## Diversity Trend" in md
        assert "## Notable Events" in md
        assert "## Forecast" in md
        assert "FleetWeatherReport v1.0" in md

    def test_markdown_cached(self, mock_conductor):
        report = FleetWeatherReport.from_conductor(mock_conductor)
        md1 = report.to_markdown()
        md2 = report.to_markdown()
        assert md1 is md2  # same cached string

    def test_node_count_in_header(self, mock_conductor):
        report = FleetWeatherReport.from_conductor(mock_conductor)
        md = report.to_markdown()
        assert "Nodes:** 3" in md

    def test_breeding_values_in_markdown(self, mock_conductor_with_breeder):
        report = FleetWeatherReport.from_conductor(mock_conductor_with_breeder)
        md = report.to_markdown()

        assert "Attempted: **10**" in md
        assert "FLUX passes: **8**" in md
        assert "FLUX fails: **1**" in md
        assert "Thermal throttled: **1**" in md
        assert "Success rate: **80.0%**" in md

    def test_diversity_with_history(self, mock_conductor_with_breeder, clean_history):
        # Seed history with yesterday's entry
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        hist = {
            "entries": [
                {
                    "date": yesterday.strftime("%Y-%m-%d"),
                    "diversity_score": 30.0,
                    "breed_success_rate": 70.0,
                }
            ]
        }
        clean_history.write_text(json.dumps(hist), encoding="utf-8")

        report = FleetWeatherReport.from_conductor(mock_conductor_with_breeder)
        md = report.to_markdown()

        assert "Current diversity score: **47.0**" in md
        assert "vs yesterday" in md
        assert "30.0" in md
        assert "47.0" in md

    def test_empty_notable_events(self, mock_conductor):
        # Clear out any events
        report = FleetWeatherReport.from_conductor(mock_conductor)
        report.stats.notable_events = []
        report.stats.errors = []
        report.stats.anomalies = []
        report.stats.high_diversity_discoveries = []
        report._markdown = None  # clear cache
        md = report.to_markdown()
        assert "No notable events." in md

    def test_no_diversity_data(self):
        stats = FleetStats(diversity_score=None)
        report = FleetWeatherReport(stats)
        md = report.to_markdown()
        assert "Diversity data unavailable." in md


class TestWeatherReportFileWrite:
    def test_write_and_read_back(self, mock_conductor, tmp_path):
        report = FleetWeatherReport.from_conductor(mock_conductor)
        path = tmp_path / "weather.md"
        returned = report.to_file(path)
        assert returned == path
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "Fleet Weather Report" in content

    def test_history_updated_on_write(self, mock_conductor, clean_history):
        report = FleetWeatherReport.from_conductor(mock_conductor)
        report.to_file("/dev/null")  # write path doesn't matter

        data = json.loads(clean_history.read_text(encoding="utf-8"))
        assert len(data["entries"]) == 1
        entry = data["entries"][0]
        assert entry["node_count"] == 3
        assert entry["diversity_score"] == 23

    def test_history_idempotent_same_day(self, mock_conductor, clean_history):
        report1 = FleetWeatherReport.from_conductor(mock_conductor)
        report2 = FleetWeatherReport.from_conductor(mock_conductor)
        report1.to_file("/dev/null")
        report2.to_file("/dev/null")

        data = json.loads(clean_history.read_text(encoding="utf-8"))
        assert len(data["entries"]) == 1  # deduped by date


class TestTrendCalculation:
    def test_trend_with_both_rates(self):
        stats = FleetStats(
            breeding=BreedingSummary(attempted=10, flux_passes=9),
            breed_success_rate_last_week=80.0,
        )
        report = FleetWeatherReport(stats)
        trend = report._calculate_trend()
        assert "up 12.5%" in trend or "up 12.50%" in trend
        assert "80.0%" in trend
        assert "90.0%" in trend

    def test_trend_down(self):
        stats = FleetStats(
            breeding=BreedingSummary(attempted=10, flux_passes=5),
            breed_success_rate_last_week=80.0,
        )
        report = FleetWeatherReport(stats)
        trend = report._calculate_trend()
        assert "down" in trend
        assert "37.5%" in trend or "37.50%" in trend

    def test_trend_no_historical_baseline(self):
        stats = FleetStats(
            breeding=BreedingSummary(attempted=10, flux_passes=7),
        )
        report = FleetWeatherReport(stats)
        trend = report._calculate_trend()
        assert "no historical baseline" in trend
        assert "70.0%" in trend

    def test_trend_no_current_data(self):
        stats = FleetStats(breeding=BreedingSummary())
        report = FleetWeatherReport(stats)
        trend = report._calculate_trend()
        assert "insufficient data" in trend

    def test_trend_zero_last_week(self):
        """Division by zero protection when last_week rate is 0."""
        stats = FleetStats(
            breeding=BreedingSummary(attempted=10, flux_passes=5),
            breed_success_rate_last_week=0.0,
        )
        report = FleetWeatherReport(stats)
        # Should not crash
        trend = report._calculate_trend()
        assert "insufficient data" in trend or "%" in trend


class TestMatrixPosting:
    def test_post_no_hook_url(self, mock_conductor):
        report = FleetWeatherReport.from_conductor(mock_conductor)
        result = report.post_to_matrix()
        assert result["posted"] is False
        assert result["reason"] == "no_hook_url_configured"
        assert "markdown" in result

    def test_post_with_url(self, mock_conductor):
        report = FleetWeatherReport.from_conductor(mock_conductor)
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_urlopen.return_value.__enter__.return_value = mock_resp
            result = report.post_to_matrix(
                hook_url="https://matrix.example.com/webhook",
                channel="#fleet-weather",
            )
            assert result["posted"] is True
            assert result["status"] == 200

    def test_post_http_failure(self, mock_conductor):
        report = FleetWeatherReport.from_conductor(mock_conductor)
        with patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
            result = report.post_to_matrix(
                hook_url="https://bad.example.com/hook",
            )
            assert result["posted"] is False
            assert "connection refused" in result["reason"]


class TestHistoryHelpers:
    def test_load_history_yesterday_diversity(self, clean_history):
        today = datetime.now(timezone.utc)
        yesterday = today - timedelta(days=1)
        two_weeks_ago = today - timedelta(days=14)

        hist = {
            "entries": [
                {
                    "date": two_weeks_ago.strftime("%Y-%m-%d"),
                    "diversity_score": 10.0,
                    "breed_success_rate": 60.0,
                },
                {
                    "date": yesterday.strftime("%Y-%m-%d"),
                    "diversity_score": 25.0,
                    "breed_success_rate": 75.0,
                },
            ]
        }
        clean_history.write_text(json.dumps(hist), encoding="utf-8")

        stats = FleetStats(timestamp=today, diversity_score=30.0)
        _maybe_load_history(stats)
        assert stats.diversity_score_yesterday == 25.0
        # Should pick the entry closest to one week ago (the two_weeks_ago one)
        assert stats.breed_success_rate_last_week == 60.0

    def test_load_history_no_file(self, monkeypatch):
        # Point to a non-existent path so we don't pick up a real history file
        fake_path = Path("/tmp/nonexistent_fleet_weather_history_12345.json")
        monkeypatch.setattr(
            "fleet.fleet_weather_report._HISTORY_PATH", fake_path
        )
        stats = FleetStats()
        # No history file should not crash
        _maybe_load_history(stats)
        assert stats.diversity_score_yesterday is None
        assert stats.breed_success_rate_last_week is None

    def test_append_history_creates_file(self, clean_history):
        stats = FleetStats(
            timestamp=datetime(2026, 5, 29, 12, 0, tzinfo=timezone.utc),
            diversity_score=42.0,
            breeding=BreedingSummary(attempted=5, flux_passes=4),
        )
        _append_history(stats)
        assert clean_history.exists()
        data = json.loads(clean_history.read_text(encoding="utf-8"))
        assert len(data["entries"]) == 1
        assert data["entries"][0]["diversity_score"] == 42.0
        assert data["entries"][0]["breed_success_rate"] == 80.0

    def test_append_history_rotates(self, clean_history):
        # Seed with max+1 entries
        entries = [
            {
                "date": f"2026-01-{i:02d}",
                "diversity_score": float(i),
                "breed_success_rate": 50.0,
            }
            for i in range(1, 92)
        ]
        clean_history.write_text(
            json.dumps({"entries": entries}), encoding="utf-8"
        )

        stats = FleetStats(
            timestamp=datetime(2026, 5, 29, 12, 0, tzinfo=timezone.utc),
            diversity_score=99.0,
            breeding=BreedingSummary(),
        )
        _append_history(stats)
        data = json.loads(clean_history.read_text(encoding="utf-8"))
        assert len(data["entries"]) == 90  # capped
        assert data["entries"][-1]["date"] == "2026-05-29"


# ── edge cases ──────────────────────────────────────────────

class TestEdgeCases:
    def test_conductor_get_status_raises(self):
        """from_conductor should handle a broken get_status gracefully."""
        conductor = MagicMock()
        conductor._node_id = "broken"
        conductor._status_log = []
        conductor._subsystems = {}
        conductor.get_status.side_effect = RuntimeError("kaboom")

        report = FleetWeatherReport.from_conductor(conductor)
        assert report.stats.fleet_name == "broken"
        assert report.stats.diversity_score is None
        # Should still produce markdown
        md = report.to_markdown()
        assert "Fleet Weather Report" in md

    def test_no_subsystems(self):
        stats = FleetStats()
        report = FleetWeatherReport(stats)
        md = report.to_markdown()
        assert "## Breeding Summary" in md
        assert "Attempted: **0**" in md
        assert "## Node Health" in md
        assert "No node health data available." in md

    def test_negative_diversity_delta(self, clean_history):
        today = datetime.now(timezone.utc)
        yesterday = today - timedelta(days=1)
        hist = {
            "entries": [
                {
                    "date": yesterday.strftime("%Y-%m-%d"),
                    "diversity_score": 50.0,
                }
            ]
        }
        clean_history.write_text(json.dumps(hist), encoding="utf-8")

        stats = FleetStats(timestamp=today, diversity_score=30.0)
        _maybe_load_history(stats)
        report = FleetWeatherReport(stats)
        md = report.to_markdown()
        assert "↓ 20.0" in md  # down arrow
