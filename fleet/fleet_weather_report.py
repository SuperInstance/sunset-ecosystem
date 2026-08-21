"""Fleet Weather Report — automated daily fleet health summaries.

Phase 5.4 implementation. Generates concise, data-dense markdown reports
from FleetConductorV2 statistics. Institutional-research style — no fluff.

Usage
-----
    conductor = FleetConductorV2(config)
    report = FleetWeatherReport.from_conductor(conductor)
    print(report.to_markdown())
    report.to_file("/tmp/fleet_weather_2026-05-29.md")
"""

from __future__ import annotations

__all__ = [
    "FleetWeatherReport",
    "FleetStats",
    "NodeHealth",
    "BreedingSummary",
]

import json
import logging
import os
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── data structures ───────────────────────────────────────────


@dataclass
class BreedingSummary:
    """Breeding activity snapshot."""

    attempted: int = 0
    flux_passes: int = 0
    flux_fails: int = 0
    thermal_throttled: int = 0

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if self.attempted == 0:
            return 0.0
        return (self.flux_passes / self.attempted) * 100.0


@dataclass
class NodeHealth:
    """Per-node health snapshot."""

    node_id: str = ""
    drift_ms: float | None = None
    beat_sync: str = "unknown"  # "synced", "drifting", "lost"
    thermal_status: str = "unknown"  # "cool", "warm", "hot", "throttled"
    drift_corrections: int = 0
    consecutive_failures: int = 0
    last_error: str = ""


@dataclass
class FleetStats:
    """Aggregated fleet statistics ingested by the report."""

    fleet_name: str = "Cocapn Fleet"
    node_count: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    breeding: BreedingSummary = field(default_factory=BreedingSummary)
    node_health: List[NodeHealth] = field(default_factory=list)
    diversity_score: float | None = None
    diversity_score_yesterday: float | None = None
    notable_events: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    anomalies: List[str] = field(default_factory=list)
    high_diversity_discoveries: List[str] = field(default_factory=list)
    # For trend calculation
    breed_success_rate_last_week: float | None = None


# ── FleetWeatherReport ──────────────────────────────────────


class FleetWeatherReport:
    """Generate a daily fleet health summary from fleet statistics.

    Design notes
    ------------
    * Information-first, institutional-research style.
    * All sections are optional — missing data renders as "N/A".
    * Historical data for trends is pulled from conductor status logs
      or a local JSON history file (~/.fleet_weather_history.json).
    """

    def __init__(self, stats: FleetStats) -> None:
        self.stats = stats
        self._markdown: str | None = None

    # ── factory methods ────────────────────────────────────

    @classmethod
    def from_conductor(cls, conductor: Any) -> "FleetWeatherReport":
        """Build a report by inspecting a FleetConductorV2 instance.

        Pulls stats via ``conductor.get_status()`` and by inspecting
        subsystems (mesh, breeder, metronome, traps, etc.).
        """
        stats = FleetStats()
        stats.fleet_name = getattr(conductor, "_node_id", "Cocapn Fleet")
        stats.timestamp = datetime.now(timezone.utc)

        # ── base status ──
        status: dict[str, Any] = {}
        try:
            status = conductor.get_status()
        except Exception as exc:
            logger.warning("conductor.get_status() failed: %s", exc)
            status = {"error": str(exc)}

        stats.node_count = len(status.get("nodes", [])) + 1  # + self

        # ── node health ──
        nodes: list[str] = status.get("nodes", [])
        drift_ms = status.get("drift_ms")
        subsystems = status.get("subsystems", {})

        # Self node
        self_health = NodeHealth(
            node_id=stats.fleet_name,
            drift_ms=drift_ms,
            beat_sync="synced"
            if drift_ms is not None and drift_ms < 10.0
            else "drifting",
            thermal_status="unknown",
            consecutive_failures=0,
        )
        if "traps" in subsystems:
            trap_detail = subsystems["traps"].get("detail", {})
            thermal = trap_detail.get("thermal", {})
            if thermal:
                temp = thermal.get("temperature", 0.0)
                if temp > 90.0:
                    self_health.thermal_status = "throttled"
                elif temp > 75.0:
                    self_health.thermal_status = "hot"
                elif temp > 60.0:
                    self_health.thermal_status = "warm"
                else:
                    self_health.thermal_status = "cool"
            self_health.consecutive_failures = subsystems["traps"].get(
                "consecutive_failures", 0
            )
            self_health.last_error = subsystems["traps"].get("last_error", "")
        stats.node_health.append(self_health)

        # Peer nodes (placeholder — real mesh would have per-node data)
        for node_id in nodes:
            stats.node_health.append(
                NodeHealth(
                    node_id=node_id,
                    drift_ms=None,
                    beat_sync="unknown",
                    thermal_status="unknown",
                )
            )

        # ── breeding summary ──
        breeder_wrapper = getattr(conductor, "_subsystems", {}).get("breeder")
        if breeder_wrapper is not None and breeder_wrapper.instance is not None:
            breeder = breeder_wrapper.instance
            try:
                bstatus = breeder.get_status() if hasattr(breeder, "get_status") else {}
                stats.breeding.attempted = bstatus.get("breeds_attempted", 0)
                stats.breeding.flux_passes = bstatus.get("flux_passes", 0)
                stats.breeding.flux_fails = bstatus.get("flux_fails", 0)
                stats.breeding.thermal_throttled = bstatus.get("thermal_throttled", 0)
            except Exception as exc:
                logger.warning("Breeder status failed: %s", exc)

        # Also infer from conductor status log (beat history)
        status_log = getattr(conductor, "_status_log", [])
        if not stats.breeding.attempted and status_log:
            # Count breeder tick markers in recent logs
            recent = status_log[-100:] if len(status_log) > 100 else status_log
            breed_ticks = sum(
                1
                for entry in recent
                if entry.get("subsystem_ticks", {}).get("breeder") == "ok"
            )
            if breed_ticks:
                stats.breeding.attempted = breed_ticks

        # ── diversity ──
        stats.diversity_score = status.get("diversity")
        if stats.diversity_score is None:
            mesh_wrapper = getattr(conductor, "_subsystems", {}).get("mesh")
            if mesh_wrapper is not None and mesh_wrapper.instance is not None:
                try:
                    mesh_stats = mesh_wrapper.instance.stats
                    stats.diversity_score = mesh_stats.get("total_entries", 0)
                except Exception:
                    pass

        # ── notable events ──
        # Errors from status log
        for entry in status_log[-50:]:
            ticks = entry.get("subsystem_ticks", {})
            for subsys, tick_status in ticks.items():
                if tick_status == "err":
                    stats.errors.append(
                        f"Beat {entry.get('beat_number')}: {subsys} subsystem error"
                    )
        # Anomalies: high consecutive failures
        for name, wrapper in getattr(conductor, "_subsystems", {}).items():
            health = wrapper.health
            if health.consecutive_failures >= 3:
                stats.anomalies.append(
                    f"{name}: {health.consecutive_failures} consecutive failures"
                )
            if health.last_error:
                stats.errors.append(f"{name}: {health.last_error}")

        # High diversity discoveries
        if stats.diversity_score is not None and stats.diversity_score > 50:
            stats.high_diversity_discoveries.append(
                f"Diversity score {stats.diversity_score:.1f} — high population diversity"
            )

        stats.notable_events = (
            stats.errors[:5]
            + stats.anomalies[:5]
            + stats.high_diversity_discoveries[:5]
        )

        # ── historical trend data ──
        _maybe_load_history(stats)

        return cls(stats)

    # ── report generation ──────────────────────────────────

    def to_markdown(self) -> str:
        """Render the report as a markdown string."""
        if self._markdown is not None:
            return self._markdown

        s = self.stats
        ts = s.timestamp.strftime("%Y-%m-%d %H:%M UTC")

        lines: List[str] = []

        # Header
        lines.append(f"# Fleet Weather Report — {s.fleet_name}")
        lines.append("")
        lines.append(f"**Date:** {ts}  |  **Nodes:** {s.node_count}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Breeding Summary
        lines.append("## Breeding Summary")
        lines.append("")
        b = s.breeding
        rate = f"{b.success_rate:.1f}%" if b.attempted > 0 else "N/A"
        lines.append(f"- Attempted: **{b.attempted}**")
        lines.append(f"- FLUX passes: **{b.flux_passes}**")
        lines.append(f"- FLUX fails: **{b.flux_fails}**")
        lines.append(f"- Thermal throttled: **{b.thermal_throttled}**")
        lines.append(f"- Success rate: **{rate}**")
        lines.append("")

        # Node Health
        lines.append("## Node Health")
        lines.append("")
        for nh in s.node_health:
            drift = f"{nh.drift_ms:.2f} ms" if nh.drift_ms is not None else "N/A"
            lines.append(
                f"- **{nh.node_id}** — drift: {drift}, "
                f"sync: {nh.beat_sync}, thermal: {nh.thermal_status}"
            )
            if nh.drift_corrections:
                lines.append(f"  - drift corrections: {nh.drift_corrections}")
            if nh.consecutive_failures:
                lines.append(f"  - ⚠️ {nh.consecutive_failures} consecutive failures")
            if nh.last_error:
                lines.append(f"  - last error: `{nh.last_error}`")
        if not s.node_health:
            lines.append("- No node health data available.")
        lines.append("")

        # Diversity Trend
        lines.append("## Diversity Trend")
        lines.append("")
        if s.diversity_score is not None:
            lines.append(f"- Current diversity score: **{s.diversity_score:.1f}**")
            if s.diversity_score_yesterday is not None:
                delta = s.diversity_score - s.diversity_score_yesterday
                arrow = "↑" if delta >= 0 else "↓"
                lines.append(
                    f"- vs yesterday: **{arrow} {abs(delta):.1f}** "
                    f"({s.diversity_score_yesterday:.1f} → {s.diversity_score:.1f})"
                )
            else:
                lines.append("- No historical comparison available.")
        else:
            lines.append("- Diversity data unavailable.")
        lines.append("")

        # Notable Events
        lines.append("## Notable Events")
        lines.append("")
        if s.notable_events:
            for ev in s.notable_events:
                prefix = "🔥" if "error" in ev.lower() or "fail" in ev.lower() else "📌"
                lines.append(f"- {prefix} {ev}")
        else:
            lines.append("- No notable events.")
        lines.append("")

        # Forecast
        lines.append("## Forecast")
        lines.append("")
        trend = self._calculate_trend()
        lines.append(f"- **Trend:** {trend}")
        lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append(f"*Report generated by FleetWeatherReport v1.0*")

        self._markdown = "\n".join(lines)
        return self._markdown

    def _calculate_trend(self) -> str:
        """Generate a simple trend sentence."""
        s = self.stats
        current_rate = s.breeding.success_rate if s.breeding.attempted > 0 else None
        last_week_rate = s.breed_success_rate_last_week

        if (
            current_rate is not None
            and last_week_rate is not None
            and last_week_rate > 0
        ):
            pct_change = ((current_rate - last_week_rate) / last_week_rate) * 100.0
            arrow = "up" if pct_change >= 0 else "down"
            return (
                f"breed success rate {arrow} {abs(pct_change):.1f}% "
                f"vs last week ({last_week_rate:.1f}% → {current_rate:.1f}%)"
            )
        elif current_rate is not None:
            return f"breed success rate at {current_rate:.1f}% (no historical baseline)"
        else:
            return "insufficient data for trend calculation"

    # ── persistence ──────────────────────────────────────────

    def to_file(self, path: str | Path) -> Path:
        """Write the markdown report to *path*.

        Also appends today's snapshot to the local history file
        (~/.fleet_weather_history.json) for trend calculations.
        """
        md = self.to_markdown()
        p = Path(path)
        p.write_text(md, encoding="utf-8")
        _append_history(self.stats)
        logger.info("Fleet weather report written to %s", p)
        return p

    # ── external posting ─────────────────────────────────────

    def post_to_matrix(
        self,
        hook_url: str | None = None,
        channel: str | None = None,
    ) -> dict[str, Any]:
        """Post the report to a Matrix room via webhook.

        If *hook_url* is None or empty, returns a stub dict with
        ``posted: False`` and the markdown payload so callers can
        retry later.
        """
        if not hook_url:
            return {
                "posted": False,
                "reason": "no_hook_url_configured",
                "markdown": self.to_markdown(),
                "channel": channel,
            }

        payload = {
            "body": self.to_markdown(),
            "formatted_body": self.to_markdown(),
            "msgtype": "m.text",
        }
        if channel:
            payload["channel"] = channel

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                hook_url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "fleet-weather-report/1.0",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                return {
                    "posted": resp.status == 200,
                    "status": resp.status,
                    "channel": channel,
                }
        except Exception as exc:
            logger.warning("Matrix post failed: %s", exc)
            return {
                "posted": False,
                "reason": str(exc),
                "markdown": self.to_markdown(),
                "channel": channel,
            }


# ── history helpers ───────────────────────────────────────────

_HISTORY_PATH = Path.home() / ".fleet_weather_history.json"
_HISTORY_MAX_ENTRIES = 90  # ~3 months of daily reports


def _maybe_load_history(stats: FleetStats) -> None:
    """Load yesterday's diversity and last week's breed rate from history."""
    if not _HISTORY_PATH.exists():
        return
    try:
        data = json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))
        entries: list[dict] = data.get("entries", [])
        if not entries:
            return

        today = stats.timestamp.strftime("%Y-%m-%d")

        # Yesterday's diversity
        for entry in reversed(entries):
            if entry.get("date") != today:
                stats.diversity_score_yesterday = entry.get("diversity_score")
                break

        # Last week's breed success rate
        one_week_ago = (stats.timestamp - timedelta(days=7)).strftime("%Y-%m-%d")
        for entry in reversed(entries):
            if entry.get("date") <= one_week_ago:
                stats.breed_success_rate_last_week = entry.get("breed_success_rate")
                break

        # Fallback: use the oldest entry if nothing older than a week
        if stats.breed_success_rate_last_week is None and entries:
            stats.breed_success_rate_last_week = entries[0].get("breed_success_rate")

    except Exception as exc:
        logger.debug("History load failed: %s", exc)


def _append_history(stats: FleetStats) -> None:
    """Append today's snapshot to the local history file."""
    entries: list[dict] = []
    if _HISTORY_PATH.exists():
        try:
            data = json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))
            entries = data.get("entries", [])
        except Exception:
            pass

    today = stats.timestamp.strftime("%Y-%m-%d")
    # Remove any existing entry for today (idempotent)
    entries = [e for e in entries if e.get("date") != today]

    entries.append(
        {
            "date": today,
            "timestamp": stats.timestamp.isoformat(),
            "node_count": stats.node_count,
            "diversity_score": stats.diversity_score,
            "breeds_attempted": stats.breeding.attempted,
            "flux_passes": stats.breeding.flux_passes,
            "flux_fails": stats.breeding.flux_fails,
            "thermal_throttled": stats.breeding.thermal_throttled,
            "breed_success_rate": stats.breeding.success_rate,
        }
    )

    if len(entries) > _HISTORY_MAX_ENTRIES:
        entries = entries[-_HISTORY_MAX_ENTRIES:]

    try:
        _HISTORY_PATH.write_text(
            json.dumps({"entries": entries}, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("History write failed: %s", exc)
