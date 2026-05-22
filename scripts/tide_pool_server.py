#!/usr/bin/env python3
"""Tide Pool Server — Flask/FastAPI endpoint for ambient fleet visualization.

Serves the Tide Pool HTML at ``localhost:8080/tide-pool`` with auto-refresh
every 5 seconds via meta refresh (or JS polling if extended).

Usage::

    python scripts/tide_pool_server.py

Or with a custom data module::

    python scripts/tide_pool_server.py --data-module sunset.swarm.live_state
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

from logos.tide_pool_viz import AgentSnapshot, FleetSnapshot, TidePoolVisualizer

# ---------------------------------------------------------------------------
# Try FastAPI first, fall back to Flask
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from uvicorn import run as uvicorn_run

    FASTAPI_AVAILABLE = True
    FLASK_AVAILABLE = False
except Exception:
    FASTAPI_AVAILABLE = False
    try:
        from flask import Flask

        FLASK_AVAILABLE = True
    except Exception:
        FLASK_AVAILABLE = False

if not FASTAPI_AVAILABLE and not FLASK_AVAILABLE:
    print(
        "Neither FastAPI nor Flask is installed.\n"
        "Install one of them:\n"
        "  pip install 'fastapi[standard]'   # preferred\n"
        "  pip install flask                  # fallback"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Mock data source (used when no real fleet state is wired in)
# ---------------------------------------------------------------------------


class MockFleetSource:
    """Generate synthetic fleet snapshots for demo / testing."""

    _domains = ["compiler", "research", "infra", "pathos", "logos", "ethos", "swarm"]
    _statuses = ["active", "active", "active", "breeding", "idle", "sunset"]

    def __init__(self, n_agents: int = 120, n_rooms: int = 48):
        self.n_agents = n_agents
        self.n_rooms = n_rooms
        self._tick = 0

    def __call__(self) -> dict:
        self._tick += 1
        agents = [
            AgentSnapshot(
                id=f"agent-{i:04d}",
                domain=self._domains[i % len(self._domains)],
                fitness=0.5 + 0.45 * (0.5 + 0.5 * (i % 7) / 7),
                age_ticks=self._tick + i,
                thermal_load=0.3 + 0.5 * ((i * 13) % 100) / 100,
                status=self._statuses[i % len(self._statuses)],
            )
            for i in range(self.n_agents)
        ]
        # Slightly vary counts each tick for realism
        if self._tick % 3 == 0:
            self.n_agents += 1
        if self._tick % 7 == 0 and self.n_agents > 20:
            self.n_agents -= 1

        events = [
            {"type": "breed", "message": f"agent-{self._tick % 100:04d} bred into research", "time": time.strftime("%H:%M:%S")},
            {"type": "info", "message": f"Metronome tick {self._tick} complete", "time": time.strftime("%H:%M:%S")},
        ]
        if self._tick % 5 == 0:
            events.append({"type": "sunset", "message": f"agent-{(self._tick * 3) % 100:04d} archived epilogue", "time": time.strftime("%H:%M:%S")})
        if self._tick % 11 == 0:
            events.append({"type": "error", "message": "Thermal threshold breached on cuda:1", "time": time.strftime("%H:%M:%S")})

        thermal = {
            "cuda:0": 0.45 + 0.1 * ((self._tick * 3) % 10) / 10,
            "cuda:1": 0.55 + 0.15 * ((self._tick * 7) % 10) / 10,
            "cpu": 0.25 + 0.05 * ((self._tick * 2) % 10) / 10,
        }

        return {
            "agents": agents,
            "n_rooms": self.n_rooms,
            "recent_events": events,
            "thermal_state": thermal,
        }


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def make_app(data_source=None, template_path: str | None = None) -> "FastAPI | Flask":
    """Create and configure the web app."""
    viz = TidePoolVisualizer()
    source = data_source if data_source is not None else MockFleetSource()
    tpl = template_path or str(Path(__file__).parent.parent / "logos" / "templates" / "tide_pool.html")

    if FASTAPI_AVAILABLE:
        app = FastAPI(title="Tide Pool")

        @app.get("/tide-pool", response_class=HTMLResponse)
        async def tide_pool():
            raw = source()
            snap = viz.generate_snapshot(
                agents=raw.get("agents", []),
                n_rooms=raw.get("n_rooms", 0),
                recent_events=raw.get("recent_events"),
                thermal_state=raw.get("thermal_state"),
            )
            # Inject a meta refresh for 5-second auto-refresh
            html = viz.render_html(snap, template_path=tpl)
            # Insert meta refresh after <head>
            meta = '<meta http-equiv="refresh" content="5">'
            html = html.replace("<head>", f"<head>\n{meta}")
            return html

        @app.get("/tide-pool/api/snapshot")
        async def api_snapshot():
            raw = source()
            snap = viz.generate_snapshot(
                agents=raw.get("agents", []),
                n_rooms=raw.get("n_rooms", 0),
                recent_events=raw.get("recent_events"),
                thermal_state=raw.get("thermal_state"),
            )
            return {
                "n_agents": snap.n_agents,
                "n_rooms": snap.n_rooms,
                "mean_fitness": snap.mean_fitness,
                "diversity": snap.diversity,
                "chaos_level": snap.chaos_level,
                "thermal_state": snap.thermal_state,
                "recent_events": snap.recent_events,
                "top_agents": snap.top_agents,
                "domains": snap.domains,
                "timestamp": snap.timestamp,
            }

        return app

    # Flask fallback
    app = Flask(__name__)

    @app.route("/tide-pool")
    def tide_pool():
        raw = source()
        snap = viz.generate_snapshot(
            agents=raw.get("agents", []),
            n_rooms=raw.get("n_rooms", 0),
            recent_events=raw.get("recent_events"),
            thermal_state=raw.get("thermal_state"),
        )
        html = viz.render_html(snap, template_path=tpl)
        meta = '<meta http-equiv="refresh" content="5">'
        html = html.replace("<head>", f"<head>\n{meta}")
        return html

    @app.route("/tide-pool/api/snapshot")
    def api_snapshot():
        raw = source()
        snap = viz.generate_snapshot(
            agents=raw.get("agents", []),
            n_rooms=raw.get("n_rooms", 0),
            recent_events=raw.get("recent_events"),
            thermal_state=raw.get("thermal_state"),
        )
        return {
            "n_agents": snap.n_agents,
            "n_rooms": snap.n_rooms,
            "mean_fitness": snap.mean_fitness,
            "diversity": snap.diversity,
            "chaos_level": snap.chaos_level,
            "thermal_state": snap.thermal_state,
            "recent_events": snap.recent_events,
            "top_agents": snap.top_agents,
            "domains": snap.domains,
            "timestamp": snap.timestamp,
        }

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Tide Pool ambient visualization server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--data-module", default=None, help="Dotted path to a callable that returns fleet raw data")
    parser.add_argument("--template", default=None, help="Path to custom HTML template")
    args = parser.parse_args()

    data_source = None
    if args.data_module:
        mod_path, attr_name = args.data_module.rsplit(".", 1)
        mod = importlib.import_module(mod_path)
        data_source = getattr(mod, attr_name)
        if callable(data_source):
            data_source = data_source()

    app = make_app(data_source=data_source, template_path=args.template)

    if FASTAPI_AVAILABLE:
        uvicorn_run(app, host=args.host, port=args.port, log_level="info")
    else:
        app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
