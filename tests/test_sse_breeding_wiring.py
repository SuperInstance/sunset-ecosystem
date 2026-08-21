"""Tests for SSE breeding event wiring."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from fleet.sse_stream_dashboard import (
    SSEStreamDashboard,
    DashboardConfig,
    StreamEvent,
    EventType,
)
from fleet.sse_breeding_wiring import SSEBreedingWiring, wire_breeder_to_sse


class TestSSEWiring:
    def test_attach_publishes_parent_select(self):
        dash = SSEStreamDashboard(DashboardConfig(history_buffer_size=10))
        breeder = MagicMock()
        breeder.select_parents = MagicMock(return_value=[(1, 2), (3, 4)])

        wiring = SSEBreedingWiring(dash)
        wiring.attach_to_breeder(breeder)

        result = breeder.select_parents(4)
        assert result == [(1, 2), (3, 4)]

        events = dash.recent_events(100)
        parent_events = [e for e in events if e.event_type == EventType.PARENT_SELECT]
        assert len(parent_events) == 2  # start + complete

    def test_attach_publishes_cycle(self):
        dash = SSEStreamDashboard(DashboardConfig(history_buffer_size=10))
        breeder = MagicMock()
        breeder.cycle = MagicMock(return_value={"new_agents": 2})

        wiring = wire_breeder_to_sse(breeder, dash)

        breeder.cycle()
        events = dash.recent_events(100)
        beat_events = [e for e in events if e.event_type == EventType.BEAT]
        assert len(beat_events) == 2

    def test_wire_breeder_to_sse_returns_wiring(self):
        dash = SSEStreamDashboard()
        breeder = MagicMock()
        wiring = wire_breeder_to_sse(breeder, dash)
        assert isinstance(wiring, SSEBreedingWiring)

    def test_thermal_event_published(self):
        dash = SSEStreamDashboard(DashboardConfig(history_buffer_size=10))
        breeder = MagicMock()
        breeder._check_thermal = MagicMock(return_value=0.75)

        wiring = SSEBreedingWiring(dash)
        wiring.attach_to_breeder(breeder)

        breeder._check_thermal()
        events = dash.recent_events(100)
        thermal = [e for e in events if e.event_type == EventType.THERMAL]
        assert len(thermal) == 1
        assert thermal[0].payload["pressure"] == 0.75
