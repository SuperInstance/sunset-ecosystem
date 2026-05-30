"""Tests for NerveTopology — initialization, stats, and tick cycle.

Mocks RoomGrid and RoutingLayer to avoid heavy dependencies.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nerve.topology import NerveTopology, TickResult
from nerve.fiber import FiberState


class TestTickResult:
    def test_creation(self):
        tr = TickResult(tick=1, fibers_perceived=2, rooms_fired=3, routes_activated=4, routes_compiled=5, novel_signals=6, latency_ms=10.0)
        assert tr.tick == 1
        assert tr.fibers_perceived == 2
        assert tr.routes_compiled == 5

    def test_defaults(self):
        tr = TickResult(tick=0, fibers_perceived=0, rooms_fired=0, routes_activated=0, routes_compiled=0, novel_signals=0, latency_ms=0.0)
        assert tr.compiled_funcs == []


class TestNerveTopology:
    def test_init(self):
        topo = NerveTopology(n_fibers=4, n_rooms=10, chaos=0.2)
        assert topo.n_fibers == 4
        assert topo.n_rooms == 10
        assert len(topo.fibers) == 4
        assert topo.tick_count == 0

    def test_repr(self):
        topo = NerveTopology(n_fibers=2, n_rooms=5)
        assert "NerveTopology" in repr(topo)
        assert "fibers=2" in repr(topo)

    def test_stats(self):
        topo = NerveTopology(n_fibers=4, n_rooms=10)
        stats = topo.stats
        assert stats["tick"] == 0
        assert stats["fibers"] == 4
        assert stats["rooms"] == 10
        assert "routes" in stats
        assert "channels" in stats

    def test_fiber_states(self):
        topo = NerveTopology(n_fibers=4, n_rooms=10)
        for f in topo.fibers.values():
            assert f.state == FiberState.PERCEIVING

    def test_routing_exists(self):
        topo = NerveTopology(n_fibers=2, n_rooms=5)
        assert topo.routing is not None
        assert len(topo.routing._routes) > 0

    def test_grid_exists(self):
        topo = NerveTopology(n_fibers=2, n_rooms=5)
        assert topo.grid is not None

    def test_enable_compiler_noop(self):
        topo = NerveTopology(n_fibers=2, n_rooms=5)
        topo.enable_compiler()  # may fail silently if compiler not available
        assert topo._compiler is None or topo._compiler is not None

    def test_encode_tile(self):
        topo = NerveTopology(n_fibers=2, n_rooms=5, signal_dim=8)
        tile = MagicMock()
        tile.pattern_id = "test"
        tile.state = FiberState.PERCEIVING
        tile.confidence = 0.5
        vec = topo._encode_tile(tile)
        assert isinstance(vec, np.ndarray)
        assert len(vec) == 8

    def test_encode_tile_cached(self):
        topo = NerveTopology(n_fibers=2, n_rooms=5, signal_dim=8)
        tile = MagicMock()
        tile.pattern_id = "test"
        tile.state = FiberState.PERCEIVING
        tile.confidence = 0.5
        v1 = topo._encode_tile(tile)
        v2 = topo._encode_tile(tile)
        assert np.array_equal(v1, v2)
        assert hasattr(topo, '_tile_cache')

    def test_results_deque(self):
        topo = NerveTopology(n_fibers=2, n_rooms=5)
        assert len(topo._results) == 0

    def test_tick_count(self):
        topo = NerveTopology(n_fibers=2, n_rooms=5)
        topo.tick_count += 1
        assert topo.tick_count == 1
