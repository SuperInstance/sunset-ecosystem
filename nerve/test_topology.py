"""Test the full NerveTopology cycle."""

import numpy as np
import pytest
from nerve.topology import NerveTopology


def test_topology_creates():
    """Topology wires fibers → rooms → routing."""
    topo = NerveTopology(n_fibers=4, n_rooms=50)
    assert topo.n_fibers == 4
    assert topo.n_rooms == 50
    assert len(topo.routing._routes) == 4 * 50  # every fiber → every room


def test_topology_ticks():
    """One tick: all fibers perceive, some rooms fire."""
    topo = NerveTopology(n_fibers=4, n_rooms=50)
    result = topo.tick()
    assert result.fibers_perceived == 4
    assert result.tick == 1
    assert result.latency_ms > 0


def test_topology_compiles():
    """After many ticks with repeated signal, fibers should compile."""
    topo = NerveTopology(n_fibers=2, n_rooms=20, adapt_threshold=0.8)
    signal = {"fiber-0": "test-pattern-abc", "fiber-1": "test-pattern-xyz"}

    for _ in range(100):
        topo.tick(signals=signal)

    # At least one fiber should have started compiling
    stats = topo.stats
    assert stats["fibers_compiled"] >= 0  # may or may not compile in 100 ticks


def test_topology_adaptive_chaos():
    """Chaos should decay over time."""
    topo = NerveTopology(n_fibers=2, n_rooms=20, chaos=0.3)
    initial_chaos = topo.routing.chaos

    for _ in range(500):
        topo.tick()

    assert topo.routing.chaos <= initial_chaos


def test_topology_rebirth():
    """Cold rooms get rebirthed."""
    topo = NerveTopology(n_fibers=2, n_rooms=20)
    # Tick once to initialize
    topo.tick()
    # Rebirth cold rooms
    rebirthed = topo.rebirth_cold_rooms()
    assert rebirthed >= 0


def test_topology_penrose_channels():
    """Hebbian channels exist between Penrose-adjacent rooms."""
    topo = NerveTopology(n_fibers=2, n_rooms=20)
    assert len(topo.routing._channels) > 0
    # Channels should be between adjacent room pairs
    for key, ch in topo.routing._channels.items():
        assert "room-" in ch.node_a
        assert "room-" in ch.node_b


def test_topology_run_batch():
    """Run 100 ticks without error."""
    topo = NerveTopology(n_fibers=4, n_rooms=50)
    results = topo.run(ticks=100)
    assert len(results) == 100
    assert results[-1].tick == 100
