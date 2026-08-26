import numpy as np
import pytest


@pytest.fixture
def room_grid_100():
    """Return a RoomGrid with 100 rooms (seeded for determinism)."""
    from nerve.room_grid import RoomGrid

    np.random.seed(42)
    return RoomGrid(100)


@pytest.fixture
def room_grid_1000():
    """Return a RoomGrid with 1000 rooms (seeded for determinism)."""
    from nerve.room_grid import RoomGrid

    np.random.seed(42)
    return RoomGrid(1000)


@pytest.fixture
def routing_layer():
    """Return a RoutingLayer with 10 routes (seeded for determinism)."""
    from nerve.routing import RoutingLayer

    np.random.seed(42)
    rl = RoutingLayer()
    for i in range(10):
        rl.add_route(f"src_{i}", f"dst_{i}")
    return rl


@pytest.fixture
def compiler():
    """Return a Compiler instance with profiler."""
    from sunset.compiler import Compiler

    return Compiler()


@pytest.fixture
def signal_64():
    """Return a 64-dim float32 signal (seeded for determinism)."""
    np.random.seed(42)
    return np.random.randn(64).astype(np.float32)


@pytest.fixture
def flux_checker():
    """Return a FLUX constraint checker with neural_bounds preset."""
    from sunset.flux_integration import FluxConstraintChecker

    return FluxConstraintChecker(preset="neural_bounds")
