"""SwarmRunner — Orchestrates the full swarm."""

from __future__ import annotations

__all__ = ["SwarmRunner", "SwarmStatus"]

import time
from dataclasses import dataclass, field
from typing import Any

from nerve.fiber import NerveFiber
from nerve.routing import RoutingLayer
from nerve.adaptation import AdaptationEngine
from .penrose import PenrosePosition, assign_positions
from .broadcast import BroadcastingChannel, BroadcastMessage


@dataclass
class SwarmStatus:
    """Current status of the running swarm.

    Attributes:
        total_agents: Number of agents in the swarm.
        active_agents: Agents currently processing.
        adaptation_score: System-wide muscle memory (0-1).
        hint_level: Current distillation hint level.
        spare_capacity: How much headroom is available (0-1).
        throughput: Tasks processed per second.
        avg_latency_ms: Average processing latency.
        backtests_run: Total backtests completed.
    """
    total_agents: int = 0
    active_agents: int = 0
    adaptation_score: float = 0.0
    hint_level: int = 10
    spare_capacity: float = 1.0
    throughput: float = 0.0
    avg_latency_ms: float = 0.0
    backtests_run: int = 0

    def __repr__(self) -> str:
        return (
            f"SwarmStatus(agents={self.total_agents}, "
            f"adapt={self.adaptation_score:.1%}, "
            f"hints={self.hint_level}, "
            f"spare={self.spare_capacity:.1%})"
        )


class SwarmRunner:
    """Orchestrates the full swarm: nerve fibers → routing → agents → feedback.

    Args:
        fibers: Dictionary of fiber_id → NerveFiber.
        routing: The routing layer.
        broadcast: The broadcasting channel.
    """

    def __init__(
        self,
        fibers: dict[str, NerveFiber] | None = None,
        routing: RoutingLayer | None = None,
        broadcast: BroadcastingChannel | None = None,
    ) -> None:
        self._fibers = fibers or {}
        self._routing = routing or RoutingLayer()
        self._broadcast = broadcast or BroadcastingChannel()
        self._engine = AdaptationEngine(dict(self._fibers))
        self._positions: list[PenrosePosition] = []
        self._ticks: int = 0
        self._tasks_processed: int = 0
        self._total_latency: float = 0.0
        self._backtests_run: int = 0

    def __repr__(self) -> str:
        return (
            f"SwarmRunner(fibers={len(self._fibers)}, "
            f"ticks={self._ticks}, "
            f"tasks={self._tasks_processed})"
        )

    def add_fiber(self, fiber: NerveFiber) -> None:
        """Add a nerve fiber to the swarm."""
        self._fibers[fiber.fiber_id] = fiber
        self._engine.register(fiber)

    def distribute(self, agent_ids: list[str]) -> list[PenrosePosition]:
        """Distribute agents on the Penrose lattice."""
        self._positions = assign_positions(agent_ids)
        return self._positions

    def tick(self, signal: Any) -> dict[str, Any]:
        """Run one swarm tick: perceive → route → process.

        Args:
            signal: The raw signal to process.

        Returns:
            Dict with processing results.
        """
        start = time.time()
        self._ticks += 1

        # Nerve fibers perceive
        tiles = {}
        for fid, fiber in self._fibers.items():
            result = self._engine.process_signal(fid, signal)
            tiles[fid] = result

        # Route to strongest destinations
        fired_routes = {}
        for fid in self._fibers:
            destinations = self._routing.fire(fid)
            if destinations:
                fired_routes[fid] = destinations

        latency = (time.time() - start) * 1000
        self._tasks_processed += 1
        self._total_latency += latency

        return {
            "tiles": tiles,
            "fired_routes": fired_routes,
            "latency_ms": latency,
        }

    def spare_capacity(self) -> float:
        """How much headroom the swarm has (0-1).

        Based on adaptation score (compiled fibers = free capacity).
        """
        return self._engine.adaptation_score

    def run_backtest_cycle(self, hint_level: int = 0) -> bool:
        """Run a backtest if there's spare capacity."""
        if self.spare_capacity() < 0.3:
            return False
        self._backtests_run += 1
        return True

    @staticmethod
    def run_forever(grid, max_ticks=-1, breed_interval=100):
        """Continuous daemon: tick grid, breed cold rooms, respect thermal.

        Args:
            grid: JEPAGrid instance.
            max_ticks: Limit for testing (-1 = infinite).
            breed_interval: How many ticks between breeding rounds.

        Yields: status dict per tick.
        """
        import numpy as np
        ticks = 0
        while max_ticks < 0 or ticks < max_ticks:
            signal = np.random.randn(grid.l).astype(np.float32)
            result = grid.tick(signal)
            ticks += 1
            result["tick"] = ticks

            # Every breed_interval ticks, breed cold rooms
            if ticks % breed_interval == 0:
                cold = grid.cold(thresh=ticks // breed_interval)
                hot = [i for i in range(grid.n) if i not in cold]
                result["cold"] = len(cold)
                result["hot"] = len(hot)
                for dst in cold[:5]:  # breed at most 5 per cycle
                    if hot:
                        src = hot[ticks % len(hot)]
                        grid.breed(src, dst)

            yield result

    @property
    def positions(self) -> list[PenrosePosition]:
        """Current agent positions on the lattice."""
        return list(self._positions)

    @property
    def adaptation_engine(self) -> AdaptationEngine:
        """The adaptation engine."""
        return self._engine

    @property
    def routing(self) -> RoutingLayer:
        """The routing layer."""
        return self._routing

    @property
    def broadcast_channel(self) -> BroadcastingChannel:
        """The broadcasting channel."""
        return self._broadcast

    def status(self) -> SwarmStatus:
        """Get current swarm status."""
        avg_latency = (
            self._total_latency / self._tasks_processed
            if self._tasks_processed > 0
            else 0.0
        )
        throughput = (
            self._tasks_processed / max(1, self._ticks)
            if self._ticks > 0
            else 0.0
        )

        return SwarmStatus(
            total_agents=len(self._positions),
            active_agents=sum(
                1 for f in self._fibers.values()
                if f.state.value != "compiled"
            ),
            adaptation_score=self._engine.adaptation_score,
            hint_level=0,
            spare_capacity=self.spare_capacity(),
            throughput=throughput,
            avg_latency_ms=avg_latency,
            backtests_run=self._backtests_run,
        )
