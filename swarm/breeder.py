"""Breeder — Connects tournament breeding to JEPAGrid room management.

Implements the breeding pipeline per SPEC-BREEDER:
  Tournament Round → Pareto frontier / sunset candidates → breed → rebirth
"""
from __future__ import annotations

__all__ = ["Breeder", "AgentLifecycle", "spawn_from_template", "BreedingDaemon"]

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from nerve.room_grid import RoomGrid
from nerve.templates import AgentTemplate
from swarm.thermal import ThermalBudget
from swarm.tournament import AgentScore, TournamentRound, breed, sunset_candidates
from swarm.breeder_daemon import BreedingDaemon

logger = logging.getLogger(__name__)


class AgentLifecycle:
    """Finite-state machine for a single agent room.

    States per SPEC-BREEDER §5:
        SPAWNED → ACTIVE → ADAPTING → COMPILED → SUNSET → (rebirth) → SPAWNED
    """

    SPAWNED = "spawned"
    ACTIVE = "active"
    ADAPTING = "adapting"
    COMPILED = "compiled"
    SUNSET = "sunset"


@dataclass
class LifecycleRecord:
    """Immutable snapshot of an agent's lifecycle state."""

    room_id: int
    state: str = AgentLifecycle.SPAWNED
    generation: int = 0
    chaos: float = 0.3
    activity: int = 0
    hint_level: int = 10
    consecutive_wins: int = 0
    tick_entered: int = 0

    def can_advance(self) -> bool:
        """Determine if the agent should transition to the next state."""
        if self.state == AgentLifecycle.SPAWNED:
            return self.activity > 0  # first tick fires → ACTIVE
        if self.state == AgentLifecycle.ACTIVE:
            return self.chaos < 0.05
        if self.state == AgentLifecycle.ADAPTING:
            return self.consecutive_wins >= 3
        if self.state == AgentLifecycle.COMPILED:
            return False  # must be externally sunset
        return False


def spawn_from_template(
    grid: RoomGrid,
    template: AgentTemplate,
    room_idx: int,
    seed: Optional[int] = None,
) -> None:
    """Spawn an agent from a template into a specific room.

    1. Reset the room via rebirth().
    2. Override chaos to template.chaos_initial.
    3. Bias weights toward the template's trinity signature.

    Args:
        grid: The JEPAGrid / RoomGrid instance.
        template: AgentTemplate to imprint on the room.
        room_idx: Which room index to spawn into.
        seed: Optional RNG seed override (defaults to hash(template.name)).
    """
    grid.rebirth(room_idx)
    grid.chaos[room_idx] = template.chaos_initial

    bias_scale = template.mean_bias
    rng_seed = seed if seed is not None else (hash(template.name) % (2 ** 31))
    rng = np.random.RandomState(rng_seed)

    # Apply template signature as a small weight bias per SPEC-BREEDER §2
    for key, shape in [("w1", (64, 32)), ("w2", (32, 16)), ("w3", (16, 16))]:
        base = grid.w[key][room_idx]
        # Slightly amplify or attenuate based on template mean bias
        grid.w[key][room_idx] = base * (0.8 + 0.4 * bias_scale)

    # Light noise injection so identical templates still diverge
    for key in ("w1", "w2", "w3"):
        noise = rng.randn(*grid.w[key][room_idx].shape).astype(np.float32) * 0.001
        grid.w[key][room_idx] += noise

    logger.info(
        "Spawned template %r into room %d (bias=%.2f)",
        template.name,
        room_idx,
        bias_scale,
    )


class Breeder:
    """Connects tournament breeding to RoomGrid room management.

    Wires together:
      - RoomGrid (room activity / rebirth)
      - TournamentRound (selection)
      - breed() (crossover)
      - ThermalBudget (slot management)
      - AgentTemplate (spawn presets)
    """

    def __init__(
        self,
        grid: RoomGrid,
        templates: Dict[str, AgentTemplate],
        thermal: ThermalBudget,
    ) -> None:
        self.grid = grid
        self.templates = templates
        self.thermal = thermal
        self.generation = 0
        self._lifecycle: Dict[int, LifecycleRecord] = {}

    # ── Lifecycle State Machine ───────────────────────────────────

    def lifecycle_state(self, room_idx: int) -> str:
        """Return the current lifecycle state of a room."""
        return self._lifecycle.get(room_idx, LifecycleRecord(room_idx=room_idx)).state

    def _advance_lifecycle(self, room_idx: int) -> None:
        """Evaluate and possibly advance the lifecycle state of a room."""
        record = self._lifecycle.get(room_idx, LifecycleRecord(room_idx=room_idx))
        if record.can_advance():
            transitions = {
                AgentLifecycle.SPAWNED: AgentLifecycle.ACTIVE,
                AgentLifecycle.ACTIVE: AgentLifecycle.ADAPTING,
                AgentLifecycle.ADAPTING: AgentLifecycle.COMPILED,
            }
            new_state = transitions.get(record.state)
            if new_state:
                self._lifecycle[room_idx] = LifecycleRecord(
                    room_id=room_idx,
                    state=new_state,
                    generation=record.generation,
                    chaos=self.grid.chaos[room_idx],
                    activity=int(self.grid.activity[room_idx]),
                    hint_level=record.hint_level,
                    consecutive_wins=record.consecutive_wins,
                    tick_entered=self.grid.ticks,
                )
                logger.debug(
                    "Room %d advanced %s → %s", room_idx, record.state, new_state
                )

    # ── Evolution ─────────────────────────────────────────────────

    def evolve(self, scores: List[AgentScore]) -> List[dict]:
        """One evolution step: tournament → breed → rebirth.

        Args:
            scores: AgentScore list representing the current population.

        Returns:
            List of placement dicts: {**child, "room": int, "generation": int}.
        """
        self.generation += 1

        # 1. Run tournament
        tournament = TournamentRound(scores)
        ranked = tournament.run()
        winners = tournament.pareto_frontier

        if not winners:
            logger.warning("Evolve: no Pareto winners, skipping generation %d", self.generation)
            return []

        # 2. Identify sunset candidates (dominated agents)
        dominated = sunset_candidates(scores)
        num_children = min(len(dominated), self.grid.n - len(winners))

        if num_children <= 0:
            logger.info("Evolve: no room for children (dominated=%d, grid.n=%d)",
                        len(dominated), self.grid.n)
            return []

        # 3. Breed from winners
        children = breed(winners, num_children)

        # 4. Place children into coldest rooms
        cold_rooms = self._pick_cold_rooms(num_children)
        placed: List[dict] = []

        for i, child in enumerate(children):
            if i >= len(cold_rooms):
                break  # thermal budget exhausted

            room_idx = cold_rooms[i]
            self.grid.rebirth(room_idx)
            self.grid.chaos[room_idx] = 0.3  # fresh exploration

            # Update lifecycle
            self._lifecycle[room_idx] = LifecycleRecord(
                room_id=room_idx,
                state=AgentLifecycle.SPAWNED,
                generation=self.generation,
                chaos=0.3,
                activity=0,
                hint_level=10,
                consecutive_wins=0,
                tick_entered=self.grid.ticks,
            )

            placed.append({
                **child,
                "room": room_idx,
                "generation": self.generation,
            })
            logger.info(
                "Evolve: placed child %s in room %d (gen %d)",
                child.get("id", "?"),
                room_idx,
                self.generation,
            )

        return placed

    def spawn_template(self, template_name: str) -> Optional[int]:
        """Spawn a specific template into the coldest available room.

        Args:
            template_name: Key in self.templates.

        Returns:
            The room index used, or None if no room available.
        """
        template = self.templates.get(template_name)
        if template is None:
            raise KeyError(f"Unknown template: {template_name!r}")

        cold = self.grid.cold(thresh=1)
        if not cold:
            # No completely inactive rooms — try the least active
            activity_copy = self.grid.activity.copy()
            room_idx = int(np.argmin(activity_copy))
        else:
            room_idx = cold[0]

        spawn_from_template(self.grid, template, room_idx)
        self._lifecycle[room_idx] = LifecycleRecord(
            room_id=room_idx,
            state=AgentLifecycle.SPAWNED,
            generation=self.generation,
            chaos=template.chaos_initial,
            activity=0,
            hint_level=template.hint_level,
            consecutive_wins=0,
            tick_entered=self.grid.ticks,
        )
        return room_idx

    def _pick_cold_rooms(self, k: int) -> List[int]:
        """Return the k coldest room indices, sorted by activity ascending."""
        idx = np.argsort(self.grid.activity)
        return [int(i) for i in idx[:k]]

    def tick_all(self) -> None:
        """Advance lifecycle for every active room after a grid tick."""
        active_mask = self.grid.activity > 0
        for i in np.where(active_mask)[0]:
            self._advance_lifecycle(int(i))

    def sunset_room(self, room_idx: int) -> None:
        """Manually sunset a room, clearing its lifecycle record."""
        self.grid.rebirth(room_idx)
        self._lifecycle.pop(room_idx, None)
        logger.info("Sunset room %d", room_idx)

    @property
    def stats(self) -> dict:
        """Summary counts per lifecycle state."""
        counts = {s: 0 for s in (
            AgentLifecycle.SPAWNED,
            AgentLifecycle.ACTIVE,
            AgentLifecycle.ADAPTING,
            AgentLifecycle.COMPILED,
            AgentLifecycle.SUNSET,
        )}
        for rec in self._lifecycle.values():
            counts[rec.state] = counts.get(rec.state, 0) + 1
        return {
            "generation": self.generation,
            "rooms": self.grid.n,
            "lifecycle": counts,
            "thermal_headroom": self.thermal.thermal_headroom(),
        }
