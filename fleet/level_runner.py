"""LevelRunner — Simulation / game level execution engine for the fleet.

An emergent application that combines xlang's event-driven runtime,
Quanta's streaming VDB for level state tracking, and caslang's
constrained execution for deterministic level logic.

Use Cases
---------
- **AI Training Scenarios**: Generate procedural levels for agent training
- **Fleet Stress Testing**: Simulate 1000-agent scenarios to find bottlenecks
- **Game Worlds**: Persistent game worlds where agents are NPCs
- **Synthetic Data**: Generate realistic simulation data for ML training

Architecture
------------
The LevelRunner operates on a "Level" abstraction:

1. **Level Definition** — A YAML/JSON spec defining terrain, entities,
   rules, and victory conditions.  Converted to caslang for deterministic
   execution.

2. **State VDB** — Quanta PartitionedVdb stores entity positions, health,
   inventory, and relationships as high-dimensional vectors.  Enables fast
   spatial queries ("find all agents within 50m of point X").

3. **Event Engine** — xlang's event bus handles real-time events
   (collision, combat, trade, discovery).  Agents react via registered
   event handlers.

4. **Tick Loop** — Deterministic simulation ticks at configurable Hz.
   Each tick: (1) process events, (2) update physics, (3) run agent
   AI, (4) commit state to VDB.

Reference
---------
- xlang events: https://github.com/xlang-foundation/xlang/Docs/xlang_spec.md
- Quanta VDB: https://github.com/CantorAI/Quanta
"""

from __future__ import annotations

__all__ = [
    "LevelRunner",
    "LevelDefinition",
    "Entity",
    "LevelState",
    "EventBus",
]

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

# ── Entity ──────────────────────────────────────────────────────


@dataclass
class Entity:
    """A single entity in the level (agent, NPC, object, terrain)."""

    entity_id: str
    entity_type: str  # "agent", "npc", "object", "terrain", "item"
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    health: float = 100.0
    inventory: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    ai_script: str = ""  # caslang script for deterministic behavior
    faction: str = "neutral"
    last_tick: int = 0

    def __post_init__(self) -> None:
        self.position = np.array(self.position, dtype=np.float32)
        self.velocity = np.array(self.velocity, dtype=np.float32)

    def to_vector(self, dim: int = 64) -> np.ndarray:
        """Serialize entity state to a high-dimensional vector for VDB storage."""
        # Position (3), velocity (3), health (1), type hash (1), faction hash (1),
        # inventory count (1), attribute count (1) = 11 base dims
        base = np.zeros(11, dtype=np.float32)
        base[0:3] = self.position
        base[3:6] = self.velocity
        base[6] = self.health / 100.0
        base[7] = float(hash(self.entity_type) % 10000) / 10000.0
        base[8] = float(hash(self.faction) % 10000) / 10000.0
        base[9] = len(self.inventory) / 100.0
        base[10] = len(self.attributes) / 100.0

        # Pad or truncate to target dim
        if dim > 11:
            vec = np.zeros(dim, dtype=np.float32)
            vec[:11] = base
            # Use remaining dims for hashed attributes
            for i, (k, v) in enumerate(self.attributes.items()):
                idx = 11 + (i % (dim - 11))
                vec[idx] += float(hash(str(v)) % 10000) / 10000.0
            return vec
        return base[:dim]

    @classmethod
    def from_vector(cls, entity_id: str, entity_type: str, vector: np.ndarray, **kwargs: Any) -> "Entity":
        """Reconstruct entity from VDB vector (partial, loses fidelity)."""
        return cls(
            entity_id=entity_id,
            entity_type=entity_type,
            position=vector[:3] if len(vector) >= 3 else np.zeros(3),
            health=max(0.0, min(100.0, vector[6] * 100.0)) if len(vector) > 6 else 100.0,
            **kwargs,
        )


# ── LevelDefinition ───────────────────────────────────────────────


@dataclass
class LevelDefinition:
    """Specification for a simulation level."""

    name: str
    bounds: tuple[float, float, float, float, float, float] = (0, 0, 0, 100, 100, 100)
    tick_rate_hz: float = 10.0
    max_entities: int = 1000
    rules: list[dict[str, Any]] = field(default_factory=list)
    victory_conditions: list[dict[str, Any]] = field(default_factory=list)
    spawn_points: list[tuple[float, float, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_caslang(self) -> str:
        """Convert level rules to a caslang script for deterministic execution."""
        # Simplified: rules become caslang flow commands
        lines = ['{"op":"caslang","version":"0.3"}']
        for rule in self.rules:
            trigger = rule.get("trigger", "tick")
            action = rule.get("action", "noop")
            params = rule.get("params", {})
            lines.append(json.dumps({
                "op": "flow.set",
                "name": f"rule_{trigger}",
                "value": json.dumps({"action": action, "params": params}),
            }))
        return "\n".join(lines) + "\n"


# ── EventBus ────────────────────────────────────────────────────


class EventBus:
    """Lightweight event bus for level simulation (xlang-inspired)."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[dict[str, Any]], None]]] = {}
        self._lock = threading.Lock()
        self._event_count = 0

    def on(self, event_name: str, handler: Callable[[dict[str, Any]], None]) -> None:
        """Register an event handler."""
        with self._lock:
            self._handlers.setdefault(event_name, []).append(handler)

    def emit(self, event_name: str, payload: dict[str, Any]) -> None:
        """Emit an event to all registered handlers."""
        with self._lock:
            handlers = list(self._handlers.get(event_name, []))
        for h in handlers:
            try:
                h(payload)
            except Exception as exc:
                logger.warning("Event handler failed for %s: %s", event_name, exc)
        self._event_count += 1

    def clear(self) -> None:
        """Remove all handlers."""
        with self._lock:
            self._handlers.clear()


# ── LevelState ────────────────────────────────────────────────────


class LevelState:
    """Mutable state container for a running level."""

    def __init__(self, definition: LevelDefinition) -> None:
        self.definition = definition
        self.entities: dict[str, Entity] = {}
        self.tick_count = 0
        self.start_time = time.time()
        self.events = EventBus()
        self.global_state: dict[str, Any] = {}
        self._lock = threading.Lock()

    def add_entity(self, entity: Entity) -> bool:
        """Add an entity to the level."""
        with self._lock:
            if len(self.entities) >= self.definition.max_entities:
                return False
            self.entities[entity.entity_id] = entity
            self.events.emit("entity_spawned", {
                "entity_id": entity.entity_id,
                "type": entity.entity_type,
                "position": entity.position.tolist(),
            })
            return True

    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity from the level."""
        with self._lock:
            if entity_id not in self.entities:
                return False
            del self.entities[entity_id]
            self.events.emit("entity_destroyed", {"entity_id": entity_id})
            return True

    def get_entities_near(
        self,
        position: np.ndarray,
        radius: float,
        entity_type: str | None = None,
    ) -> list[Entity]:
        """Spatial query: find entities within radius of position."""
        pos = np.array(position, dtype=np.float32)
        results: list[Entity] = []
        with self._lock:
            for e in self.entities.values():
                if entity_type and e.entity_type != entity_type:
                    continue
                dist = float(np.linalg.norm(e.position - pos))
                if dist <= radius:
                    results.append(e)
        return results

    def get_entities_by_faction(self, faction: str) -> list[Entity]:
        """Return all entities belonging to a faction."""
        with self._lock:
            return [e for e in self.entities.values() if e.faction == faction]

    def check_victory(self) -> dict[str, Any] | None:
        """Check if any victory condition is met."""
        for cond in self.definition.victory_conditions:
            cond_type = cond.get("type", "none")
            if cond_type == "eliminate_faction":
                target = cond.get("faction", "")
                if not self.get_entities_by_faction(target):
                    return {"victory": True, "condition": cond, "winner": "opposition"}
            elif cond_type == "survive_ticks":
                if self.tick_count >= cond.get("ticks", 1000):
                    return {"victory": True, "condition": cond, "winner": "survivors"}
            elif cond_type == "reach_position":
                target_pos = np.array(cond.get("position", [0, 0, 0]))
                radius = cond.get("radius", 5.0)
                faction = cond.get("faction", "")
                for e in self.get_entities_by_faction(faction):
                    if float(np.linalg.norm(e.position - target_pos)) <= radius:
                        return {"victory": True, "condition": cond, "winner": faction}
        return None

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "tick_count": self.tick_count,
                "elapsed_time": time.time() - self.start_time,
                "entity_count": len(self.entities),
                "event_count": self.events._event_count,
            }


# ── LevelRunner ─────────────────────────────────────────────────


class LevelRunner:
    """Execute simulation levels with deterministic tick loops.

    Parameters
    ----------
    quanta_bridge : QuantaVdbBridge | None
        If provided, entity state is persisted to Quanta VDB each tick.
    caslang_executor : CaslangExecutor | None
        If provided, entity AI scripts are executed via caslang sandbox.
    """

    def __init__(
        self,
        quanta_bridge: Any | None = None,
        caslang_executor: Any | None = None,
    ) -> None:
        self.quanta_bridge = quanta_bridge
        self.caslang_executor = caslang_executor
        self._levels: dict[str, LevelState] = {}
        self._lock = threading.Lock()
        self._running: dict[str, bool] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._tick_callbacks: dict[str, list[Callable[[LevelState], None]]] = {}

        # Stats
        self._levels_completed = 0
        self._levels_failed = 0
        self._total_ticks = 0

    # ── level lifecycle ───────────────────────────────────────────

    def load_level(self, definition: LevelDefinition) -> str:
        """Load a level definition and return a level ID."""
        level_id = f"{definition.name}_{int(time.time() * 1000)}"
        state = LevelState(definition)

        # Register default event handlers
        state.events.on("collision", self._handle_collision)
        state.events.on("combat", self._handle_combat)
        state.events.on("entity_spawned", self._handle_spawn)

        with self._lock:
            self._levels[level_id] = state
            self._running[level_id] = False
            self._tick_callbacks[level_id] = []

        return level_id

    def spawn_entity(
        self,
        level_id: str,
        entity_id: str,
        entity_type: str,
        position: tuple[float, float, float] = (0, 0, 0),
        faction: str = "neutral",
        ai_script: str = "",
        **kwargs: Any,
    ) -> bool:
        """Spawn an entity into a running level."""
        with self._lock:
            state = self._levels.get(level_id)
        if state is None:
            return False

        entity = Entity(
            entity_id=entity_id,
            entity_type=entity_type,
            position=np.array(position, dtype=np.float32),
            faction=faction,
            ai_script=ai_script,
            **kwargs,
        )
        return state.add_entity(entity)

    def start_level(self, level_id: str) -> bool:
        """Start the tick loop for a level."""
        with self._lock:
            if self._running.get(level_id, False):
                return False
            state = self._levels.get(level_id)
            if state is None:
                return False
            self._running[level_id] = True

        def tick_loop() -> None:
            tick_interval = 1.0 / state.definition.tick_rate_hz
            while self._running.get(level_id, False):
                tick_start = time.time()
                self._run_tick(level_id)
                elapsed = time.time() - tick_start
                sleep_time = max(0, tick_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        thread = threading.Thread(target=tick_loop, name=f"LevelRunner-{level_id}", daemon=True)
        thread.start()
        self._threads[level_id] = thread
        return True

    def stop_level(self, level_id: str) -> bool:
        """Stop the tick loop for a level."""
        with self._lock:
            if not self._running.get(level_id, False):
                return False
            self._running[level_id] = False

        thread = self._threads.get(level_id)
        if thread:
            thread.join(timeout=5.0)
        return True

    def get_level_state(self, level_id: str) -> LevelState | None:
        """Return the current state of a level."""
        with self._lock:
            return self._levels.get(level_id)

    def on_tick(self, level_id: str, callback: Callable[[LevelState], None]) -> None:
        """Register a callback to run after each tick."""
        with self._lock:
            self._tick_callbacks.setdefault(level_id, []).append(callback)

    # ── tick execution ────────────────────────────────────────────

    def _run_tick(self, level_id: str) -> None:
        """Execute one simulation tick."""
        with self._lock:
            state = self._levels.get(level_id)
        if state is None:
            return

        state.tick_count += 1
        self._total_ticks += 1

        # 1. Process entity AI
        self._process_ai(state)

        # 2. Update physics (simple Euler integration)
        self._update_physics(state)

        # 3. Check collisions
        self._check_collisions(state)

        # 4. Check victory
        victory = state.check_victory()
        if victory:
            state.events.emit("victory", victory)
            self._levels_completed += 1
            self.stop_level(level_id)
            return

        # 5. Persist to Quanta VDB if available
        if self.quanta_bridge is not None:
            self._persist_to_vdb(state)

        # 6. Run callbacks
        with self._lock:
            callbacks = list(self._tick_callbacks.get(level_id, []))
        for cb in callbacks:
            try:
                cb(state)
            except Exception as exc:
                logger.warning("Tick callback failed: %s", exc)

    def _process_ai(self, state: LevelState) -> None:
        """Run AI scripts for all entities."""
        for entity in list(state.entities.values()):
            if not entity.ai_script:
                continue
            if self.caslang_executor is not None:
                try:
                    # Parse and execute AI script
                    from .caslang_executor import CaslangScript
                    script = CaslangScript.from_jsonl(entity.ai_script)
                    result = self.caslang_executor.execute(script)
                    if result["status"] == "success":
                        # Apply AI output to entity
                        output = result.get("output", {})
                        if "move" in output:
                            direction = np.array(output["move"], dtype=np.float32)
                            entity.velocity += direction * 0.5
                except Exception as exc:
                    logger.debug("AI execution failed for %s: %s", entity.entity_id, exc)
            else:
                # Fallback: simple random wander
                if state.tick_count % 10 == 0:
                    entity.velocity += np.random.randn(3).astype(np.float32) * 0.2

    def _update_physics(self, state: LevelState) -> None:
        """Simple Euler physics integration."""
        bounds = state.definition.bounds
        for entity in state.entities.values():
            entity.position += entity.velocity * 0.1  # dt = 0.1s
            # Clamp to bounds
            for i in range(3):
                entity.position[i] = max(bounds[i], min(bounds[i + 3], entity.position[i]))
            # Damping
            entity.velocity *= 0.9

    def _check_collisions(self, state: LevelState) -> None:
        """Naive O(n²) collision detection (sufficient for small levels)."""
        entities = list(state.entities.values())
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                a, b = entities[i], entities[j]
                dist = float(np.linalg.norm(a.position - b.position))
                if dist < 2.0:  # collision radius
                    state.events.emit("collision", {
                        "entity_a": a.entity_id,
                        "entity_b": b.entity_id,
                        "distance": dist,
                        "position": ((a.position + b.position) / 2).tolist(),
                    })

    def _handle_collision(self, payload: dict[str, Any]) -> None:
        """Default collision handler."""
        logger.debug("Collision: %s <-> %s", payload.get("entity_a"), payload.get("entity_b"))

    def _handle_combat(self, payload: dict[str, Any]) -> None:
        """Default combat handler."""
        attacker = payload.get("attacker")
        defender = payload.get("defender")
        damage = payload.get("damage", 10.0)
        logger.debug("Combat: %s attacks %s for %.1f damage", attacker, defender, damage)

    def _handle_spawn(self, payload: dict[str, Any]) -> None:
        """Default spawn handler."""
        logger.debug("Spawned: %s (%s)", payload.get("entity_id"), payload.get("type"))

    def _persist_to_vdb(self, state: LevelState) -> None:
        """Persist entity vectors to Quanta VDB."""
        if self.quanta_bridge is None:
            return
        try:
            from .quanta_vdb_bridge import QuantaTableEntry
            for entity in state.entities.values():
                entry = QuantaTableEntry(
                    agent_id=entity.entity_id,
                    vector=entity.to_vector(dim=64),
                    timestamp=time.time(),
                    node_id="level_runner",
                    generation=state.tick_count,
                    fitness=entity.health / 100.0,
                    signature="level_runner",  # simplified
                    partition_tag=state.definition.name,
                    extra={
                        "type": entity.entity_type,
                        "faction": entity.faction,
                        "position": entity.position.tolist(),
                        "health": entity.health,
                    },
                )
                self.quanta_bridge.insert(entry)
        except Exception as exc:
            logger.debug("VDB persist failed: %s", exc)

    # ── stats ─────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "levels_loaded": len(self._levels),
                "levels_running": sum(1 for v in self._running.values() if v),
                "levels_completed": self._levels_completed,
                "levels_failed": self._levels_failed,
                "total_ticks": self._total_ticks,
            }
