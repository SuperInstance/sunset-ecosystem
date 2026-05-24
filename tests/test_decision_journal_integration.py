"""Integration tests for Decision Journal ↔ BreederDaemonV2 ↔ IntentConfirmationProtocol.

Verifies that spawn, sunset, breed, and human_command are persisted to
``data/decisions/YYYY-MM-DD.jsonl`` and can be queried by agent_id/operation.

Mocks turbovec so tests run without the native extension.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest

# ── Mock turbovec before any swarm.vector_table import ──
_mock_turbovec = types.ModuleType("turbovec")


class _MockIdMapIndex:
    """Minimal stand-in for turbovec.IdMapIndex."""

    def __init__(self, dim: int, bit_width: int = 4) -> None:
        self.dim = dim
        self.bit_width = bit_width
        self._vectors: dict[int, np.ndarray] = {}

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        for vec, aid in zip(vectors, ids):
            self._vectors[int(aid)] = vec.copy()

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        allowlist: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._vectors:
            return (
                np.zeros((1, k), dtype=np.float32),
                np.zeros((1, k), dtype=np.uint64),
            )
        q = query[0]
        candidates = list(self._vectors.items())
        if allowlist is not None:
            allowed = set(int(a) for a in allowlist)
            candidates = [(aid, v) for aid, v in candidates if aid in allowed]

        qn = q / (np.linalg.norm(q) + 1e-8)
        sims: list[tuple[int, float]] = []
        for aid, vec in candidates:
            vn = vec / (np.linalg.norm(vec) + 1e-8)
            sims.append((aid, float(np.dot(qn, vn))))
        sims.sort(key=lambda x: x[1], reverse=True)
        top = sims[:k]
        while len(top) < k:
            top.append((0, 0.0))
        scores = np.array([[s for _, s in top]], dtype=np.float32)
        ids_arr = np.array([[aid for aid, _ in top]], dtype=np.uint64)
        return scores, ids_arr

    def remove(self, agent_id: int) -> bool:
        return self._vectors.pop(agent_id, None) is not None

    def contains(self, agent_id: int) -> bool:
        return agent_id in self._vectors

    def write(self, path: str) -> None:
        pass

    @classmethod
    def load(cls, path: str) -> "_MockIdMapIndex":
        return cls(dim=256)


_mock_turbovec.IdMapIndex = _MockIdMapIndex  # type: ignore[attr-defined]
sys.modules["turbovec"] = _mock_turbovec

# Now safe to import swarm / logos modules
from logos.decision_journal import (
    log_spawn,
    log_sunset,
    log_breed,
    log_human_command,
    get_decision_history,
)
from logos.intent_protocol import FleetState, IntentConfirmationProtocol
from nerve.room_grid import RoomGrid
from swarm.breeder_daemon_v2 import (
    BreederDaemonV2,
    DiversityConfig,
    LifecycleState,
    ThermalConfig,
)
from swarm.thermal import DeviceType, ThermalBudget
from swarm.vector_table import AgentVector, FluxVectorTable


@pytest.fixture
def journal_dir():
    """Temporary directory for daily JSONL journals."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def grid():
    g = RoomGrid(n=20)
    for _ in range(20):
        for i in range(10):
            g.activity[i] += 5
    return g


@pytest.fixture
def thermal():
    return ThermalBudget({DeviceType.GPU: 10, DeviceType.CPU: 20})


@pytest.fixture
def wal_path():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


@pytest.fixture
def vector_table():
    vt = FluxVectorTable(dim=256, bit_width=4)
    rng = np.random.RandomState(42)
    for i in range(20):
        scale = 2.0 if i < 10 else 0.5
        vec = (rng.randn(256).astype(np.float32) * scale).tolist()
        vt.add(
            AgentVector(
                agent_id=i,
                vector=vec,
                fitness=0.8 if i < 10 else 0.3,
                generation=1,
                capability_mask=0xFFFF,
                thermal_pressure=0.1,
            )
        )
    return vt


def make_daemon(grid, thermal, wal_path, vector_table=None, journal_dir=None):
    """Factory for a test daemon with optional decision journal."""
    dj_path = str(journal_dir) if journal_dir else None
    return BreederDaemonV2(
        grid=grid,
        thermal=thermal,
        vector_table=vector_table,
        diversity=DiversityConfig(),
        thermal_cfg=ThermalConfig(max_agents=65, hysteresis_ticks=2),
        wal_path=wal_path,
        tick_interval=60.0,
        decision_journal_path=dj_path,
    )


# ── unit tests for standalone journal helpers ───────────────

class TestLogSpawn:
    def test_creates_record(self, journal_dir):
        path = str(journal_dir / "2026-05-23.jsonl")
        record = log_spawn(agent_id=42, parents=(1, 2), generation=3, reason="test", journal_path=path)
        assert record["operation"] == "spawn"
        assert record["agent_id"] == 42
        assert record["parents"] == [1, 2]
        assert record["generation"] == 3

    def test_appends_to_jsonl(self, journal_dir):
        path = str(journal_dir / "2026-05-23.jsonl")
        log_spawn(agent_id=1, parents=(10, 11), generation=1, journal_path=path)
        log_spawn(agent_id=2, parents=(20,), generation=2, journal_path=path)
        with open(path) as fh:
            lines = [json.loads(line) for line in fh]
        assert len(lines) == 2
        assert lines[0]["agent_id"] == 1
        assert lines[1]["agent_id"] == 2


class TestLogSunset:
    def test_creates_record_with_reason(self, journal_dir):
        path = str(journal_dir / "2026-05-23.jsonl")
        record = log_sunset(agent_id=42, reason="thermal_pressure", generation=2, journal_path=path)
        assert record["operation"] == "sunset"
        assert record["agent_id"] == 42
        assert record["reason"] == "thermal_pressure"
        assert record["generation"] == 2


class TestLogBreed:
    def test_creates_record_with_parents(self, journal_dir):
        path = str(journal_dir / "2026-05-23.jsonl")
        record = log_breed(parent_a=1, parent_b=2, child_id=99, generation=2, journal_path=path)
        assert record["operation"] == "breed"
        assert record["agent_id"] == 99
        assert record["parents"] == [1, 2]

    def test_solo_parent(self, journal_dir):
        path = str(journal_dir / "2026-05-23.jsonl")
        record = log_breed(parent_a=7, parent_b=None, child_id=88, generation=1, journal_path=path)
        assert record["parents"] == [7]


class TestLogHumanCommand:
    def test_creates_record(self, journal_dir):
        path = str(journal_dir / "2026-05-23.jsonl")
        # Mock intent object
        intent = type("Intent", (), {
            "raw_command": "sunset all agents",
            "action": "sunset",
            "is_destructive": lambda self: True,
        })()
        record = log_human_command(intent=intent, confirmed=True, scope="all", journal_path=path)
        assert record["operation"] == "human_command"
        assert record["why"] == "sunset all agents"
        assert record["actual"] == "confirmed"
        assert record["metadata"]["destructive"] is True
        assert record["metadata"]["confirmed"] is True

    def test_unconfirmed_record(self, journal_dir):
        path = str(journal_dir / "2026-05-23.jsonl")
        intent = type("Intent", (), {
            "raw_command": "breed top 10",
            "action": "breed",
            "is_destructive": lambda self: False,
        })()
        record = log_human_command(intent=intent, confirmed=False, scope="top:10", journal_path=path)
        assert record["actual"] == "pending"
        assert record["confidence"] == 0.5


class TestGetDecisionHistory:
    def test_filter_by_agent_id(self, journal_dir):
        path = str(journal_dir / "2026-05-23.jsonl")
        log_spawn(agent_id=1, parents=(10, 11), generation=1, journal_path=path)
        log_spawn(agent_id=2, parents=(20, 21), generation=1, journal_path=path)
        log_sunset(agent_id=1, reason="test", generation=1, journal_path=path)
        results = get_decision_history(agent_id=1, journal_path=path)
        assert len(results) == 2
        assert all(r["agent_id"] == 1 for r in results)

    def test_filter_by_operation(self, journal_dir):
        path = str(journal_dir / "2026-05-23.jsonl")
        log_spawn(agent_id=1, parents=(10, 11), generation=1, journal_path=path)
        log_sunset(agent_id=1, reason="test", generation=1, journal_path=path)
        results = get_decision_history(operation="sunset", journal_path=path)
        assert len(results) == 1
        assert results[0]["operation"] == "sunset"

    def test_combined_filter(self, journal_dir):
        path = str(journal_dir / "2026-05-23.jsonl")
        log_spawn(agent_id=1, parents=(10, 11), generation=1, journal_path=path)
        log_spawn(agent_id=2, parents=(20, 21), generation=1, journal_path=path)
        log_sunset(agent_id=1, reason="test", generation=1, journal_path=path)
        results = get_decision_history(agent_id=1, operation="spawn", journal_path=path)
        assert len(results) == 1
        assert results[0]["operation"] == "spawn"
        assert results[0]["agent_id"] == 1

    def test_empty_when_no_file(self, journal_dir):
        path = str(journal_dir / "nonexistent.jsonl")
        results = get_decision_history(journal_path=path)
        assert results == []


# ── integration tests with BreederDaemonV2 ──────────────────

class TestDaemonDecisionJournalIntegration:
    """BreederDaemonV2.step() writes to the decision journal."""

    def test_spawn_logged(self, grid, thermal, wal_path, vector_table, journal_dir):
        daemon = make_daemon(grid, thermal, wal_path, vector_table, journal_dir)
        daemon.start()
        daemon.queue_breed(parent_a=1, parent_b=2)
        transitions = daemon.step()
        daemon.stop()

        # Find the competed child (EGG → COMPETE)
        compete_trs = [t for t in transitions if t.to_state == LifecycleState.COMPETE]
        assert len(compete_trs) == 1
        child_id = compete_trs[0].agent_id

        # Query journal for spawn records
        records = get_decision_history(agent_id=child_id, operation="spawn", journal_path=str(journal_dir))
        assert len(records) == 1
        assert records[0]["agent_id"] == child_id
        assert records[0]["parents"] == [1, 2]

    def test_breed_logged(self, grid, thermal, wal_path, vector_table, journal_dir):
        daemon = make_daemon(grid, thermal, wal_path, vector_table, journal_dir)
        daemon.start()
        daemon.queue_breed(parent_a=1, parent_b=2)
        transitions = daemon.step()
        daemon.stop()

        # Find the competed child (EGG → COMPETE)
        compete_trs = [t for t in transitions if t.to_state == LifecycleState.COMPETE]
        assert len(compete_trs) == 1
        child_id = compete_trs[0].agent_id

        records = get_decision_history(agent_id=child_id, operation="breed", journal_path=str(journal_dir))
        assert len(records) == 1
        assert records[0]["parents"] == [1, 2]

    def test_sunset_logged_with_reason(self, grid, thermal, wal_path, vector_table, journal_dir):
        daemon = make_daemon(grid, thermal, wal_path, vector_table, journal_dir)
        daemon.start()

        # First breed — should succeed
        daemon.queue_breed(parent_a=1, parent_b=2)
        daemon.step()

        # Second breed into same room — should sunset first agent
        daemon.queue_breed(parent_a=3, parent_b=4)
        transitions = daemon.step()
        daemon.stop()

        sunset_trs = [t for t in transitions if t.to_state == LifecycleState.SUNSET]
        assert len(sunset_trs) >= 1
        old_agent_id = sunset_trs[0].agent_id

        records = get_decision_history(agent_id=old_agent_id, operation="sunset", journal_path=str(journal_dir))
        assert len(records) >= 1
        assert records[0]["reason"] == "room_reuse"


# ── integration tests with IntentConfirmationProtocol ───────

class TestIntentProtocolDecisionJournalIntegration:
    """IntentConfirmationProtocol.log_decision writes human_command records."""

    def test_human_command_logging(self, journal_dir):
        state = FleetState(
            total_agents=100,
            active_agents=80,
            rooms=["Tide-Pool"],
            avg_fitness=0.5,
        )
        protocol = IntentConfirmationProtocol(fleet_state=state)
        intent = protocol.parse_intent("sunset all agents")

        path = str(journal_dir / "2026-05-23.jsonl")
        protocol.log_decision(
            intent=intent,
            confirmed=True,
            scope="all",
            journal_path=path,
        )

        records = get_decision_history(operation="human_command", journal_path=path)
        assert len(records) == 1
        assert records[0]["why"] == "sunset all agents"
        assert records[0]["actual"] == "confirmed"
        assert records[0]["metadata"]["destructive"] is True

    def test_human_command_logging_unconfirmed(self, journal_dir):
        state = FleetState(total_agents=0, active_agents=0, rooms=[])
        protocol = IntentConfirmationProtocol(fleet_state=state)
        intent = protocol.parse_intent("breed top 10")

        path = str(journal_dir / "2026-05-23.jsonl")
        protocol.log_decision(
            intent=intent,
            confirmed=False,
            scope="top:10",
            journal_path=path,
        )

        records = get_decision_history(operation="human_command", journal_path=path)
        assert len(records) == 1
        assert records[0]["actual"] == "pending"
        assert records[0]["confidence"] == 0.5

    def test_legacy_journal_still_works(self, journal_dir):
        from logos.decision_journal import DecisionJournal
        state = FleetState(total_agents=10, active_agents=5, rooms=[])
        protocol = IntentConfirmationProtocol(fleet_state=state)
        intent = protocol.parse_intent("optimize agent 42")

        journal = DecisionJournal()
        protocol.log_decision(
            intent=intent,
            confirmed=True,
            scope="agent:42",
            journal=journal,
        )

        assert len(journal.all_entries()) == 1
        assert journal.all_entries()[0].why == "optimize agent 42"
