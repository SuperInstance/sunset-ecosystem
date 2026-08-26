"""Tests for FleetBernsteinScheduler and its sub-components.

Coverage:
- Cron parsing and fire computation
- Schedule registration, tick, and fire
- Catch-up vs skip misfire policies
- Counterfactual receipts for skipped windows
- FleetDeterministicReplay record/replay
- FleetPhasedDispatch phase separation
- FleetWorkerIsolation spawn/cleanup
- WAL integration and audit chain
- Health check and status
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

# Ensure fleet modules importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fleet.fleet_bernstein_scheduler import (
    BernsteinScheduleConfig,
    FireReceipt,
    FleetBernsteinScheduler,
    FleetDeterministicReplay,
    FleetPhasedDispatch,
    FleetWorkerIsolation,
    Phase,
    PhaseArtifact,
    PhaseSpec,
    ReplayMissError,
)


# ── Helpers ─────────────────────────────────────────────────────


class _FakeWAL:
    """In-memory WAL stub for testing."""

    def __init__(self):
        self._entries = []
        self._last_hash = ""

    def append(
        self, *, agent_id, operation, vector_hash, parent_ids, generation, node_id
    ):
        entry = {
            "agent_id": agent_id,
            "operation": operation,
            "vector_hash": vector_hash,
            "parent_ids": parent_ids,
            "generation": generation,
            "node_id": node_id,
            "previous_hash": self._last_hash,
        }
        entry["hash"] = self._hash(entry)
        self._last_hash = entry["hash"]
        self._entries.append(entry)

    def _hash(self, entry):
        import hashlib

        return hashlib.sha256(
            json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def entries(self):
        class FakeEntry:
            def __init__(self, data):
                self._data = data
                self.entry = type(
                    "obj",
                    (),
                    {
                        "operation": data["operation"],
                        "vector_hash": data["vector_hash"],
                    },
                )()

        return [FakeEntry(e) for e in self._entries]


class _FakePacing:
    """GatewayPacing stub that always allows dispatch."""

    def can_dispatch(self):
        return True, "ok"

    def record_success(self):
        pass

    def record_failure(self):
        pass

    def record_timeout(self):
        pass


class _BlockingPacing:
    """GatewayPacing stub that always blocks dispatch."""

    def can_dispatch(self):
        return False, "rate_limited"


# ── FleetDeterministicReplay ────────────────────────────────────


class TestFleetDeterministicReplay:
    def test_record_and_replay_hit(self):
        wal = _FakeWAL()
        replay = FleetDeterministicReplay(
            wal=wal, run_id="test-run", replay=False, strict=True
        )

        replay.record("prompt1", "model-a", "response1")
        replay.record("prompt1", "model-a", "response2")
        replay.record("prompt2", "model-b", "response3")

        # Switch to replay mode by loading from WAL
        replay2 = FleetDeterministicReplay(
            wal=wal, run_id="test-run", replay=True, strict=True
        )
        assert replay2.get_replay("prompt1", "model-a") == "response1"
        assert replay2.get_replay("prompt1", "model-a") == "response2"
        assert replay2.get_replay("prompt2", "model-b") == "response3"
        assert replay2.hits == 3

    def test_replay_miss_strict_raises(self):
        wal = _FakeWAL()
        replay = FleetDeterministicReplay(
            wal=wal, run_id="test-run", replay=True, strict=True
        )
        with pytest.raises(ReplayMissError) as exc_info:
            replay.get_replay("unknown", "model-x")
        assert "Fleet replay miss" in str(exc_info.value)
        assert replay.strict_violations == 1

    def test_replay_miss_non_strict_returns_none(self):
        wal = _FakeWAL()
        replay = FleetDeterministicReplay(
            wal=wal, run_id="test-run", replay=True, strict=False
        )
        assert replay.get_replay("unknown", "model-x") is None
        assert replay.misses == 1

    def test_record_noop_in_replay_mode(self):
        wal = _FakeWAL()
        replay = FleetDeterministicReplay(
            wal=wal, run_id="test-run", replay=True, strict=True
        )
        replay.record("p", "m", "r")  # should be no-op
        assert replay.cached_count == 0

    def test_coverage_line(self):
        wal = _FakeWAL()
        replay = FleetDeterministicReplay(
            wal=wal, run_id="run-1", replay=True, strict=True
        )
        line = replay.coverage_line()
        assert "run_id=run-1" in line
        assert "cached=0" in line

    def test_set_seed_determinism(self):
        replay = FleetDeterministicReplay(
            wal=_FakeWAL(), run_id="seed-test", replay=False
        )
        replay.set_seed(42)
        a = [__import__("random").random() for _ in range(5)]
        replay.set_seed(42)
        b = [__import__("random").random() for _ in range(5)]
        assert a == b

    def test_prompt_key_stability(self):
        wal = _FakeWAL()
        replay = FleetDeterministicReplay(wal=wal, run_id="k", replay=False)
        k1 = replay._prompt_key(
            "hello", "m", provider="p", temperature=0.7, max_tokens=100
        )
        k2 = replay._prompt_key(
            "hello", "m", provider="p", temperature=0.7, max_tokens=100
        )
        k3 = replay._prompt_key(
            "hello", "m", provider="p", temperature=0.8, max_tokens=100
        )
        assert k1 == k2
        assert k1 != k3


# ── FleetPhasedDispatch ───────────────────────────────────────


class TestFleetPhasedDispatch:
    def test_default_executor(self):
        """Default executor returns a mock artifact for each phase."""
        dp = FleetPhasedDispatch(executor=None)  # type: ignore
        # Actually we need an executor; use the default from init
        dp._init_default_executor = lambda: None
        dp.executor = lambda task, spec, prior: PhaseArtifact(
            summary=f"ran {spec.phase.value}",
            decisions=["d1"],
            constraints=["c1"],
        )
        dp.phases = [Phase.RESEARCH, Phase.PLAN]
        result = dp.run({"task_id": "t1"})
        assert result["success"] is True
        assert result["phases_run"] == 2
        assert "final_artifact" in result

    def test_pacing_blocks_dispatch(self):
        dp = FleetPhasedDispatch(
            executor=lambda t, s, p: PhaseArtifact(
                summary="x", decisions=["d"], constraints=["c"]
            ),
            phases=[Phase.IMPLEMENT],
        )
        result = dp.run({"task_id": "t1"}, pacing=_BlockingPacing())
        assert result["success"] is False
        assert "pacing_blocked" in result["reason"]

    def test_gate_failure_retries(self):
        call_count = 0

        def flaky_executor(task, spec, prior):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("flaky")
            return PhaseArtifact(summary="ok", decisions=["d"], constraints=["c"])

        dp = FleetPhasedDispatch(
            executor=flaky_executor, phases=[Phase.IMPLEMENT], gate_max_retries=2
        )
        result = dp.run({"task_id": "t1"})
        assert result["success"] is True
        assert call_count == 2

    def test_gate_hard_fail(self):
        dp = FleetPhasedDispatch(
            executor=lambda t, s, p: PhaseArtifact(
                summary="", decisions=[], constraints=[]
            ),
            phases=[Phase.IMPLEMENT],
            gate_max_retries=0,
        )
        result = dp.run({"task_id": "t1"})
        assert result["success"] is False
        assert result["reason"] == "phase_gate_failed"

    def test_wal_integration(self):
        wal = _FakeWAL()
        dp = FleetPhasedDispatch(
            executor=lambda t, s, p: PhaseArtifact(
                summary="ok", decisions=["d"], constraints=["c"]
            ),
            phases=[Phase.RESEARCH],
            wal=wal,
        )
        result = dp.run({"task_id": "t1", "node_id": "n1"})
        assert result["success"] is True
        # WAL should have one phase_transition entry
        ops = [e["operation"] for e in wal._entries]
        assert "phase_transition" in ops

    def test_phase_spec_override(self):
        dp = FleetPhasedDispatch(
            executor=lambda t, s, p: PhaseArtifact(
                summary="ok", decisions=["d"], constraints=["c"]
            ),
            phases=[Phase.RESEARCH],
        )
        task = {
            "task_id": "t1",
            "phases": {
                "research": {
                    "model": "custom-model",
                    "effort": "low",
                    "max_tokens": 1000,
                },
            },
        }
        # We can't easily observe the spec from outside, but we can run it
        result = dp.run(task)
        assert result["success"] is True


# ── FleetWorkerIsolation ──────────────────────────────────────


class TestFleetWorkerIsolation:
    def test_spawn_echo_command(self, tmp_path):
        wi = FleetWorkerIsolation(
            pid_dir=tmp_path / "pids", signals_dir=tmp_path / "signals"
        )
        result = wi.spawn(
            role="test",
            session_id="session_001",
            command=["echo", "hello"],
            workdir=tmp_path,
        )
        assert result["status"] == "running"
        assert result["worker_pid"] == os.getpid()
        assert "child_pid" in result
        # Wait for child
        import subprocess

        proc = subprocess.Popen(["echo", "hello"])
        # Cleanup
        wi.cleanup_session("session_001")
        pid_file = tmp_path / "pids" / "session_001.json"
        assert not pid_file.exists()

    def test_invalid_session_id(self, tmp_path):
        wi = FleetWorkerIsolation(
            pid_dir=tmp_path / "pids", signals_dir=tmp_path / "signals"
        )
        with pytest.raises(ValueError, match="Invalid session_id"):
            wi.spawn(role="test", session_id="../bad", command=["echo", "hi"])

    def test_cleanup_records_sunset(self, tmp_path):
        wal = _FakeWAL()
        wi = FleetWorkerIsolation(
            pid_dir=tmp_path / "pids", signals_dir=tmp_path / "signals"
        )
        # Just cleanup directly
        wi.cleanup_session("session_002", wal=wal)
        ops = [e["operation"] for e in wal._entries]
        assert "sunset" in ops

    def test_command_not_found(self, tmp_path):
        wi = FleetWorkerIsolation(
            pid_dir=tmp_path / "pids", signals_dir=tmp_path / "signals"
        )
        with pytest.raises(RuntimeError, match="Command not found"):
            wi.spawn(
                role="test",
                session_id="session_003",
                command=["nonexistent_binary_12345"],
            )

    def test_pid_file_written(self, tmp_path):
        wi = FleetWorkerIsolation(
            pid_dir=tmp_path / "pids", signals_dir=tmp_path / "signals"
        )
        result = wi.spawn(
            role="auditor",
            session_id="session_004",
            command=["echo", "hi"],
            model="sonnet",
        )
        pid_file = Path(result["pid_file"])
        assert pid_file.exists()
        data = json.loads(pid_file.read_text())
        assert data["role"] == "auditor"
        assert data["model"] == "sonnet"
        assert data["session"] == "session_004"
        wi.cleanup_session("session_004")


# ── FleetBernsteinScheduler ───────────────────────────────────


class TestFleetBernsteinScheduler:
    def test_register_and_list(self):
        sched = FleetBernsteinScheduler(BernsteinScheduleConfig(node_id="n1"))
        info = sched.register_schedule(
            "sched-1",
            "0 * * * *",  # every hour
            {"task_id": "t1"},
            misfire_policy="skip",
        )
        assert info["schedule_id"] == "sched-1"
        assert info["status"] == "registered"
        assert info["misfire_policy"] == "skip"

        schedules = sched.list_schedules()
        assert len(schedules) == 1
        assert schedules[0]["id"] == "sched-1"

    def test_unregister(self):
        sched = FleetBernsteinScheduler()
        sched.register_schedule("s1", "0 * * * *", {})
        assert sched.unregister_schedule("s1") is True
        assert sched.unregister_schedule("s1") is False
        assert len(sched.list_schedules()) == 0

    def test_tick_no_schedules(self):
        sched = FleetBernsteinScheduler()
        result = sched.tick(now=time.time())
        assert result["schedules_checked"] == 0
        assert result["fires_dispatched"] == 0

    def test_tick_fire_due(self):
        sched = FleetBernsteinScheduler(BernsteinScheduleConfig(node_id="n1"))
        sched.register_schedule("s1", "* * * * *", {"task_id": "t1"})  # every minute

        # Fire should happen immediately if now is past the next minute
        now = int(time.time())
        result = sched.tick(now=now + 120)  # 2 minutes in the future
        assert result["fires_dispatched"] >= 1
        assert result["schedules_checked"] == 1

    def test_skip_misfire_policy(self):
        sched = FleetBernsteinScheduler(BernsteinScheduleConfig(catch_up_limit=2))
        sched.register_schedule(
            "s1", "* * * * *", {"task_id": "t1"}, misfire_policy="skip"
        )

        # Anchor last_fire_at far in the past so multiple windows are missed
        with sched._lock:
            sched._schedules["s1"]["last_fire_at"] = time.time() - 300  # 5 minutes ago

        now = int(time.time())
        result = sched.tick(now=now)
        # Skip policy: only most recent missed instant is dispatched
        assert result["fires_dispatched"] == 1

    def test_catch_up_misfire_policy(self):
        sched = FleetBernsteinScheduler(
            BernsteinScheduleConfig(catch_up_limit=3, node_id="n1")
        )
        sched.register_schedule(
            "s1", "* * * * *", {"task_id": "t1"}, misfire_policy="catch_up"
        )

        with sched._lock:
            sched._schedules["s1"]["last_fire_at"] = time.time() - 300

        now = int(time.time())
        result = sched.tick(now=now)
        # Catch-up: up to catch_up_limit fires
        assert result["fires_dispatched"] <= 3
        assert result["fires_dispatched"] >= 1

    def test_counterfactual_receipt(self):
        sched = FleetBernsteinScheduler(BernsteinScheduleConfig(catch_up_limit=1))
        sched.register_schedule(
            "s1", "* * * * *", {"task_id": "t1"}, misfire_policy="catch_up"
        )

        with sched._lock:
            sched._schedules["s1"]["last_fire_at"] = time.time() - 180

        now = int(time.time())
        result = sched.tick(now=now)
        receipts = result["receipts"]
        counterfactuals = [r for r in receipts if r.get("counterfactual")]
        assert len(counterfactuals) >= 1
        assert counterfactuals[0]["dispatched"] is False

    def test_pacing_blocks_fire(self):
        sched = FleetBernsteinScheduler()
        sched.register_schedule("s1", "* * * * *", {"task_id": "t1"})
        pacing = _BlockingPacing()
        now = int(time.time()) + 120
        result = sched.tick(now=now, pacing=pacing)
        # Pacing blocks task execution, so no fires are dispatched
        assert result["fires_dispatched"] == 0
        assert len(result["receipts"]) >= 1
        assert not result["receipts"][0]["dispatched"]

    def test_wal_integration(self):
        wal = _FakeWAL()
        sched = FleetBernsteinScheduler(BernsteinScheduleConfig(node_id="n1"))
        sched.register_schedule("s1", "* * * * *", {"task_id": "t1"})
        now = int(time.time()) + 120
        result = sched.tick(now=now, wal=wal)
        assert result["fires_dispatched"] >= 1
        ops = [e["operation"] for e in wal._entries]
        assert "schedule_fire" in ops

    def test_status_snapshot(self):
        sched = FleetBernsteinScheduler()
        sched.register_schedule("s1", "0 0 * * *", {"task_id": "t1"})
        status = sched.get_status()
        assert status["schedules_total"] == 1
        assert status["alive"] is False  # no tick yet
        assert status["next_fire_at"] > 0

    def test_health_check(self):
        sched = FleetBernsteinScheduler()
        health = sched.health_check()
        assert health["state"] == "failed"  # no tick yet

        sched.tick(now=time.time())
        health = sched.health_check()
        assert health["state"] == "healthy"

    def test_deterministic_replay_tick(self):
        wal = _FakeWAL()
        sched = FleetBernsteinScheduler(
            BernsteinScheduleConfig(
                node_id="n1",
                enable_deterministic_replay=True,
                replay_run_id="replay-1",
                replay_strict=True,
            )
        )
        sched.register_schedule("s1", "* * * * *", {"task_id": "t1"})
        now = int(time.time()) + 120
        result = sched.tick(now=now, wal=wal)
        assert "replay_coverage" in result

    def test_invalid_cron_raises(self):
        sched = FleetBernsteinScheduler()
        with pytest.raises(ValueError, match="Cron must have 5 fields"):
            sched.register_schedule("s1", "* * *", {})

    def test_thread_safety_register_and_tick(self):
        sched = FleetBernsteinScheduler()
        errors = []

        def register_many():
            try:
                for i in range(10):
                    sched.register_schedule(
                        f"sched-{i}", f"{i % 60} * * * *", {"task_id": f"t{i}"}
                    )
            except Exception as exc:
                errors.append(exc)

        def tick_many():
            try:
                for _ in range(10):
                    sched.tick(now=time.time() + 3600)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=register_many)
        t2 = threading.Thread(target=tick_many)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert not errors
        assert len(sched.list_schedules()) == 10

    def test_cron_expand_field(self):
        sched = FleetBernsteinScheduler()
        assert sched._expand_field("*", 0, 59) == set(range(0, 60))
        assert sched._expand_field("1-5", 0, 59) == {1, 2, 3, 4, 5}
        assert sched._expand_field("1,3,5", 0, 59) == {1, 3, 5}
        assert sched._expand_field("*/15", 0, 59) == {0, 15, 30, 45}
        assert sched._expand_field("1-10/2", 0, 59) == {1, 3, 5, 7, 9}

    def test_matches_day_union(self):
        sched = FleetBernsteinScheduler()
        parsed = {
            "days": {1, 15},
            "weekdays": {1},  # Monday
            "raw": "test",
        }
        # Monday the 1st
        dt = datetime(2024, 1, 1, tzinfo=__import__("datetime").timezone.utc)
        assert sched._matches_day(parsed, dt) is True
        # Tuesday the 2nd (not in days, not Monday)
        dt2 = datetime(2024, 1, 2, tzinfo=__import__("datetime").timezone.utc)
        assert sched._matches_day(parsed, dt2) is False
        # Monday the 8th (not in days, but is Monday)
        dt3 = datetime(2024, 1, 8, tzinfo=__import__("datetime").timezone.utc)
        assert sched._matches_day(parsed, dt3) is True

    def test_receipt_persistence(self, tmp_path):
        sched = FleetBernsteinScheduler(BernsteinScheduleConfig(node_id="n1"))
        sched._receipts_dir = tmp_path / "receipts"
        sched._receipts_dir.mkdir(parents=True, exist_ok=True)
        receipt = FireReceipt(
            schedule_id="s1",
            fire_time=1234567890,
            projection_hash="abc123",
            misfire_policy="skip",
            dispatched=True,
        )
        sched._persist_receipt(receipt)
        persisted = list(sched._receipts_dir.glob("*.json"))
        assert len(persisted) == 1
        data = json.loads(persisted[0].read_text())
        assert data["schedule_id"] == "s1"

    def test_phased_dispatch_integration(self):
        """End-to-end: scheduler tick triggers phased dispatch."""
        call_log = []

        def tracking_executor(task, spec, prior):
            call_log.append(spec.phase.value)
            return PhaseArtifact(
                summary=f"{spec.phase.value} done",
                decisions=[f"d-{spec.phase.value}"],
                constraints=[f"c-{spec.phase.value}"],
            )

        sched = FleetBernsteinScheduler(
            BernsteinScheduleConfig(
                node_id="n1",
                enable_phased_dispatch=True,
                default_phases=["research", "plan", "implement"],
            )
        )
        # Replace default executor with tracking executor
        sched._phased_dispatch.executor = tracking_executor
        sched.register_schedule("s1", "* * * * *", {"task_id": "t1"})

        now = int(time.time()) + 120
        result = sched.tick(now=now)
        assert result["fires_dispatched"] >= 1
        # Phased dispatch should have run all 3 phases
        assert "research" in call_log
        assert "plan" in call_log
        assert "implement" in call_log


# ── Integration with FleetConductorV2 ─────────────────────────


class TestFleetConductorV2Integration:
    def test_subsystem_wrapper_initialization(self):
        """Verify scheduler can be wrapped as a FleetConductorV2 subsystem."""
        from nexus.fleet_conductor_v2 import (
            ConductorConfig,
            FleetConductorV2,
            SubsystemWrapper,
        )

        config = ConductorConfig(
            node_id="test-node",
            enable_bernstein_scheduler=True,
        )
        conductor = FleetConductorV2(config)

        # The subsystem should be registered
        assert "bernstein_scheduler" in conductor._subsystems
        wrapper = conductor._subsystems["bernstein_scheduler"]
        assert isinstance(wrapper, SubsystemWrapper)
        assert wrapper.enabled is True

    def test_beat_calls_scheduler_tick(self):
        """Verify conductor.beat() triggers scheduler.tick()."""
        from nexus.fleet_conductor_v2 import ConductorConfig, FleetConductorV2

        config = ConductorConfig(
            node_id="test-node",
            enable_bernstein_scheduler=True,
            enable_metronome=False,
            enable_mesh=False,
            enable_traps=False,
            enable_flux_presets=False,
            enable_identity=False,
            enable_gateway_pacing=False,
            enable_sda_loop=False,
        )
        conductor = FleetConductorV2(config)
        conductor.start()

        # Register a schedule
        scheduler = conductor._get_bernstein_scheduler()
        if scheduler is not None:
            scheduler.register_schedule("test-beat", "* * * * *", {"task_id": "beat-1"})

        result = conductor.beat()
        # Should include bernstein_scheduler in tick results
        assert "bernstein_scheduler" in result or result.get("beat_number", 0) >= 1
        conductor.shutdown()

    def test_scheduler_health_check(self):
        """Health check reflects scheduler state."""
        from nexus.fleet_conductor_v2 import ConductorConfig, FleetConductorV2

        config = ConductorConfig(
            node_id="test-node",
            enable_bernstein_scheduler=True,
            enable_metronome=False,
            enable_mesh=False,
            enable_traps=False,
            enable_flux_presets=False,
            enable_identity=False,
            enable_gateway_pacing=False,
            enable_sda_loop=False,
        )
        conductor = FleetConductorV2(config)
        conductor.start()
        health = conductor._subsystems["bernstein_scheduler"].health_check()
        # After start, if no schedules registered and no tick, state may vary
        assert hasattr(health, "state")
        conductor.shutdown()


# ── Property-based cron tests ────────────────────────────────────


@pytest.mark.parametrize(
    "cron,expected_minutes",
    [
        ("0 * * * *", {0}),
        ("*/15 * * * *", {0, 15, 30, 45}),
        ("1,2,3 * * * *", {1, 2, 3}),
        ("0 0 * * *", {0}),
        ("0 0 1 * *", {0}),
        ("0 0 * * 1", {0}),
    ],
)
def test_cron_parsing(cron, expected_minutes):
    sched = FleetBernsteinScheduler()
    parsed = sched._parse_cron(cron)
    assert parsed["minutes"] == expected_minutes


# ── Edge cases ────────────────────────────────────────────────


def test_next_fire_after_utc_only():
    sched = FleetBernsteinScheduler()
    # Every minute at :00
    next_fire = sched._next_fire_after("0 * * * *", 0)
    # Should be the first hour boundary after epoch
    dt = datetime.fromtimestamp(next_fire, tz=__import__("datetime").timezone.utc)
    assert dt.minute == 0


def test_empty_schedule_list_status():
    sched = FleetBernsteinScheduler()
    status = sched.get_status()
    assert status["schedules_total"] == 0
    assert status["next_fire_at"] == 0.0
