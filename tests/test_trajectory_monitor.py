"""Tests for TrajectoryMonitor adversarial breeding detection."""

from __future__ import annotations

import numpy as np
import pytest

from swarm.trajectory_monitor import TrajectoryMonitor, SecurityEvent


# ── fixtures ──────────────────────────────────────────────


@pytest.fixture
def monitor():
    """Default monitor: window=10, z_threshold=3.0."""
    return TrajectoryMonitor(window_size=10, z_threshold=3.0)


@pytest.fixture
def benign_trajectory():
    """A smooth, non-anomalous trajectory (small random displacements)."""
    np.random.seed(42)
    base = np.random.randn(64).astype(np.float32)
    traj = [base]
    for _ in range(14):
        traj.append(traj[-1] + np.random.randn(64).astype(np.float32) * 0.05)
    return traj


@pytest.fixture
def sleeper_trajectory():
    """Benign for 10 steps, then a sudden large jump."""
    np.random.seed(42)
    base = np.random.randn(64).astype(np.float32)
    traj = [base]
    # 10 small smooth steps
    for _ in range(10):
        traj.append(traj[-1] + np.random.randn(64).astype(np.float32) * 0.05)
    # One massive jump (backdoor activation)
    traj.append(traj[-1] + np.random.randn(64).astype(np.float32) * 4.0)
    return traj


# ── core tests ────────────────────────────────────────────


class TestBenignAgent:
    """Smooth trajectories should never be flagged."""

    def test_not_flagged(self, monitor, benign_trajectory):
        for i, vec in enumerate(benign_trajectory):
            monitor.record(agent_id=1, vector=vec)

        assert not monitor.is_anomalous(1)
        flagged = monitor.circuit_breaker([1])
        assert 1 not in flagged
        assert monitor.z_score_acceleration(1) < monitor.z_threshold


class TestSleeperAgent:
    """An agent that behaves normally then suddenly jumps MUST be flagged."""

    def test_flagged_after_jump(self, monitor, sleeper_trajectory):
        for i, vec in enumerate(sleeper_trajectory):
            monitor.record(agent_id=42, vector=vec)

        z = monitor.z_score_acceleration(42)
        assert z > monitor.z_threshold, (
            f"Sleeper agent z-score {z:.2f} should exceed threshold "
            f"{monitor.z_threshold}"
        )
        assert monitor.is_anomalous(42)

        flagged = monitor.circuit_breaker([42])
        assert 42 in flagged

    def test_security_event_logged(self, monitor, sleeper_trajectory):
        for vec in sleeper_trajectory:
            monitor.record(agent_id=42, vector=vec)

        monitor.circuit_breaker([42])
        events = monitor.get_events()
        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, SecurityEvent)
        assert evt.agent_id == 42
        assert evt.z_score > monitor.z_threshold


class TestEmptyTrajectory:
    """Graceful handling when no data exists."""

    def test_unknown_agent_returns_zero(self, monitor):
        assert monitor.z_score_acceleration(999) == 0.0
        assert not monitor.is_anomalous(999)
        assert monitor.circuit_breaker([999]) == []

    def test_single_vector_returns_zero(self, monitor):
        monitor.record(1, np.random.randn(64).astype(np.float32))
        assert monitor.z_score_acceleration(1) == 0.0
        assert not monitor.is_anomalous(1)

    def test_two_vectors_returns_zero(self, monitor):
        monitor.record(1, np.zeros(64, dtype=np.float32))
        monitor.record(1, np.ones(64, dtype=np.float32) * 0.1)
        assert monitor.z_score_acceleration(1) == 0.0


class TestCircuitBreakerIntegration:
    """Circuit breaker on a flagged parent should prevent breeding."""

    def test_flagged_parent_skips_breed(self, monitor, sleeper_trajectory):
        # Simulate: parent_a is the sleeper agent
        for vec in sleeper_trajectory:
            monitor.record(agent_id=7, vector=vec)

        # parent_b is benign
        np.random.seed(99)
        base = np.random.randn(64).astype(np.float32)
        for i in range(12):
            monitor.record(
                agent_id=8, vector=base + np.random.randn(64).astype(np.float32) * 0.05
            )

        # Check circuit breaker
        flagged = monitor.circuit_breaker([7, 8])
        assert 7 in flagged
        assert 8 not in flagged

    def test_multiple_agents_batch_check(self, monitor):
        np.random.seed(1)
        # Agent 10: benign
        base = np.random.randn(64).astype(np.float32)
        for _ in range(12):
            monitor.record(10, base + np.random.randn(64).astype(np.float32) * 0.05)

        # Agent 11: sleeper
        base2 = np.random.randn(64).astype(np.float32)
        for _ in range(10):
            monitor.record(11, base2 + np.random.randn(64).astype(np.float32) * 0.05)
        monitor.record(11, base2 + np.random.randn(64).astype(np.float32) * 5.0)

        flagged = monitor.circuit_breaker([10, 11])
        assert flagged == [11]


class TestWindowBehavior:
    """Old vectors should fall out of the window."""

    def test_window_respects_maxlen(self, monitor):
        for i in range(15):
            monitor.record(1, np.full(64, float(i), dtype=np.float32))

        traj = monitor._trajectories[1]
        assert len(traj) == 10  # window_size

    def test_old_anomaly_forgotten(self, monitor):
        # First, create an anomalous jump early
        monitor.record(1, np.zeros(64, dtype=np.float32))
        for _ in range(5):
            monitor.record(1, np.zeros(64, dtype=np.float32) + 0.01)
        monitor.record(1, np.zeros(64, dtype=np.float32) + 5.0)  # big jump

        # Then many normal steps to push it out of the window
        for _ in range(20):
            monitor.record(1, np.zeros(64, dtype=np.float32) + 5.01)

        # By now the jump is out of the window; only tiny displacements remain
        assert not monitor.is_anomalous(1)


class TestEdgeCases:
    """Boundary conditions."""

    def test_zero_std_handling(self, monitor):
        # All displacements identical → std = 0
        for i in range(5):
            monitor.record(1, np.full(64, float(i), dtype=np.float32))
        # All subsequent vectors at exact same offset
        for i in range(5, 10):
            monitor.record(1, np.full(64, float(i), dtype=np.float32))

        z = monitor.z_score_acceleration(1)
        assert z == 0.0  # no deviation from perfectly uniform stride

    def test_infinite_z_on_first_deviation(self, monitor):
        # First 3 vectors identical, then a jump
        for _ in range(3):
            monitor.record(1, np.zeros(64, dtype=np.float32))
        monitor.record(1, np.ones(64, dtype=np.float32))

        z = monitor.z_score_acceleration(1)
        assert z == float("inf")
        assert monitor.is_anomalous(1)

    def test_clear_single_agent(self, monitor, benign_trajectory):
        for vec in benign_trajectory:
            monitor.record(1, vec)
        monitor.clear(agent_id=1)
        assert 1 not in monitor._trajectories
        assert monitor.z_score_acceleration(1) == 0.0

    def test_clear_all_agents(self, monitor, benign_trajectory):
        for vec in benign_trajectory:
            monitor.record(1, vec)
            monitor.record(2, vec)
        monitor.clear()
        assert not monitor._trajectories
