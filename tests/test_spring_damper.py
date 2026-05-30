"""Tests for fleet/spring_damper.py — Spring-damper physics for smooth agent transitions."""
from __future__ import annotations

import math

import numpy as np
import pytest

from fleet.spring_damper import (
    AgentTransitionSmoother,
    MultiDimensionalSpringDamper,
    SpringDamper,
    SpringDamperConfig,
)


# ---------------------------------------------------------------------------
# 1. SpringDamper — basic physics
# ---------------------------------------------------------------------------

class TestSpringDamper:
    def test_tick_settles(self):
        """Critical damping should settle monotonically to target."""
        cfg = SpringDamperConfig(natural_freq=2.0, damping_ratio=1.0, tolerance=0.01)
        sd = SpringDamper(cfg)
        sd.set_target(10.0)

        # Run for 2 seconds at 100 Hz
        settled = False
        for _ in range(200):
            settled = sd.tick(0.01)
            if settled:
                break

        assert settled, "Should have settled within 2 seconds"
        assert sd.current == pytest.approx(10.0, abs=0.01)
        assert sd.velocity == pytest.approx(0.0, abs=0.001)
        assert sd.settled is True

    def test_tick_underdamped(self):
        """Underdamped system should oscillate before settling."""
        cfg = SpringDamperConfig(natural_freq=2.0, damping_ratio=0.3, tolerance=0.05)
        sd = SpringDamper(cfg)
        sd.set_target(10.0)

        positions = []
        for _ in range(200):
            sd.tick(0.01)
            positions.append(sd.current)

        # Should overshoot and oscillate
        max_pos = max(positions)
        assert max_pos > 10.0, "Underdamped should overshoot"
        assert sd.settled is True or sd.current > 9.0

    def test_tick_overdamped(self):
        """Overdamped system should approach target slowly without oscillation."""
        cfg = SpringDamperConfig(natural_freq=2.0, damping_ratio=2.0, tolerance=0.01)
        sd = SpringDamper(cfg)
        sd.set_target(10.0)

        positions = []
        for _ in range(400):
            sd.tick(0.01)
            positions.append(sd.current)

        # Should not overshoot
        max_pos = max(positions)
        assert max_pos <= 10.0 + 0.01, "Overdamped should not overshoot"

    def test_set_target_resets(self):
        """Changing target should reset settled flag."""
        sd = SpringDamper()
        sd.set_target(5.0)
        for _ in range(500):
            if sd.tick(0.01):
                break
        assert sd.settled is True

        sd.set_target(10.0)
        assert sd.settled is False
        assert sd.target == 10.0

    def test_settled_returns_true(self):
        """tick() should return True when settled."""
        sd = SpringDamper()
        sd.set_target(1.0)
        # Small dt, many steps
        for _ in range(1000):
            if sd.tick(0.001):
                break
        assert sd.tick(0.001) is True
        assert sd.settled is True

    def test_reset_velocity(self):
        """reset_velocity=True should zero velocity on target change."""
        sd = SpringDamper()
        sd.set_target(10.0)
        sd.tick(0.1)  # Build some velocity
        assert sd.velocity != 0.0

        sd.set_target(5.0, reset_velocity=True)
        assert sd.velocity == 0.0

    def test_direction_degrees(self):
        """8-way direction mapping from degrees."""
        sd = SpringDamper()
        test_cases = [
            (0, "N"), (45, "NE"), (90, "E"), (135, "SE"),
            (180, "S"), (225, "SW"), (270, "W"), (315, "NW"),
            (360, "N"), (22.5, "N"), (22.6, "NE"),
        ]
        for degrees, expected in test_cases:
            sd.set_target(degrees)
            sd._state.current = degrees
            assert sd.direction() == expected, f"{degrees}° should be {expected}"

    def test_direction_radians(self):
        """8-way direction mapping from radians."""
        sd = SpringDamper()
        test_cases = [
            (0, "N"), (math.pi / 4, "NE"), (math.pi / 2, "E"),
            (math.pi, "S"), (3 * math.pi / 2, "W"), (2 * math.pi, "N"),
        ]
        for radians, expected in test_cases:
            sd.set_target(radians)
            sd._state.current = radians
            assert sd.direction() == expected, f"{radians} rad should be {expected}"

    def test_repr(self):
        sd = SpringDamper()
        sd.set_target(10.0)
        sd.tick(0.1)
        r = repr(sd)
        assert "SpringDamper" in r
        assert "current=" in r
        assert "target=10.0" in r

    def test_current_property(self):
        sd = SpringDamper()
        sd.set_target(5.0)
        assert sd.current == 0.0
        sd.tick(0.1)
        assert sd.current != 0.0


# ---------------------------------------------------------------------------
# 2. MultiDimensionalSpringDamper
# ---------------------------------------------------------------------------

class TestMultiDimensionalSpringDamper:
    def test_tick_settles_all(self):
        """All dimensions should settle together."""
        msd = MultiDimensionalSpringDamper(dim=3)
        msd.set_target(np.array([1.0, 2.0, 3.0]))

        settled = False
        for _ in range(500):
            settled = msd.tick(0.01)
            if settled:
                break

        assert settled, "All dimensions should settle"
        np.testing.assert_array_almost_equal(msd.current, [1.0, 2.0, 3.0], decimal=2)
        np.testing.assert_array_almost_equal(msd.velocity, [0.0, 0.0, 0.0], decimal=3)
        assert msd.settled is True

    def test_set_target_shape_check(self):
        """Wrong shape should raise ValueError."""
        msd = MultiDimensionalSpringDamper(dim=3)
        with pytest.raises(ValueError):
            msd.set_target(np.array([1.0, 2.0]))  # Too short
        with pytest.raises(ValueError):
            msd.set_target(np.array([1.0, 2.0, 3.0, 4.0]))  # Too long

    def test_partial_settling(self):
        """Only settle when all dimensions converge."""
        cfg = SpringDamperConfig(natural_freq=1.0, tolerance=0.01)
        msd = MultiDimensionalSpringDamper(dim=2, config=cfg)
        # Target one dimension very far, one very close
        msd.set_target(np.array([100.0, 0.001]))

        # First dimension takes forever, second settles immediately
        settled_early = False
        for _ in range(50):
            settled_early = msd.tick(0.01)
            if settled_early:
                break
        assert not settled_early, "Should not settle until far dimension converges"

    def test_current_vector(self):
        msd = MultiDimensionalSpringDamper(dim=2)
        msd.set_target(np.array([5.0, 10.0]))
        # Use small dt to avoid overshoot
        msd.tick(0.001)
        current = msd.current
        assert current.shape == (2,)
        # After first small step, should be moving toward target but not past it
        assert 0.0 <= current[0] <= 5.0 + 0.1
        assert 0.0 <= current[1] <= 10.0 + 0.1

    def test_repr(self):
        msd = MultiDimensionalSpringDamper(dim=4)
        assert "dim=4" in repr(msd)
        assert "settled=False" in repr(msd)


# ---------------------------------------------------------------------------
# 3. AgentTransitionSmoother
# ---------------------------------------------------------------------------

class TestAgentTransitionSmoother:
    def test_scalar_transition(self):
        """Scalar 1D transition."""
        smoother = AgentTransitionSmoother(dim=1)
        smoother.transition_to(10.0)

        for _ in range(500):
            if smoother.tick(0.01):
                break

        assert smoother.settled is True
        assert smoother.current == pytest.approx(10.0, abs=0.01)

    def test_vector_transition(self):
        """Vector multi-dimensional transition."""
        smoother = AgentTransitionSmoother(dim=3)
        target = np.array([1.0, 2.0, 3.0])
        smoother.transition_to(target)

        for _ in range(500):
            if smoother.tick(0.01):
                break

        assert smoother.settled is True
        np.testing.assert_array_almost_equal(smoother.current, target, decimal=2)

    def test_type_error_scalar_with_vector_dim(self):
        """Passing scalar to vector dim should raise TypeError."""
        smoother = AgentTransitionSmoother(dim=3)
        with pytest.raises(TypeError):
            smoother.transition_to(5.0)  # scalar with dim=3

    def test_type_error_vector_with_scalar_dim(self):
        """Passing vector to scalar dim should raise TypeError."""
        smoother = AgentTransitionSmoother(dim=1)
        with pytest.raises(TypeError):
            smoother.transition_to(np.array([1.0, 2.0]))  # vector with dim=1

    def test_repr(self):
        smoother = AgentTransitionSmoother(dim=16)
        assert "dim=16" in repr(smoother)


# ---------------------------------------------------------------------------
# 4. Configuration
# ---------------------------------------------------------------------------

class TestSpringDamperConfig:
    def test_defaults(self):
        cfg = SpringDamperConfig()
        assert cfg.natural_freq == 2.0
        assert cfg.damping_ratio == 1.0
        assert cfg.tolerance == 0.01
        assert cfg.velocity_tolerance == 0.001

    def test_custom_values(self):
        cfg = SpringDamperConfig(natural_freq=5.0, damping_ratio=0.5, tolerance=0.1)
        assert cfg.natural_freq == 5.0
        assert cfg.damping_ratio == 0.5
        assert cfg.tolerance == 0.1


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_target(self):
        """Should settle at zero."""
        sd = SpringDamper()
        sd.set_target(0.0)
        for _ in range(500):
            if sd.tick(0.01):
                break
        assert sd.settled is True
        assert sd.current == pytest.approx(0.0, abs=0.01)

    def test_negative_target(self):
        """Should handle negative targets."""
        sd = SpringDamper()
        sd.set_target(-10.0)
        for _ in range(500):
            if sd.tick(0.01):
                break
        assert sd.settled is True
        assert sd.current == pytest.approx(-10.0, abs=0.01)

    def test_very_fast_settling(self):
        """High natural frequency should settle quickly."""
        cfg = SpringDamperConfig(
            natural_freq=10.0,
            damping_ratio=1.0,
            tolerance=0.1,
            velocity_tolerance=0.1,
        )
        sd = SpringDamper(cfg)
        sd.set_target(10.0)
        for _ in range(300):  # 300ms
            if sd.tick(0.001):
                break
        assert sd.settled is True
        assert sd.current == pytest.approx(10.0, abs=0.1)

    def test_already_at_target(self):
        """If current == target, should settle immediately."""
        sd = SpringDamper()
        sd._state.current = 5.0
        sd.set_target(5.0)
        assert sd.tick(0.01) is True
        assert sd.settled is True
