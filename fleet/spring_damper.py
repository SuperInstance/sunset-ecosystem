"""Spring-damper physics for smooth agent transitions.

Implements Pattern 5 from the SuperInstance audit: physically plausible
spring-damper transitions for agent state changes, room switches, and
parameter updates. Prevents jarring discontinuities that break temporal
continuity in JEPA models.

Reference: flux-compass + analog-spectral patterns from SuperInstance
ecosystem audit (May 30, 2026).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SpringDamperState:
    """Current state of a spring-damper system.

    Attributes:
        current: Current value (position).
        target: Target value to settle toward.
        velocity: Current velocity.
        settled: Whether the system has converged within tolerance.
    """

    current: float = 0.0
    target: float = 0.0
    velocity: float = 0.0
    settled: bool = False


@dataclass
class SpringDamperConfig:
    """Configuration for spring-damper physics.

    Attributes:
        natural_freq: Natural frequency in Hz (higher = faster settling).
        damping_ratio: Damping ratio (1.0 = critical, <1.0 = underdamped, >1.0 = overdamped).
        tolerance: Position error threshold to declare "settled".
        velocity_tolerance: Velocity threshold to declare "settled".
    """

    natural_freq: float = 2.0
    damping_ratio: float = 1.0
    tolerance: float = 0.01
    velocity_tolerance: float = 0.001


class SpringDamper:
    """Spring-damper model for smooth transitions.

    Replaces instantaneous state changes with physically plausible
transitions. Target heading exerts a restoring force; damping controls
oscillation; natural frequency controls speed.
    """

    def __init__(self, config: Optional[SpringDamperConfig] = None) -> None:
        self.config = config or SpringDamperConfig()
        self._state = SpringDamperState()

    # ── public API ───────────────────────────────────────────

    def set_target(self, target: float, reset_velocity: bool = False) -> None:
        """Set a new target to settle toward."""
        self._state.target = target
        self._state.settled = False
        if reset_velocity:
            self._state.velocity = 0.0

    def tick(self, dt: float) -> bool:
        """Advance simulation by dt seconds.

        Returns True if the system has settled within tolerance.
        """
        if self._state.settled:
            return True

        cfg = self.config
        pos = self._state.current
        vel = self._state.velocity
        target = self._state.target

        # Spring force: F = -k * displacement
        # Damping force: F = -c * velocity
        # Using omega = 2*pi*f, k = m*omega^2, c = 2*m*zeta*omega
        # Assuming m=1 for normalized dynamics:
        omega = 2.0 * math.pi * cfg.natural_freq
        k = omega * omega
        c = 2.0 * cfg.damping_ratio * omega

        displacement = pos - target
        acceleration = -k * displacement - c * vel

        # Semi-implicit Euler integration (stable for springs)
        new_vel = vel + acceleration * dt
        new_pos = pos + new_vel * dt

        self._state.velocity = new_vel
        self._state.current = new_pos

        # Check convergence
        pos_error = abs(new_pos - target)
        vel_mag = abs(new_vel)

        if pos_error < cfg.tolerance and vel_mag < cfg.velocity_tolerance:
            self._state.settled = True
            self._state.current = target
            self._state.velocity = 0.0
            return True

        return False

    @property
    def current(self) -> float:
        return self._state.current

    @property
    def target(self) -> float:
        return self._state.target

    @property
    def velocity(self) -> float:
        return self._state.velocity

    @property
    def settled(self) -> bool:
        return self._state.settled

    def direction(self) -> str:
        """Map current value to 8-way direction (N/NE/E/SE/S/SW/W/NW).

        Assumes current value is in degrees [0, 360) or radians [0, 2π).
        Auto-detects by magnitude: values > 2π are treated as degrees.
        """
        val = self._state.current
        if val > 2 * math.pi + 0.1:  # Probably degrees
            degrees = val % 360.0
        else:
            degrees = math.degrees(val) % 360.0

        # 8-way quantization: 0=N, 45=NE, 90=E, etc.
        # Use a tiny epsilon to handle exact boundary values
        idx = int((degrees + 22.499999) / 45.0) % 8
        dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        return dirs[idx]

    def __repr__(self) -> str:
        return (
            f"SpringDamper(current={self._state.current:.3f}, "
            f"target={self._state.target:.3f}, "
            f"vel={self._state.velocity:.3f}, "
            f"settled={self._state.settled})"
        )


class MultiDimensionalSpringDamper:
    """Spring-damper for vector-valued states (e.g., room state vectors)."""

    def __init__(
        self,
        dim: int,
        config: Optional[SpringDamperConfig] = None,
    ) -> None:
        self.dim = dim
        self.config = config or SpringDamperConfig()
        self._dampers = [SpringDamper(self.config) for _ in range(dim)]

    def set_target(self, target: np.ndarray, reset_velocity: bool = False) -> None:
        """Set target vector."""
        target = np.asarray(target, dtype=float)
        if target.shape != (self.dim,):
            raise ValueError(f"target shape {target.shape} != ({self.dim},)")
        for i, val in enumerate(target):
            self._dampers[i].set_target(float(val), reset_velocity=reset_velocity)

    def tick(self, dt: float) -> bool:
        """Advance all dimensions. Returns True when all settled."""
        return all(d.tick(dt) for d in self._dampers)

    @property
    def current(self) -> np.ndarray:
        return np.array([d.current for d in self._dampers], dtype=float)

    @property
    def target(self) -> np.ndarray:
        return np.array([d.target for d in self._dampers], dtype=float)

    @property
    def velocity(self) -> np.ndarray:
        return np.array([d.velocity for d in self._dampers], dtype=float)

    @property
    def settled(self) -> bool:
        return all(d.settled for d in self._dampers)

    def __repr__(self) -> str:
        return f"MultiDimensionalSpringDamper(dim={self.dim}, settled={self.settled})"


class AgentTransitionSmoother:
    """High-level wrapper for smoothing agent transitions between states.

    Uses spring-damper physics to interpolate between old and new agent
    parameters, room state vectors, or any scalar/vector quantity.
    """

    def __init__(
        self,
        dim: int = 1,
        config: Optional[SpringDamperConfig] = None,
    ) -> None:
        self.dim = dim
        if dim == 1:
            self._damper: SpringDamper | MultiDimensionalSpringDamper = SpringDamper(
                config
            )
        else:
            self._damper = MultiDimensionalSpringDamper(dim, config)

    def transition_to(self, new_state: float | np.ndarray) -> None:
        """Start transitioning to a new state."""
        if self.dim == 1 and isinstance(new_state, (int, float)):
            self._damper.set_target(float(new_state))
        elif self.dim > 1:
            arr = np.asarray(new_state, dtype=float)
            if arr.ndim == 0 or arr.shape == ():
                raise TypeError(
                    f"Expected array-like with shape ({self.dim},) for dim={self.dim}, "
                    f"got scalar"
                )
            self._damper.set_target(arr)
        else:
            raise TypeError(f"new_state type {type(new_state)} not compatible with dim={self.dim}")

    def tick(self, dt: float) -> bool:
        """Advance transition. Returns True when settled."""
        return self._damper.tick(dt)

    @property
    def current(self) -> float | np.ndarray:
        return self._damper.current

    @property
    def settled(self) -> bool:
        return self._damper.settled

    def __repr__(self) -> str:
        return f"AgentTransitionSmoother(dim={self.dim}, settled={self.settled})"
