"""Fixed-Point Auto-Scaling Bridge for FLUX VM.

Maps floating-point objective values to FLUX fixed-point integers and back.
Auto-detects dynamic range from pilot evaluations to minimize overflow/underflow.

Reference: docs/EXOTICA_NLOPT_RESEARCH_BRIEF.md (Proposal 5)
"""

from __future__ import annotations

__all__ = ["FixedPointBridge", "OverflowMode"]

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable


class OverflowMode(Enum):
    """How to handle values outside the representable fixed-point range."""

    SATURATE = auto()
    WRAP = auto()
    RAISE = auto()


@dataclass(frozen=True)
class FixedPointBridge:
    """Immutable fixed-point bridge with auto-detected scaling.

    Parameters
    ----------
    frac_bits:
        Number of fractional bits (precision). Default 16.
    total_bits:
        Total bit width (sign + integer + fractional). Default 32.
    scale_factor:
        Multiplier: ``fp_value = round(float_value * scale_factor)``.
        Computed automatically by :meth:`auto_scale`.
    overflow:
        Behavior when ``|value * scale_factor| > 2^(total_bits-1) - 1``.

    Usage
    -----
        bridge = FixedPointBridge.auto_scale([1.0, 2.5, 0.1, 100.0])
        fp = bridge.encode(42.0)
        decoded = bridge.decode(fp)
    """

    frac_bits: int = 16
    total_bits: int = 32
    scale_factor: float = 1.0
    overflow: OverflowMode = OverflowMode.SATURATE

    # ---- derived constants (computed once) ----
    _max_raw: int = 0
    _min_raw: int = 0

    def __post_init__(self) -> None:
        # dataclass(frozen=True) blocks direct mutation — use object.__setattr__
        max_val = (1 << (self.total_bits - 1)) - 1
        min_val = -(1 << (self.total_bits - 1))
        object.__setattr__(self, "_max_raw", max_val)
        object.__setattr__(self, "_min_raw", min_val)

    # ── auto-scaling ────────────────────────────────────────────

    @classmethod
    def auto_scale(
        cls,
        pilot_evaluations: list[float],
        frac_bits: int = 16,
        total_bits: int = 32,
        overflow: OverflowMode = OverflowMode.SATURATE,
        safety_margin: float = 2.0,
        sample_strategy: str = "max_abs",
    ) -> "FixedPointBridge":
        """Build a bridge from pilot objective evaluations.

        The ``safety_margin`` leaves headroom for values larger than
        anything seen in the pilot (e.g. boundary corners not sampled).

        ``sample_strategy`` controls how the pilot is reduced to a single
        scale anchor:

        - ``"max_abs"`` — use ``max(|pilot|)`` (default, robust)
        - ``"p99"`` — use 99th percentile (tolerates rare outliers)
        - ``"mean_abs"`` — use mean absolute value (aggressive, risks overflow)
        - ``"bounds_corners"`` — not implemented here; caller should pass
          evaluations from bounds corners as the pilot list
        """
        if not pilot_evaluations:
            raise ValueError("pilot_evaluations must not be empty")

        if sample_strategy == "max_abs":
            anchor = max(abs(v) for v in pilot_evaluations)
        elif sample_strategy == "p99":
            sorted_vals = sorted(abs(v) for v in pilot_evaluations)
            idx = int(math.ceil(0.99 * len(sorted_vals))) - 1
            anchor = sorted_vals[max(idx, 0)]
        elif sample_strategy == "mean_abs":
            anchor = sum(abs(v) for v in pilot_evaluations) / len(pilot_evaluations)
        else:
            raise ValueError(f"unknown sample_strategy: {sample_strategy}")

        if anchor == 0.0:
            # All pilot values are zero — arbitrary large scale
            scale = float(1 << (total_bits - 2))
        else:
            max_fp = (1 << (total_bits - 1)) - 1
            # Leave safety_margin headroom above the anchor
            scale = max_fp / (anchor * safety_margin)

        return cls(
            frac_bits=frac_bits,
            total_bits=total_bits,
            scale_factor=scale,
            overflow=overflow,
        )

    # ── encode / decode ───────────────────────────────────────

    def encode(self, value: float) -> int:
        """Float → fixed-point integer.

        Raises
        ------
        OverflowError
            If overflow mode is :attr:`OverflowMode.RAISE` and the value
            is out of range.
        """
        raw = round(value * self.scale_factor)
        if raw > self._max_raw:
            if self.overflow is OverflowMode.SATURATE:
                return self._max_raw
            if self.overflow is OverflowMode.WRAP:
                # two's complement wrap
                width = self.total_bits
                return (raw + (1 << width)) % (1 << width) - (1 << width)
            raise OverflowError(
                f"encode({value}) = {raw} exceeds max {self._max_raw} "
                f"(scale={self.scale_factor:.3e})"
            )
        if raw < self._min_raw:
            if self.overflow is OverflowMode.SATURATE:
                return self._min_raw
            if self.overflow is OverflowMode.WRAP:
                width = self.total_bits
                return (raw + (1 << width)) % (1 << width) - (1 << width)
            raise OverflowError(
                f"encode({value}) = {raw} below min {self._min_raw} "
                f"(scale={self.scale_factor:.3e})"
            )
        return raw

    def decode(self, raw: int) -> float:
        """Fixed-point integer → float."""
        return raw / self.scale_factor

    def encode_batch(self, values: list[float]) -> list[int]:
        """Vectorized encode."""
        return [self.encode(v) for v in values]

    def decode_batch(self, raws: list[int]) -> list[float]:
        """Vectorized decode."""
        return [self.decode(r) for r in raws]

    # ── properties ────────────────────────────────────────────

    @property
    def resolution(self) -> float:
        """Smallest representable float step (1 / scale_factor)."""
        return 1.0 / self.scale_factor

    @property
    def max_representable(self) -> float:
        """Maximum float value this bridge can encode without overflow."""
        return self._max_raw / self.scale_factor

    @property
    def min_representable(self) -> float:
        """Minimum float value this bridge can encode without overflow."""
        return self._min_raw / self.scale_factor

    # ── convenience for FLUX constants ─────────────────────────

    def flux_constant(self, value: float) -> dict[str, int | float]:
        """Return a dict suitable for embedding as a FLUX module constant.

        Example::

            const = bridge.flux_constant(3.14159)
            # {"raw": 205887, "scale": bridge.scale_factor}
        """
        return {
            "raw": self.encode(value),
            "scale": self.scale_factor,
            "frac_bits": self.frac_bits,
            "total_bits": self.total_bits,
        }

    def __repr__(self) -> str:
        return (
            f"FixedPointBridge("
            f"frac_bits={self.frac_bits}, "
            f"total_bits={self.total_bits}, "
            f"scale={self.scale_factor:.3e}, "
            f"range=[{self.min_representable:.3e}, {self.max_representable:.3e}], "
            f"res={self.resolution:.3e})"
        )
