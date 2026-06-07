"""TMinusBridge — Python bridge to t-minus-rs Rust crate.

Wraps the t-minus binary via subprocess JSON RPC to provide:
- Cron expression parsing and next-fire computation
- Hierarchical deadline trees with parent→child inheritance
- Token bucket and leaky bucket rate limiters

Integration targets:
- nerve.distributed_metronome_bridge — deadline propagation
- fleet.fleet_conductor — rate limiting for fleet operations
- fleet.fleet_monitor — cron-based scheduled health checks

Usage
-----
    bridge = TMinusBridge()
    next_fire = bridge.cron_next("*/15 * * * *", after=0)
    remaining = bridge.deadline_remaining(parent_secs=60, child_secs=120)
    acquired = bridge.token_bucket(burst=10.0, rate=2.0, acquire=3.0)
"""

from __future__ import annotations

__all__ = [
    "TMinusBridge",
    "DeadlineTree",
    "RateLimiter",
    "CronSchedule",
]

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DeadlineTree:
    """A hierarchical deadline tree node."""
    parent_secs: float
    child_secs: float
    remaining_secs: float


@dataclass
class RateLimiter:
    """Token bucket rate limiter state."""
    burst: float
    rate: float
    acquired: bool
    tokens_remaining: float


@dataclass
class CronSchedule:
    """Cron schedule with next fire time."""
    expr: str
    next_fire: int


class TMinusBridge:
    """Bridge to t-minus-rs Rust library.

    Parameters
    ----------
    binary_path : Path | str | None
        Path to the t_minus_bridge binary. If None, searches in:
        1. ./bin/t_minus_bridge
        2. ../bin/t_minus_bridge
        3. $PATH
    """

    def __init__(self, binary_path: Path | str | None = None) -> None:
        self.binary_path = self._resolve_binary(binary_path)
        self._version: str | None = None

    def _resolve_binary(self, path: Path | str | None) -> Path:
        """Resolve the binary path."""
        if path:
            p = Path(path)
            if p.exists():
                return p
            raise FileNotFoundError(f"Binary not found: {p}")

        # Search common locations
        candidates = [
            Path("bin/t_minus_bridge"),
            Path("../bin/t_minus_bridge"),
            Path("t_minus_bridge"),
        ]
        for c in candidates:
            if c.exists():
                return c.absolute()

        # Search PATH
        for path_dir in os.environ.get("PATH", "").split(":" + os.pathsep):
            p = Path(path_dir) / "t_minus_bridge"
            if p.exists():
                return p

        raise FileNotFoundError("t_minus_bridge binary not found. Run: cargo build --example t_minus_bridge")

    def _call(self, request: dict[str, Any]) -> dict[str, Any]:
        """Call the binary with a JSON request."""
        try:
            result = subprocess.run(
                [str(self.binary_path)],
                input=json.dumps(request),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Binary error: {result.stderr}")
            return json.loads(result.stdout)
        except subprocess.TimeoutExpired:
            raise RuntimeError("t-minus binary timeout")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON output: {e}")

    # ── Cron Scheduling ─────────────────────────────────────

    def cron_next(self, expr: str, after: int = 0) -> int:
        """Compute next fire time for a cron expression.

        Parameters
        ----------
        expr : str
            Cron expression (e.g., "*/15 * * * *").
        after : int
            Unix timestamp to compute after. Default 0 (now).

        Returns
        -------
        int
            Unix timestamp of next fire time.
        """
        resp = self._call({
            "op": "CronNext",
            "expr": expr,
            "after": after,
        })
        if not resp.get("success"):
            raise ValueError(resp.get("error", "Cron parse failed"))
        return int(resp["result"]["next_fire"])

    def cron_schedule(self, expr: str) -> CronSchedule:
        """Create a CronSchedule with next fire time."""
        next_fire = self.cron_next(expr)
        return CronSchedule(expr=expr, next_fire=next_fire)

    # ── Deadline Trees ──────────────────────────────────────

    def deadline_remaining(self, parent_secs: float, child_secs: float) -> float:
        """Compute remaining time in a hierarchical deadline tree.

        The child inherits the parent's deadline — if parent expires in 60s
        and child has 120s, the child actually has 60s remaining.

        Parameters
        ----------
        parent_secs : float
            Parent deadline in seconds.
        child_secs : float
            Child deadline in seconds.

        Returns
        -------
        float
            Remaining seconds (min of parent and child).
        """
        resp = self._call({
            "op": "DeadlineRemaining",
            "parent_secs": int(parent_secs),
            "child_secs": int(child_secs),
        })
        if not resp.get("success"):
            raise ValueError(resp.get("error", "Deadline computation failed"))
        return float(resp["result"]["remaining_secs"])

    def build_deadline_tree(self, parent_secs: float, child_secs: float) -> DeadlineTree:
        """Build a deadline tree and compute remaining time."""
        remaining = self.deadline_remaining(parent_secs, child_secs)
        return DeadlineTree(
            parent_secs=parent_secs,
            child_secs=child_secs,
            remaining_secs=remaining,
        )

    # ── Rate Limiting ───────────────────────────────────────

    def token_bucket(self, burst: float, rate: float, acquire: float) -> RateLimiter:
        """Create a token bucket and attempt to acquire tokens.

        Parameters
        ----------
        burst : float
            Maximum token capacity.
        rate : float
            Token refill rate per second.
        acquire : float
            Tokens to acquire.

        Returns
        -------
        RateLimiter
            Result with acquired flag and remaining tokens.
        """
        resp = self._call({
            "op": "TokenBucket",
            "burst": burst,
            "rate": rate,
            "acquire": acquire,
        })
        if not resp.get("success"):
            raise ValueError(resp.get("error", "Token bucket failed"))
        return RateLimiter(
            burst=burst,
            rate=rate,
            acquired=resp["result"]["acquired"],
            tokens_remaining=resp["result"]["tokens_remaining"],
        )

    def check_rate_limit(self, burst: float, rate: float, acquire: float) -> bool:
        """Quick check if tokens can be acquired.

        Returns
        -------
        bool
            True if tokens were acquired.
        """
        limiter = self.token_bucket(burst, rate, acquire)
        return limiter.acquired

    # ── Integration Helpers ─────────────────────────────────

    def schedule_fleet_beat(self, interval_mins: int = 15) -> int:
        """Schedule the next fleet beat using cron.

        Parameters
        ----------
        interval_mins : int
            Beat interval in minutes. Default 15.

        Returns
        -------
        int
            Unix timestamp of next beat.
        """
        expr = f"*/{interval_mins} * * * *"
        return self.cron_next(expr)

    def propagate_deadline(self, parent_deadline: float, child_budget: float) -> float:
        """Propagate a parent deadline to a child task.

        Used by MetronomeBridge to enforce that child tasks don't
        exceed parent deadlines.

        Parameters
        ----------
        parent_deadline : float
            Parent remaining seconds.
        child_budget : float
            Child requested seconds.

        Returns
        -------
        float
            Effective child budget (capped by parent).
        """
        return self.deadline_remaining(parent_deadline, child_budget)

    def throttle_fleet_operation(self, ops_per_sec: float, burst: int = 10) -> bool:
        """Throttle a fleet operation using token bucket.

        Parameters
        ----------
        ops_per_sec : float
            Target operations per second.
        burst : int
            Burst capacity. Default 10.

        Returns
        -------
        bool
            True if operation should proceed.
        """
        return self.check_rate_limit(burst=burst, rate=ops_per_sec, acquire=1.0)

    # ── Status ───────────────────────────────────────────────

    def is_available(self) -> bool:
        """Check if the binary is available and functional."""
        try:
            self.cron_next("0 * * * *", after=0)
            return True
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"TMinusBridge(binary={self.binary_path})"
