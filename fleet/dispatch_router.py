"""Dispatch Router — Two-Minute Test auto-routing for fleet tasks.

If a task takes < 2 minutes, do it directly; if longer, delegate to a subagent.
Prevents subagent overhead on trivial work and integrates with the gateway
circuit breaker so we don't overload shared spawn capacity.

Usage::

    from fleet.dispatch_router import DispatchRouter
    from fleet.gateway_pacing import GatewayPacing

    router = DispatchRouter(gateway=GatewayPacing())

    decision = router.route("Fix typo in README", context={"files": 1})
    # → {"mode": "direct", "reason": "OPEN — dispatch allowed", "estimated_seconds": 30}

    decision = router.route(
        "Implement mesh gossip with tests",
        context={"files": 3, "tests": True},
    )
    # → {"mode": "subagent", "reason": "...", "estimated_seconds": 345}

After completion::

    router.record_actual(task_id="abc", estimated=345, actual=400)
    # Adjusts heuristic weights based on discrepancy.
"""

from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass, field
from typing import Any

from fleet.gateway_pacing import GatewayPacing

# ── Default heuristic weights ───────────────────────────────────────
# Seconds added per keyword / activity detected in the task description.
# These are seeded from empirical fleet data and refined by record_actual().
_DEFAULT_WEIGHTS: dict[str, float] = {
    "file_creation": 60.0,     # per file mentioned
    "test_writing": 90.0,      # per test file / test suite
    "doc_writing": 45.0,       # per doc file
    "research": 180.0,         # any research / investigate / look up
    "bug_fix": 120.0,          # base bug-fix overhead
    "bug_fix_complex": 300.0,  # complex / deep / race-condition / memory-leak
    "refactor": 90.0,          # per file refactored
    "integration": 150.0,     # cross-module wiring
    "simple_edit": 30.0,       # typo, rename, one-liner
    "config_change": 45.0,     # yaml, json, env var
    "merge_conflict": 60.0,    # per conflicted file
    "dependency": 120.0,      # pip, npm, cargo, etc.
    "architecture": 240.0,     # design / scaffold / blueprint
}

# Regex patterns that trigger each weight category.
_WEIGHT_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "file_creation": [
        re.compile(r"\bcreate\b", re.I),
        re.compile(r"\bnew file\b", re.I),
        re.compile(r"\bscaffold\b", re.I),
        re.compile(r"\badd\s+(?:a\s+)?(?:module|class|function)\b", re.I),
        re.compile(r"\bimplement\b", re.I),
    ],
    "test_writing": [
        re.compile(r"\btest\w*\b", re.I),
        re.compile(r"\bassert\b", re.I),
        re.compile(r"\bpytest\b", re.I),
        re.compile(r"\bunit[- ]?test\b", re.I),
        re.compile(r"\bverify\b", re.I),
    ],
    "doc_writing": [
        re.compile(r"\bdoc\w*\b", re.I),
        re.compile(r"\breadme\b", re.I),
        re.compile(r"\bmarkdown\b", re.I),
        re.compile(r"\bguide\b", re.I),
        re.compile(r"\bcomment\b", re.I),
    ],
    "research": [
        re.compile(r"\bresearch\b", re.I),
        re.compile(r"\binvestigate\b", re.I),
        re.compile(r"\blook\s+(?:into|up)\b", re.I),
        re.compile(r"\bfind\s+out\b", re.I),
        re.compile(r"\bexplore\b", re.I),
        re.compile(r"\bsurvey\b", re.I),
    ],
    "bug_fix": [
        re.compile(r"\bfix\b", re.I),
        re.compile(r"\bbug\w*\b", re.I),
        re.compile(r"\brepair\b", re.I),
        re.compile(r"\bresolve\b", re.I),
        re.compile(r"\baddress\b", re.I),
    ],
    "bug_fix_complex": [
        re.compile(r"\brace[- ]?condition\b", re.I),
        re.compile(r"\bmemory[- ]?leak\b", re.I),
        re.compile(r"\bdeadlock\b", re.I),
        re.compile(r"\bintermittent\b", re.I),
        re.compile(r"\bheisenbug\b", re.I),
        re.compile(r"\bcomplex\b", re.I),
        re.compile(r"\bdeep\s+(?:bug|issue|root[- ]?cause)\b", re.I),
    ],
    "refactor": [
        re.compile(r"\brefactor\b", re.I),
        re.compile(r"\brename\b", re.I),
        re.compile(r"\brestructure\b", re.I),
        re.compile(r"\bclean[- ]?up\b", re.I),
        re.compile(r"\bextract\b", re.I),
    ],
    "integration": [
        re.compile(r"\bintegrat\w*\b", re.I),
        re.compile(r"\bwire\s+(?:up|together)\b", re.I),
        re.compile(r"\bbridge\b", re.I),
        re.compile(r"\bconnect\b", re.I),
        re.compile(r"\bcross[- ]?(?:module|repo|fleet)\b", re.I),
    ],
    "simple_edit": [
        re.compile(r"\btypo\b", re.I),
        re.compile(r"\bone[- ]?liner\b", re.I),
        re.compile(r"\btweak\b", re.I),
        re.compile(r"\badjust\b", re.I),
        re.compile(r"\bquick\b", re.I),
    ],
    "config_change": [
        re.compile(r"\bconfig\w*\b", re.I),
        re.compile(r"\byaml\b", re.I),
        re.compile(r"\bjson\b", re.I),
        re.compile(r"\benv\b", re.I),
        re.compile(r"\.toml\b", re.I),
    ],
    "merge_conflict": [
        re.compile(r"\bmerge[- ]?conflict\b", re.I),
        re.compile(r"\bconflicted?\b", re.I),
        re.compile(r"\brebase\b", re.I),
    ],
    "dependency": [
        re.compile(r"\bdependenc\w*\b", re.I),
        re.compile(r"\brequirements\b", re.I),
        re.compile(r"\bpackage[- ]?json\b", re.I),
        re.compile(r"\bCargo\.toml\b", re.I),
        re.compile(r"\bcomposer\b", re.I),
    ],
    "architecture": [
        re.compile(r"\barchitect\w*\b", re.I),
        re.compile(r"\bdesign\b", re.I),
        re.compile(r"\bblueprint\b", re.I),
        re.compile(r"\broadmap\b", re.I),
        re.compile(r"\bfsm\b", re.I),
        re.compile(r"\bstate[- ]?machine\b", re.I),
    ],
}

# How many files are mentioned? Look for "N file(s)" or infer from context.
_FILE_COUNT_RE = re.compile(r"(\d+)\s+(?:file|module|class|test)s?\b", re.I)


def _extract_file_count(text: str, context: dict[str, Any] | None = None) -> int:
    """Best-effort file count from description + context."""
    count = 0
    # 1. explicit counts in the text
    for m in _FILE_COUNT_RE.finditer(text):
        count = max(count, int(m.group(1)))
    # 2. context hint
    if context:
        count = max(count, context.get("files", 0))
    return max(count, 1)  # at least one file is implied by any task


# ── Feedback record ────────────────────────────────────────────────
@dataclass
class _FeedbackEntry:
    task_id: str
    estimated: float
    actual: float
    keywords: list[str] = field(default_factory=list)


class DispatchRouter:
    """Route tasks to direct execution or subagent delegation.

    Parameters
    ----------
    gateway : GatewayPacing | None
        Circuit breaker to consult before recommending "subagent".
        If None, a fresh GatewayPacing() is instantiated.
    threshold_seconds : float
        Two-minute boundary in seconds (default 120).
    weights : dict[str, float] | None
        Override default heuristic weights.
    learning_rate : float
        Per-feedback adjustment rate (0–1).  Smaller = more conservative.
    """

    def __init__(
        self,
        gateway: GatewayPacing | None = None,
        threshold_seconds: float = 120.0,
        weights: dict[str, float] | None = None,
        learning_rate: float = 0.15,
    ) -> None:
        self._gateway = gateway or GatewayPacing()
        self._threshold = threshold_seconds
        self._weights = dict(weights or _DEFAULT_WEIGHTS)
        self._learning_rate = learning_rate
        self._lock = threading.Lock()
        self._feedback: list[_FeedbackEntry] = []

    # ── Public API ────────────────────────────────────────────────────

    def estimate_duration(self, task_description: str, context: dict[str, Any] | None = None) -> int:
        """Return estimated seconds for *task_description*.

        The estimate is built from:
        1. Keyword / category matches (additive).
        2. File-count multiplier.
        3. A small base overhead (15 s) for any task.
        """
        text = task_description
        base = 15.0
        extra: dict[str, float] = {}

        for category, patterns in _WEIGHT_PATTERNS.items():
            if any(p.search(text) for p in patterns):
                w = self._weights.get(category, 0.0)
                extra[category] = w

        # If both bug_fix and bug_fix_complex match, drop the simpler one
        if "bug_fix_complex" in extra and "bug_fix" in extra:
            del extra["bug_fix"]

        # simple_edit dominance: if the task contains "typo", "tweak", "one-liner",
        # or "quick", treat it as trivial unless there are explicit complex markers.
        if "simple_edit" in extra:
            has_complex_marker = any(
                p.search(text)
                for p in _WEIGHT_PATTERNS.get("bug_fix_complex", [])
            )
            if not has_complex_marker:
                # Drop all non-simple categories; this is a trivial task
                extra = {"simple_edit": extra["simple_edit"]}
            else:
                # Complex markers present — keep simple_edit as a small bonus
                pass

        total = base + sum(extra.values())

        # File-count multiplier: each file beyond the first adds a fraction of the estimate
        file_count = _extract_file_count(text, context)
        if file_count > 1:
            # Additional files cost 40 % of the base estimate each, capped at 5×
            total *= min(1.0 + (file_count - 1) * 0.40, 5.0)

        # Round to nearest int, minimum 10 s
        return max(10, int(round(total)))

    def should_delegate(self, task_description: str, context: dict[str, Any] | None = None) -> bool:
        """True if the estimated duration exceeds the two-minute threshold."""
        return self.estimate_duration(task_description, context) > self._threshold

    def route(
        self,
        task_description: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a routing decision dict.

        Keys:
        - ``mode``: "direct" | "subagent" | "deferred"
        - ``reason``: human-readable rationale
        - ``estimated_seconds``: int
        - ``gateway_state``: current circuit-breaker state name
        """
        estimated = self.estimate_duration(task_description, context)
        gateway_ok, gateway_reason = self._gateway.can_dispatch()
        gateway_status = self._gateway.get_status()

        if estimated <= self._threshold:
            # Fast enough to do directly — no need to check gateway
            return {
                "mode": "direct",
                "reason": f"Estimated {estimated}s ≤ {int(self._threshold)}s threshold — do directly",
                "estimated_seconds": estimated,
                "gateway_state": gateway_status["state"],
            }

        # Task is slow — prefer subagent, but respect gateway
        if gateway_ok:
            return {
                "mode": "subagent",
                "reason": (
                    f"Estimated {estimated}s > {int(self._threshold)}s threshold; "
                    f"{gateway_reason}"
                ),
                "estimated_seconds": estimated,
                "gateway_state": gateway_status["state"],
            }

        # Gateway says no — defer rather than doing heavy work inline
        return {
            "mode": "deferred",
            "reason": (
                f"Estimated {estimated}s > {int(self._threshold)}s threshold, "
                f"but gateway blocks subagent: {gateway_reason}. "
                f"Queue for later or do directly if urgent."
            ),
            "estimated_seconds": estimated,
            "gateway_state": gateway_status["state"],
        }

    def record_actual(
        self,
        task_id: str,
        estimated: float,
        actual: float,
        task_description: str | None = None,
    ) -> dict[str, float]:
        """Learn from a completed task.

        Adjusts internal weights so that categories present in the task
        description move toward the observed actual/estimated ratio.

        Returns a dict of {category: old_weight → new_weight} for inspection.
        """
        with self._lock:
            ratio = actual / max(estimated, 1.0)
            # Clamp ratio to avoid runaway correction
            ratio = max(0.25, min(ratio, 4.0))

            # Which categories were active?
            categories: list[str] = []
            if task_description:
                for category, patterns in _WEIGHT_PATTERNS.items():
                    if any(p.search(task_description) for p in patterns):
                        categories.append(category)

            changes: dict[str, float] = {}
            for cat in categories:
                old = self._weights.get(cat, 0.0)
                # Move weight toward old * ratio, blended by learning_rate
                new = old + self._learning_rate * (old * ratio - old)
                new = max(5.0, new)  # floor at 5 s so we never zero-out a category
                self._weights[cat] = new
                changes[cat] = new

            self._feedback.append(
                _FeedbackEntry(
                    task_id=task_id,
                    estimated=estimated,
                    actual=actual,
                    keywords=categories,
                )
            )
            return changes

    def get_weights(self) -> dict[str, float]:
        """Return a snapshot of current heuristic weights. Thread-safe."""
        with self._lock:
            return dict(self._weights)

    def get_feedback_summary(self) -> dict[str, Any]:
        """Return aggregate learning statistics."""
        with self._lock:
            if not self._feedback:
                return {"count": 0, "mean_ratio": None, "median_ratio": None}
            ratios = [f.actual / max(f.estimated, 1.0) for f in self._feedback]
            return {
                "count": len(ratios),
                "mean_ratio": round(sum(ratios) / len(ratios), 3),
                "median_ratio": round(sorted(ratios)[len(ratios) // 2], 3),
            }

    @property
    def gateway(self) -> GatewayPacing:
        """The underlying circuit breaker (read-only reference)."""
        return self._gateway
