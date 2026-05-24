"""Constraint plugin for the superinstance-runtime event bus.

Wraps constraint-theory concepts (if available) or falls back to a
pure-Python bounds checker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from superinstance.runtime import CollectorPlugin, CompilerPlugin, SelectorPlugin


@dataclass(frozen=True, slots=True)
class ConstraintArtifact:
    """A single constraint observation."""

    source: str
    field: str
    value: float
    lower_bound: float
    upper_bound: float
    violated: bool


class ConstraintCollector(CollectorPlugin):
    """Collect constraint observations from a context dict."""

    name = "constraint"

    def collect(self, context: dict[str, Any]) -> list[ConstraintArtifact]:
        """Extract constraint observations from *context*.

        Expected context keys::

            constraints: list[dict] with keys:
                field, value, lower_bound, upper_bound
        """
        raw = context.get("constraints", [])
        artifacts: list[ConstraintArtifact] = []
        for item in raw:
            field = item.get("field", "unknown")
            value = float(item.get("value", 0.0))
            lo = float(item.get("lower_bound", float("-inf")))
            hi = float(item.get("upper_bound", float("inf")))
            artifacts.append(
                ConstraintArtifact(
                    source="context",
                    field=field,
                    value=value,
                    lower_bound=lo,
                    upper_bound=hi,
                    violated=not (lo <= value <= hi),
                )
            )
        return artifacts


class ConstraintSelector(SelectorPlugin):
    """Select only violated constraints (the ones that need action)."""

    name = "constraint"

    def select(
        self,
        artifacts: list[Any],
        context: dict[str, Any],
    ) -> list[ConstraintArtifact]:
        """Filter to violated constraints, sorted by severity."""
        violated = [a for a in artifacts if isinstance(a, ConstraintArtifact) and a.violated]
        # Severity = distance from nearest bound
        def severity(a: ConstraintArtifact) -> float:
            if a.value < a.lower_bound:
                return abs(a.lower_bound - a.value)
            return abs(a.value - a.upper_bound)
        return sorted(violated, key=severity, reverse=True)


class ConstraintCompiler(CompilerPlugin):
    """Compile violated constraints into correction directives."""

    name = "constraint"

    def compile(
        self,
        artifacts: list[Any],
        context: dict[str, Any],
    ) -> list[dict]:
        """Produce correction directives for each violated constraint."""
        directives: list[dict] = []
        for a in artifacts:
            if not isinstance(a, ConstraintArtifact):
                continue
            if a.value < a.lower_bound:
                action = "increase"
                target = a.lower_bound
            else:
                action = "decrease"
                target = a.upper_bound
            directives.append(
                {
                    "plugin": "constraint",
                    "field": a.field,
                    "action": action,
                    "delta": target - a.value,
                    "target": target,
                }
            )
        return directives
