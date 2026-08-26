"""Intent Confirmation Protocol — Disambiguates human intent before fleet-wide actions."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

__all__ = [
    "Intent",
    "FleetState",
    "IntentConfirmationProtocol",
]


# ── destructive actions ─────────────────────────────────────

DESTRUCTIVE_ACTIONS: Set[str] = {
    "sunset",
    "kill",
    "terminate",
    "destroy",
    "purge",
    "delete",
}


# ── FleetState ──────────────────────────────────────────────


@dataclass
class FleetState:
    """Lightweight snapshot of fleet status for intent disambiguation."""

    total_agents: int = 0
    active_agents: int = 0
    rooms: List[str] = field(default_factory=list)
    avg_fitness: float = 0.0
    top_fitness_threshold: float = 0.5

    def agents_above_fitness(self, threshold: float) -> int:
        """Return approximate count of agents above fitness threshold."""
        # Simplified: assume uniform distribution for quick estimate
        if self.total_agents == 0 or self.avg_fitness == 0:
            return 0
        # Heuristic: if avg is 0.5 and threshold is 0.5, ~half are above
        ratio = max(0.0, min(1.0, self.avg_fitness / max(threshold, 0.01)))
        return int(self.total_agents * ratio)


# ── Intent ──────────────────────────────────────────────────


@dataclass
class Intent:
    """Structured human intent parsed from a natural-language command."""

    action: str  # 'optimize', 'sunset', 'breed', 'migrate', 'query'
    target: str  # what the action applies to
    scope: str  # 'all', 'room:Tide-Pool', 'fitness>0.5', 'unspecified'
    urgency: str  # 'low', 'normal', 'high', 'critical'
    raw_command: str  # original human input

    # Populated after confirmation
    confirmed: bool = False
    resolved_scope: str = ""

    def is_destructive(self) -> bool:
        """True if this intent represents a destructive action."""
        return self.action in DESTRUCTIVE_ACTIONS


# ── IntentConfirmationProtocol ──────────────────────────────


class IntentConfirmationProtocol:
    """Disambiguates human intent before fleet-wide actions.

    When Casey says "make it faster", the fleet should ask
    "which agents?" rather than blindly optimizing everything.
    """

    AMBIGUITY_THRESHOLD: float = 0.7

    # Keyword → action mapping
    _ACTION_MAP: Dict[str, str] = {
        "optimize": "optimize",
        "faster": "optimize",
        "speed": "optimize",
        "tune": "optimize",
        "sunset": "sunset",
        "kill": "sunset",
        "terminate": "sunset",
        "breed": "breed",
        "spawn": "breed",
        "grow": "breed",
        "migrate": "migrate",
        "move": "migrate",
        "query": "query",
        "check": "query",
        "status": "query",
    }

    # Scope patterns
    _SCOPE_PATTERNS: List[tuple] = [
        (re.compile(r"\bagent\s+(\d+)\b", re.I), lambda m: f"agent:{m.group(1)}"),
        (re.compile(r"\broom\s+([\w\-]+)\b", re.I), lambda m: f"room:{m.group(1)}"),
        (re.compile(r"\btop\s+(\d+)\b", re.I), lambda m: f"top:{m.group(1)}"),
        (
            re.compile(r"\bfitness\s*([><=]+)\s*([\d.]+)\b", re.I),
            lambda m: f"fitness{m.group(1)}{m.group(2)}",
        ),
        (re.compile(r"\ball\b", re.I), lambda _m: "all"),
        (re.compile(r"\bevery\b", re.I), lambda _m: "all"),
        (re.compile(r"\bglobal\b", re.I), lambda _m: "all"),
    ]

    def __init__(self, fleet_state: FleetState) -> None:
        self.fleet_state = fleet_state

    # ── parsing ───────────────────────────────────────────

    def parse_intent(self, raw_command: str) -> Intent:
        """Parse a natural language command into structured intent.

        Example: "make it faster" → {
            action: 'optimize',
            target: 'performance',
            scope: 'unspecified',
            urgency: 'normal'
        }
        """
        lower = raw_command.lower()

        # Determine action
        action = "query"
        for keyword, mapped in self._ACTION_MAP.items():
            if keyword in lower:
                action = mapped
                break

        # Determine target
        target = self._extract_target(lower, action)

        # Determine scope
        scope = self._extract_scope(raw_command)

        # Determine urgency
        urgency = self._extract_urgency(lower)

        return Intent(
            action=action,
            target=target,
            scope=scope,
            urgency=urgency,
            raw_command=raw_command,
        )

    def _extract_target(self, lower: str, action: str) -> str:
        """Heuristic: map action to likely target."""
        if action == "optimize":
            if "speed" in lower or "fast" in lower:
                return "performance"
            if "memory" in lower or "ram" in lower:
                return "memory"
            return "performance"
        if action == "sunset":
            return "agents"
        if action == "breed":
            return "agents"
        if action == "migrate":
            return "agents"
        return "system"

    def _extract_scope(self, raw_command: str) -> str:
        """Extract scope from command text."""
        for pattern, extractor in self._SCOPE_PATTERNS:
            m = pattern.search(raw_command)
            if m:
                return extractor(m)
        return "unspecified"

    def _extract_urgency(self, lower: str) -> str:
        """Extract urgency level from command text."""
        if any(w in lower for w in ("critical", "emergency", "now", "immediately")):
            return "critical"
        if any(w in lower for w in ("high", "urgent", "asap", "quickly")):
            return "high"
        if any(w in lower for w in ("low", "whenever", "eventually", "later")):
            return "low"
        return "normal"

    # ── ambiguity ─────────────────────────────────────────

    def measure_ambiguity(self, intent: Intent) -> float:
        """Return 0-1 ambiguity score. High = needs confirmation.

        Factors:
        - scope unspecified → +0.5
        - scope = 'all' → +0.3
        - no explicit target → +0.2
        - urgency critical → +0.1
        """
        score = 0.0

        if intent.scope == "unspecified":
            score += 0.75
        elif intent.scope == "all":
            score += 0.3
        elif intent.scope.startswith("top:") or intent.scope.startswith("fitness"):
            score += 0.35

        if intent.target == "system":
            score += 0.2

        if intent.urgency == "critical":
            score += 0.1

        # Action-specific modifiers
        if intent.action == "sunset" and intent.scope == "all":
            score = 1.0  # maximally ambiguous AND destructive

        return min(1.0, score)

    # ── confirmation generation ───────────────────────────

    def generate_confirmation(self, intent: Intent) -> str:
        """Generate a human-readable confirmation prompt.

        Example: 'make it faster' with scope=unspecified →
        "You said 'make it faster.' There are 1,247 active agents.
         Do you want to optimize:
         (a) All agents globally
         (b) Only agents in the Tide-Pool room
         (c) Agents with fitness > 0.5
         (d) Something else?"
        """
        lines: List[str] = []
        lines.append(f'You said "{intent.raw_command}."')

        if self.fleet_state.total_agents > 0:
            lines.append(
                f"There are {self.fleet_state.total_agents:,} total agents "
                f"({self.fleet_state.active_agents:,} active)."
            )

        lines.append(f"Do you want to {intent.action}:")

        options: List[str] = []
        options.append("(a) All agents globally")

        if self.fleet_state.rooms:
            options.append(f"(b) Only agents in the {self.fleet_state.rooms[0]} room")
        else:
            options.append("(b) Only agents in a specific room")

        above = self.fleet_state.agents_above_fitness(
            self.fleet_state.top_fitness_threshold
        )
        if above > 0:
            options.append(
                f"(c) Agents with fitness > {self.fleet_state.top_fitness_threshold} "
                f"(~{above} agents)"
            )
        else:
            options.append(
                f"(c) Agents with fitness > {self.fleet_state.top_fitness_threshold}"
            )

        options.append("(d) Something else?")

        lines.extend(options)

        if intent.is_destructive():
            lines.append("\n⚠️  This is a DESTRUCTIVE action. Confirm carefully.")

        return "\n".join(lines)

    # ── confirmation requirement ────────────────────────────

    def require_confirmation(self, intent: Intent) -> bool:
        """True if ambiguity >= threshold OR action is destructive and broadly scoped."""
        ambiguity = self.measure_ambiguity(intent)
        if ambiguity >= self.AMBIGUITY_THRESHOLD:
            return True
        # Destructive actions require confirmation only when broadly scoped
        if intent.is_destructive() and intent.scope in ("all", "unspecified"):
            return True
        return False

    # ── logging ───────────────────────────────────────────

    def log_decision(
        self,
        intent: Intent,
        confirmed: bool,
        scope: str,
        journal: Optional[Any] = None,
        journal_path: Optional[str] = None,
    ) -> None:
        """Log to Decision Journal (FLAME format).

        If a journal object with a .record() method is provided, it is used.
        If *journal_path* is provided, writes to the daily JSONL journal via
        ``log_human_command``.
        """
        if journal is not None and hasattr(journal, "record"):
            journal.record(
                timestamp=time.time(),
                why=intent.raw_command,
                what=f"{intent.action} → {scope}",
                expected="fleet-wide action" if scope == "all" else "scoped action",
                actual="" if not confirmed else "pending",
                confidence=1.0 - self.measure_ambiguity(intent),
                scope=scope,
            )
        if journal_path is not None:
            from logos.decision_journal import log_human_command

            log_human_command(
                intent=intent,
                confirmed=confirmed,
                scope=scope,
                journal_path=journal_path,
            )
