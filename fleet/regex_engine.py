r"""Regex-based rule engine for fleet policy matching.

Matches structured rules against input strings with named capture groups,
rule priorities, and allow/deny actions. Used for fleet access control,
message filtering, and routing decisions.

Usage:
    engine = RegexRuleEngine()
    engine.add_rule("allow", r"^room\.(?P<room>\w+)\.trap$", priority=10)
    engine.add_rule("deny", r"^admin\..*", priority=100)
    result = engine.match("room.alpha.trap")
    # result.action == "allow", result.groups == {"room": "alpha"}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MatchResult:
    """Result of a rule match."""

    matched: bool
    action: str = ""
    rule_name: str = ""
    groups: Dict[str, str] = field(default_factory=dict)


@dataclass
class Rule:
    """A regex rule."""

    name: str
    action: str
    pattern: re.Pattern
    priority: int


class RegexRuleEngine:
    """
    Priority-ordered regex rule engine.

    Rules are evaluated highest-priority first. First match wins.
    """

    def __init__(self, default_action: str = "deny"):
        self._default_action = default_action
        self._rules: List[Rule] = []

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def add_rule(
        self,
        action: str,
        pattern: str,
        name: str = "",
        priority: int = 0,
    ) -> None:
        """Add a rule. Higher priority rules match first."""
        rule = Rule(
            name=name or f"rule-{len(self._rules)}",
            action=action,
            pattern=re.compile(pattern),
            priority=priority,
        )
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def remove_rule(self, name: str) -> bool:
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.name != name]
        return len(self._rules) < before

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match(self, text: str) -> MatchResult:
        """Evaluate text against all rules. Returns first match."""
        for rule in self._rules:
            m = rule.pattern.match(text)
            if m:
                return MatchResult(
                    matched=True,
                    action=rule.action,
                    rule_name=rule.name,
                    groups=m.groupdict(),
                )
        return MatchResult(
            matched=False,
            action=self._default_action,
        )

    def match_all(self, text: str) -> List[MatchResult]:
        """Evaluate text against all rules. Returns all matches."""
        results: List[MatchResult] = []
        for rule in self._rules:
            m = rule.pattern.match(text)
            if m:
                results.append(
                    MatchResult(
                        matched=True,
                        action=rule.action,
                        rule_name=rule.name,
                        groups=m.groupdict(),
                    )
                )
        if not results:
            results.append(
                MatchResult(
                    matched=False,
                    action=self._default_action,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def rules(self) -> List[Rule]:
        return list(self._rules)

    def rule_count(self) -> int:
        return len(self._rules)

    def __repr__(self) -> str:
        return f"<RegexRuleEngine rules={len(self._rules)}>"
