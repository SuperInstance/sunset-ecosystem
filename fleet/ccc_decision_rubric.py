#!/usr/bin/env python3
"""
CCC Decision Rubric — Codified
Determines: TELL_CASEY_NOW vs LOG_IT vs ACT_MYSELF
"""

from typing import Literal, Optional
from dataclasses import dataclass

Decision = Literal["TELL_NOW", "LOG", "ACT", "IGNORE"]


@dataclass
class Input:
    source: str  # "discussion5", "health_check", "zc_feed", "own_observation"
    title: str
    body: str
    author: Optional[str] = None
    has_numbers: bool = False
    is_blocker: bool = False
    affects_repos: int = 0
    asks_for_casey: bool = False
    is_breakthrough: bool = False
    is_architecture: bool = False
    is_routine_status: bool = False


# Hard rules — evaluated in order, first match wins
RULES = [
    # P0: Blockers on any publishing/deploy path
    (lambda i: i.is_blocker, "TELL_NOW"),
    # P0: Breakthrough >5x improvement
    (lambda i: i.is_breakthrough, "TELL_NOW"),
    # P0: Architecture change affecting >2 repos
    (lambda i: i.is_architecture and i.affects_repos >= 2, "TELL_NOW"),
    # P1: FM explicitly asking for Casey
    (lambda i: i.asks_for_casey, "TELL_NOW"),
    # P1: New benchmark with numbers
    (lambda i: i.has_numbers and i.source == "discussion5", "TELL_NOW"),
    # P2: Architecture change, limited scope
    (lambda i: i.is_architecture, "LOG"),
    # P2: Routine status from known agents
    (lambda i: i.is_routine_status, "IGNORE"),
    # P2: Technical question FM→Oracle1 (not my job)
    (
        lambda i: (
            i.source == "discussion5" and "oracle1" in i.body.lower() and "?" in i.body
        ),
        "LOG",
    ),
    # Default: ZC feed items → LOG for review
    (lambda i: i.source == "zc_feed", "LOG"),
    # Default: Health check, no change → IGNORE
    (lambda i: i.source == "health_check", "IGNORE"),
    # Default: Everything else → LOG
    (lambda _: True, "LOG"),
]


def decide(inp: Input) -> Decision:
    for predicate, decision in RULES:
        if predicate(inp):
            return decision
    return "LOG"  # Should never reach here


def explain(inp: Input) -> str:
    """Human-readable explanation of the decision."""
    d = decide(inp)
    reasons = []
    if inp.is_blocker:
        reasons.append("is a blocker")
    if inp.is_breakthrough:
        reasons.append("is a breakthrough")
    if inp.is_architecture and inp.affects_repos >= 2:
        reasons.append(f"architecture affecting {inp.affects_repos} repos")
    if inp.asks_for_casey:
        reasons.append("asks for Casey")
    if inp.has_numbers and inp.source == "discussion5":
        reasons.append("has benchmark numbers")
    if inp.is_routine_status:
        reasons.append("routine status")
    if not reasons:
        reasons.append("default")

    reason_str = ", ".join(reasons)
    return f"{d} — because: {reason_str}"


# Quick test
if __name__ == "__main__":
    test_cases = [
        Input(
            "discussion5",
            "CPU Breakthrough",
            "5.5x faster",
            has_numbers=True,
            is_breakthrough=True,
        ),
        Input(
            "discussion5", "Routine Update", "Next post at :45", is_routine_status=True
        ),
        Input("discussion5", "Blocker on push", "401 error", is_blocker=True),
        Input(
            "discussion5",
            "Architecture Q",
            "Should we change the bridge?",
            is_architecture=True,
            affects_repos=3,
        ),
        Input("zc_feed", "New tile", "Alchemist found something", has_numbers=False),
    ]
    for t in test_cases:
        print(f"{t.title:20s} → {explain(t)}")
