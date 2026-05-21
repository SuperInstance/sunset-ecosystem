"""Score how connected work is to the living code memory (Trinity connection)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from logos.codebase_state import CodebaseState, survey_codebase
from logos.decision_log import DecisionLog, DecisionRecords
from logos.generation_memory import GenerationMemory, GenerationHistory


__all__ = ["TrinityConnection", "score_trinity_connection"]


@dataclass
class TrinityConnection:
    """A score (0.0–1.0) of how connected an agent/room is to logos.

    Evaluates:
    - Codebase understanding (does the agent know the code?)
    - Integration quality (does work fit cleanly?)
    - Maintainability (can developers understand the output?)
    """

    overall: float = 0.0
    codebase_understanding: float = 0.0
    integration_quality: float = 0.0
    maintainability: float = 0.0
    details: Dict[str, str] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"TrinityConnection(overall={self.overall:.2f}, "
            f"understanding={self.codebase_understanding:.2f}, "
            f"integration={self.integration_quality:.2f}, "
            f"maintainability={self.maintainability:.2f})"
        )


def _score_codebase_understanding(
    state: CodebaseState,
    decisions: Optional[DecisionRecords] = None,
    history: Optional[GenerationHistory] = None,
) -> tuple:
    """Score how well the codebase is understood (0.0–1.0)."""
    score = 0.0
    details: Dict[str, str] = {}

    # Has files and structure
    if state.file_count > 0:
        score += 0.15
        details["has_files"] = f"{state.file_count} files surveyed"
    else:
        details["has_files"] = "No files found"

    # Language diversity — knowing 2+ languages suggests broader understanding
    lang_count = len(state.language_breakdown)
    if lang_count >= 3:
        score += 0.15
    elif lang_count >= 1:
        score += 0.08
    details["languages"] = f"{lang_count} language(s): {', '.join(state.language_breakdown.keys())}"

    # Architecture patterns detected
    mod_count = len(state.architecture_patterns.get("module_dirs", []))
    if mod_count >= 3:
        score += 0.15
    elif mod_count >= 1:
        score += 0.08
    details["modules"] = f"{mod_count} module(s) detected"

    # Decision history exists
    if decisions and decisions.total > 0:
        score += min(0.15, decisions.total * 0.03)
        details["decisions"] = f"{decisions.total} decision(s) recorded"
    else:
        details["decisions"] = "No decisions recorded"

    # Generation memory exists
    if history and history.total_generations > 0:
        score += min(0.15, history.total_generations * 0.03)
        details["generations"] = f"{history.total_generations} generation(s) tracked"
    else:
        details["generations"] = "No generation history"

    # Test awareness
    if state.test_count > 0:
        score += 0.10
        details["tests"] = f"{state.test_count} test(s) collected"
    else:
        details["tests"] = "No tests found"

    # Git history awareness
    if state.recent_commits:
        score += 0.15
        details["git"] = f"{len(state.recent_commits)} recent commit(s)"
    else:
        details["git"] = "No git history"

    return min(1.0, score), details


def _score_integration_quality(
    state: CodebaseState,
    decisions: Optional[DecisionRecords] = None,
) -> tuple:
    """Score integration quality (0.0–1.0)."""
    score = 0.0
    details: Dict[str, str] = {}

    # Low tech debt → cleaner integration
    total_debt = sum(len(v) for v in state.technical_debt.values())
    if total_debt == 0:
        score += 0.30
        details["debt"] = "No technical debt markers found"
    elif total_debt < 10:
        score += 0.20
        details["debt"] = f"Low debt: {total_debt} marker(s)"
    elif total_debt < 50:
        score += 0.10
        details["debt"] = f"Moderate debt: {total_debt} marker(s)"
    else:
        details["debt"] = f"High debt: {total_debt} marker(s)"

    # Consistent module structure
    modules = state.architecture_patterns.get("module_dirs", [])
    if modules:
        score += 0.20
        details["structure"] = "Module structure detected"
    else:
        details["structure"] = "No clear module structure"

    # Entry points exist
    entries = state.architecture_patterns.get("entry_points", [])
    if entries:
        score += 0.20
        details["entry_points"] = f"Entry point(s): {', '.join(entries)}"
    else:
        details["entry_points"] = "No entry points detected"

    # Imported packages suggest integration awareness
    imports = state.architecture_patterns.get("imported_packages", [])
    if len(imports) >= 5:
        score += 0.15
    elif imports:
        score += 0.08
    details["imports"] = f"{len(imports)} package(s) imported"

    # Recent activity (commits mean living codebase)
    if len(state.recent_commits) >= 5:
        score += 0.15
    elif state.recent_commits:
        score += 0.08
    details["activity"] = f"{len(state.recent_commits)} recent commit(s)"

    return min(1.0, score), details


def _score_maintainability(state: CodebaseState) -> tuple:
    """Score maintainability (0.0–1.0)."""
    score = 0.0
    details: Dict[str, str] = {}

    # Test coverage proxy
    if state.test_count > 0:
        ratio = state.test_count / max(state.file_count, 1)
        if ratio > 0.5:
            score += 0.30
        elif ratio > 0.1:
            score += 0.20
        else:
            score += 0.10
        details["test_ratio"] = f"~{ratio:.1%} test-to-file ratio"
    else:
        details["test_ratio"] = "No tests — maintainability at risk"

    # Codebase size — moderate is most maintainable
    if state.total_lines < 1000:
        score += 0.20
        details["size"] = "Small codebase — easy to maintain"
    elif state.total_lines < 10000:
        score += 0.25
        details["size"] = "Medium codebase — good balance"
    elif state.total_lines < 100000:
        score += 0.15
        details["size"] = "Large codebase — needs discipline"
    else:
        score += 0.05
        details["size"] = "Very large — high maintenance burden"

    # Low debt helps maintainability
    debt_items = sum(len(v) for v in state.technical_debt.values())
    hack_count = len(state.technical_debt.get("HACK", []))
    if hack_count == 0:
        score += 0.20
    elif hack_count < 3:
        score += 0.10
    details["hacks"] = f"{hack_count} HACK(s) found"

    # Module organization
    modules = state.architecture_patterns.get("module_dirs", [])
    if 2 <= len(modules) <= 20:
        score += 0.20
    elif modules:
        score += 0.10
    details["organization"] = f"{len(modules)} module(s)"

    # Recent commits → active maintenance
    if state.recent_commits:
        score += 0.10
        details["maintenance"] = "Recently active"
    else:
        details["maintenance"] = "No recent activity"

    return min(1.0, score), details


def _generate_recommendations(
    understanding: float,
    integration: float,
    maintainability: float,
    state: CodebaseState,
) -> List[str]:
    """Generate improvement recommendations based on scores."""
    recs: List[str] = []

    if understanding < 0.5:
        if not state.recent_commits:
            recs.append("Initialize git to track codebase history")
        if not state.architecture_patterns.get("module_dirs"):
            recs.append("Add module structure (__init__.py) for better architecture detection")

    if integration < 0.5:
        total_debt = sum(len(v) for v in state.technical_debt.values())
        if total_debt > 20:
            recs.append(f"Address {total_debt} technical debt markers before major changes")

    if maintainability < 0.5:
        if state.test_count == 0:
            recs.append("Add tests to improve maintainability")
        hack_count = len(state.technical_debt.get("HACK", []))
        if hack_count > 3:
            recs.append(f"Resolve {hack_count} HACK comments for cleaner code")

    if not recs:
        recs.append("Codebase is in good shape — keep it up")

    return recs


def score_trinity_connection(
    root: Optional[str] = None,
    decision_log: Optional[DecisionLog] = None,
    generation_memory: Optional[GenerationMemory] = None,
    codebase_state: Optional[CodebaseState] = None,
) -> TrinityConnection:
    """Score how connected a room/agent is to logos (code memory).

    Args:
        root: Path to codebase. Defaults to cwd.
        decision_log: Optional DecisionLog instance.
        generation_memory: Optional GenerationMemory instance.
        codebase_state: Pre-computed CodebaseState (avoids re-survey).

    Returns:
        TrinityConnection with scores 0.0–1.0.
    """
    state = codebase_state or survey_codebase(root)

    decisions: Optional[DecisionRecords] = None
    if decision_log:
        decisions = decision_log.all_records()

    history: Optional[GenerationHistory] = None
    if generation_memory:
        history = generation_memory.get_history()

    u_score, u_details = _score_codebase_understanding(state, decisions, history)
    i_score, i_details = _score_integration_quality(state, decisions)
    m_score, m_details = _score_maintainability(state)

    overall = (u_score * 0.40) + (i_score * 0.30) + (m_score * 0.30)

    all_details = {**u_details, **i_details, **m_details}
    recs = _generate_recommendations(u_score, i_score, m_score, state)

    return TrinityConnection(
        overall=round(overall, 2),
        codebase_understanding=round(u_score, 2),
        integration_quality=round(i_score, 2),
        maintainability=round(m_score, 2),
        details=all_details,
        recommendations=recs,
    )
