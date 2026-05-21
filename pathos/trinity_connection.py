"""Scores how connected a room (agent) is to PATHOS — the human interface.

Evaluates whether an agent actually serves human needs, produces useful output,
and reduces (rather than increases) cognitive load.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pathos.need_tracker import NeedState
from pathos.moment_scorer import MomentScore


@dataclass
class TrinityScore:
    """Score for how connected an agent is to the human interface.

    Attributes:
        solves_human_problems: Does the agent solve actual human problems? (0.0-1.0)
        directly_useful: Is the output directly useful, not just technically correct? (0.0-1.0)
        reduces_cognitive_load: Does it reduce mental burden on the human? (0.0-1.0)
        composite: Weighted composite score (0.0-1.0).
        reason: Human-readable explanation.
    """

    solves_human_problems: float = 0.0
    directly_useful: float = 0.0
    reduces_cognitive_load: float = 0.0
    composite: float = 0.0
    reason: str = ""

    def __repr__(self) -> str:
        return (
            f"TrinityScore(composite={self.composite:.2f}, "
            f"problems={self.solves_human_problems:.2f}, "
            f"useful={self.directly_useful:.2f}, "
            f"cognitive={self.reduces_cognitive_load:.2f})"
        )


class TrinityConnection:
    """Scores how connected an agent/room is to PATHOS (the human interface).

    Trinity asks three questions about any agent:
    1. Does it solve human problems?
    2. Is its output directly useful (not just technically correct)?
    3. Does it reduce or increase human cognitive load?

    An agent that scores high on trinity is one the human would miss if it
    disappeared. An agent that scores low is overhead — technically fine but
    disconnected from actual human benefit.

    Example::

        tc = TrinityConnection()
        score = tc.score(
            human_facing=True,
            resolves_needs=True,
            output_used_directly=True,
            requires_human_intervention=False,
        )
        print(f"Trinity score: {score.composite:.2f}")
    """

    _W_PROBLEMS = 0.40
    _W_USEFUL = 0.35
    _W_COGNITIVE = 0.25

    def score(
        self,
        human_facing: bool = False,
        resolves_needs: bool = False,
        output_used_directly: bool = False,
        requires_human_intervention: bool = True,
        adds_context_switches: bool = False,
        automates_away_work: bool = False,
        error_rate: float = 0.0,
        need_state: Optional[NeedState] = None,
        moment_score: Optional[MomentScore] = None,
    ) -> TrinityScore:
        """Score how connected an agent is to the human interface.

        Args:
            human_facing: Does the human directly interact with this agent?
            resolves_needs: Does it resolve actual human needs?
            output_used_directly: Is output used as-is (not just intermediate)?
            requires_human_intervention: Does the human need to babysit it?
            adds_context_switches: Does it force the human to switch tools/contexts?
            automates_away_work: Does it eliminate work the human would otherwise do?
            error_rate: Fraction of outputs that are wrong/unhelpful (0.0-1.0).
            need_state: Optional current need state for context.
            moment_score: Optional moment score for context.

        Returns:
            A TrinityScore with breakdown and composite.
        """
        reasons: list[str] = []

        # --- Solves Human Problems ---
        problems_score = 0.0
        if resolves_needs:
            problems_score += 0.6
            reasons.append("resolves human needs")
        if human_facing:
            problems_score += 0.2
            reasons.append("human-facing")
        if automates_away_work:
            problems_score += 0.2
            reasons.append("automates work away")
        if error_rate > 0.2:
            problems_score -= error_rate * 0.3
            reasons.append(f"high error rate ({error_rate:.0%})")
        problems_score = max(0.0, min(1.0, problems_score))

        # --- Directly Useful ---
        useful_score = 0.0
        if output_used_directly:
            useful_score += 0.7
            reasons.append("output used directly")
        if not requires_human_intervention:
            useful_score += 0.3
            reasons.append("no babysitting needed")
        else:
            useful_score -= 0.2
            reasons.append("requires human intervention")
        useful_score = max(0.0, min(1.0, useful_score))

        # --- Reduces Cognitive Load ---
        cognitive_score = 0.5  # baseline
        if not adds_context_switches:
            cognitive_score += 0.2
            reasons.append("no context switching")
        else:
            cognitive_score -= 0.3
            reasons.append("forces context switches")
        if automates_away_work:
            cognitive_score += 0.3
            reasons.append("reduces mental burden")
        if requires_human_intervention:
            cognitive_score -= 0.2
        cognitive_score = max(0.0, min(1.0, cognitive_score))

        # Boost from moment_score if available
        if moment_score is not None:
            boost = moment_score.moment_score * 0.1
            problems_score = min(1.0, problems_score + boost)

        # --- Composite ---
        composite = (
            self._W_PROBLEMS * problems_score
            + self._W_USEFUL * useful_score
            + self._W_COGNITIVE * cognitive_score
        )

        return TrinityScore(
            solves_human_problems=round(problems_score, 3),
            directly_useful=round(useful_score, 3),
            reduces_cognitive_load=round(cognitive_score, 3),
            composite=round(composite, 3),
            reason="; ".join(reasons),
        )

    def score_from_history(
        self,
        total_interactions: int,
        resolved_count: int,
        follow_up_count: int,
        human_initiated_count: int,
        avg_latency_s: float = 0.0,
    ) -> TrinityScore:
        """Score trinity connection from aggregate interaction history.

        Args:
            total_interactions: Total recorded interactions.
            resolved_count: Interactions that resolved a need.
            follow_up_count: Interactions requiring follow-up.
            human_initiated_count: Times the human actively sought the agent.
            avg_latency_s: Average response latency.

        Returns:
            A TrinityScore based on the aggregate data.
        """
        if total_interactions == 0:
            return TrinityScore(reason="no interactions recorded")

        resolve_rate = resolved_count / total_interactions
        follow_up_rate = follow_up_count / total_interactions if total_interactions > 0 else 0.0

        return self.score(
            human_facing=human_initiated_count > 0,
            resolves_needs=resolve_rate > 0.5,
            output_used_directly=resolve_rate > 0.7,
            requires_human_intervention=follow_up_rate > 0.3,
            automates_away_work=resolve_rate > 0.8 and avg_latency_s < 10,
            error_rate=follow_up_rate,
        )

    def __repr__(self) -> str:
        return "TrinityConnection()"
