"""Scores whether an agent is serving the moment.

Evaluates latency, resolution quality, frustration impact, and invisibility
to produce a 0.0-1.0 score. Invisible and effective beats visible and impressive.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from pathos.need_tracker import NeedState, Urgency, Satisfaction


@dataclass
class MomentScore:
    """Score for how well an agent serves the current moment.

    Attributes:
        latency_score: How quickly the agent responded (0.0-1.0).
        resolution_score: Whether the problem actually got solved (0.0-1.0).
        frustration_penalty: Penalty for increasing human frustration (0.0-1.0).
        invisibility_bonus: Bonus for being helpful without being noticed (0.0-1.0).
        moment_score: Weighted composite score (0.0-1.0).
        reason: Human-readable explanation of the score.
    """

    latency_score: float = 0.0
    resolution_score: float = 0.0
    frustration_penalty: float = 0.0
    invisibility_bonus: float = 0.0
    moment_score: float = 0.0
    reason: str = ""

    def __repr__(self) -> str:
        return (
            f"MomentScore(total={self.moment_score:.2f}, "
            f"latency={self.latency_score:.2f}, "
            f"resolution={self.resolution_score:.2f}, "
            f"frustration_penalty={self.frustration_penalty:.2f}, "
            f"invisibility={self.invisibility_bonus:.2f}, "
            f"reason={self.reason!r})"
        )


class MomentScorer:
    """Scores whether an agent is serving the moment.

    Combines signals from the need state and interaction history to answer:
    - Is the human waiting? → latency score
    - Did the problem get solved? → resolution score
    - Is the human frustrated? → frustration penalty
    - Does the human even know this agent exists? → invisibility bonus

    Key principle: invisible + effective > visible + impressive.

    Example::

        scorer = MomentScorer()
        score = scorer.score(need_state, resolved=True, latency_s=2.5)
        print(f"Moment score: {score.moment_score:.2f}")
    """

    # Weights for composite score
    _W_LATENCY = 0.25
    _W_RESOLUTION = 0.35
    _W_FRUSTRATION = 0.25
    _W_INVISIBILITY = 0.15

    def score(
        self,
        need_state: NeedState,
        resolved: bool = False,
        latency_s: Optional[float] = None,
        human_aware_of_agent: bool = True,
        response_quality: float = 1.0,
    ) -> MomentScore:
        """Score how well the agent served the moment.

        Args:
            need_state: Current need state from NeedTracker.
            resolved: Whether the human's problem was actually solved.
            latency_s: Response latency in seconds (None = unknown).
            human_aware_of_agent: Whether the human explicitly knows about this agent.
            response_quality: Subjective quality of the response (0.0-1.0).

        Returns:
            A MomentScore with breakdown and composite score.
        """
        reasons: list[str] = []

        # --- Latency Score ---
        if latency_s is not None:
            if latency_s <= 1.0:
                latency_score = 1.0
                reasons.append("instant response")
            elif latency_s <= 5.0:
                latency_score = 0.8
                reasons.append("fast response")
            elif latency_s <= 15.0:
                latency_score = 0.5
                reasons.append("acceptable latency")
            elif latency_s <= 60.0:
                latency_score = 0.3
                reasons.append("slow response")
            else:
                latency_score = 0.1
                reasons.append("human was waiting too long")
        else:
            # Use wait_time from need state as proxy
            wait = need_state.wait_time_seconds
            if wait <= 5.0:
                latency_score = 0.9
            elif wait <= 30.0:
                latency_score = 0.6
            elif wait <= 120.0:
                latency_score = 0.3
            else:
                latency_score = 0.1
            reasons.append(f"wait proxy: {wait:.0f}s")

        # --- Resolution Score ---
        if resolved:
            resolution_score = min(1.0, response_quality)
            reasons.append("problem resolved")
        else:
            resolution_score = 0.0
            if need_state.satisfaction == Satisfaction.POSITIVE:
                # Human seems happy even though formally unresolved
                resolution_score = 0.5
                reasons.append("human satisfied despite partial resolution")
            else:
                reasons.append("problem NOT resolved")

        # --- Frustration Penalty ---
        frustration_penalty = need_state.frustration
        if frustration_penalty > 0.5:
            reasons.append("high frustration detected")
        elif frustration_penalty > 0.2:
            reasons.append("moderate frustration")

        # --- Invisibility Bonus ---
        # Being invisible (not requiring human awareness) is GOOD
        if not human_aware_of_agent:
            invisibility_bonus = 1.0
            reasons.append("agent invisible to human (good)")
        else:
            invisibility_bonus = 0.3
            reasons.append("human aware of agent (costs cognitive load)")

        # --- Urgency multiplier ---
        urgency_multiplier = {
            Urgency.CRITICAL: 1.0,
            Urgency.HIGH: 0.9,
            Urgency.MEDIUM: 0.8,
            Urgency.LOW: 0.7,
        }.get(need_state.urgency, 0.7)

        # --- Composite Score ---
        raw = (
            self._W_LATENCY * latency_score
            + self._W_RESOLUTION * resolution_score
            - self._W_FRUSTRATION * frustration_penalty
            + self._W_INVISIBILITY * invisibility_bonus
        )
        # Clamp to [0.0, 1.0]
        composite = max(0.0, min(1.0, raw * urgency_multiplier))

        return MomentScore(
            latency_score=round(latency_score, 3),
            resolution_score=round(resolution_score, 3),
            frustration_penalty=round(frustration_penalty, 3),
            invisibility_bonus=round(invisibility_bonus, 3),
            moment_score=round(composite, 3),
            reason="; ".join(reasons),
        )

    def score_interactions(
        self,
        need_state: NeedState,
        total_interactions: int,
        resolved_count: int,
        avg_latency_s: float,
        human_aware: bool = True,
    ) -> MomentScore:
        """Score based on aggregate interaction stats.

        Convenience method for scoring from summary data rather than
        a single interaction.

        Args:
            need_state: Current need state.
            total_interactions: Total interactions seen.
            resolved_count: How many were resolved.
            avg_latency_s: Average response latency.
            human_aware: Whether the human knows about the agent.

        Returns:
            A MomentScore for the aggregate.
        """
        resolved = resolved_count > 0 and total_interactions > 0
        resolution_rate = (
            resolved_count / total_interactions if total_interactions > 0 else 0.0
        )

        return self.score(
            need_state=need_state,
            resolved=resolved,
            latency_s=avg_latency_s,
            human_aware_of_agent=human_aware,
            response_quality=resolution_rate,
        )

    def __repr__(self) -> str:
        return "MomentScorer()"
