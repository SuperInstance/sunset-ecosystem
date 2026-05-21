"""PATHOS — The human interface room.

Pathos tracks human needs, scores interactions, and ensures every agent
in the ecosystem serves the moment, not the system. A pathos agent would
rather be INVISIBLE and EFFECTIVE than VISIBLE and IMPRESSIVE.
"""

from pathos.need_tracker import NeedState, NeedTracker
from pathos.interaction_log import InteractionRecord, InteractionLog
from pathos.moment_scorer import MomentScorer, MomentScore
from pathos.trinity_connection import TrinityConnection

__all__ = [
    "NeedState",
    "NeedTracker",
    "InteractionRecord",
    "InteractionLog",
    "MomentScorer",
    "MomentScore",
    "TrinityConnection",
]
