"""LOGOS — the code memory room.

Logos agents grew up with the development team across generations.
They know WHY decisions were made.

This package surveys codebases, records architectural decisions,
tracks agent generations, and measures how connected work is to
the living code memory.
"""

from logos.codebase_state import CodebaseState, survey_codebase
from logos.decision_log import DecisionRecord, DecisionRecords, DecisionLog
from logos.generation_memory import AgentGeneration, GenerationHistory, GenerationMemory
from logos.trinity_connection import TrinityConnection, score_trinity_connection

__all__ = [
    "CodebaseState",
    "survey_codebase",
    "DecisionRecord",
    "DecisionRecords",
    "DecisionLog",
    "AgentGeneration",
    "GenerationHistory",
    "GenerationMemory",
    "TrinityConnection",
    "score_trinity_connection",
]
