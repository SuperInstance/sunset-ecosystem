"""SUNSET — the lifecycle engine where every agent lives one generation."""

from sunset.agent import Agent, AgentPhase
from sunset.sunset_documents import Epilogue, Onboarding, Summary
from sunset.seed_bank import SeedBank
from sunset.tensor_archive import TensorArchive
from sunset.trinity_scorer import trinity_score
from sunset.generation_runner import GenerationRunner, GenerationReport

__all__ = [
    "Agent",
    "AgentPhase",
    "Epilogue",
    "Onboarding",
    "Summary",
    "SeedBank",
    "TensorArchive",
    "trinity_score",
    "GenerationRunner",
    "GenerationReport",
]
