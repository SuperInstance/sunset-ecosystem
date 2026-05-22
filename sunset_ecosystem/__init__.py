"""Sunset Ecosystem — Trinity-architecture agent lifecycle.

Agents are born parallel, compete for relevance across three rooms
(ethos, pathos, logos), sunset with dignity, and their compressed
wisdom seeds the next generation.
"""

__version__ = "0.1.0"

# --- Sunset (lifecycle engine) ---
from sunset.agent import Agent, AgentPhase
from sunset.sunset_documents import Epilogue, Onboarding, Summary
from sunset.seed_bank import SeedBank
from sunset.tensor_archive import TensorArchive
from sunset.trinity_scorer import trinity_score
from sunset.generation_runner import GenerationRunner

# --- Ethos (metal surveyor) ---
from ethos.hardware_survey import HardwareProfile, survey_hardware
from ethos.stress_test import StressReport, run_stress_test
from ethos.agent_allocator import AllocationPlan, AgentAllocation, build_allocation_plan

# --- Logos (code memory) ---
from logos.codebase_state import CodebaseState, survey_codebase
from logos.decision_log import DecisionRecord, DecisionRecords, DecisionLog
from logos.generation_memory import AgentGeneration, GenerationHistory, GenerationMemory

# --- Pathos (human interface) ---
from pathos.need_tracker import NeedState, NeedTracker
from pathos.moment_scorer import MomentScorer, MomentScore

# --- Swarm (full running ecosystem) ---
from swarm.swarm_runner import SwarmRunner, SwarmStatus
from swarm.tournament import AgentScore, TournamentMatch, breed, sunset_candidates
from swarm.breeder import Breeder, AgentLifecycle, spawn_from_template
from swarm.thermal import DeviceBudget, DeviceType, ThermalBudget

# --- Nexus (federated discovery) ---
from nexus.federation import FederatedNexus, RegistrationRecord, FederationEndpoint, NexusError, ConnectionRefusedError

# --- Nerve (micro-model sensory pathways) ---
from nerve.fiber import NerveFiber, FiberState, SensoryTile
from nerve.routing import RoutingLayer, Route, HebbianChannel
from nerve.adaptation import AdaptationEngine, ShoeTracker
from nerve.templates import AgentTemplate, TemplateRegistry, BUILTIN_TEMPLATES
from nerve.topology import NerveTopology, TickResult

# --- Grammar (signal grammar) ---
from grammar.core import Production, Rule, create_rule, score_rule, evolve

# --- Triage (repo health) ---
from triage.metrics import RepoHealthMetrics, HealthScore, run_health_check
from triage.github_issues import GitHubIssues, IssueState
from triage.weekly import WeeklyTriage, TriageReport, run_triage

__all__ = [
    # Version
    "__version__",
    # Sunset
    "Agent",
    "AgentPhase",
    "Epilogue",
    "Onboarding",
    "Summary",
    "SeedBank",
    "TensorArchive",
    "trinity_score",
    "GenerationRunner",
    # Ethos
    "HardwareProfile",
    "survey_hardware",
    "StressReport",
    "run_stress_test",
    "AllocationPlan",
    "AgentAllocation",
    "build_allocation_plan",
    # Logos
    "CodebaseState",
    "survey_codebase",
    "DecisionRecord",
    "DecisionRecords",
    "DecisionLog",
    "AgentGeneration",
    "GenerationHistory",
    "GenerationMemory",
    # Pathos
    "NeedState",
    "NeedTracker",
    "MomentScorer",
    "MomentScore",
    # Swarm
    "SwarmRunner",
    "SwarmStatus",
    "AgentScore",
    "TournamentMatch",
    "breed",
    "sunset_candidates",
    "Breeder",
    "AgentLifecycle",
    "spawn_from_template",
    "DeviceBudget",
    "DeviceType",
    "ThermalBudget",
    # Nexus
    "FederatedNexus",
    "RegistrationRecord",
    "FederationEndpoint",
    "NexusError",
    "ConnectionRefusedError",
    # Nerve
    "NerveFiber",
    "FiberState",
    "SensoryTile",
    "RoutingLayer",
    "Route",
    "HebbianChannel",
    "AdaptationEngine",
    "ShoeTracker",
    "AgentTemplate",
    "TemplateRegistry",
    "BUILTIN_TEMPLATES",
    "NerveTopology",
    "TickResult",
    # Grammar
    "Production",
    "Rule",
    "create_rule",
    "score_rule",
    "evolve",
    # Triage
    "RepoHealthMetrics",
    "HealthScore",
    "run_health_check",
    "GitHubIssues",
    "IssueState",
    "WeeklyTriage",
    "TriageReport",
    "run_triage",
]
