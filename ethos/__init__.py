"""ETHOS room — the metal surveyor.

Knows the hardware. The geographic starting point of the Sunset ecosystem.
Surveys compute, benchmarks devices, allocates agents to hardware, and scores
how well any room's work aligns with the physical metal beneath it.
"""

from ethos.hardware_survey import HardwareProfile, survey_hardware
from ethos.stress_test import StressReport, run_stress_test
from ethos.agent_allocator import AllocationPlan, AgentAllocation, build_allocation_plan
from ethos.trinity_connection import score_ethos_connection

__all__ = [
    "HardwareProfile",
    "survey_hardware",
    "StressReport",
    "run_stress_test",
    "AllocationPlan",
    "AgentAllocation",
    "build_allocation_plan",
    "score_ethos_connection",
]

__version__ = "0.1.0"
