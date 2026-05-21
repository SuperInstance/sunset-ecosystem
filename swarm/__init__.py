"""Swarm — The full running ecosystem."""

from .penrose import PenrosePosition, assign_positions, compute_overlap
from .broadcast import BroadcastMessage, BroadcastingChannel
from .swarm_runner import SwarmRunner, SwarmStatus
from .tournament import (
    AgentScore,
    TournamentMatch,
    TournamentRound,
    dominated_by,
    breed,
    sunset_candidates,
)
from .thermal import DeviceBudget, DeviceType, ThermalBudget

__all__ = [
    "PenrosePosition",
    "assign_positions",
    "compute_overlap",
    "BroadcastMessage",
    "BroadcastingChannel",
    "SwarmRunner",
    "SwarmStatus",
    "AgentScore",
    "TournamentMatch",
    "TournamentRound",
    "dominated_by",
    "breed",
    "sunset_candidates",
    "DeviceBudget",
    "DeviceType",
    "ThermalBudget",
]
