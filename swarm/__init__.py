"""Swarm-level modules for agent breeding, constraint theory, and vector operations."""

from __future__ import annotations

from swarm.constraint_bridge import ConstraintBridge, SnapResult
from swarm.constraint_theory_integration import ConstraintTheoryIntegration
from swarm.eisenstein_integration import E12, HexDisk, eisenstein_norm, snap_from_angle
from swarm.fleet_bft_qd import (
    FleetBFTNetwork,
    FleetBreederConsensus,
    PBFTNode,
    QDArchive,
)

__all__ = [
    "ConstraintBridge",
    "SnapResult",
    "ConstraintTheoryIntegration",
    "E12",
    "HexDisk",
    "eisenstein_norm",
    "snap_from_angle",
    "FleetBFTNetwork",
    "FleetBreederConsensus",
    "PBFTNode",
    "QDArchive",
]
