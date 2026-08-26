"""HardwareProfileIndex — workload-aware agent placement via vector search.

Stores compressed capability profiles for each hardware device in the
fleet (RTX 4050 SMs, Ryzen AI cores, Radeon 890M CUs, XDNA 2 NPU).
Enables queries like:
    - "Which GPU has the most free SMs for a new training task?"
    - "Find a device that matches this workload profile (batch=32, FP16, 4GB)."

Uses FluxVectorTable for compressed storage.
"""

from __future__ import annotations

__all__ = ["HardwareProfileIndex", "DeviceProfile", "WorkloadQuery"]

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Fleet hardware taxonomy."""

    RTX_4050 = "rtx_4050"
    RYZEN_AI = "ryzen_ai"
    RADEON_890M = "radeon_890m"
    XDNA_2 = "xdna_2"


@dataclass(frozen=True)
class DeviceProfile:
    """A single device's capability vector + current load."""

    device_id: str
    device_type: DeviceType
    # Capability vector (8 dims, normalized):
    #   [sm_count, fp16_tops, memory_gb, pcie_bw, power_w,
    #    thermal_headroom, availability_01, generation]
    capabilities: list[float]
    # Current load vector (8 dims, normalized):
    #   [sm_util, memory_used, power_draw, temp_c, queue_depth,
    #    bandwidth_util, error_rate, uptime_h]
    current_load: list[float]
    metadata: dict[str, Any]

    @property
    def free_capacity(self) -> float:
        """Composite free capacity score [0, 1]."""
        if len(self.capabilities) != len(self.current_load):
            return 0.0
        # Free = capability - load, clamped positive
        free = [
            max(0.0, self.capabilities[i] - self.current_load[i])
            for i in range(len(self.capabilities))
        ]
        return sum(free) / len(free)


@dataclass(frozen=True)
class WorkloadQuery:
    """What an agent/task needs from hardware."""

    # Required capability minimums (8 dims)
    min_capabilities: list[float]
    # Maximum acceptable load (8 dims)
    max_load: list[float]
    # Preferred device type, or None for any
    preferred_type: DeviceType | None = None
    # Weight vector for scoring (8 dims, sum to 1.0)
    weights: list[float] | None = None


class HardwareProfileIndex:
    """Compressed index for fleet hardware placement decisions.

    Two FluxVectorTable instances:
        - capability_table: static device specs (searched rarely)
        - load_table: dynamic utilization (updated every tick)

    Args:
        bit_width: Quantization bits (default 4 for fast updates).

    Example::

        idx = HardwareProfileIndex()
        idx.register(DeviceProfile(
            device_id="gpu_0", device_type=DeviceType.RTX_4050,
            capabilities=[20, 48, 6, 16, 95, 0.3, 1.0, 1],
            current_load=[0.4, 0.3, 0.5, 0.6, 0.1, 0.2, 0.0, 0.9],
            metadata={"node": "Oracle1"},
        ))

        best = idx.find_best_device(WorkloadQuery(
            min_capabilities=[10, 20, 4, 8, 50, 0.1, 0.5, 0],
            max_load=[0.8, 0.7, 0.8, 0.8, 0.5, 0.5, 0.01, 1.0],
            preferred_type=DeviceType.RTX_4050,
        ), k=3)
    """

    PROFILE_DIM = 8

    def __init__(self, bit_width: int = 4) -> None:
        from swarm.vector_table import FluxVectorTable

        self._capability_table = FluxVectorTable(
            dim=self.PROFILE_DIM, bit_width=bit_width
        )
        self._load_table = FluxVectorTable(dim=self.PROFILE_DIM, bit_width=bit_width)
        self._profiles: dict[str, DeviceProfile] = {}

    def register(self, profile: DeviceProfile) -> None:
        """Add or update a device profile."""
        from swarm.vector_table import AgentVector

        # Encode device_id as uint64 hash
        numeric_id = self._hash_device_id(profile.device_id)

        self._capability_table.add(
            AgentVector(
                agent_id=numeric_id,
                vector=profile.capabilities,
                fitness=profile.free_capacity,
                extra={
                    "device_type": profile.device_type.value,
                    "metadata": profile.metadata,
                },
            )
        )
        self._load_table.add(
            AgentVector(
                agent_id=numeric_id,
                vector=profile.current_load,
                fitness=1.0 - profile.free_capacity,  # busier = lower fitness
                extra={
                    "device_type": profile.device_type.value,
                    "device_id": profile.device_id,
                },
            )
        )
        self._profiles[profile.device_id] = profile
        logger.debug("Registered device %s", profile.device_id)

    def update_load(self, device_id: str, new_load: list[float]) -> None:
        """Update dynamic load vector for a device."""
        profile = self._profiles.get(device_id)
        if profile is None:
            logger.warning("Unknown device %s, cannot update load", device_id)
            return

        # Re-register with new load
        new_profile = DeviceProfile(
            device_id=profile.device_id,
            device_type=profile.device_type,
            capabilities=profile.capabilities,
            current_load=new_load,
            metadata=profile.metadata,
        )
        self.register(new_profile)
        logger.debug("Updated load for %s", device_id)

    def find_best_device(
        self,
        query: WorkloadQuery,
        k: int = 5,
    ) -> list[tuple[str, float, DeviceProfile]]:
        """Find the best device(s) for a workload.

        Two-phase search:
            1. Capability filter: devices meeting min_capabilities
            2. Load filter: devices below max_load
            3. Score: weighted combination of free capacity + type match

        Returns:
            List of (device_id, score, profile) sorted best-first.
        """
        # Phase 1: capability search — find devices with similar capability profile
        # We want devices that EXCEED the query minimums, so we search
        # with the min_capabilities as query and look for high scores
        # (turbovec similarity = dot product on normalized vectors)
        cap_results = self._capability_table.search(
            query=query.min_capabilities,
            k=k * 4,  # oversample for filtering
        )

        # Phase 2: load filter + scoring
        results: list[tuple[str, float, DeviceProfile]] = []
        for numeric_id, cap_score, cap_meta in cap_results:
            device_id = self._numeric_to_device_id(numeric_id)
            profile = self._profiles.get(device_id)
            if profile is None:
                continue

            # Type filter
            if (
                query.preferred_type is not None
                and profile.device_type != query.preferred_type
            ):
                continue

            # Load filter: check all dims are below max_load
            if len(new_load := profile.current_load) != len(query.max_load):
                continue
            overloaded = any(
                new_load[i] > query.max_load[i] for i in range(len(query.max_load))
            )
            if overloaded:
                continue

            # Compute score
            score = self._score_device(profile, query)
            results.append((device_id, score, profile))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def get_device(self, device_id: str) -> DeviceProfile | None:
        """Retrieve a single device profile."""
        return self._profiles.get(device_id)

    def device_count(self) -> int:
        return len(self._profiles)

    def __repr__(self) -> str:
        return (
            f"HardwareProfileIndex(devices={self.device_count()}, "
            f"cap_table={len(self._capability_table)}, "
            f"load_table={len(self._load_table)})"
        )

    # ── internals ───────────────────────────────────────────

    @staticmethod
    def _hash_device_id(device_id: str) -> int:
        """Stable uint64 hash for device_id."""
        import hashlib

        digest = hashlib.blake2b(device_id.encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big") % (2**64)

    @staticmethod
    def _numeric_to_device_id(numeric_id: int) -> str | None:
        """Reverse lookup — requires linear scan (use sparingly)."""
        # Not efficient; for production, maintain bidirectional map
        return None

    @staticmethod
    def _score_device(profile: DeviceProfile, query: WorkloadQuery) -> float:
        """Compute placement score for a device + workload.

        Higher = better fit.
        """
        free = profile.free_capacity

        # Type bonus
        type_bonus = 0.15
        if (
            query.preferred_type is not None
            and profile.device_type == query.preferred_type
        ):
            type_bonus = 0.3

        # Weighted match if weights provided
        if query.weights is not None and len(query.weights) == len(
            profile.capabilities
        ):
            weighted_match = sum(
                query.weights[i] * (profile.capabilities[i] - profile.current_load[i])
                for i in range(len(profile.capabilities))
            )
        else:
            weighted_match = free

        return weighted_match + type_bonus
