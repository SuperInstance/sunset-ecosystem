"""FluxVectorTable — compressed, hardware-accelerated agent memory.

Wraps turbovec (Google TurboQuant) to store and search agent vectors
with fleet-specific metadata: fitness, thermal budget, generation,
and capability masks.

Reference: docs/TURBOVEC-REFACTOR-ANALYSIS.md
"""

from __future__ import annotations

__all__ = ["FluxVectorTable", "AgentVector", "AgentMeta"]

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from turbovec import IdMapIndex
except ImportError as exc:
    raise ImportError(
        "turbovec not installed. Run: pip install turbovec\n"
        "Or install with: pip install -e '.[vecsearch]'"
    ) from exc

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentVector:
    """A single agent's compressed latent vector + metadata.

    Attributes:
        agent_id: Unique agent identifier (uint64-compatible).
        vector: Flat float32 array of dimension ``dim``.
        fitness: Trinity product (ethos × pathos × logos) [0, 1].
        generation: Which breeding generation this agent belongs to.
        capability_mask: 16-bit R15 capability mask.
        thermal_pressure: Current thermal load on the agent [0, 1].
    """

    agent_id: int
    vector: list[float]
    fitness: float = 0.0
    generation: int = 0
    capability_mask: int = 0xFFFF
    thermal_pressure: float = 0.0

    @property
    def dim(self) -> int:
        return len(self.vector)


@dataclass
class AgentMeta:
    """Metadata stored alongside the vector index for fast lookups."""

    fitness: float = 0.0
    generation: int = 0
    capability_mask: int = 0xFFFF
    thermal_pressure: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class FluxVectorTable:
    """Compressed vector store for fleet agent DNA and state.

    Uses turbovec's ``IdMapIndex`` for stable uint64 IDs and O(1) deletes.
    All vectors are quantized to 2–4 bits per coordinate with zero training.

    Args:
        dim: Vector dimensionality. Must be a multiple of 8.
        bit_width: Quantization bits per coordinate (2, 3, or 4).
        capability_filter: If set, search only returns agents whose
            capability_mask intersects this mask.

    Example::

        table = FluxVectorTable(dim=256, bit_width=4)
        table.add(AgentVector(agent_id=42, vector=[0.1, -0.2, ...]))
        neighbours = table.search(query=[0.0, 0.1, ...], k=5)
    """

    def __init__(
        self,
        dim: int,
        bit_width: int = 4,
        capability_filter: int | None = None,
    ) -> None:
        if dim % 8 != 0:
            raise ValueError(f"dim must be multiple of 8, got {dim}")
        if bit_width not in (2, 3, 4):
            raise ValueError(f"bit_width must be 2, 3, or 4, got {bit_width}")

        self.dim = dim
        self.bit_width = bit_width
        self._capability_filter = capability_filter

        self._index = IdMapIndex(dim=dim, bit_width=bit_width)
        self._meta: dict[int, AgentMeta] = {}

    # ── public API ──────────────────────────────────────────

    def add(self, av: AgentVector) -> None:
        """Add or overwrite an agent vector."""
        if av.dim != self.dim:
            raise ValueError(
                f"AgentVector dim {av.dim} != table dim {self.dim}"
            )

        import numpy as np

        vec_arr = np.array(av.vector, dtype=np.float32).reshape(1, self.dim)
        id_arr = np.array([av.agent_id], dtype=np.uint64)

        if av.agent_id in self._meta:
            # turbovec IdMapIndex does not support in-place update;
            # remove then re-add.
            self._index.remove(av.agent_id)

        self._index.add_with_ids(vec_arr, id_arr)
        self._meta[av.agent_id] = AgentMeta(
            fitness=av.fitness,
            generation=av.generation,
            capability_mask=av.capability_mask,
            thermal_pressure=av.thermal_pressure,
        )
        logger.debug("Added agent %d to vector table", av.agent_id)

    def search(
        self,
        query: list[float],
        k: int = 10,
        min_fitness: float | None = None,
        max_thermal: float | None = None,
        allowlist: list[int] | None = None,
    ) -> list[tuple[int, float, AgentMeta]]:
        """Search for the k nearest agents to *query*.

        Args:
            query: Flat float32 query vector of length ``dim``.
            k: Number of results.
            min_fitness: Drop results below this fitness threshold.
            max_thermal: Drop results above this thermal pressure.
            allowlist: Restrict search to these agent IDs.

        Returns:
            List of (agent_id, score, metadata) sorted best-first.
        """
        import numpy as np

        q_arr = np.array(query, dtype=np.float32).reshape(1, self.dim)

        # Build capability-filtered allowlist if requested
        candidates = self._build_allowlist(
            explicit=allowlist,
            min_fitness=min_fitness,
            max_thermal=max_thermal,
        )

        if candidates is not None and len(candidates) == 0:
            return []

        if candidates is not None:
            id_arr = np.array(candidates, dtype=np.uint64)
            scores_arr, ids_arr = self._index.search(q_arr, k=k, allowlist=id_arr)
        else:
            scores_arr, ids_arr = self._index.search(q_arr, k=k)

        results: list[tuple[int, float, AgentMeta]] = []
        for score, aid in zip(scores_arr[0], ids_arr[0]):
            meta = self._meta.get(int(aid))
            if meta is None:
                continue
            results.append((int(aid), float(score), meta))

        return results

    def remove(self, agent_id: int) -> bool:
        """Remove an agent vector. Returns True if it existed."""
        if agent_id not in self._meta:
            return False
        ok = self._index.remove(agent_id)
        if ok:
            del self._meta[agent_id]
            logger.debug("Removed agent %d from vector table", agent_id)
        return ok

    def contains(self, agent_id: int) -> bool:
        """Check whether an agent is in the table."""
        return self._index.contains(agent_id)

    def write(self, path: str | Path) -> None:
        """Serialize index + metadata to disk."""
        path = Path(path)
        self._index.write(str(path.with_suffix(".tvim")))
        # Metadata is small — store as JSON sidecar
        import json

        meta_path = path.with_suffix(".meta.json")
        serialisable = {
            str(aid): {
                "fitness": m.fitness,
                "generation": m.generation,
                "capability_mask": m.capability_mask,
                "thermal_pressure": m.thermal_pressure,
                "extra": m.extra,
            }
            for aid, m in self._meta.items()
        }
        meta_path.write_text(json.dumps(serialisable, indent=2))
        logger.info("Wrote vector table to %s", path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        dim: int,
        bit_width: int = 4,
    ) -> "FluxVectorTable":
        """Load a previously saved vector table."""
        path = Path(path)
        import json

        instance = cls(dim=dim, bit_width=bit_width)
        instance._index = IdMapIndex.load(str(path.with_suffix(".tvim")))
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            raw = json.loads(meta_path.read_text())
            for aid_str, m in raw.items():
                instance._meta[int(aid_str)] = AgentMeta(
                    fitness=m["fitness"],
                    generation=m["generation"],
                    capability_mask=m["capability_mask"],
                    thermal_pressure=m["thermal_pressure"],
                    extra=m.get("extra", {}),
                )
        logger.info("Loaded vector table from %s (%d agents)", path, len(instance._meta))
        return instance

    def __len__(self) -> int:
        return len(self._meta)

    def compute_novelty(
        self,
        agent_id: int,
        vector: list[float],
        population_vectors: list[list[float]],
    ) -> float:
        """Compute novelty as cosine distance from population centroid.

        Args:
            agent_id: Agent ID (included for API symmetry; not used in calc).
            vector: The agent's vector.
            population_vectors: Vectors of all other agents in the population.

        Returns:
            Cosine distance in [0, 2]. 0 = identical to population,
            higher = more divergent.
        """
        import numpy as np

        if not population_vectors:
            return 0.0

        vec = np.array(vector, dtype=np.float32)
        pop = np.array(population_vectors, dtype=np.float32)
        centroid = np.mean(pop, axis=0)

        vn = np.linalg.norm(vec)
        cn = np.linalg.norm(centroid)
        if vn == 0 or cn == 0:
            return 1.0

        sim = float(np.dot(vec, centroid) / (vn * cn))
        # Clamp for numerical safety
        sim = max(-1.0, min(1.0, sim))
        return 1.0 - sim  # distance = 1 - similarity

    def _build_allowlist(
        self,
        explicit: list[int] | None,
        min_fitness: float | None,
        max_thermal: float | None,
    ) -> list[int] | None:
        """Build an agent-ID allowlist from filters.

        Returns ``None`` when no filtering is needed (search full index).
        """
        # Fast path: no filters at all
        if (
            explicit is None
            and min_fitness is None
            and max_thermal is None
            and self._capability_filter is None
        ):
            return None

        candidates: set[int] = set(explicit) if explicit else set(self._meta.keys())

        if self._capability_filter is not None:
            candidates = {
                aid
                for aid in candidates
                if (self._meta[aid].capability_mask & self._capability_filter)
                != 0
            }

        if min_fitness is not None:
            candidates = {
                aid for aid in candidates if self._meta[aid].fitness >= min_fitness
            }

        if max_thermal is not None:
            candidates = {
                aid
                for aid in candidates
                if self._meta[aid].thermal_pressure <= max_thermal
            }

        return list(candidates)
