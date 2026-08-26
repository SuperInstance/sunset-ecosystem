"""Fleet Search API — unified query interface across all memory layers.

Exposes a single search function that routes queries to the appropriate
backend based on intent:
    - "What do we know about X?" → KnowledgePipeline
    - "Which device can handle Y?" → HardwareProfileIndex
    - "What will room Z look like tomorrow?" → JepaGridMemory

Usage::

    from swarm.search_api import FleetSearch

    search = FleetSearch(knowledge=knowledge_pipeline, hardware=hw_index)
    results = search.ask("Where should I run the distillation task?")
"""

from __future__ import annotations

__all__ = ["FleetSearch", "SearchResult", "SearchIntent"]

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SearchIntent(Enum):
    """Auto-detected query intent."""

    KNOWLEDGE = "knowledge"  # Factual, what/why/how
    HARDWARE = "hardware"  # Device placement, capacity
    TEMPORAL = "temporal"  # Prediction, trends, forecasting
    AGENT = "agent"  # Agent DNA, breeding, fitness
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SearchResult:
    """One search result with provenance."""

    source: str  # "knowledge", "hardware", "temporal", "agent"
    score: float
    payload: Any  # Typed payload per source
    context: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"SearchResult({self.source}, score={self.score:.3f})"


class FleetSearch:
    """Unified fleet search across all memory layers.

    Args:
        knowledge: KnowledgePipeline instance.
        hardware: HardwareProfileIndex instance.
        temporal: JepaGridMemory instance (optional).
        agent_table: FluxVectorTable for agent DNA (optional).
    """

    def __init__(
        self,
        knowledge: Optional["KnowledgePipeline"] = None,
        hardware: Optional["HardwareProfileIndex"] = None,
        temporal: Optional["JepaGridMemory"] = None,
        agent_table: Optional["FluxVectorTable"] = None,
    ) -> None:
        self.knowledge = knowledge
        self.hardware = hardware
        self.temporal = temporal
        self.agent_table = agent_table

    # ── public API ──────────────────────────────────────────

    def ask(self, query: str, k: int = 5) -> list[SearchResult]:
        """Natural-language query across all fleet memory.

        Auto-detects intent and routes to the appropriate backend(s).
        Returns merged, scored results.
        """
        intent = self._detect_intent(query)
        logger.debug("Query '%s' → intent=%s", query, intent.value)

        results: list[SearchResult] = []

        if intent in (SearchIntent.KNOWLEDGE, SearchIntent.UNKNOWN):
            if self.knowledge is not None:
                results.extend(self._search_knowledge(query, k))

        if intent in (SearchIntent.HARDWARE, SearchIntent.UNKNOWN):
            if self.hardware is not None:
                results.extend(self._search_hardware(query, k))

        if intent in (SearchIntent.TEMPORAL, SearchIntent.UNKNOWN):
            if self.temporal is not None:
                results.extend(self._search_temporal(query, k))

        if intent in (SearchIntent.AGENT, SearchIntent.UNKNOWN):
            if self.agent_table is not None:
                results.extend(self._search_agents(query, k))

        # Re-rank by score
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def knowledge_search(
        self, query: str, room: Optional[str] = None, k: int = 5
    ) -> list[SearchResult]:
        """Explicit knowledge search."""
        if self.knowledge is None:
            return []
        return self._search_knowledge(query, k, room=room)

    def hardware_search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Explicit hardware placement search."""
        if self.hardware is None:
            return []
        return self._search_hardware(query, k)

    def predict_room(self, room_id: int, ticks_ahead: int = 1) -> SearchResult | None:
        """Predict future room state."""
        if self.temporal is None:
            return None
        prediction = self.temporal.predict(room_id, ticks_ahead)
        if prediction is None:
            return None
        return SearchResult(
            source="temporal",
            score=1.0,
            payload=prediction,
            context={"room_id": room_id, "ticks_ahead": ticks_ahead},
        )

    # ── intent detection ────────────────────────────────────

    @staticmethod
    def _detect_intent(query: str) -> SearchIntent:
        """Simple keyword-based intent classification.

        Production improvement: use a tiny classifier (DistilBERT)
        or an LLM prompt for intent detection.
        """
        q = query.lower()

        hardware_keywords = [
            "gpu",
            "device",
            "run on",
            "schedule",
            "placement",
            "which hardware",
            "rtx",
            "ryzen",
            "radeon",
            "xdna",
            "thermal",
            "capacity",
            "free",
            "available",
            "load",
        ]
        temporal_keywords = [
            "predict",
            "will",
            "future",
            "next",
            "tomorrow",
            "trend",
            "forecast",
            "look like",
            "upcoming",
        ]
        agent_keywords = [
            "agent",
            "breed",
            "fitness",
            "dna",
            "parent",
            "generation",
            "sunset",
            "tournament",
            "ethos",
            "pathos",
            "logos",
        ]

        if any(kw in q for kw in hardware_keywords):
            return SearchIntent.HARDWARE
        if any(kw in q for kw in temporal_keywords):
            return SearchIntent.TEMPORAL
        if any(kw in q for kw in agent_keywords):
            return SearchIntent.AGENT
        return SearchIntent.KNOWLEDGE

    # ── backend searchers ───────────────────────────────────

    def _search_knowledge(
        self, query: str, k: int, room: Optional[str] = None
    ) -> list[SearchResult]:
        """Search knowledge pipeline."""
        raw = self.knowledge.search(query, room=room, k=k)
        return [
            SearchResult(
                source="knowledge",
                score=score,
                payload={
                    "text": doc.text,
                    "doc_id": doc.doc_id,
                    "source": doc.source,
                },
                context={"room": room_name, "metadata": doc.metadata},
            )
            for room_name, score, doc in raw
        ]

    def _search_hardware(self, query: str, k: int) -> list[SearchResult]:
        """Search hardware index.

        Parses query for workload hints (batch size, memory, type)
        and constructs a WorkloadQuery.
        """
        from swarm.hardware_index import WorkloadQuery, DeviceType

        # Simple query parsing
        q = query.lower()
        min_caps = [0.0] * 8
        max_load = [1.0] * 8
        preferred: DeviceType | None = None

        if "gpu" in q or "rtx" in q:
            preferred = DeviceType.RTX_4050
        if "ryzen" in q or "cpu" in q:
            preferred = DeviceType.RYZEN_AI
        if "radeon" in q or "igpu" in q:
            preferred = DeviceType.RADEON_890M
        if "npu" in q or "xdna" in q:
            preferred = DeviceType.XDNA_2

        # Memory hints
        if "4gb" in q or "small" in q:
            min_caps[2] = 0.3  # 4GB / ~16GB max
        if "8gb" in q or "medium" in q:
            min_caps[2] = 0.5
        if "16gb" in q or "large" in q:
            min_caps[2] = 0.8

        wq = WorkloadQuery(
            min_capabilities=min_caps,
            max_load=max_load,
            preferred_type=preferred,
        )

        raw = self.hardware.find_best_device(wq, k=k)
        return [
            SearchResult(
                source="hardware",
                score=score,
                payload={
                    "device_id": did,
                    "device_type": profile.device_type.value,
                    "free_capacity": profile.free_capacity,
                },
                context={
                    "capabilities": profile.capabilities,
                    "load": profile.current_load,
                },
            )
            for did, score, profile in raw
        ]

    def _search_temporal(self, query: str, k: int) -> list[SearchResult]:
        """Search temporal memory for trajectory similarity."""
        # Extract room_id from query if present
        import re

        match = re.search(r"room\s*(\d+)", query.lower())
        if not match:
            return []

        room_id = int(match.group(1))
        similar = self.temporal.find_similar_trajectory(room_id, k=k)
        return [
            SearchResult(
                source="temporal",
                score=score,
                payload={"similar_room_id": rid},
                context={"query_room": room_id},
            )
            for rid, score in similar
        ]

    def _search_agents(self, query: str, k: int) -> list[SearchResult]:
        """Search agent DNA table."""
        if self.agent_table is None:
            return []

        # Encode query and search
        # Placeholder: use knowledge encoder for cross-modal search
        if self.knowledge is None:
            return []

        query_vec = self.knowledge.encoder.encode_one(query)
        results = self.agent_table.search(query_vec, k=k)

        return [
            SearchResult(
                source="agent",
                score=score,
                payload={"agent_id": aid, "fitness": meta.fitness},
                context={
                    "generation": meta.generation,
                    "capability_mask": meta.capability_mask,
                    "thermal_pressure": meta.thermal_pressure,
                },
            )
            for aid, score, meta in results
        ]

    def __repr__(self) -> str:
        return (
            f"FleetSearch(knowledge={self.knowledge is not None}, "
            f"hardware={self.hardware is not None}, "
            f"temporal={self.temporal is not None}, "
            f"agents={self.agent_table is not None})"
        )
