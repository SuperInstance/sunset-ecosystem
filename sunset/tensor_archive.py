"""Tensor archive — sunset agents as searchable tiles inside tiles."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from sunset.agent import Agent, AgentPhase
from sunset.sunset_documents import Epilogue, Summary


@dataclass
class SunsetEntry:
    """A sunset agent archived as a searchable tensor-like entry."""

    agent_id: str
    generation: int
    parent_id: Optional[str]
    epilogue: Optional[Epilogue] = None
    summary: Optional[Summary] = None
    peak_trinity_score: float = 0.0
    connections: List[str] = field(default_factory=list)
    content_blob: str = ""  # compressed text representation

    def __repr__(self) -> str:
        return (
            f"SunsetEntry(agent={self.agent_id!r}, gen={self.generation}, "
            f"peak={self.peak_trinity_score:.4f})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "peak_trinity_score": self.peak_trinity_score,
            "connections": self.connections,
            "content_blob": self.content_blob,
        }


class TensorArchive:
    """Archive of sunset agents as tiles inside tiles.

    Each sunset agent becomes a searchable entry.
    Supports search, wake (brief reanimation), and distill (compression).
    """

    def __init__(self) -> None:
        self._entries: Dict[str, SunsetEntry] = {}
        self._index: List[str] = []  # ordered by insertion

    def __repr__(self) -> str:
        return f"TensorArchive(entries={len(self._entries)})"

    def archive(self, entry: SunsetEntry) -> None:
        """Archive a sunset entry."""
        self._entries[entry.agent_id] = entry
        if entry.agent_id not in self._index:
            self._index.append(entry.agent_id)

    def search(self, query: str, top_k: int = 5) -> List[SunsetEntry]:
        """Search archived agents by content keyword matching.

        Simple bag-of-words similarity over content_blob, connections,
        and summary/epilogue text.
        """
        query_terms = set(query.lower().split())
        scored: List[Tuple[float, SunsetEntry]] = []

        for entry in self._entries.values():
            doc = _entry_text(entry).lower()
            doc_terms = set(doc.split())
            overlap = len(query_terms & doc_terms)
            if overlap == 0:
                continue
            score = overlap / (len(query_terms) + len(doc_terms) - overlap)
            scored.append((score, entry))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def wake(self, agent_id: str, question: str) -> str:
        """Briefly reanimate a sunset agent to answer a question.

        Returns a constructed response from the agent's archived knowledge.
        """
        entry = self._entries.get(agent_id)
        if entry is None:
            return f"[SUNSET] Agent {agent_id} not found in archive."

        parts = [
            f"[WAKE] Reanimating agent {agent_id} (gen {entry.generation}, "
            f"peak trinity {entry.peak_trinity_score:.4f})",
        ]
        if entry.epilogue:
            parts.append(f"  Tried: {entry.epilogue.what_i_tried}")
            parts.append(f"  Found: {entry.epilogue.what_i_found}")
        if entry.summary:
            parts.append(f"  Perspective: {entry.summary.work_from_my_perspective}")
            if entry.summary.key_insights:
                parts.append(f"  Insights: {'; '.join(entry.summary.key_insights)}")
        parts.append(f"  Question was: {question}")
        parts.append("[WAKE] Agent returned to sleep.")
        return "\n".join(parts)

    def distill(self, agent_id: str) -> bytes:
        """Compress a sunset agent's patterns into a weight blob (JSON bytes)."""
        entry = self._entries.get(agent_id)
        if entry is None:
            return b""

        blob = {
            "id": entry.agent_id,
            "gen": entry.generation,
            "score": entry.peak_trinity_score,
            "connections": entry.connections,
            "compressed": entry.content_blob[:512],  # truncate for compression
        }
        if entry.summary:
            blob["insights"] = entry.summary.key_insights
            blob["failures"] = entry.summary.failed_approaches
        if entry.epilogue:
            blob["relevance_reason"] = entry.epilogue.why_not_relevant

        return json.dumps(blob, separators=(",", ":")).encode("utf-8")

    def tensor_shape(self) -> Tuple[int, ...]:
        """Return (agents, features, generations) shape info."""
        if not self._entries:
            return (0, 0, 0)

        features_per = []
        for entry in self._entries.values():
            f = 3  # base: score, generation, num_connections
            if entry.summary:
                f += len(entry.summary.key_insights) + len(entry.summary.failed_approaches)
            if entry.epilogue:
                f += 2  # tried + found
            features_per.append(f)

        generations = len({e.generation for e in self._entries.values()})
        return (len(self._entries), max(features_per), generations)

    def get(self, agent_id: str) -> Optional[SunsetEntry]:
        """Retrieve a single entry by ID."""
        return self._entries.get(agent_id)


def _entry_text(entry: SunsetEntry) -> str:
    """Build a searchable text blob from an entry."""
    parts = [entry.content_blob]
    if entry.epilogue:
        parts.extend([entry.epilogue.what_i_tried, entry.epilogue.what_i_found, entry.epilogue.why_not_relevant])
    if entry.summary:
        parts.append(entry.summary.work_from_my_perspective)
        parts.extend(entry.summary.key_insights)
        parts.extend(entry.summary.failed_approaches)
        parts.extend(entry.summary.connections_made)
    parts.extend(entry.connections)
    return " ".join(parts)


__all__ = ["TensorArchive", "SunsetEntry"]
