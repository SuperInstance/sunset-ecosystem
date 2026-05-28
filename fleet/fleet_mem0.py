"""FleetMem0 — Semantic memory adapter for Mem0.

Wraps **mem0ai** (Mem0 v2) to provide per-agent semantic memory with
vector + BM25 + entity retrieval. Integrates with AgentIdentity and
SenseDecideAct.

Reference: https://github.com/mem0ai/mem0
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class FleetMem0Config:
    """Configuration for FleetMem0."""

    vector_store: str = "qdrant"  # qdrant, chroma, faiss, pgvector, weaviate
    vector_store_path: str = "~/.openclaw/mem0_vectors"
    llm_provider: str = "ollama"  # ollama, openai, anthropic, etc.
    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text"
    user_id: str = "fleet_default"
    agent_id: str = "kimi1"
    version: str = "v2"


@dataclass
class FleetMemoryEntry:
    """A single memory entry."""

    content: str
    memory_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None


class FleetMem0Memory:
    """Fleet wrapper around mem0.Memory.

    Usage::

        mem = FleetMem0Memory(FleetMem0Config(agent_id="kimi1"))
        mem.add("The sunset ecosystem now has 19 modules and 484 tests.")
        results = mem.search("How many modules do we have?", k=3)
        for r in results:
            print(f"{r.score:.2f}: {r.content}")
    """

    def __init__(self, config: Optional[FleetMem0Config] = None):
        self.config = config or FleetMem0Config()
        self._mem0: Optional[Any] = None
        self._try_init()

    def _try_init(self) -> None:
        try:
            from mem0.memory.main import Memory
            from mem0.configs.base import MemoryConfig

            store_path = Path(self.config.vector_store_path).expanduser()
            store_path.parent.mkdir(parents=True, exist_ok=True)

            mem0_config = MemoryConfig(
                vector_store={
                    "provider": self.config.vector_store,
                    "config": {"path": str(store_path)},
                },
                llm={
                    "provider": self.config.llm_provider,
                    "config": {"model": "qwen2.5:7b"},
                },
                embedder={
                    "provider": self.config.embedding_provider,
                    "config": {"model": self.config.embedding_model},
                },
                version=self.config.version,
            )
            self._mem0 = Memory(config=mem0_config)
            log.info("FleetMem0: mem0 backend initialized (%s)", self.config.vector_store)
        except Exception as exc:
            log.warning("FleetMem0: mem0 init failed (%s); using fallback", exc)
            self._mem0 = None
            self._fallback_memories: List[Dict[str, Any]] = []

    # ── Core API ───────────────────────────────────────────────────────

    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Add a memory. Returns memory_id or None."""
        meta = metadata or {}
        if self._mem0 is not None:
            try:
                result = self._mem0.add(
                    messages=content,
                    user_id=self.config.user_id,
                    agent_id=self.config.agent_id,
                    metadata=meta,
                )
                # mem0 returns a dict with memory_id
                return result.get("id") if isinstance(result, dict) else None
            except Exception as exc:
                log.warning("mem0.add failed: %s", exc)
                return None
        else:
            entry = {
                "content": content,
                "metadata": meta,
                "timestamp": __import__("time").time(),
            }
            self._fallback_memories.append(entry)
            return str(len(self._fallback_memories))

    def search(self, query: str, k: int = 5) -> List[FleetMemoryEntry]:
        """Search memories."""
        if self._mem0 is not None:
            try:
                results = self._mem0.search(
                    query=query,
                    user_id=self.config.user_id,
                    agent_id=self.config.agent_id,
                    limit=k,
                )
                return [
                    FleetMemoryEntry(
                        content=r.get("memory", ""),
                        memory_id=r.get("id"),
                        metadata=r.get("metadata", {}),
                        score=r.get("score"),
                    )
                    for r in (results.get("results", []) if isinstance(results, dict) else results)
                ]
            except Exception as exc:
                log.warning("mem0.search failed: %s", exc)
                return []
        else:
            return self._fallback_search(query, k)

    def get_all(self) -> List[FleetMemoryEntry]:
        """Retrieve all memories."""
        if self._mem0 is not None:
            try:
                results = self._mem0.get_all(
                    user_id=self.config.user_id,
                    agent_id=self.config.agent_id,
                )
                return [
                    FleetMemoryEntry(
                        content=r.get("memory", ""),
                        memory_id=r.get("id"),
                        metadata=r.get("metadata", {}),
                    )
                    for r in (results if isinstance(results, list) else [])
                ]
            except Exception as exc:
                log.warning("mem0.get_all failed: %s", exc)
                return []
        else:
            return [
                FleetMemoryEntry(content=e["content"], metadata=e["metadata"])
                for e in self._fallback_memories
            ]

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        if self._mem0 is not None:
            try:
                self._mem0.delete(memory_id=memory_id)
                return True
            except Exception as exc:
                log.warning("mem0.delete failed: %s", exc)
                return False
        else:
            return False

    def history(self, memory_id: str) -> List[Dict[str, Any]]:
        """Get history of a memory."""
        if self._mem0 is not None:
            try:
                return self._mem0.history(memory_id=memory_id) or []
            except Exception as exc:
                log.warning("mem0.history failed: %s", exc)
                return []
        return []

    # ── Fallback ───────────────────────────────────────────────────────

    def _fallback_search(self, query: str, k: int) -> List[FleetMemoryEntry]:
        """Pure-Python fallback: simple keyword search."""
        words = query.lower().split()
        scored: List[tuple] = []
        for entry in self._fallback_memories:
            content_lower = entry["content"].lower()
            score = sum(content_lower.count(w) for w in words) / max(len(words), 1)
            scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            FleetMemoryEntry(content=e["content"], metadata=e["metadata"], score=s)
            for s, e in scored[:k]
        ]

    # ── Integration helpers ──────────────────────────────────────────────

    @classmethod
    def from_agent_identity(cls, identity: Any, **kwargs) -> "FleetMem0Memory":
        """Create a FleetMem0Memory from an AgentIdentity card."""
        agent_id = getattr(identity, "agent_id", "unknown")
        config = FleetMem0Config(agent_id=agent_id, **kwargs)
        return cls(config)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": "mem0" if self._mem0 is not None else "fallback",
            "config": {
                "vector_store": self.config.vector_store,
                "llm_provider": self.config.llm_provider,
                "agent_id": self.config.agent_id,
            },
            "n_memories": len(self.get_all()),
        }
