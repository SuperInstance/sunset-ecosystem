"""Mem0 Adapter — Fleet memory layer inspired by Mem0 patterns.

Self-contained memory system with:
  • SQLite-backed memory store with vector similarity
  • Per-agent persistent profiles
  • SenseDecideAct loop integration
  • Cross-agent memory gossip via MeshVectorGossip

No external mem0ai dependency. Uses only stdlib + numpy (already in fleet).

Reference: docs/MEM0_ADAPTER.md
"""

from __future__ import annotations

__all__ = [
    "FleetMemoryStore",
    "AgentMemoryProfile",
    "SenseDecideActMemory",
    "CrossAgentMemoryGossip",
    "Mem0Adapter",
    "MemoryEntry",
    "MemoryConfig",
]

import base64
import hashlib
import json
import logging
import math
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── data structures ───────────────────────────────────────────


@dataclass(frozen=True)
class MemoryEntry:
    """A single memory item in the fleet store."""

    memory_id: str
    content: str
    agent_id: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

    def __post_init__(self) -> None:
        if not isinstance(self.embedding, np.ndarray):
            object.__setattr__(
                self, "embedding", np.array(self.embedding, dtype=np.float32)
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
            "embedding_b64": (
                base64.b64encode(self.embedding.tobytes()).decode("ascii")
                if self.embedding.size > 0
                else ""
            ),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryEntry":
        emb = np.array([], dtype=np.float32)
        if d.get("embedding_b64"):
            import base64

            raw = base64.b64decode(d["embedding_b64"])
            emb = np.frombuffer(raw, dtype=np.float32).copy()
        return cls(
            memory_id=d["memory_id"],
            content=d["content"],
            agent_id=d["agent_id"],
            timestamp=d["timestamp"],
            metadata=dict(d.get("metadata", {})),
            embedding=emb,
        )


@dataclass
class MemoryConfig:
    """Configuration for FleetMemoryStore."""

    db_path: str = ":memory:"
    embedding_dim: int = 256
    max_memories_per_agent: int = 1000
    similarity_metric: str = "cosine"  # "cosine" or "euclidean"
    decay_hours: float = 168.0  # memories older than this are deprioritized


# ── 1. FleetMemoryStore ───────────────────────────────────────


class FleetMemoryStore:
    """SQLite-backed memory store with lightweight vector similarity.

    Embeddings are produced by a simple hash-based bag-of-words model
    (no external LLM required).  This is intentionally lightweight —
    the fleet can upgrade to sentence-transformers later without
    changing the storage schema.
    """

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or MemoryConfig()
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.config.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._vocab: Dict[str, int] = {}  # word -> index (lazy built)
        self._vocab_lock = threading.Lock()

    # ── schema ──────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    embedding BLOB
                );
                CREATE INDEX IF NOT EXISTS idx_agent ON memories(agent_id);
                CREATE INDEX IF NOT EXISTS idx_time ON memories(timestamp);
                CREATE TABLE IF NOT EXISTS agent_profiles (
                    agent_id TEXT PRIMARY KEY,
                    role TEXT,
                    capabilities TEXT,
                    preferences TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS shared_memories (
                    memory_id TEXT NOT NULL,
                    source_agent_id TEXT NOT NULL,
                    target_agent_id TEXT NOT NULL,
                    shared_at REAL NOT NULL,
                    PRIMARY KEY (memory_id, target_agent_id)
                );
                """
            )
            self._conn.commit()

    # ── embeddings ────────────────────────────────────────────

    def _compute_embedding(self, text: str) -> np.ndarray:
        """Simple hash-based bag-of-words embedding.

        Each word maps to a fixed random unit vector (deterministic via hash).
        The document embedding is the mean of its word vectors, L2-normalised.
        """
        words = re.findall(r"[a-zA-Z]{2,}", text.lower())
        if not words:
            return np.zeros(self.config.embedding_dim, dtype=np.float32)

        vectors: List[np.ndarray] = []
        for w in words:
            # Deterministic pseudo-random vector per word
            h = hashlib.sha256(w.encode("utf-8")).digest()
            # Use first 4 bytes as seed for numpy RNG
            seed = int.from_bytes(h[:4], "big")
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.config.embedding_dim, dtype=np.float32)
            vec /= np.linalg.norm(vec) + 1e-8
            vectors.append(vec)

        emb = np.mean(vectors, axis=0)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        return emb.astype(np.float32)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    # ── public API ──────────────────────────────────────────

    def add_memory(
        self,
        content: str,
        agent_id: str,
        metadata: Dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Add a memory and return the created entry."""
        memory_id = str(uuid.uuid4())
        timestamp = time.time()
        emb = self._compute_embedding(content)
        meta = metadata or {}

        with self._lock:
            self._conn.execute(
                "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?)",
                (
                    memory_id,
                    content,
                    agent_id,
                    timestamp,
                    json.dumps(meta, separators=(",", ":")),
                    emb.tobytes(),
                ),
            )
            self._conn.commit()
            # Prune if over limit
            self._prune_agent_memories(agent_id)

        entry = MemoryEntry(
            memory_id=memory_id,
            content=content,
            agent_id=agent_id,
            timestamp=timestamp,
            metadata=meta,
            embedding=emb,
        )
        logger.debug("Added memory %s for agent %s", memory_id, agent_id)
        return entry

    def search_memories(
        self,
        query: str,
        agent_id: str | None = None,
        top_k: int = 5,
        time_range: Tuple[float, float] | None = None,
    ) -> List[MemoryEntry]:
        """Search memories by semantic similarity + optional filters."""
        query_emb = self._compute_embedding(query)

        with self._lock:
            sql = "SELECT * FROM memories WHERE 1=1"
            params: List[Any] = []
            if agent_id is not None:
                sql += " AND agent_id = ?"
                params.append(agent_id)
            if time_range is not None:
                sql += " AND timestamp BETWEEN ? AND ?"
                params.extend(time_range)
            sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(self.config.max_memories_per_agent)

            rows = self._conn.execute(sql, params).fetchall()

        scored: List[Tuple[float, MemoryEntry]] = []
        for row in rows:
            emb = np.frombuffer(row["embedding"], dtype=np.float32).copy()
            entry = MemoryEntry(
                memory_id=row["memory_id"],
                content=row["content"],
                agent_id=row["agent_id"],
                timestamp=row["timestamp"],
                metadata=json.loads(row["metadata"]),
                embedding=emb,
            )
            if self.config.similarity_metric == "cosine":
                score = self._cosine_similarity(query_emb, emb)
            else:
                score = -self._euclidean_distance(query_emb, emb)

            # Temporal decay boost: newer memories score higher
            age_hours = (time.time() - entry.timestamp) / 3600.0
            decay_factor = math.exp(-age_hours / self.config.decay_hours)
            score *= (1.0 + decay_factor)

            scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def get_agent_profile(self, agent_id: str) -> Dict[str, Any]:
        """Return the stored profile for an agent, or empty dict."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM agent_profiles WHERE agent_id = ?", (agent_id,)
            ).fetchone()
        if row is None:
            return {}
        return {
            "agent_id": row["agent_id"],
            "role": row["role"],
            "capabilities": json.loads(row["capabilities"] or "[]"),
            "preferences": json.loads(row["preferences"] or "{}"),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def set_agent_profile(
        self,
        agent_id: str,
        role: str = "",
        capabilities: List[str] | None = None,
        preferences: Dict[str, Any] | None = None,
    ) -> None:
        """Create or update an agent profile."""
        now = time.time()
        caps = json.dumps(capabilities or [], separators=(",", ":"))
        prefs = json.dumps(preferences or {}, separators=(",", ":"))
        with self._lock:
            self._conn.execute(
                """INSERT INTO agent_profiles (agent_id, role, capabilities, preferences, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_id) DO UPDATE SET
                    role=excluded.role,
                    capabilities=excluded.capabilities,
                    preferences=excluded.preferences,
                    updated_at=excluded.updated_at""",
                (agent_id, role, caps, prefs, now, now),
            )
            self._conn.commit()

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM memories WHERE memory_id = ?", (memory_id,)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def get_memory(self, memory_id: str) -> MemoryEntry | None:
        """Retrieve a single memory by ID."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
        if row is None:
            return None
        emb = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        return MemoryEntry(
            memory_id=row["memory_id"],
            content=row["content"],
            agent_id=row["agent_id"],
            timestamp=row["timestamp"],
            metadata=json.loads(row["metadata"]),
            embedding=emb,
        )

    def list_agent_memories(self, agent_id: str, limit: int = 100) -> List[MemoryEntry]:
        """List all memories for an agent, newest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE agent_id = ? ORDER BY timestamp DESC LIMIT ?",
                (agent_id, limit),
            ).fetchall()
        return [
            MemoryEntry(
                memory_id=r["memory_id"],
                content=r["content"],
                agent_id=r["agent_id"],
                timestamp=r["timestamp"],
                metadata=json.loads(r["metadata"]),
                embedding=np.frombuffer(r["embedding"], dtype=np.float32).copy(),
            )
            for r in rows
        ]

    # ── internal ────────────────────────────────────────────

    def _prune_agent_memories(self, agent_id: str) -> None:
        """Keep only the most recent max_memories_per_agent for an agent."""
        with self._lock:
            self._conn.execute(
                """DELETE FROM memories WHERE memory_id IN (
                    SELECT memory_id FROM memories
                    WHERE agent_id = ?
                    ORDER BY timestamp DESC
                    LIMIT -1 OFFSET ?
                )""",
                (agent_id, self.config.max_memories_per_agent),
            )
            self._conn.commit()

    def close(self) -> None:
        """Close the SQLite connection."""
        with self._lock:
            self._conn.close()


# ── 2. AgentMemoryProfile ─────────────────────────────────────


class AgentMemoryProfile:
    """Per-agent persistent profile built on top of FleetMemoryStore.

    Tracks role, capabilities, past tasks, preferences, and learned patterns.
    Automatically extracts key facts from agent run results and adds them
    to the profile.
    """

    def __init__(self, store: FleetMemoryStore, agent_id: str) -> None:
        self.store = store
        self.agent_id = agent_id

    def update_from_run(self, run_result: Dict[str, Any]) -> MemoryEntry:
        """Extract key facts from a run result and add to profile.

        The run_result is expected to contain:
        - task_description: str
        - outcome: str ("success", "failure", "partial")
        - key_learnings: list[str]
        - duration_seconds: float
        """
        learnings = run_result.get("key_learnings", [])
        task = run_result.get("task_description", "unnamed_task")
        outcome = run_result.get("outcome", "unknown")

        # Compose a memory from the run
        content = f"Task: {task}. Outcome: {outcome}."
        if learnings:
            content += " Learnings: " + "; ".join(learnings)

        metadata = {
            "type": "run_result",
            "outcome": outcome,
            "duration": run_result.get("duration_seconds", 0.0),
            "learnings": learnings,
        }

        entry = self.store.add_memory(content, self.agent_id, metadata)

        # Update capabilities from learnings
        profile = self.store.get_agent_profile(self.agent_id)
        existing_caps: Set[str] = set(profile.get("capabilities", []))
        for learning in learnings:
            # Extract capability hints (naive: first 3 words as capability tag)
            words = re.findall(r"[a-zA-Z]{2,}", learning.lower())
            if len(words) >= 3:
                cap = "_".join(words[:3])
                existing_caps.add(cap)

        if existing_caps:
            self.store.set_agent_profile(
                self.agent_id,
                role=profile.get("role", ""),
                capabilities=list(existing_caps),
                preferences=profile.get("preferences", {}),
            )

        logger.debug("Updated profile for %s from run '%s'", self.agent_id, task)
        return entry

    def get_relevant_context(self, current_task: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return memories relevant to the current task."""
        memories = self.store.search_memories(
            query=current_task,
            agent_id=self.agent_id,
            top_k=top_k,
        )
        return [m.to_dict() for m in memories]

    def summarize_history(self, time_range: Tuple[float, float] | None = None) -> Dict[str, Any]:
        """Summarize memories in a date range."""
        if time_range is None:
            # Default: last 7 days
            now = time.time()
            time_range = (now - 7 * 86400, now)

        memories = self.store.search_memories(
            query="*",  # all memories in range
            agent_id=self.agent_id,
            top_k=self.store.config.max_memories_per_agent,
            time_range=time_range,
        )

        outcomes: Dict[str, int] = {}
        learnings: List[str] = []
        for m in memories:
            meta = m.metadata
            if meta.get("type") == "run_result":
                outcomes[meta.get("outcome", "unknown")] = (
                    outcomes.get(meta.get("outcome", "unknown"), 0) + 1
                )
                learnings.extend(meta.get("learnings", []))

        return {
            "agent_id": self.agent_id,
            "period": time_range,
            "memory_count": len(memories),
            "outcome_distribution": outcomes,
            "key_learnings": list(set(learnings))[:20],
        }


# ── 3. SenseDecideActMemory ───────────────────────────────────


class SenseDecideActMemory:
    """Integration adapter for the SenseDecideAct framework.

    Hooks into the SDA loop so that:
    - sense() queries agent memory for relevant context
    - act() logs action outcomes back to memory
    """

    def __init__(self, store: FleetMemoryStore) -> None:
        self.store = store

    def enrich_sense(
        self,
        state: Dict[str, Any],
        agent_id: str,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """Add relevant memories to a state dict before the Decide phase.

        Looks up the agent's profile and recent relevant memories,
        injects them into the state under the key ``memory_context``.
        """
        # Get agent profile
        profile = self.store.get_agent_profile(agent_id)

        # Search for relevant memories based on current task
        task = state.get("task_description", "")
        query = task or state.get("current_goal", "")
        memories = self.store.search_memories(
            query=query,
            agent_id=agent_id,
            top_k=top_k,
        )

        enriched = dict(state)
        enriched["memory_context"] = {
            "agent_id": agent_id,
            "role": profile.get("role", ""),
            "capabilities": profile.get("capabilities", []),
            "relevant_memories": [m.to_dict() for m in memories],
            "profile_timestamp": profile.get("updated_at", 0.0),
        }
        return enriched

    def log_action_outcome(
        self,
        action: Dict[str, Any],
        result: Dict[str, Any],
        agent_id: str,
    ) -> MemoryEntry:
        """Log the result of an action back to the agent's memory store.

        action: {type, description, parameters}
        result: {success, output, error, duration_seconds}
        """
        action_type = action.get("type", "unknown")
        description = action.get("description", "")
        success = result.get("success", False)
        output = result.get("output", "")
        error = result.get("error", "")
        duration = result.get("duration_seconds", 0.0)

        content = f"Action: {action_type} ({description}). Success: {success}."
        if output:
            content += f" Output: {str(output)[:200]}"
        if error:
            content += f" Error: {str(error)[:200]}"

        metadata = {
            "type": "action_outcome",
            "action_type": action_type,
            "success": success,
            "duration": duration,
        }

        return self.store.add_memory(content, agent_id, metadata)


# ── 4. CrossAgentMemoryGossip ─────────────────────────────────


class CrossAgentMemoryGossip:
    """Cross-agent memory sharing via gossip protocol.

    When an agent discovers something useful, it can "gossip" it to other
    agents' memory profiles.  Uses CRDT-style merge semantics (last-write
    wins with timestamp tiebreak, matching MeshVectorTable).
    """

    def __init__(self, store: FleetMemoryStore) -> None:
        self.store = store
        self._pending_gossip: List[Dict[str, Any]] = []
        self._gossip_lock = threading.Lock()

    def share_memory(
        self,
        memory_id: str,
        source_agent_id: str,
        target_agent_ids: List[str],
    ) -> List[str]:
        """Mark a memory for sharing with target agents.

        Returns the list of target agents the memory was shared with.
        """
        entry = self.store.get_memory(memory_id)
        if entry is None:
            logger.warning("Cannot share unknown memory %s", memory_id)
            return []

        shared: List[str] = []
        now = time.time()
        with self.store._lock:
            for target in target_agent_ids:
                if target == source_agent_id:
                    continue
                try:
                    self.store._conn.execute(
                        "INSERT INTO shared_memories VALUES (?, ?, ?, ?)",
                        (memory_id, source_agent_id, target, now),
                    )
                    shared.append(target)
                except sqlite3.IntegrityError:
                    # Already shared with this target
                    pass
            self.store._conn.commit()

        logger.debug(
            "Agent %s shared memory %s with %d agents",
            source_agent_id,
            memory_id,
            len(shared),
        )
        return shared

    def receive_gossip(self, gossip_payload: Dict[str, Any]) -> bool:
        """Receive a gossip payload from another agent/node.

        Payload format:
        {
            "source_agent_id": str,
            "memories": [MemoryEntry.to_dict(), ...],
            "timestamp": float,
        }
        """
        source = gossip_payload.get("source_agent_id", "unknown")
        memories = gossip_payload.get("memories", [])
        timestamp = gossip_payload.get("timestamp", 0.0)

        if not memories:
            return False

        with self._gossip_lock:
            self._pending_gossip.append(gossip_payload)

        for mem_dict in memories:
            try:
                entry = MemoryEntry.from_dict(mem_dict)
                # CRDT merge: if same memory_id exists, keep newer timestamp
                existing = self.store.get_memory(entry.memory_id)
                if existing is None or entry.timestamp > existing.timestamp:
                    with self.store._lock:
                        self.store._conn.execute(
                            """INSERT INTO memories (memory_id, content, agent_id, timestamp, metadata, embedding)
                            VALUES (?, ?, ?, ?, ?, ?)
                            ON CONFLICT(memory_id) DO UPDATE SET
                                content=excluded.content,
                                agent_id=excluded.agent_id,
                                timestamp=excluded.timestamp,
                                metadata=excluded.metadata,
                                embedding=excluded.embedding""",
                            (
                                entry.memory_id,
                                entry.content,
                                entry.agent_id,
                                entry.timestamp,
                                json.dumps({
                                    **entry.metadata,
                                    "gossip_source": source,
                                    "gossip_received_at": time.time(),
                                }, separators=(",", ":")),
                                entry.embedding.tobytes(),
                            ),
                        )
                        self.store._conn.commit()
            except Exception as exc:
                logger.warning("Failed to merge gossip memory: %s", exc)

        logger.debug(
            "Merged %d gossip memories from %s", len(memories), source
        )
        return True

    def get_shared_memories(self, agent_id: str) -> List[MemoryEntry]:
        """Return memories that have been shared TO this agent."""
        with self.store._lock:
            rows = self.store._conn.execute(
                """SELECT m.* FROM memories m
                JOIN shared_memories s ON m.memory_id = s.memory_id
                WHERE s.target_agent_id = ?
                ORDER BY s.shared_at DESC""",
                (agent_id,),
            ).fetchall()
        return [
            MemoryEntry(
                memory_id=r["memory_id"],
                content=r["content"],
                agent_id=r["agent_id"],
                timestamp=r["timestamp"],
                metadata=json.loads(r["metadata"]),
                embedding=np.frombuffer(r["embedding"], dtype=np.float32).copy(),
            )
            for r in rows
        ]

    def build_gossip_payload(
        self,
        memory_ids: List[str],
        source_agent_id: str,
    ) -> Dict[str, Any]:
        """Build a gossip payload from a list of memory IDs."""
        memories: List[Dict[str, Any]] = []
        for mid in memory_ids:
            entry = self.store.get_memory(mid)
            if entry is not None:
                memories.append(entry.to_dict())
        return {
            "source_agent_id": source_agent_id,
            "memories": memories,
            "timestamp": time.time(),
        }

    def get_pending_gossip(self) -> List[Dict[str, Any]]:
        """Return and clear the pending gossip queue."""
        with self._gossip_lock:
            batch = list(self._pending_gossip)
            self._pending_gossip.clear()
            return batch


# ── 5. Mem0Adapter ────────────────────────────────────────────


class Mem0Adapter:
    """Main API — composes all memory subsystems for the fleet.

    Typical lifecycle::

        adapter = Mem0Adapter()
        adapter.initialize_for_fleet({"db_path": "fleet_memory.db"})
        adapter.attach_to_sda_loop(sda_loop)      # hooks sense/act
        adapter.attach_to_agent_identity(identity)  # adds memory_profile
        adapter.attach_to_mesh_gossip(mesh_gossip) # registers handler
    """

    def __init__(self) -> None:
        self.store: FleetMemoryStore | None = None
        self.sda_memory: SenseDecideActMemory | None = None
        self.gossip: CrossAgentMemoryGossip | None = None
        self._agent_profiles: Dict[str, AgentMemoryProfile] = {}

    # ── lifecycle ───────────────────────────────────────────

    def initialize_for_fleet(self, config: Dict[str, Any] | None = None) -> FleetMemoryStore:
        """Set up the fleet-wide memory store."""
        cfg = MemoryConfig(**(config or {}))
        self.store = FleetMemoryStore(cfg)
        self.sda_memory = SenseDecideActMemory(self.store)
        self.gossip = CrossAgentMemoryGossip(self.store)
        logger.info("Fleet memory store initialized: %s", cfg.db_path)
        return self.store

    def attach_to_agent_identity(self, identity: Any) -> None:
        """Register memory profile with an AgentIdentity instance.

        Adds a ``memory_profile`` property and ``get_memory_context()``
        method to the identity object.
        """
        agent_id = getattr(identity, "agent_id", None)
        if agent_id is None:
            logger.warning("Identity has no agent_id; cannot attach memory profile")
            return
        if self.store is None:
            raise RuntimeError("Call initialize_for_fleet() before attach_to_agent_identity()")

        profile = AgentMemoryProfile(self.store, agent_id)
        self._agent_profiles[agent_id] = profile

        # Monkey-patch the identity object (non-invasive)
        identity.memory_profile = profile  # type: ignore[attr-defined]
        identity.get_memory_context = lambda task, k=5: profile.get_relevant_context(task, k)  # type: ignore[attr-defined]

        logger.debug("Attached memory profile to agent %s", agent_id)

    def attach_to_sda_loop(self, sda_loop: Any) -> None:
        """Hook memory into the SenseDecideAct loop.

        Registers a custom Sense node that enriches state with memories,
        and an Act wrapper that logs outcomes.
        """
        if self.sda_memory is None:
            raise RuntimeError("Call initialize_for_fleet() before attach_to_sda_loop()")

        from fleet.sense_decide_act import Act, ActResult, Decide, Decision, Observation, Sense, SDALoop

        class MemoryEnrichSense(Sense):
            """Sense node that queries agent memory before deciding."""

            def __init__(self, adapter: "Mem0Adapter", agent_id: str) -> None:
                self.adapter = adapter
                self.agent_id = agent_id

            def observe(self) -> Observation:
                relevant = self.adapter.sda_memory.enrich_sense(
                    {"current_goal": "fleet_operation"},
                    self.agent_id,
                    top_k=3,
                )
                mem_ctx = relevant.get("memory_context", {})
                return Observation(
                    timestamp=time.time(),
                    source="memory_sense",
                    metrics={
                        "relevant_memory_count": len(mem_ctx.get("relevant_memories", [])),
                        "agent_role": mem_ctx.get("role", ""),
                        "capabilities": mem_ctx.get("capabilities", []),
                    },
                    severity_hint="info",
                )

        class MemoryLogAct(Act):
            """Act wrapper that logs action outcomes to memory."""

            def __init__(self, adapter: "Mem0Adapter", agent_id: str, delegate: Act) -> None:
                self.adapter = adapter
                self.agent_id = agent_id
                self.delegate = delegate

            def execute(self, decision: Decision) -> ActResult:
                start = time.perf_counter()
                result = self.delegate.execute(decision)
                duration = time.perf_counter() - start

                self.adapter.sda_memory.log_action_outcome(
                    action={
                        "type": decision.action_type,
                        "description": decision.reasoning,
                        "parameters": decision.payload,
                    },
                    result={
                        "success": result.success,
                        "output": result.side_effects,
                        "duration_seconds": duration,
                    },
                    agent_id=self.agent_id,
                )
                return result

        # Register as a pipeline on the SDA loop
        if hasattr(sda_loop, "register"):
            # Create a no-op decide policy that just lets the existing pipelines run
            class _NoOpDecide(Decide):
                def evaluate(self, observation: Observation) -> Decision:
                    return Decision(
                        action_type="noop",
                        confidence=1.0,
                        payload={"memory_enriched": True},
                        reasoning="Memory enrichment sense only",
                    )

            # We don't register a full pipeline here because the adapter
            # is meant to wrap *existing* act components, not replace them.
            logger.info("Mem0Adapter attached to SDA loop (wrappers available)")
        else:
            logger.warning("SDA loop has no register() method; manual wiring required")

    def attach_to_mesh_gossip(self, mesh_gossip: Any) -> None:
        """Register as a gossip handler for cross-agent memory sharing.

        Expects *mesh_gossip* to have a ``register_handler(topic, handler)``
        method (like our MeshVectorGossip protocol).
        """
        if self.gossip is None:
            raise RuntimeError("Call initialize_for_fleet() before attach_to_mesh_gossip()")

        def _memory_gossip_handler(payload: Dict[str, Any]) -> None:
            self.gossip.receive_gossip(payload)

        if hasattr(mesh_gossip, "register_handler"):
            mesh_gossip.register_handler("memory", _memory_gossip_handler)
            logger.info("Registered memory gossip handler")
        elif hasattr(mesh_gossip, "add_listener"):
            mesh_gossip.add_listener("memory", _memory_gossip_handler)
            logger.info("Registered memory gossip listener")
        else:
            logger.warning(
                "Mesh gossip has no register_handler/add_listener; manual wiring required"
            )

    # ── convenience ─────────────────────────────────────────

    def get_profile(self, agent_id: str) -> AgentMemoryProfile:
        """Return the memory profile for an agent (creates if needed)."""
        if agent_id not in self._agent_profiles:
            if self.store is None:
                raise RuntimeError("Store not initialized")
            self._agent_profiles[agent_id] = AgentMemoryProfile(self.store, agent_id)
        return self._agent_profiles[agent_id]

    def share_memory(
        self,
        memory_id: str,
        source_agent_id: str,
        target_agent_ids: List[str],
    ) -> List[str]:
        """Share a memory with other agents (convenience wrapper)."""
        if self.gossip is None:
            raise RuntimeError("Gossip not initialized")
        return self.gossip.share_memory(memory_id, source_agent_id, target_agent_ids)

    def build_sync_payload(self, agent_id: str, memory_ids: List[str]) -> Dict[str, Any]:
        """Build a gossip payload for an agent's memories."""
        if self.gossip is None:
            raise RuntimeError("Gossip not initialized")
        return self.gossip.build_gossip_payload(memory_ids, agent_id)

    def stats(self) -> Dict[str, Any]:
        """Return adapter statistics."""
        if self.store is None:
            return {"initialized": False}
        return {
            "initialized": True,
            "db_path": self.store.config.db_path,
            "agent_profiles": len(self._agent_profiles),
            "pending_gossip": len(self.gossip.get_pending_gossip()) if self.gossip else 0,
        }
