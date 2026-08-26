"""Tests for fleet.mem0_adapter.

Covers: FleetMemoryStore, AgentMemoryProfile, SenseDecideActMemory,
CrossAgentMemoryGossip, Mem0Adapter.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any, Dict, List

import numpy as np
import pytest

from fleet.mem0_adapter import (
    AgentMemoryProfile,
    CrossAgentMemoryGossip,
    FleetMemoryStore,
    Mem0Adapter,
    MemoryConfig,
    MemoryEntry,
    SenseDecideActMemory,
)


# ═══════════════════════════════════════════════════════════
# FleetMemoryStore
# ═══════════════════════════════════════════════════════════


class TestFleetMemoryStore:
    def test_add_memory_basic(self) -> None:
        store = FleetMemoryStore(MemoryConfig(db_path=":memory:"))
        entry = store.add_memory("Learned about pytest fixtures", "agent_1")
        assert entry.content == "Learned about pytest fixtures"
        assert entry.agent_id == "agent_1"
        assert entry.memory_id
        assert entry.timestamp > 0

    def test_search_memories_basic(self) -> None:
        store = FleetMemoryStore(MemoryConfig(db_path=":memory:"))
        store.add_memory("pytest fixtures are powerful", "agent_1")
        store.add_memory("asyncio can be tricky", "agent_1")
        store.add_memory("fixtures help with setup", "agent_1")

        results = store.search_memories("pytest setup", agent_id="agent_1", top_k=2)
        assert len(results) <= 2
        # The pytest-related memories should rank higher
        assert any("pytest" in r.content.lower() for r in results)

    def test_search_memories_agent_isolation(self) -> None:
        store = FleetMemoryStore(MemoryConfig(db_path=":memory:"))
        store.add_memory("agent alpha secret", "agent_alpha")
        store.add_memory("agent beta secret", "agent_beta")

        alpha_results = store.search_memories("secret", agent_id="agent_alpha")
        assert all(r.agent_id == "agent_alpha" for r in alpha_results)

    def test_get_memory_by_id(self) -> None:
        store = FleetMemoryStore(MemoryConfig(db_path=":memory:"))
        entry = store.add_memory("unique content", "agent_1")
        fetched = store.get_memory(entry.memory_id)
        assert fetched is not None
        assert fetched.content == "unique content"

    def test_delete_memory(self) -> None:
        store = FleetMemoryStore(MemoryConfig(db_path=":memory:"))
        entry = store.add_memory("to be deleted", "agent_1")
        assert store.delete_memory(entry.memory_id) is True
        assert store.get_memory(entry.memory_id) is None
        assert store.delete_memory(entry.memory_id) is False

    def test_agent_profile_roundtrip(self) -> None:
        store = FleetMemoryStore(MemoryConfig(db_path=":memory:"))
        store.set_agent_profile(
            "agent_1",
            role="auditor",
            capabilities=["code_review", "security_scan"],
            preferences={"language": "python"},
        )
        profile = store.get_agent_profile("agent_1")
        assert profile["role"] == "auditor"
        assert "code_review" in profile["capabilities"]
        assert profile["preferences"]["language"] == "python"

    def test_pruning_respects_limit(self) -> None:
        config = MemoryConfig(db_path=":memory:", max_memories_per_agent=3)
        store = FleetMemoryStore(config)
        for i in range(5):
            store.add_memory(f"memory {i}", "agent_1")

        all_mems = store.list_agent_memories("agent_1")
        assert len(all_mems) <= 3

    def test_embedding_consistency(self) -> None:
        store = FleetMemoryStore(MemoryConfig(db_path=":memory:"))
        emb1 = store._compute_embedding("hello world")
        emb2 = store._compute_embedding("hello world")
        assert np.allclose(emb1, emb2)
        assert abs(np.linalg.norm(emb1) - 1.0) < 1e-6  # unit vector

    def test_temporal_decay_boost(self) -> None:
        store = FleetMemoryStore(MemoryConfig(db_path=":memory:"))
        old = store.add_memory("old memory about python", "agent_1")
        # Manually backdate
        with store._lock:
            store._conn.execute(
                "UPDATE memories SET timestamp = ? WHERE memory_id = ?",
                (old.timestamp - 86400 * 10, old.memory_id),
            )
            store._conn.commit()

        recent = store.add_memory("recent memory about python", "agent_1")

        results = store.search_memories("python", agent_id="agent_1", top_k=2)
        # Recent should rank higher despite same keyword
        assert results[0].memory_id == recent.memory_id

    def test_search_time_range_filter(self) -> None:
        store = FleetMemoryStore(MemoryConfig(db_path=":memory:"))
        now = time.time()
        store.add_memory("today memory", "agent_1")
        # Backdate another
        old_entry = store.add_memory("old memory", "agent_1")
        with store._lock:
            store._conn.execute(
                "UPDATE memories SET timestamp = ? WHERE memory_id = ?",
                (now - 86400 * 30, old_entry.memory_id),
            )
            store._conn.commit()

        results = store.search_memories(
            "memory",
            agent_id="agent_1",
            time_range=(now - 86400, now + 1),
        )
        assert all(r.timestamp > now - 86400 for r in results)


# ═══════════════════════════════════════════════════════════
# AgentMemoryProfile
# ═══════════════════════════════════════════════════════════


class TestAgentMemoryProfile:
    def test_update_from_run_extracts_learnings(self) -> None:
        store = FleetMemoryStore(MemoryConfig(db_path=":memory:"))
        profile = AgentMemoryProfile(store, "agent_1")

        run_result = {
            "task_description": "fix wal query",
            "outcome": "success",
            "key_learnings": [
                "bisect insort preserves ordering",
                "wal indexes need parsers",
            ],
            "duration_seconds": 45.0,
        }
        entry = profile.update_from_run(run_result)
        assert "fix wal query" in entry.content
        assert "bisect insort" in entry.content

    def test_get_relevant_context(self) -> None:
        store = FleetMemoryStore(MemoryConfig(db_path=":memory:"))
        profile = AgentMemoryProfile(store, "agent_1")
        profile.update_from_run(
            {
                "task_description": "refactor pytest collection",
                "outcome": "success",
                "key_learnings": ["conftest.py controls fixtures"],
                "duration_seconds": 30.0,
            }
        )

        ctx = profile.get_relevant_context("pytest fixtures", top_k=1)
        assert len(ctx) > 0
        assert (
            "conftest" in ctx[0]["content"].lower()
            or "pytest" in ctx[0]["content"].lower()
        )

    def test_summarize_history(self) -> None:
        store = FleetMemoryStore(MemoryConfig(db_path=":memory:"))
        profile = AgentMemoryProfile(store, "agent_1")
        profile.update_from_run(
            {
                "task_description": "task A",
                "outcome": "success",
                "key_learnings": ["learning A"],
                "duration_seconds": 10.0,
            }
        )
        profile.update_from_run(
            {
                "task_description": "task B",
                "outcome": "failure",
                "key_learnings": ["learning B"],
                "duration_seconds": 20.0,
            }
        )

        summary = profile.summarize_history()
        assert summary["agent_id"] == "agent_1"
        assert summary["memory_count"] >= 2
        assert summary["outcome_distribution"]["success"] >= 1
        assert summary["outcome_distribution"]["failure"] >= 1
        assert "learning A" in summary["key_learnings"]

    def test_capabilities_accumulation(self) -> None:
        store = FleetMemoryStore(MemoryConfig(db_path=":memory:"))
        profile = AgentMemoryProfile(store, "agent_1")
        profile.update_from_run(
            {
                "task_description": "test",
                "outcome": "success",
                "key_learnings": ["distributed consensus requires quorum"],
                "duration_seconds": 5.0,
            }
        )

        prof = store.get_agent_profile("agent_1")
        # The first 3 words become a capability tag
        assert any(
            "distributed_consensus_requires" in cap for cap in prof["capabilities"]
        )


# ═══════════════════════════════════════════════════════════
# SenseDecideActMemory
# ═══════════════════════════════════════════════════════════


class TestSenseDecideActMemory:
    def test_enrich_sense_adds_memory_context(self) -> None:
        store = FleetMemoryStore(MemoryConfig(db_path=":memory:"))
        sda_mem = SenseDecideActMemory(store)
        store.add_memory("Always check pytest version before running", "agent_1")

        state = {"task_description": "run tests"}
        enriched = sda_mem.enrich_sense(state, "agent_1", top_k=1)
        assert "memory_context" in enriched
        assert enriched["memory_context"]["agent_id"] == "agent_1"
        assert len(enriched["memory_context"]["relevant_memories"]) > 0

    def test_log_action_outcome(self) -> None:
        store = FleetMemoryStore(MemoryConfig(db_path=":memory:"))
        sda_mem = SenseDecideActMemory(store)

        entry = sda_mem.log_action_outcome(
            action={"type": "dispatch", "description": "spawn subagent"},
            result={"success": True, "output": "done", "duration_seconds": 2.5},
            agent_id="agent_1",
        )
        assert entry.agent_id == "agent_1"
        assert "dispatch" in entry.content
        assert entry.metadata["success"] is True

    def test_enrich_sense_with_profile(self) -> None:
        store = FleetMemoryStore(MemoryConfig(db_path=":memory:"))
        sda_mem = SenseDecideActMemory(store)
        store.set_agent_profile("agent_1", role="test_builder", capabilities=["pytest"])
        store.add_memory("pytest -x stops on first failure", "agent_1")

        state = {"task_description": "debug failing test"}
        enriched = sda_mem.enrich_sense(state, "agent_1")
        assert enriched["memory_context"]["role"] == "test_builder"
        assert "pytest" in enriched["memory_context"]["capabilities"]


# ═══════════════════════════════════════════════════════════
# CrossAgentMemoryGossip
# ═══════════════════════════════════════════════════════════


class TestCrossAgentMemoryGossip:
    def test_share_memory_creates_shared_record(self) -> None:
        store = FleetMemoryStore(MemoryConfig(db_path=":memory:"))
        gossip = CrossAgentMemoryGossip(store)
        entry = store.add_memory("shared secret", "agent_alpha")

        targets = gossip.share_memory(
            entry.memory_id, "agent_alpha", ["agent_beta", "agent_gamma"]
        )
        assert "agent_beta" in targets
        assert "agent_gamma" in targets
        assert "agent_alpha" not in targets  # skip self

    def test_receive_gossip_merges_memories(self) -> None:
        store = FleetMemoryStore(MemoryConfig(db_path=":memory:"))
        gossip = CrossAgentMemoryGossip(store)

        payload = {
            "source_agent_id": "remote_node",
            "memories": [
                {
                    "memory_id": "mem_123",
                    "content": "Remote discovery: bisect is faster",
                    "agent_id": "remote_agent",
                    "timestamp": time.time(),
                    "metadata": {},
                    "embedding_b64": "",
                }
            ],
            "timestamp": time.time(),
        }
        ok = gossip.receive_gossip(payload)
        assert ok is True
        merged = store.get_memory("mem_123")
        assert merged is not None
        assert merged.metadata.get("gossip_source") == "remote_node"

    def test_receive_gossip_crdt_newer_wins(self) -> None:
        store = FleetMemoryStore(MemoryConfig(db_path=":memory:"))
        gossip = CrossAgentMemoryGossip(store)

        now = time.time()
        # Existing older memory
        store.add_memory("old content", "agent_1")
        with store._lock:
            store._conn.execute(
                "UPDATE memories SET memory_id = 'mem_same' WHERE agent_id = 'agent_1'"
            )
            store._conn.commit()

        # Newer gossip memory with same ID
        payload = {
            "source_agent_id": "peer",
            "memories": [
                {
                    "memory_id": "mem_same",
                    "content": "newer content",
                    "agent_id": "agent_1",
                    "timestamp": now + 10,
                    "metadata": {},
                    "embedding_b64": "",
                }
            ],
            "timestamp": now + 10,
        }
        gossip.receive_gossip(payload)
        merged = store.get_memory("mem_same")
        assert merged.content == "newer content"

    def test_get_shared_memories(self) -> None:
        store = FleetMemoryStore(MemoryConfig(db_path=":memory:"))
        gossip = CrossAgentMemoryGossip(store)
        entry = store.add_memory("share me", "agent_alpha")
        gossip.share_memory(entry.memory_id, "agent_alpha", ["agent_beta"])

        shared = gossip.get_shared_memories("agent_beta")
        assert len(shared) == 1
        assert shared[0].content == "share me"

    def test_build_gossip_payload(self) -> None:
        store = FleetMemoryStore(MemoryConfig(db_path=":memory:"))
        gossip = CrossAgentMemoryGossip(store)
        e1 = store.add_memory("fact one", "agent_1")
        e2 = store.add_memory("fact two", "agent_1")

        payload = gossip.build_gossip_payload([e1.memory_id, e2.memory_id], "agent_1")
        assert payload["source_agent_id"] == "agent_1"
        assert len(payload["memories"]) == 2
        assert "timestamp" in payload


# ═══════════════════════════════════════════════════════════
# Mem0Adapter
# ═══════════════════════════════════════════════════════════


class TestMem0Adapter:
    def test_initialize_for_fleet(self) -> None:
        adapter = Mem0Adapter()
        store = adapter.initialize_for_fleet({"db_path": ":memory:"})
        assert store is not None
        assert adapter.store is not None
        assert adapter.sda_memory is not None
        assert adapter.gossip is not None

    def test_attach_to_agent_identity(self) -> None:
        adapter = Mem0Adapter()
        adapter.initialize_for_fleet({"db_path": ":memory:"})

        class FakeIdentity:
            agent_id = "test_agent"

        identity = FakeIdentity()
        adapter.attach_to_agent_identity(identity)
        assert hasattr(identity, "memory_profile")
        assert hasattr(identity, "get_memory_context")

    def test_get_profile_creates_on_demand(self) -> None:
        adapter = Mem0Adapter()
        adapter.initialize_for_fleet({"db_path": ":memory:"})
        profile = adapter.get_profile("agent_42")
        assert profile.agent_id == "agent_42"

    def test_share_memory_convenience(self) -> None:
        adapter = Mem0Adapter()
        adapter.initialize_for_fleet({"db_path": ":memory:"})
        entry = adapter.store.add_memory("shared fact", "agent_1")
        targets = adapter.share_memory(
            entry.memory_id, "agent_1", ["agent_2", "agent_3"]
        )
        assert len(targets) == 2

    def test_build_sync_payload(self) -> None:
        adapter = Mem0Adapter()
        adapter.initialize_for_fleet({"db_path": ":memory:"})
        e1 = adapter.store.add_memory("one", "agent_1")
        e2 = adapter.store.add_memory("two", "agent_1")
        payload = adapter.build_sync_payload("agent_1", [e1.memory_id, e2.memory_id])
        assert len(payload["memories"]) == 2

    def test_stats(self) -> None:
        adapter = Mem0Adapter()
        assert adapter.stats()["initialized"] is False
        adapter.initialize_for_fleet({"db_path": ":memory:"})
        adapter.get_profile("a1")
        stats = adapter.stats()
        assert stats["initialized"] is True
        assert stats["agent_profiles"] == 1

    def test_memory_entry_roundtrip(self) -> None:
        emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        entry = MemoryEntry(
            memory_id="id",
            content="test",
            agent_id="a",
            timestamp=1.0,
            metadata={"k": "v"},
            embedding=emb,
        )
        d = entry.to_dict()
        restored = MemoryEntry.from_dict(d)
        assert restored.content == "test"
        assert np.allclose(restored.embedding, emb)

    def test_gossip_handler_registration_warns_on_missing_method(
        self, caplog: Any
    ) -> None:
        adapter = Mem0Adapter()
        adapter.initialize_for_fleet({"db_path": ":memory:"})

        class NoopGossip:
            pass

        adapter.attach_to_mesh_gossip(NoopGossip())
        assert "manual wiring required" in caplog.text

    def test_sda_attach_warns_on_missing_register(self, caplog: Any) -> None:
        adapter = Mem0Adapter()
        adapter.initialize_for_fleet({"db_path": ":memory:"})

        class NoopSDA:
            pass

        adapter.attach_to_sda_loop(NoopSDA())
        assert "manual wiring required" in caplog.text
