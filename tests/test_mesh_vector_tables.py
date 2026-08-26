"""Tests for MeshVectorTables — federated CRDT-based vector tables.

Run with: pytest tests/test_mesh_vector_tables.py -v
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any

import numpy as np
import pytest

from swarm.mesh_vector_tables import (
    VectorTableEntry,
    MeshVectorTable,
    FleetVectorIndex,
    SignatureError,
)

# AgentIdentity is optional (cryptography may not be installed)
try:
    from a2a.identity import AgentIdentity

    _HAS_IDENTITY = True
except Exception:
    _HAS_IDENTITY = False


# ── fixtures ────────────────────────────────────────────────────


@pytest.fixture
def sample_vector() -> np.ndarray:
    """A deterministic 64-dim float32 vector."""
    return np.linspace(-1, 1, 64, dtype=np.float32)


@pytest.fixture
def diverse_vectors() -> list[np.ndarray]:
    """10 diverse 64-dim vectors."""
    return [np.random.randn(64).astype(np.float32) for _ in range(10)]


@pytest.fixture
def mock_identity(tmp_path) -> Any:
    """Return a real AgentIdentity if crypto is available, else None."""
    if _HAS_IDENTITY:
        return AgentIdentity(agent_id="test_node", base_dir=str(tmp_path / "keys"))
    return None


@pytest.fixture
def sample_entry(sample_vector, mock_identity) -> VectorTableEntry:
    """A valid VectorTableEntry (signed if identity available)."""
    payload = {
        "agent_id": "Oracle1::agent_42",
        "vector_b64": "",
        "timestamp": 1000.0,
        "node_id": "Oracle1",
        "generation": 3,
        "fitness": 0.85,
        "capability_mask": 0xFFFF,
        "thermal_pressure": 0.2,
        "extra": {},
    }
    sig = ""
    if mock_identity is not None:
        sig = mock_identity.sign_task(payload)
    else:
        # SHA-256 fallback when crypto unavailable
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        sig = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:64]

    return VectorTableEntry(
        agent_id="Oracle1::agent_42",
        vector=sample_vector,
        timestamp=1000.0,
        node_id="Oracle1",
        generation=3,
        fitness=0.85,
        signature=sig,
        capability_mask=0xFFFF,
        thermal_pressure=0.2,
    )


# ── VectorTableEntry tests ──────────────────────────────────────


class TestVectorTableEntry:
    def test_to_dict_from_dict_roundtrip(self, sample_entry):
        """Serializing and deserializing an entry yields the same data."""
        d = sample_entry.to_dict()
        restored = VectorTableEntry.from_dict(d)
        assert restored.agent_id == sample_entry.agent_id
        assert restored.timestamp == sample_entry.timestamp
        assert restored.node_id == sample_entry.node_id
        assert restored.generation == sample_entry.generation
        assert restored.fitness == pytest.approx(sample_entry.fitness)
        assert restored.signature == sample_entry.signature
        assert restored.capability_mask == sample_entry.capability_mask
        assert restored.thermal_pressure == pytest.approx(sample_entry.thermal_pressure)
        np.testing.assert_array_almost_equal(restored.vector, sample_entry.vector)

    def test_canonical_payload_excludes_signature(self, sample_entry):
        """The canonical payload must not contain the signature field."""
        payload = sample_entry.canonical_payload()
        assert "signature" not in payload
        assert payload["agent_id"] == sample_entry.agent_id


# ── MeshVectorTable basic tests ─────────────────────────────────


class TestMeshVectorTableBasics:
    def test_insert_and_query_roundtrip(self, sample_entry):
        """Inserting an entry makes it queryable."""
        table = MeshVectorTable(table_id="test")
        ok = table.insert(sample_entry, skip_verify=True)
        assert ok is True
        found = table.query(sample_entry.agent_id)
        assert found is not None
        assert found.agent_id == sample_entry.agent_id

    def test_query_missing_returns_none(self):
        """Querying a non-existent agent returns None."""
        table = MeshVectorTable(table_id="test")
        assert table.query("ghost") is None

    def test_query_by_fitness_sorted_descending(self, diverse_vectors):
        """Fitness query returns results ordered best-first."""
        table = MeshVectorTable(table_id="test")
        for i, vec in enumerate(diverse_vectors):
            entry = VectorTableEntry(
                agent_id=f"agent_{i}",
                vector=vec,
                timestamp=1000.0 + i,
                node_id="Oracle1",
                generation=1,
                fitness=0.5 + i * 0.05,
                signature=f"sig_{i:04d}",
            )
            table.insert(entry, skip_verify=True)

        results = table.query_by_fitness(min_fitness=0.6, max_results=5)
        assert len(results) == 5
        fitnesses = [e.fitness for e in results]
        assert fitnesses == sorted(fitnesses, reverse=True)
        assert all(f >= 0.6 for f in fitnesses)

    def test_query_by_diversity_filters_distance(self, diverse_vectors):
        """Diversity query excludes vectors too close to the reference."""
        table = MeshVectorTable(table_id="test")
        ref = diverse_vectors[0]
        for i, vec in enumerate(diverse_vectors):
            entry = VectorTableEntry(
                agent_id=f"agent_{i}",
                vector=vec,
                timestamp=1000.0,
                node_id="Oracle1",
                generation=1,
                fitness=0.8,
                signature=f"sig_{i:04d}",
            )
            table.insert(entry, skip_verify=True)

        # Use a large min_distance so some entries are excluded
        results = table.query_by_diversity(
            reference_vector=ref, min_distance=1.0, max_results=50
        )
        assert len(results) <= 9  # at least the reference itself is excluded
        for entry in results:
            dist = float(np.linalg.norm(ref - entry.vector))
            assert dist >= 1.0

    def test_population_summary_accurate(self, diverse_vectors):
        """Population summary reflects actual inserted data."""
        table = MeshVectorTable(table_id="test")
        for i, vec in enumerate(diverse_vectors):
            entry = VectorTableEntry(
                agent_id=f"agent_{i}",
                vector=vec,
                timestamp=1000.0,
                node_id="Oracle1" if i < 5 else "ProArt",
                generation=1,
                fitness=0.5 + i * 0.01,
                signature=f"sig_{i:04d}",
            )
            table.insert(entry, skip_verify=True)

        summary = table.get_population_summary()
        assert summary["count"] == 10
        assert summary["mean_fitness"] == pytest.approx(0.545, abs=0.01)
        assert summary["node_breakdown"]["Oracle1"] == 5
        assert summary["node_breakdown"]["ProArt"] == 5
        assert summary["generation_range"] == (1, 1)
        assert 0.0 <= summary["diversity_score"] <= 1.0

    def test_empty_table_summary(self):
        """An empty table returns sensible zero-valued summary."""
        table = MeshVectorTable(table_id="test")
        summary = table.get_population_summary()
        assert summary["count"] == 0
        assert summary["mean_fitness"] == 0.0
        assert summary["diversity_score"] == 0.0
        assert summary["node_breakdown"] == {}
        assert summary["generation_range"] is None


# ── CRDT merge tests ────────────────────────────────────────────


class TestCRDTMerge:
    def test_newer_timestamp_wins(self, sample_vector):
        """When timestamps differ, the newer entry wins."""
        table = MeshVectorTable(table_id="test")
        old = VectorTableEntry(
            agent_id="agent_A",
            vector=sample_vector,
            timestamp=1000.0,
            node_id="Oracle1",
            generation=1,
            fitness=0.5,
            signature="sig_old_000",
        )
        new = VectorTableEntry(
            agent_id="agent_A",
            vector=sample_vector + 0.1,
            timestamp=2000.0,
            node_id="ProArt",
            generation=2,
            fitness=0.9,
            signature="sig_new_000",
        )
        table.insert(old, skip_verify=True)
        table.insert(new, skip_verify=True)

        found = table.query("agent_A")
        assert found is not None
        assert found.timestamp == 2000.0
        assert found.fitness == pytest.approx(0.9)
        assert found.node_id == "ProArt"

    def test_tiebreak_by_signature_hash(self, sample_vector):
        """When timestamps are equal, lower signature hash wins."""
        table = MeshVectorTable(table_id="test")
        # Two entries with identical timestamps
        sig_a = "aaaaaaaa" + "a" * 56  # 64 chars
        sig_b = "bbbbbbbb" + "b" * 56  # 64 chars
        hash_a = hashlib.sha256(sig_a.encode("utf-8")).hexdigest()
        hash_b = hashlib.sha256(sig_b.encode("utf-8")).hexdigest()
        # Deterministic ordering: we sort the actual hashes
        if hash_a > hash_b:
            sig_a, sig_b = sig_b, sig_a
            hash_a, hash_b = hash_b, hash_a

        entry_a = VectorTableEntry(
            agent_id="agent_A",
            vector=sample_vector,
            timestamp=1000.0,
            node_id="Oracle1",
            generation=1,
            fitness=0.5,
            signature=sig_a,
        )
        entry_b = VectorTableEntry(
            agent_id="agent_A",
            vector=sample_vector + 0.1,
            timestamp=1000.0,
            node_id="ProArt",
            generation=2,
            fitness=0.9,
            signature=sig_b,
        )
        table.insert(entry_b, skip_verify=True)
        table.insert(entry_a, skip_verify=True)

        found = table.query("agent_A")
        # entry_a should win because its signature hash is lower
        assert found is not None
        assert found.signature == sig_a
        assert found.node_id == "Oracle1"

    def test_merge_remote_table(self, diverse_vectors):
        """Merging a full remote table brings in all entries."""
        local = MeshVectorTable(table_id="gen_1")
        remote = MeshVectorTable(table_id="gen_1")

        for i, vec in enumerate(diverse_vectors[:5]):
            entry = VectorTableEntry(
                agent_id=f"agent_{i}",
                vector=vec,
                timestamp=1000.0 + i,
                node_id="Oracle1",
                generation=1,
                fitness=0.6,
                signature=f"sig_{i:04d}",
            )
            local.insert(entry, skip_verify=True)

        for i, vec in enumerate(diverse_vectors[5:]):
            entry = VectorTableEntry(
                agent_id=f"agent_{i + 5}",
                vector=vec,
                timestamp=1000.0 + i,
                node_id="ProArt",
                generation=1,
                fitness=0.7,
                signature=f"sig_{i + 5:04d}",
            )
            remote.insert(entry, skip_verify=True)

        stats = local.merge_remote_table(remote)
        assert stats["merged"] == 5
        assert stats["rejected"] == 0
        assert len(local) == 10

    def test_merge_remote_with_conflict(self, sample_vector):
        """Remote merge respects CRDT rules for overlapping agents."""
        local = MeshVectorTable(table_id="gen_1")
        remote = MeshVectorTable(table_id="gen_1")

        shared_id = "agent_shared"
        local_entry = VectorTableEntry(
            agent_id=shared_id,
            vector=sample_vector,
            timestamp=1000.0,
            node_id="Oracle1",
            generation=1,
            fitness=0.5,
            signature="sig_local_000",
        )
        remote_entry = VectorTableEntry(
            agent_id=shared_id,
            vector=sample_vector + 0.1,
            timestamp=2000.0,
            node_id="ProArt",
            generation=2,
            fitness=0.9,
            signature="sig_remote_00",
        )
        local.insert(local_entry, skip_verify=True)
        remote.insert(remote_entry, skip_verify=True)

        stats = local.merge_remote_table(remote)
        # Remote wins (newer timestamp), so the existing entry is overwritten
        assert stats["merged"] == 1  # existing overwritten by winner
        assert stats["rejected"] == 0
        found = local.query(shared_id)
        assert found.fitness == pytest.approx(0.9)


# ── sync payload tests ──────────────────────────────────────────


class TestSyncPayload:
    def test_sync_payload_roundtrip(self, diverse_vectors):
        """Compressing and decompressing a payload preserves all entries."""
        table = MeshVectorTable(table_id="gen_1")
        for i, vec in enumerate(diverse_vectors):
            entry = VectorTableEntry(
                agent_id=f"agent_{i}",
                vector=vec,
                timestamp=1000.0 + i,
                node_id="Oracle1",
                generation=1,
                fitness=0.6,
                signature=f"sig_{i:04d}",
            )
            table.insert(entry, skip_verify=True)

        payload = table.get_sync_payload()
        assert isinstance(payload, bytes)
        assert len(payload) > 0

        # Apply to a fresh table (use skip_verify=False with long signatures)
        fresh = MeshVectorTable(table_id="gen_1")
        stats = fresh.apply_sync_payload(payload)
        assert stats["merged"] == 10
        assert stats["errors"] == []
        assert len(fresh) == 10

        # Verify individual entry
        found = fresh.query("agent_3")
        assert found is not None
        np.testing.assert_array_almost_equal(found.vector, diverse_vectors[3])

    def test_fleet_sync_payload_roundtrip(self, diverse_vectors):
        """Fleet-wide sync covers all generation tables."""
        index = FleetVectorIndex(node_id="Oracle1")
        for i, vec in enumerate(diverse_vectors[:5]):
            entry = VectorTableEntry(
                agent_id=f"agent_{i}",
                vector=vec,
                timestamp=1000.0,
                node_id="Oracle1",
                generation=1,
                fitness=0.6,
                signature=f"sig_{i:04d}",
            )
            index.insert_fleet_entry(entry)

        for i, vec in enumerate(diverse_vectors[5:]):
            entry = VectorTableEntry(
                agent_id=f"agent_{i + 5}",
                vector=vec,
                timestamp=1000.0,
                node_id="ProArt",
                generation=2,
                fitness=0.7,
                signature=f"sig_{i + 5:04d}",
            )
            index.insert_fleet_entry(entry)

        payload = index.get_fleet_sync_payload()
        fresh = FleetVectorIndex(node_id="Jetson1")
        stats = fresh.apply_fleet_sync_payload(payload)
        assert "per_gen" in stats
        assert stats["per_gen"]["1"]["merged"] == 5
        assert stats["per_gen"]["2"]["merged"] == 5
        assert fresh.stats["total_entries"] == 10


# ── FleetVectorIndex tests ──────────────────────────────────────


class TestFleetVectorIndex:
    def test_breedable_pool_respects_all_filters(self, diverse_vectors):
        """get_breedable_pool filters by fitness, thermal, and diversity."""
        index = FleetVectorIndex(node_id="Oracle1")
        for i, vec in enumerate(diverse_vectors):
            entry = VectorTableEntry(
                agent_id=f"agent_{i}",
                vector=vec,
                timestamp=1000.0,
                node_id="Oracle1" if i < 5 else "ProArt",
                generation=1,
                fitness=0.5 + i * 0.05,
                signature=f"sig_{i:04d}",
                thermal_pressure=0.1 if i < 5 else 0.8,
            )
            index.insert_fleet_entry(entry)

        pool = index.get_breedable_pool(
            min_fitness=0.65,
            max_thermal=0.5,
            diversity_target=0.0,  # disable diversity filter for this test
            max_results=10,
        )
        # Only agents 0-4 have thermal <= 0.5; of those, only i>=3 have fitness >= 0.65
        assert all(e.fitness >= 0.65 for e in pool)
        assert all(e.thermal_pressure <= 0.5 for e in pool)
        assert all(e.node_id == "Oracle1" for e in pool)

    def test_capability_map_spans_multiple_nodes(self, diverse_vectors):
        """get_capability_map aggregates agents from all nodes."""
        index = FleetVectorIndex(node_id="Oracle1")
        # Use a fixed skill name so we know which bit it maps to
        for i, vec in enumerate(diverse_vectors):
            # Set capability mask so all agents have the skill
            entry = VectorTableEntry(
                agent_id=f"agent_{i}",
                vector=vec,
                timestamp=1000.0,
                node_id="Oracle1" if i < 3 else "ProArt",
                generation=1,
                fitness=0.6,
                signature=f"sig_{i:04d}",
                capability_mask=0xFFFF,  # all bits set = all skills
            )
            index.insert_fleet_entry(entry)

        cap_map = index.get_capability_map("flux_vm")
        assert "Oracle1" in cap_map
        assert "ProArt" in cap_map
        assert len(cap_map["Oracle1"]) == 3
        assert len(cap_map["ProArt"]) == 7

    def test_novelty_score_decreases_as_population_grows(self, sample_vector):
        """Adding more agents lowers the novelty of a fixed vector."""
        index = FleetVectorIndex(node_id="Oracle1")
        vec = sample_vector

        score_empty = index.get_novelty_score(vec)
        assert score_empty == 1.0  # empty fleet = max novelty

        # Add one agent with a very different vector
        entry1 = VectorTableEntry(
            agent_id="agent_1",
            vector=np.ones_like(vec) * 2.0,
            timestamp=1000.0,
            node_id="Oracle1",
            generation=1,
            fitness=0.6,
            signature="sig_0001",
        )
        index.insert_fleet_entry(entry1)
        score_one = index.get_novelty_score(vec)

        # Add many more agents clustered around the centroid
        for i in range(20):
            v = np.random.randn(len(vec)).astype(np.float32) * 0.1
            entry = VectorTableEntry(
                agent_id=f"agent_{i + 2}",
                vector=v,
                timestamp=1000.0,
                node_id="Oracle1",
                generation=1,
                fitness=0.6,
                signature=f"sig_{i + 2:04d}",
            )
            index.insert_fleet_entry(entry)

        score_many = index.get_novelty_score(vec)
        assert score_many < score_one
        assert 0.0 <= score_many <= 1.0

    def test_insert_fleet_entry_cross_generations(self, diverse_vectors):
        """Entries are routed to the correct generation table."""
        index = FleetVectorIndex(node_id="Oracle1")
        for i, vec in enumerate(diverse_vectors[:5]):
            entry = VectorTableEntry(
                agent_id=f"gen1_{i}",
                vector=vec,
                timestamp=1000.0,
                node_id="Oracle1",
                generation=1,
                fitness=0.6,
                signature=f"sig_{i:04d}",
            )
            index.insert_fleet_entry(entry)

        for i, vec in enumerate(diverse_vectors[5:]):
            entry = VectorTableEntry(
                agent_id=f"gen2_{i}",
                vector=vec,
                timestamp=1000.0,
                node_id="ProArt",
                generation=2,
                fitness=0.7,
                signature=f"sig_{i + 5:04d}",
            )
            index.insert_fleet_entry(entry)

        assert len(index.get_gen_table(1)) == 5
        assert len(index.get_gen_table(2)) == 5
        assert index.stats["gen_tables"] == 2


# ── thread-safety tests ─────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_inserts(self, diverse_vectors):
        """Many threads inserting simultaneously do not corrupt the table."""
        table = MeshVectorTable(table_id="stress")
        errors: list[Exception] = []

        def worker(start_idx: int) -> None:
            try:
                for i in range(20):
                    vec = np.random.randn(64).astype(np.float32)
                    entry = VectorTableEntry(
                        agent_id=f"thread_{start_idx}_agent_{i}",
                        vector=vec,
                        timestamp=time.time(),
                        node_id=f"node_{start_idx}",
                        generation=1,
                        fitness=0.5 + i * 0.01,
                        signature=f"sig_{start_idx}_{i:04d}",
                    )
                    table.insert(entry, skip_verify=True)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(table) == 160  # 8 threads × 20 entries
        summary = table.get_population_summary()
        assert summary["count"] == 160


# ── signature tests ────────────────────────────────────────────


class TestSignatures:
    def test_invalid_signature_rejected(self, sample_vector):
        """An entry with a bad signature is rejected when verification is on."""
        table = MeshVectorTable(table_id="test")
        entry = VectorTableEntry(
            agent_id="bad_agent",
            vector=sample_vector,
            timestamp=1000.0,
            node_id="Oracle1",
            generation=1,
            fitness=0.5,
            signature="INVALID",  # 7 chars < 8 minimum for fallback
        )
        # Without an identity, the fallback just checks length >= 8
        # "INVALID" is only 7 chars, so it should raise
        with pytest.raises(SignatureError):
            table.insert(entry, skip_verify=False)

    def test_insert_signed_auto_signs(self, mock_identity, sample_vector):
        """insert_signed produces a valid signature when identity is present."""
        if mock_identity is None:
            pytest.skip("cryptography not available")
        table = MeshVectorTable(table_id="test", identity=mock_identity)
        entry = table.insert_signed(
            agent_id="auto_agent",
            vector=sample_vector,
            node_id="Oracle1",
            generation=1,
            fitness=0.9,
        )
        assert entry.signature != ""
        assert len(entry.signature) >= 16
        # Should be queryable
        found = table.query("auto_agent")
        assert found is not None
        assert found.fitness == pytest.approx(0.9)


# ── distance metric tests ───────────────────────────────────────


class TestDistanceMetrics:
    def test_euclidean_vs_cosine_distance(self):
        """Distance helper returns different values for different metrics."""
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        euclid = MeshVectorTable._compute_distance(a, b, metric="euclidean")
        cosine = MeshVectorTable._compute_distance(a, b, metric="cosine")
        assert euclid == pytest.approx(np.sqrt(2))
        assert cosine == pytest.approx(1.0)  # orthogonal vectors

    def test_cosine_identical_vectors(self):
        """Cosine distance of identical vectors is 0."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        d = MeshVectorTable._compute_distance(a, a, metric="cosine")
        assert d == pytest.approx(0.0, abs=1e-6)


# ── edge case tests ─────────────────────────────────────────────


class TestEdgeCases:
    def test_identical_entries_noop(self, sample_vector):
        """Inserting the exact same entry twice is a no-op after the first."""
        table = MeshVectorTable(table_id="test")
        entry = VectorTableEntry(
            agent_id="same",
            vector=sample_vector,
            timestamp=1000.0,
            node_id="Oracle1",
            generation=1,
            fitness=0.5,
            signature="sig_same_00",
        )
        ok1 = table.insert(entry, skip_verify=True)
        ok2 = table.insert(entry, skip_verify=True)
        assert ok1 is True
        # CRDT winner is existing because timestamps are equal and
        # signature hashes are identical — the first entry wins
        assert ok2 is False
        assert len(table) == 1

    def test_merge_empty_table(self):
        """Merging an empty remote table is harmless."""
        local = MeshVectorTable(table_id="gen_1")
        remote = MeshVectorTable(table_id="gen_1")
        stats = local.merge_remote_table(remote)
        assert stats == {"merged": 0, "rejected": 0, "skipped": 0}

    def test_diversity_query_empty_table(self):
        """Diversity query on empty table returns empty list."""
        table = MeshVectorTable(table_id="test")
        results = table.query_by_diversity(
            reference_vector=np.zeros(64, dtype=np.float32),
            min_distance=0.1,
        )
        assert results == []

    def test_breedable_pool_empty_fleet(self, sample_vector):
        """Breedable pool on empty fleet returns empty list."""
        index = FleetVectorIndex(node_id="Oracle1")
        pool = index.get_breedable_pool(min_fitness=0.0, max_thermal=1.0)
        assert pool == []


# ── performance / scale smoke tests ───────────────────────────


class TestScale:
    def test_large_population_summary(self):
        """Population summary stays fast for 500 agents."""
        table = MeshVectorTable(table_id="scale")
        dim = 256
        for i in range(500):
            vec = np.random.randn(dim).astype(np.float32)
            entry = VectorTableEntry(
                agent_id=f"agent_{i}",
                vector=vec,
                timestamp=1000.0 + i,
                node_id="Oracle1",
                generation=1,
                fitness=0.5 + (i % 100) / 200.0,
                signature=f"sig_{i:04d}",
            )
            table.insert(entry, skip_verify=True)

        summary = table.get_population_summary()
        assert summary["count"] == 500
        assert 0.0 < summary["diversity_score"] <= 1.0
        # Trigger fitness index rebuild
        _ = table.query_by_fitness(min_fitness=0.0, max_results=1)
        assert len(table._fitness_index) == 500
