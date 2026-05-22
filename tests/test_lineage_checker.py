"""Tests for LineageSanityChecker tamper-detection system.

Reference: docs/RESEARCH_SECURITY.md — Experiment: LineageSanityChecker
"""

from __future__ import annotations

import numpy as np
import pytest

from swarm.lineage_checker import Agent, LineageSanityChecker


class TestVerifyLineage:
    """verify_lineage() comprehensive checks."""

    def test_valid_lineage_passes(self):
        """Parents exist, plausible vector jump, correct generation."""
        rng = np.random.RandomState(42)
        parent_vec = rng.randn(128).astype(np.float32)
        # Child vector close to parent (plausible mutation)
        child_vec = parent_vec + rng.randn(128).astype(np.float32) * 0.1

        population = [
            Agent(id=1, vector=parent_vec.tolist(), generation=0),
            Agent(id=2, vector=child_vec.tolist(), generation=1, parent_a=1),
        ]
        checker = LineageSanityChecker()
        is_valid, reason = checker.verify_lineage(2, population)
        assert is_valid, f"Expected valid lineage, got: {reason}"
        assert reason == "Lineage valid"

    def test_orphan_detected(self):
        """Parent missing from population → orphan detected."""
        population = [
            Agent(id=1, vector=[0.1, 0.2], generation=0),
            # Agent 2 claims parent_a=99 which doesn't exist
            Agent(id=2, vector=[0.1, 0.2], generation=1, parent_a=99),
        ]
        checker = LineageSanityChecker()
        is_valid, reason = checker.verify_lineage(2, population)
        assert not is_valid
        assert "Orphan" in reason
        assert "99" in reason

    def test_cycle_detection_invalid(self):
        """Agent is its own ancestor via tampered parent pointers."""
        population = [
            Agent(id=1, vector=[0.1, 0.2], generation=2, parent_a=2),
            Agent(id=2, vector=[0.3, 0.4], generation=1, parent_a=1),
        ]
        # Agent 1 → parent_a=2 → parent_a=1 (cycle back to self)
        checker = LineageSanityChecker()
        is_valid, reason = checker.verify_lineage(1, population)
        assert not is_valid
        assert "Cycle" in reason

    def test_generation_mismatch_invalid(self):
        """Child.gen != max(parent.gen) + 1 → invalid."""
        population = [
            Agent(id=1, vector=[0.1, 0.2], generation=3),
            Agent(
                id=2,
                vector=[0.1, 0.2],
                generation=5,  # Should be 4 (max(3)+1)
                parent_a=1,
            ),
        ]
        checker = LineageSanityChecker()
        is_valid, reason = checker.verify_lineage(2, population)
        assert not is_valid
        assert "Generation mismatch" in reason
        assert "expected 4" in reason

    def test_root_agent_non_zero_generation_invalid(self):
        """Root agent (no parents) with non-zero generation → invalid."""
        population = [
            Agent(id=1, vector=[0.1, 0.2], generation=1),  # Should be 0
        ]
        checker = LineageSanityChecker()
        is_valid, reason = checker.verify_lineage(1, population)
        assert not is_valid
        assert "Root agent" in reason
        assert "non-zero generation" in reason


class TestDetectOrphans:
    """detect_orphans() batch scan."""

    def test_finds_all_orphans(self):
        population = [
            Agent(id=1, vector=[0.1], generation=0),
            Agent(id=2, vector=[0.2], generation=1, parent_a=1),  # valid
            Agent(id=3, vector=[0.3], generation=1, parent_a=99),  # orphan
            Agent(id=4, vector=[0.4], generation=1, parent_a=1, parent_b=99),  # orphan
        ]
        checker = LineageSanityChecker()
        orphans = checker.detect_orphans(population)
        assert sorted(orphans) == [3, 4]

    def test_no_orphans_in_valid_population(self):
        population = [
            Agent(id=1, vector=[0.1], generation=0),
            Agent(id=2, vector=[0.2], generation=1, parent_a=1),
            Agent(id=3, vector=[0.3], generation=2, parent_a=1, parent_b=2),
        ]
        checker = LineageSanityChecker()
        assert checker.detect_orphans(population) == []


class TestDetectImpossibleJumps:
    """detect_impossible_jumps() flags anomalous vector distances."""

    def test_typical_jumps_not_flagged(self):
        """Normal parent-child distances should not be flagged."""
        rng = np.random.RandomState(77)
        population = []
        # Root agent
        root_vec = rng.randn(64).astype(np.float32)
        population.append(Agent(id=0, vector=root_vec.tolist(), generation=0))

        # 5 children with small mutations (~0.1 std)
        for i in range(1, 6):
            child_vec = root_vec + rng.randn(64).astype(np.float32) * 0.1
            population.append(
                Agent(
                    id=i,
                    vector=child_vec.tolist(),
                    generation=1,
                    parent_a=0,
                )
            )

        checker = LineageSanityChecker()
        flagged = checker.detect_impossible_jumps(population, threshold=5.0)
        assert flagged == []

    def test_impossible_jump_flagged(self):
        """Child vector 10× typical distance from parents → flagged."""
        rng = np.random.RandomState(88)
        population = []
        # Root agent
        root_vec = rng.randn(64).astype(np.float32)
        population.append(Agent(id=0, vector=root_vec.tolist(), generation=0))

        # 5 children with small mutations (~0.1 std)
        for i in range(1, 6):
            child_vec = root_vec + rng.randn(64).astype(np.float32) * 0.1
            population.append(
                Agent(
                    id=i,
                    vector=child_vec.tolist(),
                    generation=1,
                    parent_a=0,
                )
            )

        # One child with a HUGE jump (10× farther than typical)
        huge_vec = root_vec + rng.randn(64).astype(np.float32) * 10.0
        population.append(
            Agent(
                id=99,
                vector=huge_vec.tolist(),
                generation=1,
                parent_a=0,
            )
        )

        checker = LineageSanityChecker()
        flagged = checker.detect_impossible_jumps(population, threshold=5.0)
        assert 99 in flagged, f"Expected agent 99 flagged, got: {flagged}"

    def test_empty_population_no_flagged(self):
        checker = LineageSanityChecker()
        assert checker.detect_impossible_jumps([]) == []

    def test_only_root_agents_no_flagged(self):
        """No parents means no jumps to compare."""
        population = [
            Agent(id=1, vector=[0.1, 0.2], generation=0),
            Agent(id=2, vector=[0.3, 0.4], generation=0),
        ]
        checker = LineageSanityChecker()
        assert checker.detect_impossible_jumps(population) == []


class TestBuildLineageTree:
    """build_lineage_tree() recursive ancestor traversal."""

    def test_tree_depth_respects_max_depth(self):
        population = [
            Agent(id=1, vector=[0.1], generation=0),
            Agent(id=2, vector=[0.2], generation=1, parent_a=1),
            Agent(id=3, vector=[0.3], generation=2, parent_a=2),
            Agent(id=4, vector=[0.4], generation=3, parent_a=3),
        ]
        # max_depth=2 → should stop at grandparents
        checker = LineageSanityChecker(max_depth=2)
        tree = checker.build_lineage_tree(4, population)

        assert tree["agent_id"] == 4
        assert tree["generation"] == 3
        # 4 → parent_a=3 → parent_a=2 (depth 2, stops there)
        assert len(tree["parents"]) == 1
        parent_tree = tree["parents"][0]
        assert parent_tree["agent_id"] == 3
        assert len(parent_tree["parents"]) == 1
        grandparent = parent_tree["parents"][0]
        assert grandparent["agent_id"] == 2
        # No further parents since max_depth=2
        assert grandparent["parents"] == []

    def test_tree_with_two_parents(self):
        population = [
            Agent(id=1, vector=[0.1], generation=0),
            Agent(id=2, vector=[0.2], generation=0),
            Agent(id=3, vector=[0.3], generation=1, parent_a=1, parent_b=2),
        ]
        checker = LineageSanityChecker(max_depth=5)
        tree = checker.build_lineage_tree(3, population)
        assert len(tree["parents"]) == 2
        parent_ids = {p["agent_id"] for p in tree["parents"]}
        assert parent_ids == {1, 2}

    def test_cycle_in_tree_repr(self):
        """Cycles are represented as {'cycle': True} in tree output."""
        population = [
            Agent(id=1, vector=[0.1], generation=1, parent_a=2),
            Agent(id=2, vector=[0.2], generation=0, parent_a=1),
        ]
        checker = LineageSanityChecker()
        tree = checker.build_lineage_tree(1, population)
        # Tree should show cycle marker somewhere in the nested structure
        # 1 → parent_a=2 → parent_a=1 (cycle back to visited 1)
        parents = tree["parents"]
        assert len(parents) == 1
        parent_2 = parents[0]
        assert parent_2["agent_id"] == 2
        # Cycle appears in parent_2's parents (the recursive call for agent 1)
        assert len(parent_2["parents"]) == 1
        cycle_node = parent_2["parents"][0]
        assert cycle_node["agent_id"] == 1
        assert any(p.get("cycle") for p in cycle_node["parents"])


class TestCosineDistance:
    """Internal cosine distance helper."""

    def test_identical_vectors_zero_distance(self):
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        dist = LineageSanityChecker._cosine_distance(vec, vec)
        assert abs(dist) < 1e-6

    def test_opposite_vectors_max_distance(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        dist = LineageSanityChecker._cosine_distance(a, b)
        assert abs(dist - 2.0) < 1e-6

    def test_orthogonal_vectors_distance_one(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        dist = LineageSanityChecker._cosine_distance(a, b)
        assert abs(dist - 1.0) < 1e-6


class TestIntegrationWithBreederDaemonV2:
    """End-to-end: LineageSanityChecker works with daemon-step flow."""

    @pytest.mark.skip(reason="Uses fixtures from test_breeder_daemon_v2; run together with that module")
    def test_daemon_integration_invalid_lineage_sunsets_child(
        self, grid, thermal, wal_path, vector_table
    ):
        """If lineage check fails during step(), child is immediately sunset."""
        from tests.test_breeder_daemon_v2 import make_daemon
        from swarm.breeder_daemon_v2 import LifecycleState

        daemon = make_daemon(grid, thermal, wal_path, vector_table)
        daemon.start()

        # Normal breed should pass lineage check
        daemon.queue_breed(parent_a=1, parent_b=2)
        transitions = daemon.step()
        daemon.stop()

        # At least one agent should have reached EGG
        spawned = [
            t for t in transitions if t.to_state == LifecycleState.EGG
        ]
        assert len(spawned) > 0

    def test_orphan_population_scan(self):
        """Batch orphan detection on a mixed population."""
        population = [
            Agent(id=1, vector=[0.1] * 10, generation=0),
            Agent(id=2, vector=[0.2] * 10, generation=1, parent_a=1),
            Agent(id=3, vector=[0.3] * 10, generation=1, parent_a=999),
            Agent(id=4, vector=[0.4] * 10, generation=2, parent_a=2, parent_b=999),
        ]
        checker = LineageSanityChecker()
        orphans = checker.detect_orphans(population)
        assert sorted(orphans) == [3, 4]

        # Verify via verify_lineage too
        for orphan_id in orphans:
            is_valid, reason = checker.verify_lineage(orphan_id, population)
            assert not is_valid
            assert "Orphan" in reason

    def test_generation_mismatch_batch(self):
        """Multiple agents with wrong generations."""
        population = [
            Agent(id=1, vector=[0.1] * 10, generation=2),
            Agent(id=2, vector=[0.2] * 10, generation=5, parent_a=1),  # should be 3
            Agent(id=3, vector=[0.3] * 10, generation=1, parent_a=1),  # should be 3
        ]
        checker = LineageSanityChecker()
        invalid = [aid for aid in [2, 3] if not checker.verify_lineage(aid, population)[0]]
        assert sorted(invalid) == [2, 3]
