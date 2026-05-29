"""Tests for Mercury Consensus Specification.

Covers safety, liveness, QD proofs, and simulation helpers.
"""

import pytest

from fleet.mercury_consensus import ConsensusSpec, Determinism, NodeState


class TestDeterminism:
    def test_det(self):
        assert Determinism.DET == "det"

    def test_semidet(self):
        assert Determinism.SEMIDET == "semidet"


class TestInit:
    def test_three_nodes(self):
        spec = ConsensusSpec(nodes=["a", "b", "c"], f=1)
        assert spec.n == 3
        assert spec.quorum == 3
        assert len(spec.states) == 3

    def test_five_nodes(self):
        spec = ConsensusSpec(nodes=["a", "b", "c", "d", "e"], f=1)
        assert spec.n == 5
        assert spec.quorum == 3

    def test_seven_nodes(self):
        spec = ConsensusSpec(nodes=["a", "b", "c", "d", "e", "f", "g"], f=2)
        assert spec.n == 7
        assert spec.quorum == 5


class TestSafety:
    def test_empty_safe(self):
        spec = ConsensusSpec(nodes=["a", "b", "c"], f=1)
        assert spec.check_safety() is True

    def test_single_commit_safe(self):
        spec = ConsensusSpec(nodes=["a", "b", "c"], f=1)
        spec.simulate_commit("a", "value1", view=0)
        assert spec.check_safety() is True

    def test_same_value_same_view_safe(self):
        spec = ConsensusSpec(nodes=["a", "b", "c"], f=1)
        spec.simulate_commit("a", "value1", view=0)
        spec.simulate_commit("b", "value1", view=0)
        assert spec.check_safety() is True

    def test_different_values_same_view_unsafe(self):
        spec = ConsensusSpec(nodes=["a", "b", "c"], f=1)
        spec.simulate_commit("a", "value1", view=0)
        spec.simulate_commit("b", "value2", view=0)
        assert spec.check_safety() is False

    def test_byzantine_different_value_ignored(self):
        spec = ConsensusSpec(nodes=["a", "b", "c"], f=1)
        spec.simulate_byzantine("c")
        spec.simulate_commit("a", "value1", view=0)
        spec.simulate_commit("b", "value1", view=0)
        spec.simulate_commit("c", "evil_value", view=0)
        assert spec.check_safety() is True

    def test_different_views_safe(self):
        spec = ConsensusSpec(nodes=["a", "b", "c"], f=1)
        spec.simulate_commit("a", "value1", view=0)
        spec.simulate_commit("b", "value2", view=1)
        assert spec.check_safety() is True

    def test_safety_proof_format(self):
        spec = ConsensusSpec(nodes=["a", "b", "c"], f=1)
        proof = spec.safety_proof()
        assert "Safety Proof" in proof
        assert "3" in proof


class TestLiveness:
    def test_f_less_than_n_third(self):
        spec = ConsensusSpec(nodes=["a", "b", "c", "d"], f=1)
        assert spec.f < spec.n / 3

    def test_f_equal_n_third(self):
        spec = ConsensusSpec(nodes=["a", "b", "c"], f=1)
        assert spec.f >= spec.n / 3

    def test_liveness_holds(self):
        spec = ConsensusSpec(nodes=["a", "b", "c", "d"], f=1)
        spec.simulate_commit("a", "v1", view=0)
        spec.simulate_commit("b", "v1", view=0)
        assert spec.check_liveness(max_views=5) is True

    def test_liveness_violated_by_fault_tolerance(self):
        spec = ConsensusSpec(nodes=["a", "b", "c"], f=1)
        assert spec.check_liveness(max_views=5) is False

    def test_liveness_violated_by_timeout(self):
        spec = ConsensusSpec(nodes=["a", "b", "c", "d"], f=1)
        # No commits within 5 views — but view counter is 0, so we need to simulate view changes
        spec.simulate_view_change("a", 5)
        spec.simulate_view_change("b", 5)
        assert spec.check_liveness(max_views=5) is False

    def test_liveness_proof_format(self):
        spec = ConsensusSpec(nodes=["a", "b", "c", "d"], f=1)
        proof = spec.liveness_proof()
        assert "Liveness Proof" in proof
        assert "4" in proof


class TestQualityDiversity:
    def test_empty_archive(self):
        spec = ConsensusSpec(nodes=["a", "b"], f=0)
        assert spec.check_quality_diversity([]) is True

    def test_single_entry(self):
        spec = ConsensusSpec(nodes=["a", "b"], f=0)
        assert spec.check_quality_diversity([{"coverage": 0.5}]) is True

    def test_increasing_coverage(self):
        spec = ConsensusSpec(nodes=["a", "b"], f=0)
        archive = [
            {"coverage": 0.5},
            {"coverage": 0.6},
            {"coverage": 0.7},
        ]
        assert spec.check_quality_diversity(archive) is True

    def test_decreasing_coverage(self):
        spec = ConsensusSpec(nodes=["a", "b"], f=0)
        archive = [
            {"coverage": 0.7},
            {"coverage": 0.6},
            {"coverage": 0.5},
        ]
        assert spec.check_quality_diversity(archive) is False

    def test_qd_proof_format(self):
        spec = ConsensusSpec(nodes=["a", "b"], f=0)
        proof = spec.qd_proof([{"coverage": 0.5}])
        assert "Quality Diversity" in proof


class TestFullProof:
    def test_full_proof(self):
        spec = ConsensusSpec(nodes=["a", "b", "c", "d"], f=1)
        spec.simulate_commit("a", "v1", view=0)
        spec.simulate_commit("b", "v1", view=0)
        archive = [{"coverage": 0.5}, {"coverage": 0.6}]
        proof = spec.full_proof(archive)
        assert "Safety Proof" in proof
        assert "Liveness Proof" in proof
        assert "Quality Diversity" in proof

    def test_full_proof_without_archive(self):
        spec = ConsensusSpec(nodes=["a", "b", "c", "d"], f=1)
        proof = spec.full_proof()
        assert "Safety Proof" in proof
        assert "Liveness Proof" in proof


class TestSimulation:
    def test_simulate_commit(self):
        spec = ConsensusSpec(nodes=["a", "b"], f=0)
        spec.simulate_commit("a", "val", view=1)
        assert spec.states["a"].committed_value == "val"
        assert spec.states["a"].view == 1

    def test_simulate_byzantine(self):
        spec = ConsensusSpec(nodes=["a", "b"], f=0)
        spec.simulate_byzantine("a")
        assert spec.states["a"].is_byzantine is True

    def test_simulate_view_change(self):
        spec = ConsensusSpec(nodes=["a", "b"], f=0)
        spec.simulate_view_change("a", 5)
        assert spec.states["a"].view == 5

    def test_committed_log(self):
        spec = ConsensusSpec(nodes=["a", "b"], f=0)
        spec.simulate_commit("a", "v1", view=0)
        spec.simulate_commit("b", "v1", view=0)
        assert len(spec.committed_log) == 2

    def test_node_state_defaults(self):
        state = NodeState(node_id="a")
        assert state.view == 0
        assert state.committed_value is None
        assert state.is_byzantine is False
