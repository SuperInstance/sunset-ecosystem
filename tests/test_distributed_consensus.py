"""Tests for HolonomyConsensus distributed consensus."""
from __future__ import annotations

import pytest

from nexus.distributed_consensus import HolonomyConsensus, Proposal, Vote


class TestProposal:
    def test_proposal_digest_is_stable(self):
        p1 = Proposal(seq_num=1, operation="test", payload={"a": 1}, proposer="n1", timestamp=1.0)
        p2 = Proposal(seq_num=1, operation="test", payload={"a": 1}, proposer="n1", timestamp=1.0)
        assert p1.digest() == p2.digest()

    def test_proposal_digest_changes_with_content(self):
        p1 = Proposal(seq_num=1, operation="test", payload={"a": 1}, proposer="n1", timestamp=1.0)
        p2 = Proposal(seq_num=1, operation="test", payload={"a": 2}, proposer="n1", timestamp=1.0)
        assert p1.digest() != p2.digest()


class TestBasicConsensus:
    def test_quorum_for_4_nodes_is_3(self):
        c = HolonomyConsensus("n1", ["n2", "n3", "n4"])
        assert c.n_nodes == 4
        assert c.f_byzantine == 1
        assert c.quorum == 3

    def test_quorum_for_7_nodes_is_5(self):
        c = HolonomyConsensus("n1", ["n2", "n3", "n4", "n5", "n6", "n7"])
        assert c.n_nodes == 7
        assert c.f_byzantine == 2
        assert c.quorum == 5

    def test_commit_reaches_quorum(self):
        c = HolonomyConsensus("n1", ["n2", "n3", "n4"])
        prop = c.propose_state_change("resize", {"n": 100})

        # Simulate 3 votes (quorum = 3 for N=4)
        for _ in range(3):
            c.vote_on_proposal(prop.digest(), approve=True)

        result = c.commit_if_quorum(prop.digest())
        assert result.committed
        assert result.votes_for == 3
        assert result.proposal is not None

    def test_commit_fails_without_quorum(self):
        c = HolonomyConsensus("n1", ["n2", "n3", "n4"])
        prop = c.propose_state_change("resize", {"n": 100})
        c.vote_on_proposal(prop.digest(), approve=True)

        result = c.commit_if_quorum(prop.digest())
        assert not result.committed
        assert result.votes_for == 1


class TestByzantineFaults:
    def test_tolerates_one_byzantine_in_4_nodes(self):
        c = HolonomyConsensus("n1", ["n2", "n3", "n4"])
        prop = c.propose_state_change("resize", {"n": 100})

        # 2 honest + 1 byzantine (no vote)
        c.vote_on_proposal(prop.digest(), approve=True)
        c.vote_on_proposal(prop.digest(), approve=True)
        # Byzantine node withholds vote

        # Need 3 votes for quorum, only have 2
        result = c.commit_if_quorum(prop.digest())
        assert not result.committed

    def test_detects_conflicting_votes(self):
        c = HolonomyConsensus("n1", ["n2", "n3", "n4"])
        prop = c.propose_state_change("resize", {"n": 100})

        c.vote_on_proposal(prop.digest(), approve=True)
        c.vote_on_proposal(prop.digest(), approve=False)  # conflicting

        status = c.detect_emergence()
        assert status["conflicting_votes"] >= 1


class TestPartitionHandling:
    def test_partition_reduces_quorum(self):
        c = HolonomyConsensus("n1", ["n2", "n3", "n4", "n5", "n6", "n7"])
        assert c.quorum == 5  # N=7

        c.handle_partition(["n2", "n3"])  # only 2 reachable + self = 3
        assert c.n_nodes == 3
        assert c.quorum == 1  # f=(3-1)//3=0, quorum=2*0+1=1

    def test_partition_detected_in_status(self):
        c = HolonomyConsensus("n1", ["n2", "n3"])
        c.handle_partition(["n2"])
        status = c.get_status()
        assert status["view_number"] == 1
        assert status["n_nodes"] == 2


class TestEmergenceDetection:
    def test_no_emergence_without_commits(self):
        c = HolonomyConsensus("n1", ["n2", "n3"])
        result = c.detect_emergence()
        assert not result["emergence_detected"]

    def test_emergence_with_high_convergence(self):
        c = HolonomyConsensus("n1", ["n2", "n3"])
        for i in range(5):
            prop = c.propose_state_change(f"op_{i}", {})
            for _ in range(3):
                c.vote_on_proposal(prop.digest(), approve=True)
            c.commit_if_quorum(prop.digest())

        result = c.detect_emergence()
        assert result["convergence_rate"] == 1.0
        assert result["emergence_detected"]


class TestIntegration:
    def test_full_lifecycle(self):
        c = HolonomyConsensus("node-1", ["node-2", "node-3", "node-4"])
        
        # Propose
        prop = c.propose_state_change("room_grid_resize", {"n": 200})
        assert prop.seq_num == 1
        
        # Vote
        c.vote_on_proposal(prop.digest(), approve=True)
        
        # Check status
        status = c.get_status()
        assert status["n_proposals"] == 1
        
        # Try commit (no quorum yet)
        result = c.commit_if_quorum(prop.digest())
        assert not result.committed
        
        # Add more votes to reach quorum
        for _ in range(2):
            c.vote_on_proposal(prop.digest(), approve=True)
        
        result = c.commit_if_quorum(prop.digest())
        assert result.committed
        assert result.votes_for == 3
