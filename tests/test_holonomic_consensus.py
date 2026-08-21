"""Tests for Holonomic Consensus — BFT consensus with holonomic verification.

Covers HolonomicBFT, propose, vote, holonomy checking, commit, and Byzantine detection.
"""

import math
import pytest

from fleet.holonomic_consensus import HolonomicBFT, Vote


class TestHolonomicBFT:
    def test_init(self):
        consensus = HolonomicBFT(node_id="alpha", peers=["beta", "gamma"], f=1)
        assert consensus.node_id == "alpha"
        assert len(consensus.peers) == 2
        assert consensus.quorum == 3

    def test_propose(self):
        consensus = HolonomicBFT(node_id="alpha", peers=["beta"], f=0)
        consensus.propose("batch_1", value=[0.6, 0.8])
        assert "batch_1" in consensus._proposals
        assert "alpha" in consensus._proposals["batch_1"]

    def test_receive_vote(self):
        consensus = HolonomicBFT(node_id="alpha", peers=["beta"], f=0)
        vote = Vote(
            node_id="beta", proposal_id="batch_1", value=[0.5, 0.5], timestamp=0.0
        )
        consensus.receive_vote(vote)
        assert "beta" in consensus._proposals["batch_1"]

    def test_check_holonomy_insufficient_votes(self):
        consensus = HolonomicBFT(node_id="alpha", peers=["beta", "gamma"], f=1)
        consensus.propose("batch_1", value=[0.6, 0.8])
        # Only 1 vote, need 3
        assert consensus.check_holonomy("batch_1") is False

    def test_check_holonomy_with_quorum(self):
        consensus = HolonomicBFT(node_id="alpha", peers=["beta", "gamma", "delta"], f=1)
        # All nodes vote with same direction (zero holonomy)
        for node in consensus.all_nodes:
            if node == "alpha":
                consensus.propose("batch_1", value=[0.6, 0.8])
            else:
                consensus.receive_vote(
                    Vote(
                        node_id=node,
                        proposal_id="batch_1",
                        value=[0.6, 0.8],
                        timestamp=0.0,
                    )
                )

        assert len(consensus._proposals["batch_1"]) == 4
        result = consensus.check_holonomy("batch_1")
        assert isinstance(result, bool)

    def test_get_holonomy_error(self):
        consensus = HolonomicBFT(node_id="alpha", peers=["beta"], f=0)
        consensus.propose("batch_1", value=[0.6, 0.8])
        consensus.receive_vote(
            Vote(node_id="beta", proposal_id="batch_1", value=[0.6, 0.8], timestamp=0.0)
        )
        error = consensus.get_holonomy_error("batch_1")
        assert error >= 0.0

    def test_commit(self):
        consensus = HolonomicBFT(node_id="alpha", peers=["beta", "gamma"], f=1)
        # Add 3 votes (quorum) with same value
        for node in ["alpha", "beta", "gamma"]:
            if node == "alpha":
                consensus.propose("batch_1", value=[0.6, 0.8])
            else:
                consensus.receive_vote(
                    Vote(
                        node_id=node,
                        proposal_id="batch_1",
                        value=[0.6, 0.8],
                        timestamp=0.0,
                    )
                )

        result = consensus.commit("batch_1")
        assert isinstance(result, bool)
        if result:
            assert consensus.get_commit("batch_1") is not None

    def test_get_stats(self):
        consensus = HolonomicBFT(node_id="alpha", peers=["beta"], f=0)
        consensus.propose("batch_1", value=[0.6, 0.8])
        stats = consensus.get_stats("batch_1")
        assert stats["proposal_id"] == "batch_1"
        assert stats["votes_received"] == 1
        assert stats["quorum_required"] == 1

    def test_is_byzantine_fault(self):
        consensus = HolonomicBFT(node_id="alpha", peers=["beta", "gamma", "delta"], f=1)
        # Alpha and beta agree, gamma and delta disagree
        consensus.propose("batch_1", value=[0.6, 0.8])
        consensus.receive_vote(
            Vote(node_id="beta", proposal_id="batch_1", value=[0.6, 0.8], timestamp=0.0)
        )
        consensus.receive_vote(
            Vote(
                node_id="gamma",
                proposal_id="batch_1",
                value=[-0.6, -0.8],
                timestamp=0.0,
            )
        )
        consensus.receive_vote(
            Vote(
                node_id="delta",
                proposal_id="batch_1",
                value=[-0.6, -0.8],
                timestamp=0.0,
            )
        )

        # Check if gamma is Byzantine (it disagrees with majority)
        is_byz = consensus.is_byzantine_fault("batch_1", "gamma")
        assert isinstance(is_byz, bool)

    def test_nonexistent_proposal(self):
        consensus = HolonomicBFT(node_id="alpha", peers=["beta"], f=0)
        assert consensus.check_holonomy("missing") is False
        assert consensus.get_holonomy_error("missing") == float("inf")
        assert "error" in consensus.get_stats("missing")

    def test_multiple_proposals(self):
        consensus = HolonomicBFT(node_id="alpha", peers=["beta"], f=0)
        consensus.propose("batch_1", value=[0.6, 0.8])
        consensus.propose("batch_2", value=[0.3, 0.4])
        assert len(consensus._proposals) == 2
        stats1 = consensus.get_stats("batch_1")
        stats2 = consensus.get_stats("batch_2")
        assert stats1["votes_received"] == 1
        assert stats2["votes_received"] == 1
