"""
Tests for Commit-Caster I2I Router.

Covers: FleetCast, CommitCaster.
"""

import pytest

from fleet.commit_caster import FleetCast, CommitCaster
from fleet.vessel_handshake import NetworkTopology


class TestFleetCast:
    def test_init(self):
        cast = FleetCast(
            source_vessel="alpha",
            target_vessel="beta",
            payload={"msg": "hello"},
            timestamp=1000.0,
        )
        assert cast.source_vessel == "alpha"
        assert cast.target_vessel == "beta"
        assert cast.cast_hash != ""

    def test_hash_computation(self):
        cast1 = FleetCast(
            source_vessel="alpha", target_vessel="beta",
            payload={"msg": "hello"}, timestamp=1000.0
        )
        cast2 = FleetCast(
            source_vessel="alpha", target_vessel="beta",
            payload={"msg": "hello"}, timestamp=1000.0
        )
        assert cast1.cast_hash == cast2.cast_hash

    def test_hash_different(self):
        cast1 = FleetCast(
            source_vessel="alpha", target_vessel="beta",
            payload={"msg": "hello"}, timestamp=1000.0
        )
        cast2 = FleetCast(
            source_vessel="alpha", target_vessel="gamma",
            payload={"msg": "hello"}, timestamp=1000.0
        )
        assert cast1.cast_hash != cast2.cast_hash

    def test_to_commit_message(self):
        cast = FleetCast(
            source_vessel="alpha", target_vessel="beta",
            payload={"msg": "hello"}, timestamp=1000.0
        )
        msg = cast.to_commit_message()
        assert "[FLEET-CAST]" in msg
        assert "alpha -> beta" in msg
        assert "hello" in msg

    def test_to_commit_message_broadcast(self):
        cast = FleetCast(
            source_vessel="alpha", target_vessel=None,
            payload={"msg": "hello"}, timestamp=1000.0
        )
        msg = cast.to_commit_message()
        assert "BROADCAST" in msg

    def test_from_commit_message(self):
        cast = FleetCast(
            source_vessel="alpha", target_vessel="beta",
            payload={"msg": "hello"}, timestamp=1000.0, sequence=1
        )
        msg = cast.to_commit_message()
        parsed = FleetCast.from_commit_message(msg)
        assert parsed is not None
        assert parsed.source_vessel == "alpha"
        assert parsed.target_vessel == "beta"
        assert parsed.payload == {"msg": "hello"}
        assert parsed.sequence == 1

    def test_from_commit_message_invalid(self):
        parsed = FleetCast.from_commit_message("not a fleet cast")
        assert parsed is None

    def test_to_dict(self):
        cast = FleetCast(
            source_vessel="alpha", target_vessel="beta",
            payload={"msg": "hello"}, timestamp=1000.0
        )
        d = cast.to_dict()
        assert d["source"] == "alpha"
        assert d["target"] == "beta"
        assert d["payload"] == {"msg": "hello"}


class TestCommitCaster:
    def test_init(self):
        caster = CommitCaster("alpha")
        assert caster.vessel_id == "alpha"
        assert caster.sequence == 0

    def test_cast(self):
        caster = CommitCaster("alpha")
        cast = caster.cast("beta", {"msg": "hello"})
        assert cast.source_vessel == "alpha"
        assert cast.target_vessel == "beta"
        assert cast.payload == {"msg": "hello"}
        assert caster.sequence == 1
        assert cast in caster.pending_casts

    def test_broadcast(self):
        caster = CommitCaster("alpha")
        cast = caster.broadcast({"alert": "test"})
        assert cast.target_vessel is None
        assert caster.sequence == 1

    def test_receive_direct(self):
        caster = CommitCaster("beta")
        cast = FleetCast(
            source_vessel="alpha", target_vessel="beta",
            payload={"msg": "hello"}, timestamp=1000.0
        )
        result = caster.receive(cast)
        assert result is True
        assert cast in caster.delivered_casts

    def test_receive_broadcast(self):
        caster = CommitCaster("beta")
        cast = FleetCast(
            source_vessel="alpha", target_vessel=None,
            payload={"msg": "hello"}, timestamp=1000.0
        )
        result = caster.receive(cast)
        assert result is True

    def test_receive_deduplication(self):
        caster = CommitCaster("beta")
        cast = FleetCast(
            source_vessel="alpha", target_vessel="beta",
            payload={"msg": "hello"}, timestamp=1000.0
        )
        caster.receive(cast)
        result = caster.receive(cast)
        assert result is False

    def test_receive_wrong_target(self):
        caster = CommitCaster("beta")
        cast = FleetCast(
            source_vessel="alpha", target_vessel="gamma",
            payload={"msg": "hello"}, timestamp=1000.0
        )
        result = caster.receive(cast)
        assert result is False

    def test_get_route(self):
        topo = NetworkTopology()
        topo.add_edge("alpha", "beta")
        topo.add_edge("beta", "gamma")
        caster = CommitCaster("alpha", network=topo)
        route = caster.get_route("gamma")
        assert route == ["alpha", "beta", "gamma"]

    def test_get_route_no_network(self):
        caster = CommitCaster("alpha")
        route = caster.get_route("beta")
        assert route == ["alpha", "beta"]

    def test_find_next_hop(self):
        topo = NetworkTopology()
        topo.add_edge("alpha", "beta")
        topo.add_edge("beta", "gamma")
        caster = CommitCaster("alpha", network=topo)
        hop = caster._find_next_hop("gamma")
        assert hop == "beta"

    def test_get_pending(self):
        caster = CommitCaster("alpha")
        caster.cast("beta", {"msg": "hello"})
        pending = caster.get_pending()
        assert len(pending) == 1

    def test_get_delivered(self):
        caster = CommitCaster("beta")
        cast = FleetCast(
            source_vessel="alpha", target_vessel="beta",
            payload={"msg": "hello"}, timestamp=1000.0
        )
        caster.receive(cast)
        delivered = caster.get_delivered()
        assert len(delivered) == 1

    def test_clear_pending(self):
        caster = CommitCaster("alpha")
        caster.cast("beta", {"msg": "hello"})
        caster.clear_pending()
        assert len(caster.pending_casts) == 0

    def test_get_stats(self):
        caster = CommitCaster("alpha")
        caster.cast("beta", {"msg": "hello"})
        stats = caster.get_stats()
        assert stats["vessel_id"] == "alpha"
        assert stats["sequence"] == 1
        assert stats["pending"] == 1

    def test_to_dict(self):
        caster = CommitCaster("alpha")
        caster.cast("beta", {"msg": "hello"})
        d = caster.to_dict()
        assert d["vessel_id"] == "alpha"
        assert len(d["pending"]) == 1

    def test_sequence_increment(self):
        caster = CommitCaster("alpha")
        caster.cast("beta", {"msg": "1"})
        caster.cast("beta", {"msg": "2"})
        caster.broadcast({"msg": "3"})
        assert caster.sequence == 3

    def test_max_hops(self):
        caster = CommitCaster("alpha")
        assert caster.max_hops == 10
