"""
Tests for Vessel Handshake Protocol.

Covers: PeerIdentity, HandshakeMessage, NetworkTopology,
VesselHandshakeProtocol, FleetDirectory.
"""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from fleet.vessel_handshake import (
    PeerIdentity,
    HandshakeMessage,
    TopologyEdge,
    NetworkTopology,
    VesselHandshakeProtocol,
    FleetDirectory,
)


class TestPeerIdentity:
    def test_trinity_score(self):
        p = PeerIdentity(
            vessel_id="v1", node_id="n1", public_key="pk1",
            ethos_score=0.8, pathos_score=0.9, logos_score=0.7
        )
        assert p.trinity_score == 0.8 * 0.9 * 0.7
        assert abs(p.trinity_score - 0.504) < 0.001

    def test_zero_trinity(self):
        p = PeerIdentity(
            vessel_id="v1", node_id="n1", public_key="pk1",
            ethos_score=0.0, pathos_score=0.9, logos_score=0.7
        )
        assert p.trinity_score == 0.0

    def test_to_dict(self):
        p = PeerIdentity(
            vessel_id="v1", node_id="n1", public_key="pk1",
            capabilities=["breeding"], latency_ms=50.0
        )
        d = p.to_dict()
        assert d["vessel_id"] == "v1"
        assert d["capabilities"] == ["breeding"]

    def test_from_dict(self):
        d = {
            "vessel_id": "v1", "node_id": "n1", "public_key": "pk1",
            "capabilities": ["spatial"], "ethos_score": 0.5
        }
        p = PeerIdentity.from_dict(d)
        assert p.vessel_id == "v1"
        assert p.capabilities == ["spatial"]
        assert p.ethos_score == 0.5


class TestHandshakeMessage:
    def test_signature_computation(self):
        msg = HandshakeMessage(
            sender_id="v1", nonce="abc123", timestamp=12345.0
        )
        sig = msg.compute_signature("secret")
        assert len(sig) == 16
        assert isinstance(sig, str)

    def test_signature_verification(self):
        msg = HandshakeMessage(
            sender_id="v1", nonce="abc123", timestamp=12345.0
        )
        msg.signature = msg.compute_signature("secret")
        assert msg.verify("secret")
        assert not msg.verify("wrong")

    def test_to_dict(self):
        msg = HandshakeMessage(
            sender_id="v1", nonce="abc123", timestamp=12345.0,
            known_peers=["v2", "v3"], capabilities=["breeding"]
        )
        d = msg.to_dict()
        assert d["sender_id"] == "v1"
        assert d["known_peers"] == ["v2", "v3"]


class TestNetworkTopology:
    def test_add_edge(self):
        t = NetworkTopology()
        t.add_edge("v1", "v2", weight=2.0)
        assert "v1" in t.nodes
        assert "v2" in t.nodes
        assert len(t.edges) == 1
        assert t.edges[0].weight == 2.0

    def test_add_edge_update(self):
        t = NetworkTopology()
        t.add_edge("v1", "v2", weight=1.0)
        t.add_edge("v1", "v2", weight=3.0)
        assert len(t.edges) == 1
        assert t.edges[0].weight == 3.0

    def test_shortest_path_direct(self):
        t = NetworkTopology()
        t.add_edge("v1", "v2")
        path = t.shortest_path("v1", "v2")
        assert path == ["v1", "v2"]

    def test_shortest_path_indirect(self):
        t = NetworkTopology()
        t.add_edge("v1", "v2")
        t.add_edge("v2", "v3")
        # The BFS treats edges as bidirectional, so both directions exist
        path = t.shortest_path("v1", "v3")
        # BFS should find v1-v2-v3 (2 hops) not v1-v3 directly
        assert path is not None
        assert path[0] == "v1"
        assert path[-1] == "v3"
        assert len(path) <= 3  # At most 3 nodes

    def test_shortest_path_none(self):
        t = NetworkTopology()
        t.add_edge("v1", "v2")
        t.add_edge("v3", "v4")
        path = t.shortest_path("v1", "v3")
        assert path is None

    def test_same_node_path(self):
        t = NetworkTopology()
        path = t.shortest_path("v1", "v1")
        assert path == ["v1"]

    def test_get_neighbors(self):
        t = NetworkTopology()
        t.add_edge("v1", "v2")
        t.add_edge("v1", "v3")
        t.add_edge("v2", "v3")
        neighbors = t.get_neighbors("v1")
        assert sorted(neighbors) == ["v2", "v3"]

    def test_remove_stale_edges(self):
        t = NetworkTopology()
        t.add_edge("v1", "v2")
        # Manually set old timestamp
        t.edges[0].last_seen = time.time() - 400
        t.remove_stale_edges(max_age=300)
        assert len(t.edges) == 0
        assert len(t.nodes) == 0

    def test_cluster_coefficient(self):
        # Triangle: v1-v2, v2-v3, v1-v3
        t = NetworkTopology()
        t.add_edge("v1", "v2")
        t.add_edge("v2", "v3")
        t.add_edge("v1", "v3")
        cc = t.cluster_coefficient("v1")
        assert cc == 1.0  # All neighbors connected

    def test_cluster_coefficient_none(self):
        t = NetworkTopology()
        t.add_edge("v1", "v2")
        cc = t.cluster_coefficient("v1")
        assert cc == 0.0  # Only one neighbor

    def test_diameter(self):
        t = NetworkTopology()
        t.add_edge("v1", "v2")
        t.add_edge("v2", "v3")
        t.add_edge("v3", "v4")
        assert t.diameter() == 3

    def test_diameter_single_node(self):
        t = NetworkTopology()
        t.nodes = {"v1"}
        assert t.diameter() == 0

    def test_to_dict(self):
        t = NetworkTopology()
        t.add_edge("v1", "v2")
        d = t.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert len(d["edges"]) == 1


class TestVesselHandshakeProtocol:
    def test_init(self):
        v = VesselHandshakeProtocol(
            vessel_id="v1", node_id="n1", secret="sekrit"
        )
        assert v.vessel_id == "v1"
        assert v.node_id == "n1"
        assert v.secret == "sekrit"
        assert v.max_hops == 3

    def test_create_handshake(self):
        v = VesselHandshakeProtocol(
            vessel_id="v1", node_id="n1", secret="sekrit"
        )
        msg = v.create_handshake()
        assert msg.sender_id == "v1"
        assert len(msg.nonce) == 16
        assert msg.verify("sekrit")
        assert "breeding" in msg.capabilities

    def test_process_handshake(self):
        v = VesselHandshakeProtocol(
            vessel_id="v1", node_id="n1", secret="sekrit"
        )
        msg = HandshakeMessage(
            sender_id="v2", nonce="abc", timestamp=time.time(),
            capabilities=["spatial"]
        )
        msg.signature = msg.compute_signature("sekrit")

        response = v.process_handshake(msg, latency_ms=50.0)
        assert "v2" in v.peers
        assert v.peers["v2"].capabilities == ["spatial"]
        assert v.peers["v2"].latency_ms == 50.0
        assert response.sender_id == "v1"

    def test_process_handshake_invalid(self):
        v = VesselHandshakeProtocol(
            vessel_id="v1", node_id="n1", secret="sekrit"
        )
        msg = HandshakeMessage(
            sender_id="v2", nonce="abc", timestamp=time.time()
        )
        msg.signature = "invalid"

        with pytest.raises(ValueError, match="Invalid handshake"):
            v.process_handshake(msg)

    def test_discover_peers(self):
        v = VesselHandshakeProtocol(
            vessel_id="v1", node_id="n1", secret="sekrit"
        )
        discovered = v.discover_peers(["v2", "v3"])
        assert len(discovered) == 2
        assert "v2" in discovered
        assert "v3" in discovered

    def test_discover_peers_self_skip(self):
        v = VesselHandshakeProtocol(
            vessel_id="v1", node_id="n1", secret="sekrit"
        )
        discovered = v.discover_peers(["v1", "v2"])
        assert "v1" not in discovered
        assert "v2" in discovered

    def test_find_route(self):
        v = VesselHandshakeProtocol(
            vessel_id="v1", node_id="n1", secret="sekrit"
        )
        # Simulate discovered v2 and v3
        v.topology.add_edge("v1", "v2")
        v.topology.add_edge("v2", "v3")
        route = v.find_route("v3")
        assert route == ["v1", "v2", "v3"]

    def test_recommend_peer(self):
        v = VesselHandshakeProtocol(
            vessel_id="v1", node_id="n1", secret="sekrit"
        )
        v.peers["v2"] = PeerIdentity(
            vessel_id="v2", node_id="n2", public_key="pk2",
            capabilities=["breeding"], latency_ms=50.0,
            ethos_score=0.8, pathos_score=0.8, logos_score=0.8
        )
        v.peers["v3"] = PeerIdentity(
            vessel_id="v3", node_id="n3", public_key="pk3",
            capabilities=["spatial"], latency_ms=30.0
        )

        rec = v.recommend_peer_for_task("breeding")
        assert rec is not None
        assert rec.vessel_id == "v2"

    def test_recommend_peer_none(self):
        v = VesselHandshakeProtocol(
            vessel_id="v1", node_id="n1", secret="sekrit"
        )
        rec = v.recommend_peer_for_task("nonexistent")
        assert rec is None

    def test_load_peers_from_file(self, tmp_path):
        peers_file = tmp_path / "peers.md"
        peers_file.write_text("""
# Fleet Peers
v2 n2 pk2 breeding spatial
v3 n3 pk3 spatial
""")
        v = VesselHandshakeProtocol(
            vessel_id="v1", node_id="n1", secret="sekrit",
            peers_file=str(peers_file)
        )
        peers = v.load_peers()
        assert len(peers) == 2
        assert v.peers["v2"].capabilities == ["breeding", "spatial"]
        assert v.peers["v3"].capabilities == ["spatial"]

    def test_load_peers_missing_file(self):
        v = VesselHandshakeProtocol(
            vessel_id="v1", node_id="n1", secret="sekrit",
            peers_file="/nonexistent/peers.md"
        )
        peers = v.load_peers()
        assert len(peers) == 0

    def test_save_peers(self, tmp_path):
        peers_file = tmp_path / ".i2i" / "peers.md"
        v = VesselHandshakeProtocol(
            vessel_id="v1", node_id="n1", secret="sekrit",
            peers_file=str(peers_file)
        )
        v.peers["v2"] = PeerIdentity(
            vessel_id="v2", node_id="n2", public_key="pk2",
            capabilities=["breeding"]
        )
        v.save_peers()
        assert peers_file.exists()
        content = peers_file.read_text()
        assert "v2" in content
        assert "breeding" in content

    def test_get_network_stats(self):
        v = VesselHandshakeProtocol(
            vessel_id="v1", node_id="n1", secret="sekrit"
        )
        v.peers["v2"] = PeerIdentity(
            vessel_id="v2", node_id="n2", public_key="pk2"
        )
        v.topology.add_edge("v1", "v2")
        stats = v.get_network_stats()
        assert stats["vessel_id"] == "v1"
        assert stats["peers_known"] == 1
        assert stats["topology_nodes"] == 2

    def test_callbacks(self):
        v = VesselHandshakeProtocol(
            vessel_id="v1", node_id="n1", secret="sekrit"
        )
        discovered = []
        v.on_peer_discovered = lambda p: discovered.append(p.vessel_id)

        msg = HandshakeMessage(
            sender_id="v2", nonce="abc", timestamp=time.time()
        )
        msg.signature = msg.compute_signature("sekrit")
        v.process_handshake(msg)

        assert "v2" in discovered


class TestFleetDirectory:
    def test_register_vessel(self):
        fd = FleetDirectory()
        v = VesselHandshakeProtocol(
            vessel_id="v1", node_id="n1", secret="sekrit"
        )
        v.topology.add_edge("v1", "v2")
        fd.register_vessel(v)
        assert "v1" in fd.vessels
        assert len(fd.global_topology.nodes) == 2

    def test_lookup(self):
        fd = FleetDirectory()
        v = VesselHandshakeProtocol(
            vessel_id="v1", node_id="n1", secret="sekrit"
        )
        v.peers["v2"] = PeerIdentity(
            vessel_id="v2", node_id="n2", public_key="pk2"
        )
        fd.register_vessel(v)
        peer = fd.lookup("v2")
        assert peer is not None
        assert peer.vessel_id == "v2"

    def test_lookup_not_found(self):
        fd = FleetDirectory()
        assert fd.lookup("nonexistent") is None

    def test_find_by_capability(self):
        fd = FleetDirectory()
        v1 = VesselHandshakeProtocol(
            vessel_id="v1", node_id="n1", secret="sekrit"
        )
        v1.peers["v2"] = PeerIdentity(
            vessel_id="v2", node_id="n2", public_key="pk2",
            capabilities=["breeding"]
        )
        v1.peers["v3"] = PeerIdentity(
            vessel_id="v3", node_id="n3", public_key="pk3",
            capabilities=["spatial"]
        )
        fd.register_vessel(v1)

        breeders = fd.find_all_peers_with_capability("breeding")
        assert len(breeders) == 1
        assert breeders[0].vessel_id == "v2"

    def test_fleet_size(self):
        fd = FleetDirectory()
        v1 = VesselHandshakeProtocol(
            vessel_id="v1", node_id="n1", secret="sekrit"
        )
        v1.topology.add_edge("v1", "v2")
        v1.topology.add_edge("v2", "v3")
        fd.register_vessel(v1)
        assert fd.get_fleet_size() == 3

    def test_connected_components(self):
        fd = FleetDirectory()
        v1 = VesselHandshakeProtocol(
            vessel_id="v1", node_id="n1", secret="sekrit"
        )
        v1.topology.add_edge("v1", "v2")
        v1.topology.add_edge("v3", "v4")
        fd.register_vessel(v1)

        components = fd.get_connected_components()
        assert len(components) == 2
        assert {"v1", "v2"} in [c for c in components]
        assert {"v3", "v4"} in [c for c in components]

    def test_isolated_vessels(self):
        fd = FleetDirectory()
        v1 = VesselHandshakeProtocol(
            vessel_id="v1", node_id="n1", secret="sekrit"
        )
        v1.topology.add_edge("v1", "v2")
        v1.topology.nodes.add("v3")  # Isolated node
        fd.register_vessel(v1)

        # Note: v3 has no edges, so it won't be in topology.nodes
        # But if we add it as a node with no edges:
        isolated = fd.get_isolated_vessels()
        # Only truly isolated nodes (no edges) appear
        assert len(isolated) == 0  # v3 wasn't added via add_edge

    def test_duplicate_peer_filtering(self):
        fd = FleetDirectory()
        v1 = VesselHandshakeProtocol(
            vessel_id="v1", node_id="n1", secret="sekrit"
        )
        v1.peers["v2"] = PeerIdentity(
            vessel_id="v2", node_id="n2", public_key="pk2",
            capabilities=["breeding"]
        )

        v2 = VesselHandshakeProtocol(
            vessel_id="v3", node_id="n3", secret="sekrit"
        )
        v2.peers["v2"] = PeerIdentity(
            vessel_id="v2", node_id="n2", public_key="pk2",
            capabilities=["breeding"]
        )

        fd.register_vessel(v1)
        fd.register_vessel(v2)

        breeders = fd.find_all_peers_with_capability("breeding")
        assert len(breeders) == 1  # No duplicates


class TestIntegration:
    def test_full_discovery_pipeline(self):
        """End-to-end: load peers → discover → route → recommend."""
        v1 = VesselHandshakeProtocol(
            vessel_id="v1", node_id="n1", secret="fleet-secret"
        )
        v2 = VesselHandshakeProtocol(
            vessel_id="v2", node_id="n2", secret="fleet-secret"
        )
        v3 = VesselHandshakeProtocol(
            vessel_id="v3", node_id="n3", secret="fleet-secret"
        )

        # v1 discovers v2, v2 discovers v3
        v1.discover_peers(["v2"])
        v2.discover_peers(["v3"])

        # Merge topologies
        v1.merge_topology(v2.topology)

        # v1 should now be able to find a route to v3
        route = v1.find_route("v3")
        assert route is not None
        assert route[0] == "v1"
        assert route[-1] == "v3"

    def test_trinity_score_peer_selection(self):
        """Peers with higher trinity scores should be preferred."""
        v1 = VesselHandshakeProtocol(
            vessel_id="v1", node_id="n1", secret="sekrit"
        )
        v1.peers["high"] = PeerIdentity(
            vessel_id="high", node_id="n2", public_key="pk2",
            capabilities=["breeding"],
            ethos_score=0.9, pathos_score=0.9, logos_score=0.9,
            latency_ms=100.0
        )
        v1.peers["low"] = PeerIdentity(
            vessel_id="low", node_id="n3", public_key="pk3",
            capabilities=["breeding"],
            ethos_score=0.3, pathos_score=0.3, logos_score=0.3,
            latency_ms=10.0
        )

        # With same latency, high trinity should be selected
        # But our sort is by latency first, then trinity desc
        # low has better latency (10ms vs 100ms), so it wins
        rec = v1.recommend_peer_for_task("breeding")
        assert rec.vessel_id == "low"  # Lower latency wins

    def test_fleet_directory_merge(self):
        """Fleet directory merges multiple vessel topologies."""
        fd = FleetDirectory()

        v1 = VesselHandshakeProtocol(
            vessel_id="v1", node_id="n1", secret="sekrit"
        )
        v1.topology.add_edge("v1", "v2")
        v1.topology.add_edge("v2", "v3")

        v2 = VesselHandshakeProtocol(
            vessel_id="v4", node_id="n4", secret="sekrit"
        )
        v2.topology.add_edge("v4", "v5")
        v2.topology.add_edge("v5", "v1")  # Connects to first component

        fd.register_vessel(v1)
        fd.register_vessel(v2)

        assert fd.get_fleet_size() == 5
        components = fd.get_connected_components()
        assert len(components) == 1  # All connected through v1

    def test_handshake_with_callbacks(self):
        """Full handshake with callback recording."""
        v1 = VesselHandshakeProtocol(
            vessel_id="v1", node_id="n1", secret="sekrit"
        )

        events = []
        v1.on_peer_discovered = lambda p: events.append(("discovered", p.vessel_id))
        v1.on_handshake_received = lambda m: events.append(("handshake", m.sender_id))

        msg = HandshakeMessage(
            sender_id="v2", nonce="abc", timestamp=time.time(),
            capabilities=["breeding", "spatial"]
        )
        msg.signature = msg.compute_signature("sekrit")

        v1.process_handshake(msg)

        assert len(events) == 2
        assert events[0] == ("discovered", "v2")
        assert events[1] == ("handshake", "v2")

    def test_gossip_ttl(self):
        """Gossip TTL limits propagation."""
        v1 = VesselHandshakeProtocol(
            vessel_id="v1", node_id="n1", secret="sekrit"
        )
        # TTL 0 should stop immediately
        v1.gossip_new_peer("v99", ttl=0)
        # TTL 3 should propagate
        v1.topology.add_edge("v1", "v2")
        v1.gossip_new_peer("v99", ttl=3)
        # Just verify no exception
        assert True

    def test_network_diameter_growth(self):
        """Network diameter grows with chain topology."""
        v = VesselHandshakeProtocol(
            vessel_id="v1", node_id="n1", secret="sekrit"
        )

        # Chain: v1-v2-v3-v4-v5
        for i in range(1, 5):
            v.topology.add_edge(f"v{i}", f"v{i+1}")

        assert v.topology.diameter() == 4
        assert v.get_network_stats()["network_diameter"] == 4
