"""Tests for Arrow Flight Mesh.

Covers node lifecycle, peer registration, table storage, JSON fallback,
heartbeat, and peer list.
"""

import time

import pytest

from swarm.arrow_flight_mesh import ArrowFlightMeshNode, HAS_PYARROW_FLIGHT, MeshPeer


# ---------------------------------------------------------------------------
# Node lifecycle
# ---------------------------------------------------------------------------

class TestNodeLifecycle:
    def test_node_init(self):
        node = ArrowFlightMeshNode(node_id="alpha")
        assert node.node_id == "alpha"
        assert node.listen_port > 0

    def test_node_start_stop(self):
        node = ArrowFlightMeshNode(node_id="alpha", listen_port=0)
        node.start()
        assert node._running is True
        node.stop()
        assert node._running is False

    def test_node_custom_port(self):
        node = ArrowFlightMeshNode(node_id="alpha", listen_port=54321)
        assert node.listen_port == 54321

    def test_node_location(self):
        node = ArrowFlightMeshNode(node_id="alpha", host="127.0.0.1", listen_port=50051)
        assert node.location == "grpc://127.0.0.1:50051"

    def test_heartbeat(self):
        node = ArrowFlightMeshNode(node_id="alpha")
        node.start()
        try:
            status = node.heartbeat()
            assert status["node_id"] == "alpha"
            assert status["tables"] == 0
            assert status["peers"] == 0
            assert status["running"] is True
            assert status["flight_enabled"] == HAS_PYARROW_FLIGHT
        finally:
            node.stop()


# ---------------------------------------------------------------------------
# Peer registration
# ---------------------------------------------------------------------------

class TestPeerRegistration:
    def test_register_peer(self):
        node = ArrowFlightMeshNode(node_id="alpha")
        peer = MeshPeer("beta", "127.0.0.1", 50052)
        node.register_peer(peer)
        assert len(node.get_peer_list()) == 1
        assert node.get_peer_list()[0].node_id == "beta"

    def test_register_multiple_peers(self):
        node = ArrowFlightMeshNode(node_id="alpha")
        node.register_peer(MeshPeer("beta", "127.0.0.1", 50052))
        node.register_peer(MeshPeer("gamma", "127.0.0.1", 50053))
        assert len(node.get_peer_list()) == 2

    def test_register_peer_updates_last_seen(self):
        node = ArrowFlightMeshNode(node_id="alpha")
        peer = MeshPeer("beta", "127.0.0.1", 50052)
        node.register_peer(peer)
        assert peer.last_seen > 0.0

    def test_register_peer_overwrite(self):
        node = ArrowFlightMeshNode(node_id="alpha")
        node.register_peer(MeshPeer("beta", "127.0.0.1", 50052))
        node.register_peer(MeshPeer("beta", "127.0.0.1", 50053))
        peers = node.get_peer_list()
        assert len(peers) == 1
        assert peers[0].port == 50053


# ---------------------------------------------------------------------------
# Table storage
# ---------------------------------------------------------------------------

class TestTableStorage:
    def test_store_table(self):
        node = ArrowFlightMeshNode(node_id="alpha")
        node.store_table("test_table", {"rows": 10})
        assert node._get_local_table("test_table") == {"rows": 10}

    def test_store_table_names(self):
        node = ArrowFlightMeshNode(node_id="alpha")
        node.store_table("a", [1, 2, 3])
        node.store_table("b", [4, 5, 6])
        names = node.get_local_table_names()
        assert sorted(names) == ["a", "b"]

    def test_store_table_overwrite(self):
        node = ArrowFlightMeshNode(node_id="alpha")
        node.store_table("t", [1])
        node.store_table("t", [2])
        assert node._get_local_table("t") == [2]

    def test_get_missing_table(self):
        node = ArrowFlightMeshNode(node_id="alpha")
        assert node._get_local_table("missing") is None


# ---------------------------------------------------------------------------
# Push / pull with JSON fallback
# ---------------------------------------------------------------------------

class TestPushPullFallback:
    def test_push_to_unknown_peer(self):
        node = ArrowFlightMeshNode(node_id="alpha")
        node.start()
        try:
            result = node.push_to("unknown", "table", [1, 2, 3])
            assert result is False
        finally:
            node.stop()

    def test_pull_from_unknown_peer(self):
        node = ArrowFlightMeshNode(node_id="alpha")
        node.start()
        try:
            result = node.pull_from("unknown", "table")
            assert result is None
        finally:
            node.stop()

    def test_push_missing_table(self):
        node = ArrowFlightMeshNode(node_id="alpha")
        node.register_peer(MeshPeer("beta", "127.0.0.1", 1))
        node.start()
        try:
            result = node.push_to("beta", "missing")
            assert result is False
        finally:
            node.stop()


# ---------------------------------------------------------------------------
# MeshPeer
# ---------------------------------------------------------------------------

class TestMeshPeer:
    def test_peer_location(self):
        peer = MeshPeer("beta", "192.168.1.10", 50051)
        assert peer.location == "grpc://192.168.1.10:50051"

    def test_peer_defaults(self):
        peer = MeshPeer("beta", "127.0.0.1", 50051)
        assert peer.last_seen == 0.0


# ---------------------------------------------------------------------------
# Concurrent access
# ---------------------------------------------------------------------------

class TestConcurrency:
    def test_concurrent_store_and_list(self):
        node = ArrowFlightMeshNode(node_id="alpha")
        import threading

        def store():
            for i in range(50):
                node.store_table(f"t{i}", i)
                time.sleep(0.001)

        t1 = threading.Thread(target=store)
        t2 = threading.Thread(target=store)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert len(node.get_local_table_names()) >= 50

    def test_concurrent_register_peer(self):
        node = ArrowFlightMeshNode(node_id="alpha")
        import threading

        def reg():
            for i in range(50):
                node.register_peer(MeshPeer(f"p{i}", "127.0.0.1", 50000 + i))
                time.sleep(0.001)

        t1 = threading.Thread(target=reg)
        t2 = threading.Thread(target=reg)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert len(node.get_peer_list()) >= 50
