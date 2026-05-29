"""
Vessel Handshake Protocol

Fleet discovery protocol: how vessels find each other without
a central authority. Each agent maintains a `.i2i/peers.md`
file with known peers. On startup, agents traverse peers and
exchange HANDSHAKE messages, building a network topology map.

References:
- Fleet Workshop item #8: "vessel-handshake — Fleet Discovery Protocol"
- SuperInstance ecosystem: vessel-template, babel-vessel
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class PeerIdentity:
    """Identity of a peer in the fleet."""
    vessel_id: str
    node_id: str
    public_key: str
    capabilities: List[str] = field(default_factory=list)
    last_seen: float = 0.0
    latency_ms: float = 0.0
    # Trinity score components
    ethos_score: float = 0.0
    pathos_score: float = 0.0
    logos_score: float = 0.0

    @property
    def trinity_score(self) -> float:
        return self.ethos_score * self.pathos_score * self.logos_score

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "PeerIdentity":
        return cls(**d)


@dataclass
class HandshakeMessage:
    """A handshake message exchanged between vessels."""
    sender_id: str
    nonce: str
    timestamp: float
    # Network topology: who the sender knows
    known_peers: List[str] = field(default_factory=list)
    # Capability advertisement
    capabilities: List[str] = field(default_factory=list)
    # Trinity score for self-advertisement
    ethos_score: float = 0.0
    pathos_score: float = 0.0
    logos_score: float = 0.0
    # Signature (simplified: HMAC would be used in production)
    signature: str = ""

    def compute_signature(self, secret: str) -> str:
        payload = f"{self.sender_id}:{self.nonce}:{self.timestamp}"
        return hashlib.sha256((payload + secret).encode()).hexdigest()[:16]

    def verify(self, secret: str) -> bool:
        expected = self.compute_signature(secret)
        return self.signature == expected

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "HandshakeMessage":
        return cls(**d)


@dataclass
class TopologyEdge:
    """An edge in the network topology graph."""
    source: str
    target: str
    weight: float = 1.0
    last_seen: float = 0.0
    latency_ms: float = 0.0


@dataclass
class NetworkTopology:
    """Discovered network topology of the fleet."""
    nodes: Set[str] = field(default_factory=set)
    edges: List[TopologyEdge] = field(default_factory=list)
    # Path cache: shortest paths between nodes
    _path_cache: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)

    def add_edge(self, source: str, target: str, weight: float = 1.0):
        self.nodes.add(source)
        self.nodes.add(target)
        # Update existing edge or add new
        for edge in self.edges:
            if (edge.source == source and edge.target == target) or \
               (edge.source == target and edge.target == source):
                edge.weight = weight
                edge.last_seen = time.time()
                return
        self.edges.append(TopologyEdge(source, target, weight, time.time()))

    def remove_stale_edges(self, max_age: float = 300.0):
        """Remove edges not seen in max_age seconds."""
        now = time.time()
        self.edges = [e for e in self.edges if now - e.last_seen < max_age]
        # Update nodes set
        self.nodes = set()
        for e in self.edges:
            self.nodes.add(e.source)
            self.nodes.add(e.target)

    def shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest path using BFS."""
        if source == target:
            return [source]

        # BFS
        queue = [(source, [source])]
        visited = {source}

        while queue:
            current, path = queue.pop(0)
            for edge in self.edges:
                if edge.source == current and edge.target not in visited:
                    if edge.target == target:
                        return path + [target]
                    visited.add(edge.target)
                    queue.append((edge.target, path + [edge.target]))
                elif edge.target == current and edge.source not in visited:
                    if edge.source == target:
                        return path + [target]
                    visited.add(edge.source)
                    queue.append((edge.source, path + [target]))

        return None

    def get_neighbors(self, node: str) -> List[str]:
        """Get all neighbors of a node."""
        neighbors = []
        for edge in self.edges:
            if edge.source == node:
                neighbors.append(edge.target)
            elif edge.target == node:
                neighbors.append(edge.source)
        return neighbors

    def cluster_coefficient(self, node: str) -> float:
        """Compute local clustering coefficient."""
        neighbors = self.get_neighbors(node)
        if len(neighbors) < 2:
            return 0.0

        # Count triangles
        triangles = 0
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i + 1:]:
                if n2 in self.get_neighbors(n1):
                    triangles += 1

        max_triangles = len(neighbors) * (len(neighbors) - 1) / 2
        return triangles / max_triangles if max_triangles > 0 else 0.0

    def diameter(self) -> int:
        """Approximate network diameter (longest shortest path)."""
        max_dist = 0
        for n1 in self.nodes:
            for n2 in self.nodes:
                if n1 != n2:
                    path = self.shortest_path(n1, n2)
                    if path:
                        max_dist = max(max_dist, len(path) - 1)
        return max_dist

    def to_dict(self) -> Dict:
        return {
            "nodes": list(self.nodes),
            "edges": [
                {"source": e.source, "target": e.target,
                 "weight": e.weight, "last_seen": e.last_seen}
                for e in self.edges
            ],
        }


class VesselHandshakeProtocol:
    """
    Fleet discovery protocol.

    Each vessel:
    1. Maintains a `.i2i/peers.md` file with known peers
    2. On startup, reads peers and sends HANDSHAKE to each
    3. On receiving HANDSHAKE, responds with ACK + known peers
    4. Merges discovered peers into local topology
    5. Gossips new discoveries to neighbors
    """

    def __init__(self,
                 vessel_id: str,
                 node_id: str,
                 secret: str,
                 peers_file: Optional[str] = None,
                 max_hops: int = 3):
        self.vessel_id = vessel_id
        self.node_id = node_id
        self.secret = secret
        self.max_hops = max_hops

        self.peers_file = peers_file or f".i2i/peers.md"
        self.peers: Dict[str, PeerIdentity] = {}
        self.topology = NetworkTopology()

        # Callbacks for handshake events
        self.on_peer_discovered: Optional[Callable[[PeerIdentity], None]] = None
        self.on_handshake_received: Optional[Callable[[HandshakeMessage], None]] = None

    # ── Peer File Management ──

    def load_peers(self) -> List[PeerIdentity]:
        """Load peers from `.i2i/peers.md`."""
        path = Path(self.peers_file)
        if not path.exists():
            return []

        peers = []
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    # Parse: vessel_id node_id public_key [capabilities...]
                    parts = line.split()
                    if len(parts) >= 3:
                        peers.append(PeerIdentity(
                            vessel_id=parts[0],
                            node_id=parts[1],
                            public_key=parts[2],
                            capabilities=parts[3:] if len(parts) > 3 else []
                        ))
        except (IOError, OSError):
            pass

        for peer in peers:
            self.peers[peer.vessel_id] = peer

        return peers

    def save_peers(self):
        """Save peers to `.i2i/peers.md`."""
        path = Path(self.peers_file)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            f.write("# Fleet Peers\n")
            f.write("# Format: vessel_id node_id public_key [capabilities...]\n\n")
            for peer in self.peers.values():
                caps = " ".join(peer.capabilities)
                f.write(f"{peer.vessel_id} {peer.node_id} {peer.public_key} {caps}\n")

    # ── Handshake Protocol ──

    def create_handshake(self, known_peers: Optional[List[str]] = None) -> HandshakeMessage:
        """Create a handshake message to send to peers."""
        nonce = hashlib.sha256(str(time.time() + random.random()).encode()).hexdigest()[:16]
        msg = HandshakeMessage(
            sender_id=self.vessel_id,
            nonce=nonce,
            timestamp=time.time(),
            known_peers=known_peers or list(self.peers.keys()),
            capabilities=["breeding", "spatial", "causal-discovery"]
        )
        msg.signature = msg.compute_signature(self.secret)
        return msg

    def process_handshake(self, msg: HandshakeMessage, latency_ms: float = 0.0) -> HandshakeMessage:
        """
        Process incoming handshake and return response.
        Returns ACK handshake with our known peers.
        """
        if not msg.verify(self.secret):
            raise ValueError("Invalid handshake signature")

        # Record sender as peer
        peer = PeerIdentity(
            vessel_id=msg.sender_id,
            node_id="unknown",  # Would be resolved in real implementation
            public_key="",  # Would be exchanged
            capabilities=msg.capabilities,
            last_seen=time.time(),
            latency_ms=latency_ms,
            ethos_score=msg.ethos_score,
            pathos_score=msg.pathos_score,
            logos_score=msg.logos_score,
        )
        self.peers[msg.sender_id] = peer
        self.topology.add_edge(self.vessel_id, msg.sender_id, weight=1.0)

        if self.on_peer_discovered:
            self.on_peer_discovered(peer)
        if self.on_handshake_received:
            self.on_handshake_received(msg)

        # Respond with ACK + our known peers (excluding sender)
        our_peers = [p for p in self.peers.keys() if p != msg.sender_id]
        response = self.create_handshake(known_peers=our_peers)
        return response

    # ── Gossip Protocol ──

    def gossip_new_peer(self, new_peer_id: str, ttl: int = 3):
        """
        Gossip a new peer discovery to neighbors.
        ttl: time-to-live (hops remaining)
        """
        if ttl <= 0:
            return

        neighbors = self.topology.get_neighbors(self.vessel_id)
        for neighbor in neighbors:
            # In real implementation, send network message
            # For now, just record the gossip
            pass

    def merge_topology(self, other_topology: NetworkTopology):
        """Merge another vessel's topology into ours."""
        for edge in other_topology.edges:
            self.topology.add_edge(edge.source, edge.target, edge.weight)

    # ── Discovery Utilities ──

    def discover_peers(self, peer_ids: List[str]) -> Dict[str, PeerIdentity]:
        """
        Discover peers by initiating handshakes.
        Returns discovered peers.
        """
        discovered = {}
        for peer_id in peer_ids:
            if peer_id == self.vessel_id:
                continue

            # Create and send handshake
            handshake = self.create_handshake()

            # Simulate receiving response (in real impl, network call)
            # For testing, we create a mock response
            response = HandshakeMessage(
                sender_id=peer_id,
                nonce=handshake.nonce,
                timestamp=time.time(),
                known_peers=[],
                capabilities=["breeding"]
            )
            response.signature = response.compute_signature(self.secret)

            try:
                self.process_handshake(response, latency_ms=random.uniform(10, 100))
                discovered[peer_id] = self.peers[peer_id]
            except ValueError:
                pass

        return discovered

    def find_route(self, target: str) -> Optional[List[str]]:
        """Find route to target vessel through topology."""
        return self.topology.shortest_path(self.vessel_id, target)

    def recommend_peer_for_task(self, task_capability: str) -> Optional[PeerIdentity]:
        """
        Recommend a peer capable of handling a task.
        Uses trinity score as tiebreaker.
        """
        candidates = [
            peer for peer in self.peers.values()
            if task_capability in peer.capabilities
        ]

        if not candidates:
            return None

        # Sort by latency, then by trinity score
        candidates.sort(key=lambda p: (p.latency_ms, -p.trinity_score))
        return candidates[0]

    def get_network_stats(self) -> Dict:
        """Get statistics about the discovered network."""
        return {
            "vessel_id": self.vessel_id,
            "peers_known": len(self.peers),
            "topology_nodes": len(self.topology.nodes),
            "topology_edges": len(self.topology.edges),
            "network_diameter": self.topology.diameter(),
            "avg_clustering": np.mean([
                self.topology.cluster_coefficient(n)
                for n in self.topology.nodes
            ]) if self.topology.nodes else 0.0,
            "trinity_scores": {
                pid: peer.trinity_score
                for pid, peer in self.peers.items()
            }
        }

    def to_dict(self) -> Dict:
        return {
            "vessel_id": self.vessel_id,
            "node_id": self.node_id,
            "peers": {k: v.to_dict() for k, v in self.peers.items()},
            "topology": self.topology.to_dict(),
        }


class FleetDirectory:
    """
    Shared fleet directory that aggregates all vessel discoveries.
    Acts as a lightweight, distributed DNS for the fleet.
    """

    def __init__(self):
        self.vessels: Dict[str, VesselHandshakeProtocol] = {}
        self.global_topology = NetworkTopology()

    def register_vessel(self, vessel: VesselHandshakeProtocol):
        self.vessels[vessel.vessel_id] = vessel
        self._merge_vessel_topology(vessel)

    def _merge_vessel_topology(self, vessel: VesselHandshakeProtocol):
        for edge in vessel.topology.edges:
            self.global_topology.add_edge(edge.source, edge.target, edge.weight)

    def lookup(self, vessel_id: str) -> Optional[PeerIdentity]:
        for v in self.vessels.values():
            if vessel_id in v.peers:
                return v.peers[vessel_id]
        return None

    def find_all_peers_with_capability(self, capability: str) -> List[PeerIdentity]:
        results = []
        seen = set()
        for v in self.vessels.values():
            for peer in v.peers.values():
                if capability in peer.capabilities and peer.vessel_id not in seen:
                    results.append(peer)
                    seen.add(peer.vessel_id)
        return results

    def get_fleet_size(self) -> int:
        return len(self.global_topology.nodes)

    def get_connected_components(self) -> List[Set[str]]:
        """Find connected components in the fleet topology."""
        visited = set()
        components = []

        for node in self.global_topology.nodes:
            if node in visited:
                continue

            component = set()
            stack = [node]
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    component.add(current)
                    stack.extend(self.global_topology.get_neighbors(current))

            components.append(component)

        return components

    def get_isolated_vessels(self) -> List[str]:
        """Find vessels with no connections."""
        components = self.get_connected_components()
        isolated = []
        for comp in components:
            if len(comp) == 1:
                isolated.extend(comp)
        return isolated
