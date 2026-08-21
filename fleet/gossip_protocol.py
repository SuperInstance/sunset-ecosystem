"""Epidemic gossip protocol for fleet state propagation.

Propagates state updates through the fleet using random peer selection.
Each gossip round picks a subset of neighbors and exchanges digests.
Converges in O(log N) rounds for N nodes.

Usage:
    gossip = GossipProtocol(node_id="node-1", fanout=3)
    gossip.add_peer("node-2", "http://node-2:8080")
    gossip.set_state({"trap_count": 5})
    for _ in range(10):
        gossip.round()
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class GossipMessage:
    """A gossip payload."""

    sender: str
    digest: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class GossipProtocol:
    """
    Epidemic gossip for state synchronization.

    :param node_id: Unique identifier for this node.
    :param fanout: Number of peers to gossip with each round.
    :param digest_fn: Function to compute state digest.
    """

    def __init__(
        self,
        node_id: str,
        fanout: int = 3,
        digest_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
    ):
        self.node_id = node_id
        self.fanout = fanout
        self._peers: Dict[str, str] = {}  # node_id -> address
        self._state: Dict[str, Any] = {}
        self._digest_fn = digest_fn or self._default_digest
        self._stats: Dict[str, int] = {"rounds": 0, "messages_sent": 0, "updates": 0}

    # ------------------------------------------------------------------
    # Peer management
    # ------------------------------------------------------------------

    def add_peer(self, node_id: str, address: str) -> None:
        self._peers[node_id] = address

    def remove_peer(self, node_id: str) -> bool:
        if node_id in self._peers:
            del self._peers[node_id]
            return True
        return False

    def peers(self) -> List[str]:
        return list(self._peers.keys())

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def set_state(self, state: Dict[str, Any]) -> None:
        self._state = dict(state)

    def get_state(self) -> Dict[str, Any]:
        return dict(self._state)

    def update_state(self, key: str, value: Any) -> None:
        self._state[key] = value

    # ------------------------------------------------------------------
    # Gossip rounds
    # ------------------------------------------------------------------

    def round(self) -> List[GossipMessage]:
        """
        Execute one gossip round.

        Returns list of messages that would be sent (for simulation).
        In production, this would actually send over the network.
        """
        self._stats["rounds"] += 1
        messages: List[GossipMessage] = []
        if not self._peers:
            return messages

        targets = self._select_targets()
        digest = self._digest_fn(self._state)

        for target in targets:
            msg = GossipMessage(
                sender=self.node_id,
                digest=digest,
                payload=dict(self._state),
            )
            messages.append(msg)
            self._stats["messages_sent"] += 1

        return messages

    def receive(self, msg: GossipMessage) -> bool:
        """
        Process an incoming gossip message.

        Returns True if state was updated.
        """
        if msg.sender == self.node_id:
            return False

        my_digest = self._digest_fn(self._state)
        if msg.digest != my_digest:
            # Merge incoming state (last-write-wins for simplicity)
            updated = False
            for key, value in msg.payload.items():
                if key not in self._state or self._state[key] != value:
                    self._state[key] = value
                    updated = True
            if updated:
                self._stats["updates"] += 1
            return updated
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_targets(self) -> List[str]:
        peers = list(self._peers.keys())
        if len(peers) <= self.fanout:
            return peers
        return random.sample(peers, self.fanout)

    def _default_digest(self, state: Dict[str, Any]) -> str:
        data = json.dumps(state, sort_keys=True)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def __repr__(self) -> str:
        return f"<GossipProtocol node={self.node_id} peers={len(self._peers)}>"
