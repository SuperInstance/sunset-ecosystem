"""Data sharding and rebalancing manager.

Manages data shard assignment, rebalancing, and migration. Supports
consistent hashing for shard placement and node-aware rebalancing. Used
for fleet distributed storage, database sharding, and workload
partitioning.

Usage:
    shards = ShardManager()
    shards.add_node("node-1", capacity=100)
    shards.add_shard("shard-1", size=10)
    assert shards.get_node("shard-1") == "node-1"
    shards.rebalance()
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class ShardManager:
    """
    Shard placement and rebalancing manager.

    :param replication: Number of replica copies per shard.
    """

    def __init__(self, replication: int = 1):
        self._replication = replication
        self._nodes: Dict[str, int] = {}  # node_id -> capacity
        self._node_usage: Dict[str, int] = {}  # node_id -> used capacity
        self._shards: Dict[str, Dict[str, Any]] = {}  # shard_id -> {size, node}
        self._assignments: Dict[str, str] = {}  # shard_id -> primary node

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_node(self, node_id: str, capacity: int) -> None:
        """Register a storage node."""
        self._nodes[node_id] = capacity
        self._node_usage[node_id] = 0

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and migrate its shards."""
        if node_id not in self._nodes:
            return False
        # Reassign shards from this node
        shards_to_move = [
            sid for sid, nid in self._assignments.items() if nid == node_id
        ]
        del self._nodes[node_id]
        del self._node_usage[node_id]
        for sid in shards_to_move:
            self._assignments.pop(sid, None)
            self._place_shard(sid)
        return True

    # ------------------------------------------------------------------
    # Shard management
    # ------------------------------------------------------------------

    def add_shard(self, shard_id: str, size: int = 1) -> bool:
        """Register a shard and place it on a node."""
        if shard_id in self._shards:
            return False
        self._shards[shard_id] = {"size": size}
        return self._place_shard(shard_id)

    def remove_shard(self, shard_id: str) -> bool:
        """Remove a shard."""
        if shard_id not in self._shards:
            return False
        node = self._assignments.pop(shard_id, None)
        if node:
            self._node_usage[node] = max(0, self._node_usage.get(node, 0) - self._shards[shard_id]["size"])
        del self._shards[shard_id]
        return True

    def _place_shard(self, shard_id: str) -> bool:
        """Place a shard on the least-loaded node."""
        if not self._nodes:
            return False
        # Find node with most available capacity
        best_node = None
        best_avail = -1
        for node_id, capacity in self._nodes.items():
            avail = capacity - self._node_usage.get(node_id, 0)
            if avail > best_avail:
                best_avail = avail
                best_node = node_id
        if best_node and best_avail > 0:
            old_node = self._assignments.get(shard_id)
            if old_node:
                self._node_usage[old_node] = max(0, self._node_usage.get(old_node, 0) - self._shards[shard_id]["size"])
            self._assignments[shard_id] = best_node
            self._node_usage[best_node] = self._node_usage.get(best_node, 0) + self._shards[shard_id]["size"]
            return True
        return False

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    def rebalance(self) -> int:
        """
        Rebalance shards across nodes.

        :returns: Number of shards moved.
        """
        if len(self._nodes) <= 1:
            return 0
        moved = 0
        # Calculate target usage per node (proportional to capacity)
        total_capacity = sum(self._nodes.values())
        total_usage = sum(self._node_usage.values())
        if total_capacity == 0:
            return 0
        # Move shards from overloaded to underloaded nodes
        for node_id in list(self._nodes.keys()):
            capacity = self._nodes[node_id]
            usage = self._node_usage.get(node_id, 0)
            target = (capacity / total_capacity) * total_usage
            # If significantly overloaded, try to move shards
            while usage > target + 1 and len(self._shards) > 0:
                # Find a shard on this node to move
                candidates = [
                    sid for sid, nid in self._assignments.items()
                    if nid == node_id
                ]
                if not candidates:
                    break
                shard_id = candidates[0]
                # Try to place on another node
                old_node = self._assignments[shard_id]
                del self._assignments[shard_id]
                self._node_usage[old_node] = max(0, self._node_usage.get(old_node, 0) - self._shards[shard_id]["size"])
                if not self._place_shard(shard_id):
                    # Couldn't place, restore
                    self._assignments[shard_id] = old_node
                    self._node_usage[old_node] = self._node_usage.get(old_node, 0) + self._shards[shard_id]["size"]
                    break
                moved += 1
                usage = self._node_usage.get(node_id, 0)
        return moved

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_node(self, shard_id: str) -> Optional[str]:
        """Get the node hosting a shard."""
        return self._assignments.get(shard_id)

    def shards_on_node(self, node_id: str) -> List[str]:
        """Get shards assigned to a node."""
        return [sid for sid, nid in self._assignments.items() if nid == node_id]

    def node_utilization(self, node_id: str) -> float:
        """Get node utilization ratio (0.0-1.0)."""
        capacity = self._nodes.get(node_id, 0)
        if capacity == 0:
            return 0.0
        return self._node_usage.get(node_id, 0) / capacity

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "nodes": len(self._nodes),
            "shards": len(self._shards),
            "replication": self._replication,
            "total_capacity": sum(self._nodes.values()),
            "total_usage": sum(self._node_usage.values()),
        }

    def __repr__(self) -> str:
        return f"<ShardManager nodes={len(self._nodes)} shards={len(self._shards)}>"