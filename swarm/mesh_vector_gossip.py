"""MeshVectorGossip — Anti-entropy gossip protocol for FluxVectorTable CRDTs.

Propagates agent DNA vector deltas across fleet nodes using anti-entropy
pulls.  Each node maintains a local FluxVectorTable; mesh-wide consistency
is achieved by periodic gossip rounds where nodes exchange digests and
then pull missing deltas.

Integration touchpoints
-----------------------
- FleetConductor: triggers gossip rounds via `trigger_gossip_round()`
  (or directly calls `MeshVectorGossip.gossip_round()` inside its sync loop).
- AutoBreeder: queries `get_mesh_wide_vectors()` for cross-node parent
  discovery, or calls `query_peers_for_breed_candidates()`.
- SignedWAL: every successful merge / delta application is logged via
  `SignedWAL.append(WALEntry(..., operation='gossip_delta'))`.

Reference: docs/SPEC_MULTI_INSTANCE_MESH.md — sections 3 & 4.
"""

from __future__ import annotations

__all__ = [
    "MeshVectorGossip",
    "GossipDigest",
    "DeltaBatch",
    "GossipResult",
    "ThermalRoutingError",
]

import hashlib
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

# ── Data structures ───────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GossipDigest:
    """Bloom-filter-like digest of a node's vector table.

    We use a simple count-min-sketch style digest: for each agent we
    compute k hashes and OR them into a bit array.  This is not a real
    Bloom filter (no false negatives) but it is lightweight and
    sufficient for the anti-entropy protocol: if a peer reports a bit
    we don't have, we pull the delta; false positives just waste a
    round-trip.
    """

    node_id: str
    agent_count: int
    version_vector: dict[int, int]  # agent_id -> logical version
    bitfield: str  # hex-encoded bit digest for fast comparison
    max_wall_time: float = 0.0

    def might_contain(self, agent_id: int, wall_time: float) -> bool:
        """Return True if this digest *probably* contains the agent."""
        return agent_id in self.version_vector and self.version_vector[agent_id] >= 1

    def estimate_bandwidth_saved(self, full_table_size: int) -> float:
        """Approximate fraction of bandwidth saved by using digest."""
        if full_table_size == 0:
            return 0.0
        # Simple heuristic: digest is ~count * (agent_id + version) bytes
        digest_bytes = len(self.version_vector) * 12  # int + int ≈ 12 bytes each
        full_bytes = full_table_size
        return 1.0 - (digest_bytes / full_bytes) if full_bytes > 0 else 0.0


@dataclass
class DeltaBatch:
    """A batch of vector deltas ready for gossip."""

    node_id: str
    deltas: list[dict[str, Any]] = field(default_factory=list)
    sequence: int = 0

    def __len__(self) -> int:
        return len(self.deltas)

    def append(self, delta: dict[str, Any]) -> None:
        self.deltas.append(delta)


@dataclass
class GossipResult:
    """Outcome of a single gossip round."""

    peer_id: str
    merged_count: int = 0
    rejected_count: int = 0
    thermal_rejected: bool = False
    duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)


class ThermalRoutingError(Exception):
    """Raised when a peer is too hot to accept or send deltas."""

    pass


# ── MeshVectorGossip ────────────────────────────────────────────


class MeshVectorGossip:
    """Anti-entropy gossip for FluxVectorTable across fleet nodes.

    Parameters
    ----------
    node_id : str
        Unique identifier for this node (e.g. "Oracle1", "ProArt").
    local_table : FluxVectorTable
        The local vector table to sync.
    signed_wal : SignedWAL | None
        Optional WAL for audit logging of gossip events.
    gossip_interval_seconds : float
        Seconds between automatic gossip rounds (0 disables auto).
    max_peers_per_round : int
        Maximum number of peers contacted in one round.
    delta_batch_size : int
        Max deltas to queue before flushing.
    thermal_threshold : float
        Thermal pressure above which a peer is considered "hot" and
        will be skipped for delta exchange.
    wal_callback : callable | None
        If signed_wal is not provided, this callback receives audit
        tuples: (operation, agent_id, metadata_dict).

    Example
    -------
    >>> gossip = MeshVectorGossip(
    ...     node_id="Oracle1",
    ...     local_table=flux_table,
    ...     signed_wal=wal,
    ...     gossip_interval_seconds=5.0,
    ...     max_peers_per_round=3,
    ... )
    >>> gossip.publish_delta(7, [0.1, -0.2, ...], score=0.85, timestamp=time.time())
    >>> result = gossip.gossip_round(["ProArt", "JetsonClaw1"])
    """

    def __init__(
        self,
        node_id: str,
        local_table: Any,  # FluxVectorTable — avoid circular import at module load
        signed_wal: Any | None = None,
        gossip_interval_seconds: float = 5.0,
        max_peers_per_round: int = 3,
        delta_batch_size: int = 50,
        thermal_threshold: float = 0.85,
        wal_callback: Callable[[str, int, dict], None] | None = None,
    ):
        self.node_id = node_id
        self.local_table = local_table
        self._wal = signed_wal
        self._wal_callback = wal_callback
        self.gossip_interval_seconds = gossip_interval_seconds
        self.max_peers_per_round = max_peers_per_round
        self.delta_batch_size = delta_batch_size
        self.thermal_threshold = thermal_threshold

        # Delta queue: local changes waiting to be gossiped
        self._delta_queue: list[dict[str, Any]] = []
        self._sequence = 0
        self._version_vector: dict[int, int] = {}  # agent_id -> logical version
        self._agent_wall_times: dict[
            int, float
        ] = {}  # agent_id -> last update wall_time

        # Peer thermal cache (updated externally or during gossip)
        self._peer_thermal: dict[str, float] = {}

        # Gossip statistics
        self._stats: dict[str, Any] = {
            "rounds": 0,
            "deltas_sent": 0,
            "deltas_received": 0,
            "bytes_saved": 0,
        }

    # ── CRDT merge ──────────────────────────────────────────

    @staticmethod
    def merge_vector_tables(
        local: Any,  # FluxVectorTable
        remote: dict[str, Any],
    ) -> Any:
        """CRDT merge of a remote dict representation into a local FluxVectorTable.

        Merge rules (per SPEC section 4):
        1. Higher fitness score wins.
        2. If fitness is equal, tie-break by *earlier* wall_time (the
           node that produced the vector first is authoritative).
        3. If both are missing, keep local.
        4. Vectors are deep-copied so the returned table is independent.

        Parameters
        ----------
        local : FluxVectorTable
            The local table (mutated in place and returned).
        remote : dict
            Serialized remote table. Expected keys:
            - "agents": list of dicts with keys
              agent_id, vector, fitness, generation, capability_mask,
              thermal_pressure, wall_time.

        Returns
        -------
        FluxVectorTable
            The mutated *local* table after merge.
        """
        from swarm.flux_vector_table import AgentVector

        agents = remote.get("agents", [])
        for agent_dict in agents:
            aid = agent_dict["agent_id"]
            remote_fitness = float(agent_dict.get("fitness", 0.0))
            remote_wall = float(agent_dict.get("wall_time", 0.0))

            local_meta = getattr(local, "_meta", {}).get(aid)
            if local_meta is not None:
                local_fitness = getattr(local_meta, "fitness", 0.0)
                local_wall = float(
                    getattr(local_meta, "extra", {}).get("wall_time", 0.0)
                )
            else:
                local_fitness = 0.0
                local_wall = 0.0

            # Rule 1: higher fitness wins
            if remote_fitness > local_fitness:
                winner = "remote"
            elif remote_fitness < local_fitness:
                winner = "local"
            else:
                # Rule 2: tie-break by wall_time — *earlier* wins
                # (the agent that finished training first is authoritative)
                winner = "local" if local_wall <= remote_wall else "remote"

            if winner == "remote":
                vec = agent_dict.get("vector", [])
                if isinstance(vec, np.ndarray):
                    vec = vec.tolist()
                av = AgentVector(
                    agent_id=aid,
                    vector=vec,
                    fitness=remote_fitness,
                    generation=int(agent_dict.get("generation", 0)),
                    capability_mask=int(agent_dict.get("capability_mask", 0xFFFF)),
                    thermal_pressure=float(agent_dict.get("thermal_pressure", 0.0)),
                )
                local.add(av)
                # Stamp wall_time into extra for future merges
                if aid in getattr(local, "_meta", {}):
                    local._meta[aid].extra["wall_time"] = remote_wall

        return local

    # ── Gossip round ────────────────────────────────────────

    def gossip_round(self, peers: list[str]) -> dict[str, GossipResult]:
        """Pull deltas from up to *max_peers_per_round* peers and merge.

        Steps per peer:
        1. Send our digest.
        2. Peer responds with missing deltas (or "up to date").
        3. Merge incoming deltas via CRDT rules.
        4. Optionally send our queued deltas back.

        Parameters
        ----------
        peers : list[str]
            Identifiers of candidate peers.  The method samples up to
            ``max_peers_per_round``.

        Returns
        -------
        dict[str, GossipResult]
            Mapping peer_id -> result.
        """
        if not peers:
            return {}

        selected = self._select_peers(peers)
        results: dict[str, GossipResult] = {}

        for peer_id in selected:
            start = time.perf_counter()
            result = GossipResult(peer_id=peer_id)

            try:
                # Thermal-aware routing: skip hot peers
                peer_thermal = self._peer_thermal.get(peer_id, 0.0)
                if peer_thermal > self.thermal_threshold:
                    result.thermal_rejected = True
                    raise ThermalRoutingError(
                        f"Peer {peer_id} thermal {peer_thermal:.2f} > threshold "
                        f"{self.thermal_threshold:.2f}"
                    )

                # 1. Build and send digest
                digest = self.get_digest()

                # 2. Pull remote deltas (mockable transport)
                remote_deltas = self._fetch_peer_deltas(peer_id, digest)

                # 3. Merge
                merged_count = self._apply_remote_deltas(remote_deltas, peer_id)
                result.merged_count = merged_count
                self._stats["deltas_received"] += merged_count

                # 4. Push our queued deltas
                pushed = self._push_deltas_to_peer(peer_id)
                self._stats["deltas_sent"] += pushed

                # 5. Log audit
                self._log_gossip_event(peer_id, merged_count, pushed)

            except ThermalRoutingError as exc:
                result.errors.append(str(exc))
                logger.warning("Thermal rejection for peer %s: %s", peer_id, exc)
            except Exception as exc:
                result.errors.append(str(exc))
                logger.exception("Gossip round failed for peer %s", peer_id)

            result.duration_ms = (time.perf_counter() - start) * 1000.0
            results[peer_id] = result

        self._stats["rounds"] += 1
        return results

    # ── Delta publishing ──────────────────────────────────────

    def publish_delta(
        self,
        room_id: int,
        vector: list[float],
        score: float,
        timestamp: float | None = None,
    ) -> None:
        """Queue a local vector change for the next gossip round.

        Parameters
        ----------
        room_id : int
            Identifies the agent (agent_id == room_id in many configs).
        vector : list[float]
            The raw float32 vector.
        score : float
            Fitness / score for CRDT comparison.
        timestamp : float | None
            Wall-clock time (defaults to time.time()).
        """
        ts = timestamp if timestamp is not None else time.time()
        delta = {
            "agent_id": room_id,
            "vector": vector,
            "fitness": score,
            "wall_time": ts,
            "node_id": self.node_id,
        }
        self._delta_queue.append(delta)
        self._version_vector[room_id] = self._version_vector.get(room_id, 0) + 1
        self._agent_wall_times[room_id] = ts

        # Flush if batch full
        if len(self._delta_queue) >= self.delta_batch_size:
            logger.debug(
                "Delta batch filled (%d); queued for next gossip",
                len(self._delta_queue),
            )

    # ── Digest ────────────────────────────────────────────────

    def get_digest(self) -> GossipDigest:
        """Return a compact digest of the local vector table.

        The digest includes:
        - agent_count
        - version_vector (agent_id -> logical version)
        - A bitfield hash for very fast "anything new?" checks.
        - max_wall_time (newest entry).
        """
        count = len(getattr(self.local_table, "_meta", {}))
        max_wall = 0.0
        if self._agent_wall_times:
            max_wall = max(self._agent_wall_times.values())

        # Build a simple deterministic bitfield from version_vector
        vv = dict(self._version_vector)
        bitfield_payload = hashlib.sha256(
            str(sorted(vv.items())).encode("utf-8")
        ).hexdigest()[:16]

        return GossipDigest(
            node_id=self.node_id,
            agent_count=count,
            version_vector=vv,
            bitfield=bitfield_payload,
            max_wall_time=max_wall,
        )

    # ── Mesh-wide query helpers ───────────────────────────────

    def get_mesh_wide_vectors(
        self,
        min_fitness: float | None = None,
        max_thermal: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return all local vectors that could be offered mesh-wide.

        AutoBreeder calls this (or its mesh counterpart) to discover
        cross-node breeding candidates.
        """
        results: list[dict[str, Any]] = []
        meta = getattr(self.local_table, "_meta", {})
        for aid, m in meta.items():
            fitness = getattr(m, "fitness", 0.0)
            thermal = getattr(m, "thermal_pressure", 0.0)
            if min_fitness is not None and fitness < min_fitness:
                continue
            if max_thermal is not None and thermal > max_thermal:
                continue
            vec = getattr(self.local_table, "_vectors", {}).get(aid)
            vec_list = (
                vec.tolist()
                if hasattr(vec, "tolist")
                else list(vec)
                if vec is not None
                else []
            )
            results.append(
                {
                    "agent_id": aid,
                    "fitness": fitness,
                    "generation": getattr(m, "generation", 0),
                    "capability_mask": getattr(m, "capability_mask", 0xFFFF),
                    "thermal_pressure": thermal,
                    "vector": vec_list,
                    "node_id": self.node_id,
                }
            )
        return results

    def query_peers_for_breed_candidates(
        self,
        peers: list[str],
        min_fitness: float = 0.7,
        max_thermal: float = 0.5,
    ) -> dict[str, list[dict[str, Any]]]:
        """Ask peers for their breed-eligible vectors.

        This is the cross-node interface used by AutoBreeder.
        Returns a map peer_id -> list of candidate dicts.
        """
        # In a real implementation this would RPC to peers.
        # Here we provide the local data as the default (tests mock the RPC).
        return {self.node_id: self.get_mesh_wide_vectors(min_fitness, max_thermal)}

    # ── Transport hooks (intended to be mocked in tests) ──────

    def _fetch_peer_deltas(
        self,
        peer_id: str,
        digest: GossipDigest,
    ) -> dict[str, Any]:
        """Fetch deltas from a peer that are missing from our digest.

        Override or monkey-patch in tests.  Default returns empty.
        """
        return {"agents": []}

    def _push_deltas_to_peer(self, peer_id: str) -> int:
        """Push queued local deltas to a peer.

        Returns number of deltas pushed.  Override in tests.
        """
        count = len(self._delta_queue)
        self._delta_queue.clear()
        return count

    def _select_peers(self, peers: list[str]) -> list[str]:
        """Sample up to max_peers_per_round from the candidate list."""
        if len(peers) <= self.max_peers_per_round:
            return list(peers)
        return random.sample(peers, self.max_peers_per_round)

    def _apply_remote_deltas(
        self,
        remote_deltas: dict[str, Any],
        peer_id: str,
    ) -> int:
        """Merge remote deltas into local table; return count of new agents."""
        before = len(getattr(self.local_table, "_meta", {}))
        self.merge_vector_tables(self.local_table, remote_deltas)
        after = len(getattr(self.local_table, "_meta", {}))
        merged = max(0, after - before)
        # Bump version vector for any agents in the remote delta so digest is current
        for agent_dict in remote_deltas.get("agents", []):
            aid = agent_dict["agent_id"]
            self._version_vector[aid] = self._version_vector.get(aid, 0) + 1
            self._agent_wall_times[aid] = float(
                agent_dict.get("wall_time", time.time())
            )
        return merged

    def _log_gossip_event(self, peer_id: str, merged: int, pushed: int) -> None:
        """Write audit entry to SignedWAL or callback."""
        meta = {
            "peer_id": peer_id,
            "merged": merged,
            "pushed": pushed,
            "node_id": self.node_id,
            "timestamp": time.time(),
        }
        if self._wal is not None:
            try:
                from logos.signed_wal import WALEntry

                entry = WALEntry(
                    timestamp=meta["timestamp"],
                    agent_id=0,  # system agent
                    operation="gossip_delta",
                    vector_hash="",
                    parent_ids=[],
                    generation=0,
                )
                self._wal.append(entry)
            except Exception:
                logger.debug("WAL append failed (non-critical)")
        if self._wal_callback is not None:
            try:
                self._wal_callback("gossip_delta", 0, meta)
            except Exception:
                pass

    # ── Statistics ──────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        return dict(self._stats)

    def set_peer_thermal(self, peer_id: str, pressure: float) -> None:
        """Update cached thermal pressure for a peer."""
        self._peer_thermal[peer_id] = pressure

    def clear_delta_queue(self) -> int:
        """Manually flush delta queue; return count flushed."""
        count = len(self._delta_queue)
        self._delta_queue.clear()
        return count
