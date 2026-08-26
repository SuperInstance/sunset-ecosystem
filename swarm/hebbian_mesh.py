"""Hebbian Mesh — diversity-aware stochastic routing for MeshVectorGossip.

Wires the ``hebbian-router`` pattern into ``MeshVectorGossip`` so that
peer affinity evolves from interaction quality and mesh-wide diversity
automatically modulates routing chaos.  When the population collapses
around a few centroids (low diversity), the layer injects more randomness
into peer selection to force exploration.

Integration touchpoints
-----------------------
- FleetConductor: reads ``get_diversity_score()`` for fleet-wide health.
- AutoBreeder: uses ``chaos_factor`` for parent-selection exploration.
- MeshVectorGossip: optionally wraps with ``HebbianMeshLayer``.

Reference: docs/HEBBIAN_MESH.md — integration guide.
"""

from __future__ import annotations

__all__ = [
    "HebbianAffinity",
    "HebbianOutcome",
    "HebbianMeshLayer",
    "DiversityError",
]

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── constants ───────────────────────────────────────────────────

# Affinity bounds
AFFINITY_MIN = 0.0
AFFINITY_MAX = 1.0

# Affinity delta rules
DELTA_SUCCESS = +0.10
DELTA_TIMEOUT = -0.20
DELTA_VIOLATION = -0.30
DELTA_NOVELTY = +0.15

# Blacklist threshold — any peer whose affinity drops below this is
# immediately blacklisted and will not be selected for gossip.
BLACKLIST_THRESHOLD = 0.10

# Chaos factor bounds
CHAOS_MIN = 0.10  # high diversity → orderly routing
CHAOS_MAX = 0.50  # low diversity → wild exploration

# Diversity → chaos interpolation thresholds
# diversity <= DIVERSITY_LOW  → chaos = CHAOS_MAX
# diversity >= DIVERSITY_HIGH → chaos = CHAOS_MIN
DIVERSITY_LOW = 0.20
DIVERSITY_HIGH = 0.60

# ── data structures ─────────────────────────────────────────────


class HebbianOutcome(Enum):
    """Result of an interaction with a peer."""

    SUCCESS = auto()  # Gossip round completed, deltas merged
    TIMEOUT = auto()  # Peer did not respond in time
    VIOLATION = auto()  # Peer sent malformed / violating deltas
    NOVELTY = auto()  # Peer contributed genuinely new information


@dataclass
class HebbianAffinity:
    """Mutable affinity record for a single peer."""

    peer_id: str
    strength: float = 0.50  # 0.0 – 1.0, higher = more trusted
    last_interaction: float = 0.0  # Unix timestamp
    trust_score: float = 0.50  # rolling average of recent outcomes
    interaction_count: int = 0  # number of recorded outcomes
    blacklisted: bool = False

    def __post_init__(self) -> None:
        # Clamp initial strength
        self.strength = max(AFFINITY_MIN, min(AFFINITY_MAX, self.strength))
        self.trust_score = max(AFFINITY_MIN, min(AFFINITY_MAX, self.trust_score))


class DiversityError(Exception):
    """Raised when diversity cannot be computed (empty table, etc.)."""

    pass


# ── HebbianMeshLayer ──────────────────────────────────────────


class HebbianMeshLayer:
    """Wraps ``MeshVectorGossip`` with affinity-based, chaos-aware routing.

    The layer sits *between* the caller and the underlying gossip instance.
    It intercepts peer selection, tracks interaction outcomes, and
    dynamically adjusts routing randomness based on population diversity.

    Parameters
    ----------
    gossip : MeshVectorGossip
        The underlying gossip instance to wrap.
    chaos_min : float
        Minimum chaos factor (default 0.1).
    chaos_max : float
        Maximum chaos factor (default 0.5).
    diversity_low : float
        Diversity score below which chaos hits ``chaos_max``.
    diversity_high : float
        Diversity score above which chaos hits ``chaos_min``.

    Example
    -------
    >>> gossip = MeshVectorGossip(node_id="Oracle1", local_table=table)
    >>> mesh = HebbianMeshLayer(gossip)
    >>> peers = mesh.select_peers_for_gossip(["ProArt", "Jetson1"], k=1)
    >>> results = gossip.gossip_round(peers)
    >>> for peer, res in results.items():
    ...     mesh.update_affinity(peer, HebbianOutcome.SUCCESS)
    """

    def __init__(
        self,
        gossip: Any,  # MeshVectorGossip — avoid circular import at module load
        chaos_min: float = CHAOS_MIN,
        chaos_max: float = CHAOS_MAX,
        diversity_low: float = DIVERSITY_LOW,
        diversity_high: float = DIVERSITY_HIGH,
    ) -> None:
        self.gossip = gossip
        self._affinities: dict[str, HebbianAffinity] = {}
        self._lock = threading.Lock()

        self._cached_peer_ids: tuple[str, ...] = ()
        self._cached_weights: np.ndarray = np.array([], dtype=np.float32)
        self._cached_blacklisted_mask: np.ndarray = np.array([], dtype=bool)
        self._cached_valid_mask: np.ndarray = np.array([], dtype=bool)

        # Chaos calibration
        self._chaos_min = chaos_min
        self._chaos_max = chaos_max
        self._diversity_low = diversity_low
        self._diversity_high = diversity_high
        self._chaos_factor: float = chaos_min

        # Cached diversity score (updated on demand)
        self._last_diversity: float | None = None
        self._last_diversity_time: float = 0.0
        self._diversity_ttl_seconds: float = 2.0

        logger.debug(
            "HebbianMeshLayer initialised (chaos %.2f–%.2f, diversity %.2f–%.2f)",
            chaos_min,
            chaos_max,
            diversity_low,
            diversity_high,
        )

    # ── cache rebuild (write path only) ─────────────────────

    def _rebuild_cache_locked(self) -> None:
        """Rebuild cached routing arrays. Must be called with self._lock held."""
        if not self._affinities:
            self._cached_peer_ids = ()
            self._cached_weights = np.array([], dtype=np.float32)
            self._cached_blacklisted_mask = np.array([], dtype=bool)
            self._cached_valid_mask = np.array([], dtype=bool)
            return

        peer_ids = list(self._affinities.keys())
        n = len(peer_ids)

        strengths = np.empty(n, dtype=np.float32)
        blacklisted = np.empty(n, dtype=bool)

        for i, pid in enumerate(peer_ids):
            aff = self._affinities[pid]
            strengths[i] = 0.0 if aff.blacklisted else max(0.01, aff.strength)
            blacklisted[i] = aff.blacklisted

        self._cached_peer_ids = tuple(peer_ids)
        self._cached_weights = strengths
        self._cached_blacklisted_mask = blacklisted
        self._cached_valid_mask = ~blacklisted

    # ── affinity management ─────────────────────────────────

    def update_affinity(self, peer_id: str, outcome: HebbianOutcome) -> None:
        """Record an interaction outcome and adjust peer affinity.

        Rules (per SPEC):
        - SUCCESS  : +0.1, capped at 1.0
        - TIMEOUT  : -0.2, floored at 0.0
        - VIOLATION: -0.3, immediate blacklist if strength < 0.1
        - NOVELTY  : +0.15 (reward exploration)
        """
        with self._lock:
            aff = self._affinities.setdefault(
                peer_id,
                HebbianAffinity(peer_id=peer_id),
            )

            if aff.blacklisted and outcome != HebbianOutcome.NOVELTY:
                # Only NOVELTY can un-blacklist a peer (second chance)
                logger.debug(
                    "Peer %s is blacklisted; ignoring %s", peer_id, outcome.name
                )
                return

            if outcome == HebbianOutcome.SUCCESS:
                delta = DELTA_SUCCESS
            elif outcome == HebbianOutcome.TIMEOUT:
                delta = DELTA_TIMEOUT
            elif outcome == HebbianOutcome.VIOLATION:
                delta = DELTA_VIOLATION
            elif outcome == HebbianOutcome.NOVELTY:
                delta = DELTA_NOVELTY
            else:
                raise ValueError(f"Unknown outcome: {outcome}")

            new_strength = aff.strength + delta
            new_strength = max(AFFINITY_MIN, min(AFFINITY_MAX, new_strength))

            # Update rolling trust score (exponential moving average)
            alpha = 0.3  # smoothing factor
            aff.trust_score = alpha * new_strength + (1 - alpha) * aff.trust_score
            aff.strength = new_strength
            aff.last_interaction = time.time()
            aff.interaction_count += 1

            # Blacklist check after VIOLATION
            if (
                outcome == HebbianOutcome.VIOLATION
                and aff.strength < BLACKLIST_THRESHOLD
            ):
                aff.blacklisted = True
                logger.warning(
                    "Peer %s BLACKLISTED (strength %.2f < %.2f) after VIOLATION",
                    peer_id,
                    aff.strength,
                    BLACKLIST_THRESHOLD,
                )

            # NOVELTY un-blacklists (graceful recovery)
            if outcome == HebbianOutcome.NOVELTY and aff.blacklisted:
                aff.blacklisted = False
                logger.info("Peer %s un-blacklisted by NOVELTY", peer_id)

            self._rebuild_cache_locked()

    def get_affinity(self, peer_id: str) -> HebbianAffinity:
        """Return the current affinity record for a peer."""
        import dataclasses

        with self._lock:
            aff = self._affinities.get(
                peer_id,
                HebbianAffinity(peer_id=peer_id),
            )
            return dataclasses.replace(aff)

    def is_blacklisted(self, peer_id: str) -> bool:
        """Check whether a peer is currently blacklisted."""
        peer_ids = self._cached_peer_ids
        mask = self._cached_blacklisted_mask
        try:
            idx = peer_ids.index(peer_id)
        except ValueError:
            return False  # Unknown peer = not blacklisted
        return bool(mask[idx])

    def list_blacklisted(self) -> list[str]:
        """Return all currently blacklisted peer IDs."""
        peer_ids = self._cached_peer_ids
        mask = self._cached_blacklisted_mask
        indices = np.where(mask)[0]
        return [peer_ids[i] for i in indices]

    def reset_affinity(self, peer_id: str) -> None:
        """Reset a peer to default affinity (useful for manual recovery)."""
        with self._lock:
            self._affinities[peer_id] = HebbianAffinity(peer_id=peer_id)
            self._rebuild_cache_locked()

    # ── diversity & chaos ─────────────────────────────────────

    def get_diversity_score(self) -> float:
        """Compute a 0.0–1.0 diversity score from the local vector table.

        The score is the average Euclidean distance of all agent vectors
        from the population centroid, normalised by the expected spread
        of a healthy fleet (0.5 * sqrt(dim)).

        Returns
        -------
        float
            0.0 = all vectors are identical (collapse),
            1.0 = vectors are maximally spread (exploration).

        Raises
        ------
        DiversityError
            If the local table has no vectors.
        """
        # Use cached value if recent
        now = time.time()
        if (
            self._last_diversity is not None
            and (now - self._last_diversity_time) < self._diversity_ttl_seconds
        ):
            return self._last_diversity

        vectors = self._collect_vectors()
        if len(vectors) == 0:
            raise DiversityError("No vectors in local table; cannot compute diversity")

        # Compute centroid
        centroid = np.mean(vectors, axis=0)
        distances = np.linalg.norm(vectors - centroid, axis=1)
        avg_distance = float(np.mean(distances))

        # Normalise: expected max spread ~ sqrt(dim) for unit-variance vectors,
        # but real vectors are smaller.  Use 0.5 * sqrt(dim) as "healthy".
        dim = vectors.shape[1]
        healthy_spread = 0.5 * np.sqrt(dim)
        score = min(1.0, avg_distance / healthy_spread)

        self._last_diversity = score
        self._last_diversity_time = now
        return score

    @property
    def chaos_factor(self) -> float:
        """Current chaos factor, auto-updated from diversity score."""
        try:
            diversity = self.get_diversity_score()
        except DiversityError:
            diversity = 0.0  # no data = assume collapse, max chaos

        self._chaos_factor = self._compute_chaos(diversity)
        return self._chaos_factor

    def _compute_chaos(self, diversity: float) -> float:
        """Map diversity score to chaos factor (linear interpolation)."""
        if diversity <= self._diversity_low:
            return self._chaos_max
        if diversity >= self._diversity_high:
            return self._chaos_min
        # Linear interpolation between the two thresholds
        t = (diversity - self._diversity_low) / (
            self._diversity_high - self._diversity_low
        )
        return self._chaos_max - t * (self._chaos_max - self._chaos_min)

    # ── routing ─────────────────────────────────────────────

    def route_with_chaos(self, peer_pool: list[str], n_routes: int) -> list[str]:
        """Select ``n_routes`` peers with base affinity weighting + chaos.

        The selection works in two stages:
        1. Affinity-weighted sampling (higher-strength peers are more likely).
        2. Chaos injection: with probability ``chaos_factor``, replace the
           selected peer with a uniformly random peer from the pool.

        This read path is lock-free — it snapshots cached routing arrays
        that are atomically replaced after every write.
        """
        if not peer_pool or n_routes <= 0:
            return []

        n_routes = min(n_routes, len(peer_pool))
        chaos = self.chaos_factor

        # Snapshot cached arrays (atomic pointer copy — no lock)
        peer_ids = self._cached_peer_ids
        weights_arr = self._cached_weights
        valid_mask = self._cached_valid_mask

        # Build eligible pool using cache (lock-free)
        known_index = {pid: i for i, pid in enumerate(peer_ids)}
        eligible: list[str] = []
        eligible_weights: list[float] = []

        for pid in peer_pool:
            if pid in known_index:
                idx = known_index[pid]
                if valid_mask[idx]:
                    eligible.append(pid)
                    eligible_weights.append(float(weights_arr[idx]))
            else:
                # Unknown peer — not blacklisted, default weight
                eligible.append(pid)
                eligible_weights.append(0.5)

        if not eligible:
            # All peers blacklisted — fall back to random from full pool
            logger.warning("All peers blacklisted; routing randomly")
            eligible = list(peer_pool)
            eligible_weights = [0.5] * len(eligible)

        selected: list[str] = []
        temp_ids = list(eligible)
        temp_weights = np.array(eligible_weights, dtype=np.float32)

        while len(selected) < n_routes and temp_ids:
            if temp_weights.sum() <= 0:
                idx = random.randrange(len(temp_ids))
            else:
                probs = temp_weights / temp_weights.sum()
                idx = int(np.random.choice(len(temp_ids), p=probs))

            picked = temp_ids.pop(idx)
            temp_weights = np.delete(temp_weights, idx)

            # Stage 2: chaos injection
            if random.random() < chaos:
                picked = random.choice(eligible)
                logger.debug("Chaos injection: replaced with %s", picked)

            if picked not in selected:
                selected.append(picked)

        return selected

    def select_peers_for_gossip(self, peer_pool: list[str], k: int) -> list[str]:
        """Convenience alias: select *k* peers for a gossip round.

        This is the drop-in replacement for ``MeshVectorGossip._select_peers``.
        """
        return self.route_with_chaos(peer_pool, n_routes=k)

    # ── internal helpers ──────────────────────────────────────

    def _collect_vectors(self) -> np.ndarray:
        """Gather all float32 vectors from the wrapped gossip's local table."""
        local_table = getattr(self.gossip, "local_table", None)
        if local_table is None:
            return np.empty((0, 0))

        vectors_dict = getattr(local_table, "_vectors", {})
        if not vectors_dict:
            return np.empty((0, 0))

        # Stack into (N, dim) array
        vecs = []
        for v in vectors_dict.values():
            arr = np.array(v, dtype=np.float32)
            if arr.ndim == 1:
                vecs.append(arr)
        if not vecs:
            return np.empty((0, 0))
        return np.stack(vecs, axis=0)

    def _affinity_weights(self, peer_ids: list[str]) -> list[float]:
        """Return routing weights for each peer (higher = more likely)."""
        weights: list[float] = []
        for pid in peer_ids:
            aff = self.get_affinity(pid)
            # Base weight = strength, but blacklisted peers get 0
            if aff.blacklisted:
                w = 0.0
            else:
                # Small epsilon so zero-affinity peers still have a chance
                w = max(0.01, aff.strength)
            weights.append(w)
        return weights

    # ── statistics ──────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        """Return a snapshot of mesh-layer statistics."""
        with self._lock:
            total = len(self._affinities)
            blacklisted = sum(1 for a in self._affinities.values() if a.blacklisted)
            avg_strength = 0.0
            avg_trust = 0.0
            if total > 0:
                avg_strength = (
                    sum(a.strength for a in self._affinities.values()) / total
                )
                avg_trust = (
                    sum(a.trust_score for a in self._affinities.values()) / total
                )

            return {
                "peer_count": total,
                "blacklisted_count": blacklisted,
                "avg_strength": round(avg_strength, 3),
                "avg_trust": round(avg_trust, 3),
                "chaos_factor": round(self._chaos_factor, 3),
                "last_diversity": round(self._last_diversity, 3)
                if self._last_diversity is not None
                else None,
            }

    # ── gossip-round wrapper (optional) ───────────────────────

    def gossip_round(self, peers: list[str]) -> dict[str, Any]:
        """Run a gossip round through the wrapped instance and auto-update affinities.

        This is a convenience wrapper that:
        1. Selects peers via ``select_peers_for_gossip()``.
        2. Calls ``gossip.gossip_round()``.
        3. Maps each ``GossipResult`` to a ``HebbianOutcome`` and updates affinity.

        Returns the raw ``GossipResult`` dict so callers can inspect details.
        """
        selected = self.select_peers_for_gossip(
            peers, k=self.gossip.max_peers_per_round
        )
        results = self.gossip.gossip_round(selected)

        for peer_id, result in results.items():
            if result.thermal_rejected:
                self.update_affinity(peer_id, HebbianOutcome.TIMEOUT)
            elif result.errors:
                # Distinguish violation from timeout by error content
                err_text = " ".join(result.errors).lower()
                if "thermal" in err_text or "timeout" in err_text:
                    self.update_affinity(peer_id, HebbianOutcome.TIMEOUT)
                elif "violat" in err_text or "malform" in err_text:
                    self.update_affinity(peer_id, HebbianOutcome.VIOLATION)
                else:
                    self.update_affinity(peer_id, HebbianOutcome.TIMEOUT)
            elif result.merged_count > 0:
                self.update_affinity(peer_id, HebbianOutcome.SUCCESS)
            else:
                # No error but nothing merged — neutral, slight timeout penalty
                self.update_affinity(peer_id, HebbianOutcome.TIMEOUT)

        return results
