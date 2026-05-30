"""Spectral Graph Wave Coherence Monitoring.

Implements Pattern 2 from the SuperInstance audit: model the agent fleet as a
graph where nodes are agents/rooms and edges are communication links. Propagate
discrete waves through the graph; wave behavior is governed by the eigenvalue
spectrum. Conservation ratio (CR) directly predicts fleet coherence halflife.

Reference: wave-conservation + analog-spectral patterns from SuperInstance
ecosystem audit (May 30, 2026).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np


class ThermostatAction(Enum):
    """Actions emitted by the spectral thermostat when CR drifts."""

    NoOp = auto()
    IncreaseCR = auto()
    DecreaseCR = auto()


@dataclass
class WaveState:
    """Spectral state of a fleet communication graph.

    Attributes:
        adjacency: Weighted adjacency matrix (N×N). A[i,j] > 0 means agents
            i and j can communicate; higher weight = stronger link.
        node_labels: Optional human-readable labels for each node.
        _eigenvalues: Cached sorted eigenvalues of the graph Laplacian.
        _eigenvectors: Cached eigenvectors of the graph Laplacian.
        _history: Prior eigenvalue snapshots for topology-change detection.
    """

    adjacency: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    node_labels: list[str] = field(default_factory=list)
    _eigenvalues: Optional[np.ndarray] = None
    _eigenvectors: Optional[np.ndarray] = None
    _history: list[np.ndarray] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    @classmethod
    def from_edges(
        cls,
        n_nodes: int,
        edges: list[tuple[int, int, float]],
        labels: Optional[list[str]] = None,
    ) -> "WaveState":
        """Build a WaveState from an edge list.

        Args:
            n_nodes: Number of agents/rooms in the fleet.
            edges: List of (i, j, weight) tuples. Undirected; symmetry is
                enforced automatically.
            labels: Optional node labels.
        """
        adj = np.zeros((n_nodes, n_nodes), dtype=float)
        for i, j, w in edges:
            adj[i, j] = w
            adj[j, i] = w
        return cls(adjacency=adj, node_labels=labels or [f"node_{k}" for k in range(n_nodes)])

    @classmethod
    def from_fleet_topology(
        cls,
        agents: list[str],
        links: list[tuple[str, str, float]],
    ) -> "WaveState":
        """Build a WaveState from human-readable fleet topology.

        Args:
            agents: List of agent identifiers (names).
            links: List of (agent_a, agent_b, weight) communication links.
        """
        idx = {name: i for i, name in enumerate(agents)}
        edges = [(idx[a], idx[b], w) for a, b, w in links if a in idx and b in idx]
        return cls.from_edges(len(agents), edges, labels=agents)

    # ------------------------------------------------------------------
    # Spectral core
    # ------------------------------------------------------------------

    def _invalidate_cache(self) -> None:
        self._eigenvalues = None
        self._eigenvectors = None

    @property
    def n_nodes(self) -> int:
        return self.adjacency.shape[0]

    @property
    def degree_matrix(self) -> np.ndarray:
        """Diagonal degree matrix D where D[i,i] = Σ_j A[i,j]."""
        return np.diag(self.adjacency.sum(axis=1))

    @property
    def laplacian(self) -> np.ndarray:
        """Graph Laplacian L = D - A."""
        return self.degree_matrix - self.adjacency

    def compute_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute and cache eigenvalues / eigenvectors of the Laplacian.

        Returns (eigenvalues, eigenvectors) where eigenvalues are sorted
        ascending.  The first eigenvalue is always 0 for a connected graph.
        """
        if self._eigenvalues is None or self._eigenvectors is None:
            vals, vecs = np.linalg.eigh(self.laplacian)
            # Sort ascending (smallest eigenvalue first)
            order = np.argsort(vals)
            self._eigenvalues = vals[order]
            self._eigenvectors = vecs[:, order]
        return self._eigenvalues.copy(), self._eigenvectors.copy()

    def fiedler_eigenvalue(self) -> float:
        """Return λ₂ — the second-smallest eigenvalue of L.

        Also called the algebraic connectivity.  Higher λ₂ means the
        graph is more strongly connected; λ₂ = 0 indicates the graph is
        disconnected.
        """
        vals, _ = self.compute_spectrum()
        if len(vals) < 2:
            return 0.0
        return float(vals[1])

    def wave_speed(self) -> float:
        """Wave propagation speed on the graph.

        From the wave-conservation pattern: speed = √λ₂.
        """
        lam2 = self.fiedler_eigenvalue()
        return math.sqrt(max(lam2, 0.0))

    def frequency_sweep(
        self,
        frequencies: np.ndarray,
        node_source: int = 0,
    ) -> np.ndarray:
        """Simulate a frequency sweep through the graph.

        Drive the graph at `node_source` with a sinusoidal forcing term at
        each frequency in `frequencies`.  Returns the steady-state response
        amplitude at every node, shaped (len(frequencies), n_nodes).

        The standing-wave peaks occur at the eigenfrequencies ω_k = √λ_k,
        producing resonance when the drive frequency matches.
        """
        vals, vecs = self.compute_spectrum()
        n = self.n_nodes
        # Avoid division by zero for λ=0 (rigid-body mode)
        safe_vals = np.where(vals < 1e-12, 1e-12, vals)
        # Eigenfrequencies
        omega_k = np.sqrt(safe_vals)  # shape (n,)

        response = np.zeros((len(frequencies), n), dtype=float)
        for fi, f in enumerate(frequencies):
            # For each mode, compute the driven amplitude
            # Resonance: amplitude ~ 1 / |ω_k² - f²|  (simplified)
            denom = np.abs(omega_k**2 - f**2)
            denom = np.where(denom < 1e-12, 1e-12, denom)
            amplitudes = 1.0 / denom  # shape (n,)

            # Project source excitation onto eigenbasis
            source_vec = vecs[node_source, :]  # mode weights at source
            mode_weights = source_vec * amplitudes

            # Reconstruct physical-space response
            response[fi, :] = vecs @ mode_weights
        return response

    def standing_wave_peaks(
        self,
        freq_resolution: float = 0.01,
        max_freq: float = 10.0,
        threshold_ratio: float = 0.3,
    ) -> list[tuple[float, float]]:
        """Detect standing-wave resonance peaks.

        Scans frequencies [0, max_freq] and returns a list of
        (frequency, peak_amplitude) tuples where the global response shows
        a local maximum exceeding `threshold_ratio * global_max`.

        These peaks correspond to the eigenfrequencies of the graph.
        """
        freqs = np.arange(0.0, max_freq, freq_resolution)
        if len(freqs) == 0:
            return []
        response = self.frequency_sweep(freqs)
        # Global response = sum of squared amplitudes across all nodes
        global_resp = np.sum(response**2, axis=1)
        if global_resp.max() <= 0:
            return []
        # Find local maxima
        peaks = []
        for i in range(1, len(global_resp) - 1):
            if (
                global_resp[i] > global_resp[i - 1]
                and global_resp[i] > global_resp[i + 1]
                and global_resp[i] >= threshold_ratio * global_resp.max()
            ):
                peaks.append((float(freqs[i]), float(global_resp[i])))
        return peaks

    # ------------------------------------------------------------------
    # Topology change detection
    # ------------------------------------------------------------------

    def topology_change_detected(
        self,
        significant_shift: float = 0.15,
        window_size: int = 3,
    ) -> bool:
        """Return True if the eigenvalue spectrum has shifted significantly.

        Compares the current spectrum against a rolling window of prior
        snapshots.  A shift is declared when the relative change in any
        non-zero eigenvalue exceeds `significant_shift`.
        """
        vals, _ = self.compute_spectrum()
        if not self._history:
            self._snapshot(vals)
            return False

        # Compare against recent snapshots
        recent = self._history[-window_size:]
        for snap in recent:
            # Align lengths (graphs can grow / shrink)
            min_len = min(len(vals), len(snap))
            if min_len < 2:
                continue
            # Skip the zero eigenvalue (rigid-body mode)
            cur = vals[1:min_len]
            prev = snap[1:min_len]
            # Relative change
            denom = np.abs(prev)
            denom = np.where(denom < 1e-12, 1e-12, denom)
            rel_change = np.abs(cur - prev) / denom
            if np.any(rel_change > significant_shift):
                return True
        return False

    def _snapshot(self, vals: np.ndarray) -> None:
        self._history.append(vals.copy())

    def record_snapshot(self) -> None:
        """Store the current spectrum for future change detection."""
        vals, _ = self.compute_spectrum()
        self._snapshot(vals)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add_node(self, label: Optional[str] = None) -> int:
        """Add an isolated node. Returns the new node index."""
        n = self.n_nodes
        new_adj = np.zeros((n + 1, n + 1), dtype=float)
        new_adj[:n, :n] = self.adjacency
        self.adjacency = new_adj
        self.node_labels.append(label or f"node_{n}")
        self._invalidate_cache()
        return n

    def remove_node(self, index: int) -> None:
        """Remove a node and all its edges."""
        n = self.n_nodes
        if not (0 <= index < n):
            raise IndexError(f"Node index {index} out of range [0, {n})")
        mask = np.ones(n, dtype=bool)
        mask[index] = False
        self.adjacency = self.adjacency[np.ix_(mask, mask)]
        self.node_labels.pop(index)
        self._invalidate_cache()

    def add_edge(self, i: int, j: int, weight: float = 1.0) -> None:
        """Add or strengthen an undirected edge."""
        n = self.n_nodes
        if not (0 <= i < n and 0 <= j < n):
            raise IndexError(f"Edge ({i},{j}) out of range for {n} nodes")
        self.adjacency[i, j] = weight
        self.adjacency[j, i] = weight
        self._invalidate_cache()

    def remove_edge(self, i: int, j: int) -> None:
        """Remove an edge."""
        self.adjacency[i, j] = 0.0
        self.adjacency[j, i] = 0.0
        self._invalidate_cache()


@dataclass
class SpectralThermostatConfig:
    """Configuration for the spectral thermostat.

    Attributes:
        target_cr: Target conservation ratio (default 0.5, battle-tested).
        deadband: Half-width of the no-action zone around target_cr.
        significant_shift: Threshold for topology_change_detected().
    """

    target_cr: float = 0.5
    deadband: float = 0.05
    significant_shift: float = 0.15


class SpectralThermostat:
    """Deadband controller for fleet spectral health.

    When the conservation ratio drifts outside [target - deadband,
    target + deadband], the thermostat emits `IncreaseCR` (heating) or
    `DecreaseCR` (cooling) actions.  Topology-change detection is also
    monitored so that structural shifts (new rooms, dropped agents) are
    flagged immediately.
    """

    def __init__(
        self,
        config: Optional[SpectralThermostatConfig] = None,
        wave_state: Optional[WaveState] = None,
    ) -> None:
        self.config = config or SpectralThermostatConfig()
        self.wave_state = wave_state or WaveState()
        self.last_action: ThermostatAction = ThermostatAction.NoOp
        self._last_cr: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_graph(self, wave_state: WaveState) -> None:
        """Replace the monitored graph."""
        self.wave_state = wave_state

    def tick(self) -> ThermostatAction:
        """Evaluate current spectral state and emit an action.

        Returns:
            ThermostatAction.NoOp, IncreaseCR, or DecreaseCR.
        """
        cr = conservation_ratio(self.wave_state)
        self._last_cr = cr

        # Topology change takes precedence — it's a structural alarm
        if self.wave_state.topology_change_detected(
            significant_shift=self.config.significant_shift
        ):
            # Structural shifts usually *lower* CR temporarily; heat to compensate
            self.last_action = ThermostatAction.IncreaseCR
            return self.last_action

        low = self.config.target_cr - self.config.deadband
        high = self.config.target_cr + self.config.deadband

        if cr < low:
            self.last_action = ThermostatAction.IncreaseCR
        elif cr > high:
            self.last_action = ThermostatAction.DecreaseCR
        else:
            self.last_action = ThermostatAction.NoOp

        # Record snapshot for next tick's change detection
        self.wave_state.record_snapshot()
        return self.last_action

    def predicted_halflife(self) -> float:
        """Predicted coherence halflife (seconds) of the current fleet."""
        cr = self._last_cr
        if cr is None:
            cr = conservation_ratio(self.wave_state)
        return coherence_halflife(cr)


# ------------------------------------------------------------------
# Standalone utility functions
# ------------------------------------------------------------------


def conservation_ratio(wave_state: WaveState) -> float:
    """Compute the conservation ratio for a fleet graph.

    CR is defined as the Fiedler eigenvalue normalized by the average
    degree:  CR = λ₂ / <deg>.  This is bounded in [0, 1] for connected
    graphs and directly correlates with how well information propagates.
    """
    lam2 = wave_state.fiedler_eigenvalue()
    avg_deg = float(wave_state.adjacency.sum() / max(wave_state.n_nodes, 1))
    if avg_deg <= 0:
        return 0.0
    cr = lam2 / avg_deg
    # For some graph families the ratio can slightly exceed 1; clamp
    return float(np.clip(cr, 0.0, 1.0))


def coherence_halflife(cr: float, time_unit: float = 1.0) -> float:
    """Map conservation ratio to predicted coherence halflife.

    Higher CR → longer-lived coherent states.  Uses the mapping:
        halflife = time_unit * (cr / (1 - cr + ε))
    where ε = 1e-6 prevents division by zero.

    At CR = 0.5 (target), halflife ≈ 1.0 time_unit.
    At CR → 1.0, halflife diverges (perfect coherence).
    At CR → 0.0, halflife → 0 (immediate decoherence).
    """
    cr = float(np.clip(cr, 0.0, 0.999999))
    return time_unit * (cr / (1.0 - cr + 1e-6))


def fleet_coherence_forecast(
    wave_state: WaveState,
    horizon_steps: int = 10,
) -> list[float]:
    """Predict coherence level over a discrete horizon.

    Models exponential decay of coherence from the current state,
    using the halflife derived from the current CR.  Returns a list of
    predicted coherence values at steps 1..horizon_steps.
    """
    cr = conservation_ratio(wave_state)
    hl = coherence_halflife(cr)
    if hl <= 0 or cr <= 0:
        return [0.0] * horizon_steps
    # Coherence decays as  (1/2)^(t / hl)
    return [float((0.5) ** (t / hl)) for t in range(1, horizon_steps + 1)]


def detect_topology_change(
    before: WaveState,
    after: WaveState,
    significant_shift: float = 0.15,
) -> bool:
    """Compare two wave states and flag significant eigenvalue shifts.

    A convenience wrapper for external callers who maintain two explicit
    snapshots rather than using WaveState's internal history.
    """
    v1, _ = before.compute_spectrum()
    v2, _ = after.compute_spectrum()
    min_len = min(len(v1), len(v2))
    if min_len < 2:
        return len(v1) != len(v2)
    cur = v1[1:min_len]
    prev = v2[1:min_len]
    denom = np.abs(prev)
    denom = np.where(denom < 1e-12, 1e-12, denom)
    rel_change = np.abs(cur - prev) / denom
    return bool(np.any(rel_change > significant_shift))
