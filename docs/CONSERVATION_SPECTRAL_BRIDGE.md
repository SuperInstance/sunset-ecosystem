# Conservation Spectral Bridge

Integrates the [SuperInstance Conservation Spectral Framework](https://github.com/SuperInstance/SuperInstance) into the sunset-ecosystem breeding loop.

## What This Does

The Conservation Spectral Framework uses **Laplacian eigenvalue decomposition** to measure how much structural information survives spectral projection. In plain terms: it gives every agent a **spectral fingerprint** — a mathematical identity derived from its capability graph.

This bridge connects that framework to our breeding diversity metrics:

| Framework Concept | Breeding Use |
|---|---|
| Spectral fingerprint | Agent identity — derived from capability graph |
| Spectral alignment (cosine similarity of eigenvalue spectra) | **Diversity score** for parent selection |
| Conservation ratio | **Trinity alignment** monitor — drops when ethos/pathos/logos drift |
| Fiedler vector | **Routing signal** — which agents should collaborate |

## Quickstart

```python
from fleet.conservation_spectral_bridge import (
    SpectralBreederDiversity,
    SpectralFingerprint,
)

# Register agents with their capabilities
sbd = SpectralBreederDiversity()

sbd.register_agent(
    agent_id="vision_specialist",
    capabilities=["vision", "detection", "tracking"],
    capability_links=[("vision", "detection", 0.9), ("detection", "tracking", 0.8)],
)

sbd.register_agent(
    agent_id="language_specialist",
    capabilities=["nlp", "translation", "summarization"],
    capability_links=[("nlp", "translation", 0.7), ("nlp", "summarization", 0.9)],
)

# Select diverse parents for breeding
parents = sbd.select_parents(n=2, min_diversity=0.3)
print(f"Selected: {parents}")

# Detect anomalies in agent coherence
anomalies = sbd.detect_anomalies()
for agent_id, severity in anomalies:
    print(f"ANOMALY: {agent_id} severity={severity:.2f}")
```

## Architecture

```
┌─────────────────────────────────────────┐
│   SuperInstance Conservation Spectral   │
│   Framework (Python/Rust/C/JS/...)      │
│   · Laplacian decomposition             │
│   · Eigenvalue spectra                  │
│   · Conservation ratios                 │
└──────────────┬──────────────────────────┘
               │
               │ conservation-spectral-python
               │ (optional — pure-Python fallback)
               ▼
┌─────────────────────────────────────────┐
│   SpectralBreederDiversity              │
│   · SpectralFingerprint per agent       │
│   · SpectralAlignmentScorer             │
│   · ConservationRatioMonitor            │
│   · Archive serialization (git/WAL)     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Sunset-Ecosystem Breeding Loop      │
│   · Parent selection via diversity      │
│   · Trinity anomaly detection           │
│   · Fiedler-based collaboration routing │
└─────────────────────────────────────────┘
```

## Classes

### `SpectralFingerprint`

An agent's spectral identity.

```python
fp = SpectralFingerprint.from_agent(
    agent_id="my_agent",
    capabilities=["vision", "audio", "reasoning"],
    capability_links=[("vision", "reasoning", 0.8)],
)

print(fp.conservation_ratio)  # structural coherence
print(fp.spectral_gap)  # λ₂ - λ₁
print(fp.alignment_coefficient)  # α = λ₂ / CR(a)
print(fp.fiedler_vector)  # routing signal

# Serialize for git/WAL storage
d = fp.to_dict()
fp2 = SpectralFingerprint.from_dict(d)
```

### `SpectralAlignmentScorer`

Score diversity between two agents. Higher = more diverse = better breeding pair.

```python
score = SpectralAlignmentScorer.score(fp_a, fp_b)  # 0.0 (identical) → 1.0 (orthogonal)
mat = SpectralAlignmentScorer.score_batch([fp_a, fp_b, fp_c])  # pairwise matrix
parents = SpectralAlignmentScorer.select_diverse_parents(
    [fp_a, fp_b, fp_c], n_parents=2, min_diversity=0.3
)
```

### `ConservationRatioMonitor`

Detect when an agent's structural coherence drops — a trinity violation.

```python
monitor = ConservationRatioMonitor(window_size=10, threshold=0.5)
monitor.record("agent_1", 0.8)
monitor.record("agent_1", 0.8)
monitor.record("agent_1", 0.4)  # sudden drop!
is_anom, severity = monitor.is_anomaly("agent_1")
# → (True, 0.5)
```

### `SpectralBreederDiversity`

Integration layer: register agents, score diversity, select parents, detect anomalies.

```python
sbd = SpectralBreederDiversity()
sbd.register_agent("a", ["vision", "audio"])
sbd.register_agent("b", ["reasoning", "planning"])

# Diversity score
sbd.score_diversity("a", "b")  # 0.0 → 1.0

# Select parents
sbd.select_parents(n=2, min_diversity=0.3)

# Detect anomalies
sbd.detect_anomalies()

# Archive / restore
archive = sbd.to_archive()
sbd2 = SpectralBreederDiversity.from_archive(archive)
```

### `ConservationSpectralEngine`

Optional wrapper around the real `conservation-spectral-python` package. Falls back to pure-Python if not installed.

```python
engine = ConservationSpectralEngine()
result = engine.analyze(adjacency_matrix)
print(result["conservation_ratio"])
print(result["spectral_gap"])
print(result["fiedler_vector"])
```

## Integration with Existing Modules

| Existing Module | How This Bridge Connects |
|---|---|
| `swarm/breeder.py` | Replace HDC novelty with `SpectralAlignmentScorer.score()` for diversity |
| `swarm/fleet_bft_qd.py` | Use conservation ratio as QD behavior descriptor |
| `fleet/sense_decide_act.py` | `Sense` → spectral fingerprint; `Decide` → alignment threshold; `Act` → parent selection |
| `nexus/fleet_conductor_v2.py` | Monitor conservation ratios as health metric |
| `fleet/sse_stream_dashboard.py` | Stream `conservation_ratio`, `spectral_gap`, `alignment_coefficient` |

## Mathematical Background

**Laplacian:** L = D - A, where D is degree matrix, A is adjacency matrix.

**Eigenvalue spectrum:** Solve Lv = λv. Sorted: λ₁ ≤ λ₂ ≤ ... ≤ λₙ.

**Conservation ratio:** CR = ||V·diag(λ)·Vᵀ||²_F / ||A||²_F — how much of the graph structure is captured by its spectral projection.

**Alignment coefficient:** α = λ₂ / CR(a) — predicts whether conservation will work in a new domain.

**Spectral alignment:** cos_sim(λ_A, λ_B) = ⟨λ_A, λ_B⟩ / (||λ_A|| · ||λ_B||) — how similar two agents' spectral identities are. Diversity = 1 - cos_sim.

## Tests

38 tests covering:
- Core spectral math (Laplacian, eigendecomposition, conservation ratio)
- Fingerprint construction from adjacency and capabilities
- Alignment and diversity scoring
- Parent selection with diversity thresholds
- Anomaly detection via conservation ratio drops
- Archive serialization/deserialization
- Engine fallback behavior

Run: `pytest tests/test_conservation_spectral_bridge.py -v`

## References

- [SuperInstance/SuperInstance](https://github.com/SuperInstance/SuperInstance) — Conservation Spectral Framework
- [SuperInstance/conservation-spectral-python](https://github.com/SuperInstance/conservation-spectral-python) — Python SDK
- `docs/ECOSYSTEM_INTEGRATION.md` — Sunset ecosystem integration map
