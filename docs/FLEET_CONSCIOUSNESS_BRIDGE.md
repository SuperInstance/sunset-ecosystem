# Fleet Consciousness Bridge

*Integration target: `fleet-consciousness-dashboard`*

Brings the Fleet Consciousness Index (FCI) into sunset-ecosystem as a zero-dependency Python module.

## What is FCI?

The Fleet Consciousness Index is a weighted composite score (0.0–1.0) measuring fleet-wide consciousness across four dimensions:

| Metric | Weight | Description |
|--------|--------|-------------|
| Room Phi | 40% | Room integration via tile count |
| Attention | 20% | Agent participation in attention tracking |
| Learning | 25% | Ratio of positive to total learning passes |
| Meta | 15% | Average meta-level depth of tiles |

### Consciousness Levels

| Range | Level | Recommendation |
|-------|-------|----------------|
| < 0.15 | dormant | Activate rooms and seed initial tiles |
| 0.15–0.30 | emerging | Increase agent participation and room density |
| 0.30–0.45 | aware | Enable attention tiles from all agents |
| 0.45–0.60 | conscious | Deepen meta-level tiles and cross-room correlations |
| 0.60–0.75 | self-aware | Optimize learning passes and Penrose correlations |
| > 0.75 | transcendent | Monitor for degradation and maintain diversity |

## Quick Start

```python
from fleet.fleet_consciousness_bridge import FleetConsciousnessIndex

fci = FleetConsciousnessIndex()

# Compute from component scores
score = fci.compute(
    room_phi_score=0.50,
    attention_score=0.30,
    learning_score=0.50,
    meta_score=0.20,
)
print(score.fci)        # 0.415
print(score.level)      # "aware"
print(score.recommendation)

# Or compute from raw fleet metrics
score = FleetConsciousnessIndex.from_fleet_metrics(
    rooms=30,
    total_rooms_capacity=100,
    active_agents=5,
    total_agents=10,
    positive_learning_passes=8,
    total_learning_passes=10,
    meta_tile_depth_sum=20.0,
    total_tiles=50,
)

# Render dashboard
print(fci.render_text(score))
print(fci.render_json(score))
print(fci.render_oneline(score))
```

## Custom Weights

```python
fci = FleetConsciousnessIndex(weights={
    "room_phi": 0.25,
    "attention": 0.25,
    "learning": 0.25,
    "meta": 0.25,
})
```

## Integration with SSE Stream Dashboard

The FCI score can be emitted as a fleet-wide metric:

```python
from fleet.fleet_consciousness_bridge import FleetConsciousnessIndex
from fleet.sse_stream_dashboard import SSEStreamDashboard

fci = FleetConsciousnessIndex()
score = fci.compute(...)
dashboard = SSEStreamDashboard()
dashboard.emit_metric("fci", score.fci, level=score.level)
```

## API Reference

### `FleetConsciousnessIndex(weights=None)`

Constructor. Optional `weights` dict overrides default weights.

### `compute(room_phi_score, attention_score, learning_score, meta_score, **details)`

Compute FCI from four normalized component scores (0.0–1.0). Returns `ConsciousnessScore`.

### `from_fleet_metrics(rooms, total_rooms_capacity, active_agents, total_agents, positive_learning_passes, total_learning_passes, meta_tile_depth_sum, total_tiles, **details)`

Class method. Compute FCI from raw fleet metrics. Handles zero-division safely.

### `render_text(score)`, `render_json(score)`, `render_oneline(score)`

Render a `ConsciousnessScore` in three formats: text dashboard, JSON, or one-line summary.

## Tests

```bash
python3 -m pytest tests/test_fleet_consciousness_bridge.py -v
```

14 tests covering all 6 consciousness levels, rendering formats, raw metrics, edge cases.

---

*Zero dependencies. Compatible with fleet-consciousness-dashboard weights and levels.*
