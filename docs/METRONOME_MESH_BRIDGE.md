# Metronome Mesh Gossip Bridge

Unifies two reverse-actualization P0 modules:

- **Distributed Metronome Bridge** — cross-node beat synchronization
- **Mesh Vector Gossip** — federated CRDT-based vector table updates

## What It Does

The bridge transports metronome sync messages (beat ticks, drift corrections)
over the mesh gossip protocol. This means **one gossip channel handles both**:

1. Vector table CRDT updates
2. Metronome beat synchronization

## Quick Start

```python
from nerve.metronome_mesh_bridge import MetronomeGossipBridge, BridgeConfig

bridge = MetronomeGossipBridge(
    BridgeConfig(
        enable_beat_gossip=True,
        enable_drift_gossip=True,
        enable_vector_passthrough=True,
    )
)

bridge.attach_metronome(metronome)
bridge.attach_gossip(gossip)
bridge.start()

# On each local beat, metronome calls:
bridge.on_metronome_beat(bpm=120.0, beat_number=42, drift_ms=0.5)
# → broadcasts to all peers via gossip

# Incoming gossip messages are decoded and routed:
bridge.on_gossip_receive(raw_payload)
# → BEAT → update local metronome
# → DRIFT_CORRECTION → apply correction
# → VECTOR_UPDATE → pass through to vector table
```

## Message Types

| Type | Source | Action |
|------|--------|--------|
| `BEAT` | Local metronome | Broadcast to mesh |
| `DRIFT_CORRECTION` | Local PID controller | Broadcast to mesh |
| `NODE_ANNOUNCE` | Startup / rejoin | Broadcast presence |
| `VECTOR_UPDATE` | Local vector table | Passthrough to mesh |

## Deduplication

The bridge deduplicates messages using a `{node_id}:{type}:{beat_number}` key
with a 60-second window. This prevents echo loops in dense meshes.

## Stale Message Rejection

Messages older than `max_gossip_age_sec` (default 30s) are dropped. This
prevents delayed beat corrections from causing oscillations.

## Configuration

```python
BridgeConfig(
    enable_drift_gossip=True,  # forward drift corrections
    enable_beat_gossip=True,  # forward beat ticks
    enable_vector_passthrough=True,  # forward vector updates
    max_gossip_age_sec=30.0,  # drop stale messages
    dedup_window_sec=60.0,  # deduplication window
)
```

## Metrics

```python
bridge.get_metrics()
# {
#   "node_id": "node-1",
#   "metronome_attached": True,
#   "gossip_attached": True,
#   "running": True,
#   "dedup_window_size": 12,
#   "config": { ... }
# }
```

## Integration with FleetConductorV2

```python
from fleet.sse_stream_dashboard import wire_to_fleet_conductor
from nerve.metronome_mesh_bridge import MetronomeGossipBridge

bridge = MetronomeGossipBridge()
bridge.attach_metronome(conductor._get_metronome())
bridge.attach_gossip(conductor._get_mesh())
bridge.start()

# Now every beat and every vector update propagates fleet-wide
```

## Reference

- `nerve/metronome_mesh_bridge.py` — implementation
- `tests/test_metronome_mesh_bridge.py` — 19 tests
- `docs/DISTRIBUTED_METRONOME_BRIDGE.md` — metronome details
- `docs/MESH_VECTOR_TABLES.md` — gossip details
