# SSE Stream Dashboard

Real-time breeding progress via Server-Sent Events (SSE).

## Quick Start

```python
from fleet.sse_stream_dashboard import SSEStreamDashboard, StreamEvent, EventType

dash = SSEStreamDashboard()
sub = dash.subscribe()

# Publish events
dash.publish(StreamEvent(EventType.BEAT, payload={"n": 1}, node_id="node-1"))

# Or use the simple API
dash.publish_simple(EventType.THERMAL, {"temp": 42.0}, node_id="node-1")

# Consume (in an HTTP handler, yield these)
for msg in sub.get(timeout=1.0):
    yield msg.to_sse()
```

## Event Types

| Type | When |
|------|------|
| `BEAT` | Conductor tick |
| `PARENT_SELECT` | Breeder selected parents |
| `MUTATION` | Mutation applied |
| `FLUX_GATE` | FLUX constraint check |
| `THERMAL` | Thermal state change |
| `FLEET_STATUS` | Full fleet snapshot |
| `AGENT_SPAWN` | New agent dispatched |
| `ERROR` | Something went wrong |
| `INFO` | General information |

## HTTP Handler Example (Flask/FastAPI)

```python
from flask import Flask, Response

app = Flask(__name__)
dash = SSEStreamDashboard()

@app.route("/stream")
def stream():
    def event_stream():
        sub = dash.subscribe()
        while True:
            ev = sub.get(timeout=30.0)
            yield ev.to_sse()
    return Response(event_stream(), mimetype="text/event-stream")
```

## Configuration

```python
from fleet.sse_stream_dashboard import DashboardConfig

cfg = DashboardConfig(
    max_queue_size=1000,        # drop old events when full
    heartbeat_interval_sec=15,  # keepalive ping
    history_buffer_size=100,    # replay last N events to new subscribers
    filter_event_types=["BEAT", "FLEET_STATUS"],  # only stream these
    enable_backpressure=True,   # drop instead of evict when full
)
```

## Integration with FleetConductorV2

```python
from fleet.sse_stream_dashboard import SSEStreamDashboard, wire_to_fleet_conductor

dash = SSEStreamDashboard()
wire_to_fleet_conductor(dash, conductor)

# Now every conductor.beat() publishes BEAT + FLEET_STATUS events
```

## Integration with BreederDaemonV2

```python
from fleet.sse_stream_dashboard import wire_to_breeder

wire_to_breeder(dash, breeder)
# Now every breeder.cycle() publishes start/end events
```

## Metrics

```python
dash.get_metrics()
# {
#   "subscribers": 3,
#   "queue_depth": 12,
#   "history_size": 100,
#   "max_queue_size": 1000,
#   "heartbeat_interval": 15.0,
# }
```

## Architecture

```
Publisher → Queue → Fan-out → Subscribers
                ↓
            History Buffer (replay to new subscribers)
```

- Thread-safe: publish from any thread
- Backpressure: configurable drop vs evict
- Subscriber cleanup: slow subscribers auto-removed

## Reference

- `fleet/sse_stream_dashboard.py` — implementation
- `tests/test_sse_stream_dashboard.py` — 18 tests
