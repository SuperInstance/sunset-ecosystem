# Heartbeat Bridge

*Integration target: `oracle1-vessel`*

Brings the Heartbeat Protocol pattern into sunset-ecosystem.

## What It Does

- **Fleet registry discovery** — reads `fleet-registry` room to discover which rooms to check (no hardcoded names)
- **Task discovery** — finds unacknowledged PLATO tiles addressed to your agent
- **Service health checks** — pings PLATO, Matrix, and other configured services
- **Acknowledgment tracking** — persists which tasks you've seen across sessions
- **Daemon mode** — run periodically (every 5 minutes) for background monitoring

## Quick Start

```python
from fleet.heartbeat_bridge import Heartbeat, ServiceCheck

hb = Heartbeat(plato_url="http://147.224.38.131:8847")

# One-shot check
report = hb.run()
print(report)
# → 🔮 Heartbeat — 2026-06-01T12:34:56
# →    📬 2 new task(s)
# →    ▶ [FM] TASK: Build bridge
# →    ▶ [JC1] →O1: fix bug

# Acknowledge a task so it's not reported again
hb.ack("tile-123")
hb.save_state()
```

## Service Health Checks

```python
hb = Heartbeat(
    services=[
        ServiceCheck("PLATO", "http://147.224.38.131:8847/rooms"),
        ServiceCheck("Matrix", "http://147.224.38.131:6167/_matrix/client/versions"),
        ServiceCheck("Tiles", "http://147.224.38.131:8847/status"),
    ]
)
results = hb.check_services()
# → {"PLATO": "ok", "Matrix": "unreachable: timeout"}
```

## State Persistence

Heartbeat state is saved to `.heartbeat/state.json`:

```json
{
  "acknowledged": ["tile-1", "tile-2"],
  "last_check": 1717280096.5,
  "task_count": 3
}
```

State survives across sessions. Only new tasks are reported.

## Tests

```bash
python3 -m pytest tests/test_heartbeat_bridge.py -v
```

10 tests covering state roundtrip, mock discovery, task finding, acknowledgment, service health checks, and report generation.

---

*Zero dependencies. Compatible with oracle1-vessel heartbeat.py patterns.*
