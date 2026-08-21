# Health Bridge

*Integration target: `cocapn-health`*

Brings the cocapn-health `CheckResult` pattern into sunset-ecosystem as a zero-dependency Python module.

## What It Does

- Standardized health check results (name, ok, latency_ms, status, details)
- Fleet service definitions with metric extraction from JSON responses
- Event bus bridge for service transition notifications (UP→DOWN, DOWN→UP)
- REST API with cache TTL
- System checks (disk, memory, CPU)

## Quick Start

```python
from fleet.health_bridge import HealthChecker, ServiceDef, CheckResult, FLEET_SERVICES

# Check all 18 fleet services
checker = HealthChecker(FLEET_SERVICES)
results = checker.check_all()

# Report in multiple formats
print(HealthChecker.report(results, format="md"))
print(HealthChecker.report(results, format="json"))
print(HealthChecker.report(results, format="oneline"))

# Check a single service
result = checker.check_one(FLEET_SERVICES[0])
print(result.name, result.ok, result.status)

# System checks
system_results = HealthChecker.check_system()
for r in system_results:
    print(r.name, r.status, r.details)
```

## Event Bus Integration

```python
from fleet.health_bridge import EventBusHealthChecker


class MyEventBus:
    def emit(self, event_type, payload):
        print(
            f"EVENT: {event_type} — {payload['name']} is {'UP' if payload['ok'] else 'DOWN'}"
        )


bus = MyEventBus()
checker = EventBusHealthChecker(FLEET_SERVICES, bus=bus, emit_on_every_check=False)
results = checker.check_all()  # emits only on transitions
```

## Health Cache

```python
from fleet.health_bridge import HealthCache

cache = HealthCache(ttl=30.0)
cache.set_services(FLEET_SERVICES)

# First call checks all services
results = cache.get(checker)

# Second call within 30s returns cached results
results = cache.get(checker)

# Force refresh
cache.get(checker, force=True)
```

## Service Definitions

The 18 fleet services are pre-defined in `FLEET_SERVICES`:

| Service | Host | Port | Path |
|---------|------|------|------|
| MUD v3 | <BOAT_IP> | 4042 | /status |
| The Lock v2 | <BOAT_IP> | 4043 | /status |
| Arena | <BOAT_IP> | 4044 | /stats |
| Grammar Engine | <BOAT_IP> | 4045 | /grammar |
| Dashboard | <BOAT_IP> | 4046 | / |
| Federated Nexus | <BOAT_IP> | 4047 | / |
| Harbor | <BOAT_IP> | 4050 | / |
| Grammar Compactor | <BOAT_IP> | 4055 | /status |
| Rate-Attention | <BOAT_IP> | 4056 | /streams |
| Skill Forge | <BOAT_IP> | 4057 | /status |
| PLATO Terminal | <BOAT_IP> | 4060 | / |
| PLATO Gate | <BOAT_IP> | 8847 | /rooms |
| PLATO Shell | <BOAT_IP> | 8848 | / |
| Service Guard | <BOAT_IP> | 8899 | / |
| Task Queue | <BOAT_IP> | 8900 | / |
| Steward | <BOAT_IP> | 8901 | / |
| Matrix Bridge | <BOAT_IP> | 6168 | /status |
| Conduwuit | <BOAT_IP> | 6167 | / |

## Adding a Custom Service

```python
from fleet.health_bridge import ServiceDef

my_service = ServiceDef(
    name="My API",
    host="127.0.0.1",
    port=8080,
    path="/health",
    extract={"version": "api_version"},  # json_path → metric_name
)
```

## API Reference

### `CheckResult(name, ok, latency_ms, status, details={})`

Standardized health check result. `to_dict()` and `from_dict(d)` for serialization.

### `ServiceDef(name, host, port, path="/", timeout=5.0, expect_status=None, extract=None, headers=None)`

Fleet service definition. `url()` builds the full URL. `extract` is a dict of `{json_path: metric_name}` for JSON response metric extraction.

### `HealthChecker(services)`

Fleet health checker with methods:
- `check_all()` — check all services
- `check_one(svc)` — check one service
- `check_http(url, ...)` — static method for HTTP checks
- `check_tcp(host, port)` — static method for TCP checks
- `check_system()` — static method for disk/memory/CPU
- `report(results, format="md")` — format results as markdown/json/oneline

### `EventBusHealthChecker(services, bus=None, emit_on_every_check=False)`

Extends `HealthChecker` with event emission on service transitions. Supports `emit()`, `publish()`, `send()`, or callable bus interfaces.

### `HealthCache(ttl=30.0)`

Caches check results with TTL. `set_services(services)`, `get(checker, force=False)`, `clear()`.

## Tests

```bash
python3 -m pytest tests/test_health_bridge.py -v
```

38 tests covering HTTP checks, TCP checks, system checks, fleet-wide checking, reporting, event bus transitions, cache behavior.

---

*Zero dependencies. Compatible with cocapn-health data structures.*
