# A2A Protocol

Google A2A (Agent-to-Agent) protocol implementation for the Cocapn Fleet.

Implements core A2A spec: agent cards, task lifecycle, SSE streaming, and fleet conductor integration.

---

## Quick Start

```python
from logos.a2a_protocol import A2AAgentCard, A2AServer, A2AClient

# Server
server = A2AServer(
    agent_card=A2AAgentCard(
        name="FleetNode",
        description="Cocapn fleet node",
        version="1.0.0",
        url="http://node.local",
        capabilities={"streaming": True, "pushNotifications": False},
        skills=[{"id": "breed", "name": "Agent Breeding"}],
        authentication={"schemes": ["bearer"]},
    )
)
response = server.handle_request({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tasks/send",
    "params": {"id": "task_1", "sessionId": "sess_1"},
})
```

---

## Classes

### `A2AAgentCard`

Agent metadata per A2A spec.

- `to_dict()` / `to_json()` — canonical JSON (sorted keys)
- `from_dict(data)` — deserialize
- `from_agent_identity(agent_identity)` — build from fleet AgentIdentity

Fields:
```python
name, description, version, url,
capabilities, skills, authentication,
default_input_content_type, default_output_content_type
```

### `A2ATask`

Task lifecycle model.

Statuses: `submitted`, `working`, `input_required`, `completed`, `cancelled`, `failed`, `unknown`

- `set_status(status)` — updates + appends to history
- `add_artifact(artifact)` — attach output
- `to_dict()` / `from_dict(data)` — serialize / deserialize

### `A2AServer`

JSON-RPC request handler.

Methods:
- `agent/cards` → return agent card
- `tasks/send` → create and process task
- `tasks/status` → get task by ID
- `tasks/cancel` → cancel task

`task_handler` callback: `Callable[[A2ATask], A2ATask]` — runs on each `tasks/send`.

SSE listeners: `register_sse_listener(callback)` — emits events for every state change.

### `A2AClient`

Async client for remote agent discovery and task dispatch.

- `discover_agent(url)` → `A2AAgentCard`
- `send_task(url, task_name, params)` → `A2ATask`
- `get_task_status(url, task_id)` → `A2ATask`

Transport: injectable async callable `(url, request) -> response`. Defaults to HTTP if `httpx` / `aiohttp` available, otherwise raises `A2AError`.

### `A2AProtocolAdapter`

Fleet-specific wrapper that registers multiple agents as a single A2A server.

- `register_agent(agent_identity)` → `A2AAgentCard`
- `get_server_routes()` → `Dict[str, Callable]` — FastAPI/Flask route handlers
- `attach_to_fleet_conductor(fleet_conductor)` — auto-register conductor as agent
- `attach_to_sse_dashboard(sse_dashboard)` — wire SSE events
- `discover_and_send(url, task_name, params)` — one-shot discover + dispatch

---

## Task Lifecycle

```
submitted → working → [input_required] → completed
                        ↓
                    cancelled / failed
```

Each transition is logged in `task.history` with timestamp.

---

## SSE Events

Server-Sent Events emitted by `A2AServer`:

| Event Type | Trigger |
|-----------|---------|
| `task_update` | Any status change |
| `task_artifact` | Artifact added |
| `task_complete` | Status → completed |
| `task_failed` | Status → failed |
| `task_cancelled` | Status → cancelled |

Listeners receive `(event_type: str, data: dict)`.

---

## Integration Points

### FleetConductorV2
```python
from logos.a2a_protocol import A2AProtocolAdapter
adapter = A2AProtocolAdapter()
adapter.attach_to_fleet_conductor(conductor)
```

### SSE Stream Dashboard
```python
from fleet.sse_stream_dashboard import SSEStreamDashboard
sse = SSEStreamDashboard()
adapter.attach_to_sse_dashboard(sse)
# Now every A2A task event also goes to the SSE stream
```

### Agent Identity
```python
from logos.agent_identity import AgentIdentity
identity = AgentIdentity(agent_id="scout_1")
adapter.register_agent(identity)
```

---

## Testing

Run: `python3 -m pytest tests/test_a2a_protocol.py -v`

31 tests covering:
- Agent card roundtrip, canonical JSON, from identity
- Task status lifecycle, artifact addition
- Server request dispatch (cards, send, status, cancel)
- Custom task handler (success + failure)
- SSE listener notification
- Client discover / send / status with mock transport
- Protocol adapter (register, routes, dispatch, unknown agent)
- Fleet conductor attachment
- SSE dashboard wiring
- Stats and discover-and-send

---

## Protocol Notes

- JSON-RPC 2.0 request/response format
- Standard error codes: -32601 (method not found), -32603 (internal error)
- Fleet extension: -32002 (agent not found)
- All timestamps are Unix epoch seconds
- No streaming push notification support yet (capability advertised as False)
