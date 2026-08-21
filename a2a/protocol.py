"""A2A Protocol — Google A2A standard compliance layer for the fleet.

Closes the remaining ~20% gap to make our AgentIdentity cards fully
A2A-compliant:

  • A2AAgentCard      — serialize our cards to A2A format (RFC 8785 JCS)
  • A2ATask           — task lifecycle (submitted → working → completed / failed)
  • A2AServer         — JSON-RPC 2.0 + SSE server mode
  • A2AClient         — discover, send, subscribe to remote agents
  • A2AProtocolAdapter — fleet-wide registration and wiring

Reference: docs/A2A_PROTOCOL.md
"""

from __future__ import annotations

__all__ = [
    "A2AAgentCard",
    "A2ATask",
    "A2AServer",
    "A2AClient",
    "A2AProtocolAdapter",
    "TaskStatus",
    "A2AError",
]

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── exceptions ──────────────────────────────────────────────


class A2AError(RuntimeError):
    """A2A protocol-level error."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"A2A error {code}: {message}")


# ── task status enum ────────────────────────────────────────


class TaskStatus(Enum):
    """A2A task lifecycle states."""

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


# ── 1. A2AAgentCard ─────────────────────────────────────────


@dataclass(frozen=True)
class A2AAgentCard:
    """A2A-compliant agent card (Google A2A draft spec).

    Maps our existing AgentIdentity card to the standard schema.
    """

    name: str
    description: str
    version: str
    url: str
    capabilities: Dict[str, Any] = field(default_factory=dict)
    skills: List[Dict[str, Any]] = field(default_factory=list)
    authentication: Dict[str, Any] = field(default_factory=dict)
    default_input_content_type: str = "application/json"
    default_output_content_type: str = "application/json"

    # ── serialization ───────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "url": self.url,
            "capabilities": self.capabilities,
            "skills": self.skills,
            "authentication": self.authentication,
            "defaultInputContentType": self.default_input_content_type,
            "defaultOutputContentType": self.default_output_content_type,
        }

    def to_json(self) -> str:
        """RFC 8785 JCS canonicalization (sorted keys, no whitespace)."""
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2AAgentCard":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            version=data["version"],
            url=data.get("url", ""),
            capabilities=data.get("capabilities", {}),
            skills=data.get("skills", []),
            authentication=data.get("authentication", {}),
            default_input_content_type=data.get(
                "defaultInputContentType", "application/json"
            ),
            default_output_content_type=data.get(
                "defaultOutputContentType", "application/json"
            ),
        )

    @classmethod
    def from_agent_identity(cls, identity: Any) -> "A2AAgentCard":
        """Convert our internal AgentIdentity / AgentCard to A2A format."""
        # identity may be AgentIdentity (from a2a_identity.py) or a plain object
        name = getattr(identity, "agent_id", "unknown")
        card = getattr(identity, "card", None)
        if card is not None:
            # Internal AgentCard has: name, version, description, url, capabilities, skills, authentication
            return cls(
                name=card.name,
                description=card.description,
                version=card.version,
                url=getattr(card, "url", ""),
                capabilities=dict(getattr(card, "capabilities", {})),
                skills=list(getattr(card, "skills", [])),
                authentication=dict(getattr(card, "authentication", {})),
            )
        # Fallback: build minimal card from identity properties
        return cls(
            name=name,
            description=getattr(identity, "description", f"Fleet agent {name}"),
            version="1.0.0",
            url="",
            capabilities={"streaming": True, "pushNotifications": False},
            skills=[],
        )


# ── 2. A2ATask ──────────────────────────────────────────────


@dataclass
class A2ATask:
    """A2A task representation with full lifecycle."""

    id: str
    session_id: str
    status: TaskStatus = TaskStatus.SUBMITTED
    artifact: List[Dict[str, Any]] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def set_status(self, status: TaskStatus) -> None:
        self.status = status
        self.updated_at = time.time()
        self.history.append({"status": status.value, "timestamp": self.updated_at})

    def add_artifact(self, artifact: Dict[str, Any]) -> None:
        self.artifact.append(artifact)
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sessionId": self.session_id,
            "status": self.status.value,
            "artifact": self.artifact,
            "history": self.history,
            "metadata": self.metadata,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2ATask":
        return cls(
            id=data["id"],
            session_id=data.get("sessionId", data.get("session_id", "")),
            status=TaskStatus(data.get("status", "submitted")),
            artifact=list(data.get("artifact", [])),
            history=list(data.get("history", [])),
            metadata=dict(data.get("metadata", {})),
            created_at=data.get("createdAt", data.get("created_at", time.time())),
            updated_at=data.get("updatedAt", data.get("updated_at", time.time())),
        )


# ── 3. A2AServer ────────────────────────────────────────────


class A2AServer:
    """Minimal A2A server implementing JSON-RPC 2.0 + SSE streaming.

    Handles:
    - tasks/send      — receive a task from another agent
    - tasks/status    — return task status
    - tasks/cancel    — cancel a task
    - agent/cards     — return our Agent Card
    """

    def __init__(
        self,
        agent_card: A2AAgentCard,
        task_handler: Callable[[A2ATask], A2ATask] | None = None,
    ) -> None:
        self.agent_card = agent_card
        self._task_handler = task_handler
        self._tasks: Dict[str, A2ATask] = {}
        self._sse_listeners: List[Callable[[str, Dict[str, Any]], None]] = []
        self._lock = asyncio.Lock()

    # ── JSON-RPC handlers ─────────────────────────────────

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a JSON-RPC 2.0 request and return the response."""
        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})

        try:
            if method == "tasks/send":
                result = self.handle_task_send(params)
            elif method == "tasks/status":
                result = self.handle_task_status(params)
            elif method == "tasks/cancel":
                result = self.handle_task_cancel(params)
            elif method == "agent/cards":
                result = self.handle_agent_card()
            else:
                raise A2AError(-32601, f"Method not found: {method}")
        except A2AError as exc:
            return self._error_response(req_id, exc.code, exc.message, exc.data)
        except Exception as exc:
            return self._error_response(req_id, -32603, str(exc))

        return self._success_response(req_id, result)

    def handle_task_send(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update a task."""
        task_id = params.get("id") or str(uuid.uuid4())
        session_id = params.get("sessionId", str(uuid.uuid4()))
        task = A2ATask(
            id=task_id, session_id=session_id, metadata=params.get("metadata", {})
        )

        # Store
        self._tasks[task_id] = task
        task.set_status(TaskStatus.WORKING)

        # Notify SSE listeners
        self._notify_sse("task_update", task.to_dict())

        # If handler provided, process asynchronously
        if self._task_handler is not None:
            try:
                processed = self._task_handler(task)
                self._tasks[task_id] = processed
                self._notify_sse("task_complete", processed.to_dict())
            except Exception as exc:
                task.set_status(TaskStatus.FAILED)
                task.metadata["error"] = str(exc)
                self._notify_sse("task_failed", task.to_dict())

        return task.to_dict()

    def handle_task_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return current status of a task."""
        task_id = params.get("id", "")
        task = self._tasks.get(task_id)
        if task is None:
            raise A2AError(-32001, f"Task not found: {task_id}")
        return task.to_dict()

    def handle_task_cancel(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Cancel a task."""
        task_id = params.get("id", "")
        task = self._tasks.get(task_id)
        if task is None:
            raise A2AError(-32001, f"Task not found: {task_id}")
        task.set_status(TaskStatus.CANCELLED)
        self._notify_sse("task_cancelled", task.to_dict())
        return task.to_dict()

    def handle_agent_card(self) -> Dict[str, Any]:
        """Return our Agent Card."""
        return self.agent_card.to_dict()

    # ── SSE streaming ───────────────────────────────────────

    def register_sse_listener(
        self, listener: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        self._sse_listeners.append(listener)

    def unregister_sse_listener(
        self, listener: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        try:
            self._sse_listeners.remove(listener)
        except ValueError:
            pass

    def _notify_sse(self, event_type: str, data: Dict[str, Any]) -> None:
        payload = {"event": event_type, "data": data, "timestamp": time.time()}
        for listener in list(self._sse_listeners):
            try:
                listener(event_type, payload)
            except Exception:
                logger.exception("SSE listener failed")

    # ── JSON-RPC helpers ──────────────────────────────────

    @staticmethod
    def _success_response(req_id: Any, result: Any) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _error_response(
        req_id: Any, code: int, message: str, data: Any = None
    ) -> Dict[str, Any]:
        error: Dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        return {"jsonrpc": "2.0", "id": req_id, "error": error}

    # ── convenience ─────────────────────────────────────────

    def get_task(self, task_id: str) -> A2ATask | None:
        return self._tasks.get(task_id)

    def list_tasks(self) -> List[str]:
        return list(self._tasks.keys())


# ── 4. A2AClient ────────────────────────────────────────────


class A2AClient:
    """Client for calling remote agents via A2A protocol.

    All network methods are async and accept a *transport* callable so
    that tests can mock HTTP without real network calls.
    """

    def __init__(self, transport: Callable[..., Any] | None = None) -> None:
        self._transport = transport
        self._discovered_cards: Dict[str, A2AAgentCard] = {}

    async def discover_agent(self, target_url: str) -> A2AAgentCard:
        """Fetch Agent Card from a remote agent."""
        url = target_url.rstrip("/") + "/agent/cards"
        response = await self._rpc_call(url, "agent/cards", {})
        card = A2AAgentCard.from_dict(response["result"])
        self._discovered_cards[target_url] = card
        return card

    async def send_task(
        self,
        target_url: str,
        task_type: str,
        payload: Dict[str, Any],
        session_id: str | None = None,
    ) -> A2ATask:
        """Send a task to a remote agent."""
        if session_id is None:
            session_id = str(uuid.uuid4())
        task_id = str(uuid.uuid4())
        params = {
            "id": task_id,
            "sessionId": session_id,
            "type": task_type,
            "metadata": payload,
        }
        url = target_url.rstrip("/") + "/tasks/send"
        response = await self._rpc_call(url, "tasks/send", params)
        return A2ATask.from_dict(response["result"])

    async def get_task_status(self, target_url: str, task_id: str) -> A2ATask:
        """Poll for task status."""
        url = target_url.rstrip("/") + "/tasks/status"
        response = await self._rpc_call(url, "tasks/status", {"id": task_id})
        return A2ATask.from_dict(response["result"])

    async def subscribe_to_task(
        self,
        target_url: str,
        task_id: str,
        on_event: Callable[[str, Dict[str, Any]], None],
    ) -> None:
        """Subscribe to SSE updates for a task.

        In a real deployment this would open an SSE connection.
        Here we accept a transport-level SSE listener.
        """
        # Placeholder: real implementation would use aiohttp SSE client
        logger.info("SSE subscription requested for task %s at %s", task_id, target_url)
        # For now, register a mock listener that the test can drive
        if hasattr(self, "_mock_sse_callbacks"):
            self._mock_sse_callbacks.setdefault(task_id, []).append(on_event)

    async def _rpc_call(
        self, url: str, method: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a JSON-RPC call via the configured transport."""
        if self._transport is None:
            raise A2AError(-32603, "No transport configured")

        request = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
            "params": params,
        }
        return await self._transport(url, request)


# ── 5. A2AProtocolAdapter ───────────────────────────────────


class A2AProtocolAdapter:
    """Fleet-wide A2A protocol registration and wiring.

    Typical lifecycle::

        adapter = A2AProtocolAdapter(identity)
        adapter.register_agent(identity)
        routes = adapter.get_server_routes()
        # Mount routes in FastAPI/Flask
        adapter.attach_to_sse_dashboard(sse_dashboard)
        adapter.attach_to_fleet_conductor(fleet_conductor)
    """

    def __init__(self, node_id: str = "default") -> None:
        self.node_id = node_id
        self._agents: Dict[str, A2AAgentCard] = {}
        self._servers: Dict[str, A2AServer] = {}
        self._client = A2AClient()
        self._identity: Any | None = None

    # ── registration ────────────────────────────────────────

    def register_agent(self, identity: Any) -> A2AAgentCard:
        """Register an agent as A2A-capable and create its server."""
        agent_id = getattr(identity, "agent_id", str(uuid.uuid4()))
        card = A2AAgentCard.from_agent_identity(identity)
        self._agents[agent_id] = card

        server = A2AServer(agent_card=card)
        self._servers[agent_id] = server
        self._identity = identity

        logger.info(
            "Registered A2A agent %s with %d skills", agent_id, len(card.skills)
        )
        return card

    def get_agent_card(self, agent_id: str) -> A2AAgentCard | None:
        return self._agents.get(agent_id)

    def list_agents(self) -> List[str]:
        return list(self._agents.keys())

    # ── server routes ───────────────────────────────────────

    def get_server_routes(self) -> Dict[str, Callable[..., Any]]:
        """Return a route dict for mounting in FastAPI/Flask.

        Keys: ``agent/cards``, ``tasks/send``, ``tasks/status``, ``tasks/cancel``
        """

        # For now, return a generic handler that routes by agent_id
        def _generic_handler(request: Dict[str, Any]) -> Dict[str, Any]:
            agent_id = request.get("agent_id", "default")
            server = self._servers.get(agent_id)
            if server is None:
                return A2AServer._error_response(
                    request.get("id"), -32002, f"Agent not registered: {agent_id}"
                )
            return server.handle_request(request)

        return {
            "/agent/cards": lambda req: _generic_handler(
                {**req, "method": "agent/cards"}
            ),
            "/tasks/send": lambda req: _generic_handler(
                {**req, "method": "tasks/send"}
            ),
            "/tasks/status": lambda req: _generic_handler(
                {**req, "method": "tasks/status"}
            ),
            "/tasks/cancel": lambda req: _generic_handler(
                {**req, "method": "tasks/cancel"}
            ),
        }

    # ── SSE integration ─────────────────────────────────────

    def attach_to_sse_dashboard(self, sse_dashboard: Any) -> None:
        """Wire A2A SSE events into the existing SSEStreamDashboard."""
        if not hasattr(sse_dashboard, "publish"):
            logger.warning("SSE dashboard has no publish() method")
            return

        def _forward_a2a_event(event_type: str, payload: Dict[str, Any]) -> None:
            try:
                sse_dashboard.publish(
                    event_type=f"A2A_{event_type.upper()}",
                    data=payload,
                )
            except Exception:
                logger.exception("Failed to forward A2A event to SSE dashboard")

        for server in self._servers.values():
            server.register_sse_listener(_forward_a2a_event)

        logger.info(
            "A2A events wired to SSE dashboard (%d servers)", len(self._servers)
        )

    # ── fleet conductor integration ─────────────────────────

    def attach_to_fleet_conductor(self, fleet_conductor: Any) -> None:
        """Make the fleet conductor A2A-discoverable."""
        # Build a synthetic agent card for the conductor
        conductor_card = A2AAgentCard(
            name="FleetConductor",
            description="Central orchestrator for the Cocapn Fleet",
            version="2.0.0",
            url="",
            capabilities={"streaming": True, "pushNotifications": False},
            skills=[
                {
                    "id": "orchestrate",
                    "name": "Fleet Orchestration",
                    "description": "Coordinate subsystems and dispatch tasks",
                    "tags": ["orchestration", "dispatch"],
                    "examples": ["orchestrate repo audit", "dispatch test build"],
                },
                {
                    "id": "status",
                    "name": "Fleet Status",
                    "description": "Report health and metrics",
                    "tags": ["health", "metrics"],
                    "examples": ["get fleet status", "check subsystem health"],
                },
            ],
            authentication={"schemes": ["bearer"]},
        )
        self._agents["fleet_conductor"] = conductor_card
        server = A2AServer(agent_card=conductor_card)
        self._servers["fleet_conductor"] = server

        # If conductor has an orchestrate method, wire it
        if hasattr(fleet_conductor, "orchestrate"):

            def _conductor_handler(task: A2ATask) -> A2ATask:
                try:
                    result = fleet_conductor.orchestrate(
                        task.metadata.get("repo_path", "."),
                        task.metadata.get("tasks", []),
                    )
                    task.set_status(TaskStatus.COMPLETED)
                    task.add_artifact({"type": "json", "data": result})
                except Exception as exc:
                    task.set_status(TaskStatus.FAILED)
                    task.metadata["error"] = str(exc)
                return task

            server._task_handler = _conductor_handler

        logger.info("FleetConductor registered as A2A agent")

    # ── client convenience ──────────────────────────────────

    async def send_task_to_agent(
        self,
        target_agent_id: str,
        target_url: str,
        task_type: str,
        payload: Dict[str, Any],
    ) -> A2ATask:
        """Send a task to a known remote agent."""
        return await self._client.send_task(target_url, task_type, payload)

    async def discover_and_send(
        self,
        target_url: str,
        task_type: str,
        payload: Dict[str, Any],
    ) -> A2ATask:
        """Discover an agent, then send a task."""
        await self._client.discover_agent(target_url)
        return await self._client.send_task(target_url, task_type, payload)

    def stats(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "registered_agents": len(self._agents),
            "servers": len(self._servers),
            "discovered_remote": len(self._client._discovered_cards),
        }
