"""XlangAgentBridge — Bridge between sunset-ecosystem and xlang/xMind runtime.

Provides:
- Python → xlang module importing (GIL-bypassed C++ interop)
- JSON agent graph → xMind YAML blueprint conversion
- Session memory synchronization between our system and xMind AgentFlow
- Distributed execution via xlang's LRPC IPC layer

Architecture
------------
The bridge has three layers:

1. **Runtime Layer** — lazy-loaded xlang C++ engine via `import xlang`.
   Handles module loading, IPC connections, and tensor marshalling.

2. **AgentFlow Layer** — converts our `json_agent_graph.py` JSON graphs
   into xMind's YAML blueprints (nodes, pins, actions, agents).

3. **Session Layer** — bidirectional session memory sync.  Our fleet
   sessions (agent_id, context, history) map to xMind's session IDs.

Reference
---------
- xlang runtime: https://github.com/xlang-foundation/xlang
- xMind AgentFlow: https://github.com/xlang-foundation/xMind/AgentFlow.md
- xlang IPC: https://github.com/xlang-foundation/xlang/Docs/DISTRIBUTED.md
"""

from __future__ import annotations

__all__ = [
    "XlangAgentBridge",
    "AgentFlowBlueprint",
    "SessionSyncAdapter",
]

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ── AgentFlowBlueprint ──────────────────────────────────────────


@dataclass
class AgentFlowBlueprint:
    """xMind AgentFlow YAML blueprint, generated from a JSON agent graph.

    Nodes: function | action | agent
    Pins:  input / output interfaces with X::Value data
    """

    name: str
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)
    session_memory: dict[str, Any] = field(default_factory=dict)

    def to_yaml(self) -> str:
        return yaml.dump(
            {
                "blueprint": {
                    "name": self.name,
                    "variables": self.variables,
                    "nodes": self.nodes,
                    "edges": self.edges,
                }
            },
            default_flow_style=False,
            sort_keys=False,
        )

    @classmethod
    def from_json_graph(
        cls, graph: dict[str, Any], name: str = "fleet_graph"
    ) -> "AgentFlowBlueprint":
        """Convert a sunset-ecosystem JSON agent graph to xMind YAML.

        JSON graph format (from json_agent_graph.py):
        {
            "nodes": [{"id": "n1", "type": "agent", "config": {...}}],
            "edges": [{"source": "n1", "target": "n2", "relation": "delegates"}]
        }
        """
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        variables: dict[str, Any] = {}

        for n in graph.get("nodes", []):
            node_type = n.get("type", "function")
            node: dict[str, Any] = {
                "id": n["id"],
                "type": node_type,
            }
            if node_type == "agent":
                node["llm_config"] = n.get("config", {})
                node["prompt_template"] = n.get("prompt", "")
                node["output_pins"] = ["result", "status"]
            elif node_type == "action":
                node["action_type"] = n.get("action", "rest_api")
                node["endpoint"] = n.get("endpoint", "")
                node["timeout_ms"] = n.get("timeout_ms", 5000)
            else:  # function
                node["function"] = n.get("function", "pass_through")
                node["input_pins"] = ["input"]
                node["output_pins"] = ["output"]
            nodes.append(node)

        for e in graph.get("edges", []):
            edges.append(
                {
                    "source": e.get("source", ""),
                    "source_pin": e.get("source_pin", "output"),
                    "target": e.get("target", ""),
                    "target_pin": e.get("target_pin", "input"),
                    "relation": e.get("relation", "flows_to"),
                }
            )

        return cls(name=name, nodes=nodes, edges=edges, variables=variables)


# ── SessionSyncAdapter ──────────────────────────────────────────


@dataclass
class SessionSyncAdapter:
    """Bidirectional session memory sync between fleet and xMind."""

    session_id: str
    fleet_context: dict[str, Any] = field(default_factory=dict)
    xmind_session_handle: Any | None = None

    def to_xmind_payload(self) -> dict[str, Any]:
        """Serialize fleet context for xMind session binding."""
        return {
            "session_id": self.session_id,
            "history": self.fleet_context.get("history", []),
            "variables": self.fleet_context.get("variables", {}),
            "metadata": {
                "node_id": self.fleet_context.get("node_id", "unknown"),
                "agent_id": self.fleet_context.get("agent_id", "unknown"),
                "timestamp": time.time(),
            },
        }

    def from_xmind_payload(self, payload: dict[str, Any]) -> None:
        """Update fleet context from xMind session output."""
        self.fleet_context["history"] = payload.get("history", [])
        self.fleet_context["variables"] = payload.get("variables", {})
        self.fleet_context["last_xmind_update"] = time.time()


# ── XlangAgentBridge ────────────────────────────────────────────


class XlangAgentBridge:
    """Bridge to the xlang runtime and xMind AgentFlow framework.

    Parameters
    ----------
    node_id : str
        Fleet node identifier.
    xmind_path : Path | str | None
        Path to xMind installation (for local embedding mode).
    lrpc_endpoint : str | None
        Remote LRPC endpoint for distributed mode (e.g. "lrpc:9090").
    """

    def __init__(
        self,
        node_id: str,
        xmind_path: Path | str | None = None,
        lrpc_endpoint: str | None = None,
    ) -> None:
        self.node_id = node_id
        self.xmind_path = Path(xmind_path) if xmind_path else None
        self.lrpc_endpoint = lrpc_endpoint

        self._lock = threading.RLock()
        self._xlang: Any | None = None  # lazy-loaded xlang module
        self._xmind: Any | None = None  # lazy-loaded xMind module
        self._sessions: dict[str, SessionSyncAdapter] = {}
        self._graphs: dict[str, AgentFlowBlueprint] = {}

        # Stats
        self._graphs_executed = 0
        self._sessions_synced = 0
        self._remote_calls = 0

    # ── lazy xlang loading ────────────────────────────────────────

    def _load_xlang(self) -> Any | None:
        """Lazy-import the xlang C++ runtime."""
        if self._xlang is not None:
            return self._xlang
        try:
            import xlang  # type: ignore[import-untyped]

            self._xlang = xlang
            logger.info("xlang C++ runtime loaded")
        except Exception as exc:
            logger.warning("Failed to load xlang runtime: %s", exc)
            self._xlang = None
        return self._xlang

    def _load_xmind(self) -> Any | None:
        """Lazy-import the xMind AgentFlow framework."""
        if self._xmind is not None:
            return self._xmind
        xlang = self._load_xlang()
        if xlang is None:
            return None
        try:
            if self.xmind_path:
                self._xmind = xlang.importModule("xMind", fromPath=str(self.xmind_path))
            else:
                self._xmind = xlang.importModule("xMind")
            logger.info("xMind AgentFlow framework loaded")
        except Exception as exc:
            logger.warning("Failed to load xMind: %s", exc)
            self._xmind = None
        return self._xmind

    # ── graph conversion ────────────────────────────────────────

    def convert_graph(
        self, json_graph: dict[str, Any], name: str
    ) -> AgentFlowBlueprint:
        """Convert a JSON agent graph to xMind YAML blueprint."""
        blueprint = AgentFlowBlueprint.from_json_graph(json_graph, name=name)
        with self._lock:
            self._graphs[name] = blueprint
        return blueprint

    def save_blueprint(self, name: str, path: Path | str) -> None:
        """Save a blueprint to a YAML file."""
        with self._lock:
            blueprint = self._graphs.get(name)
            if blueprint is None:
                raise KeyError(f"Blueprint '{name}' not found")
        Path(path).write_text(blueprint.to_yaml(), encoding="utf-8")

    def load_blueprint(self, path: Path | str) -> AgentFlowBlueprint:
        """Load a blueprint from a YAML file."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        bp = AgentFlowBlueprint(
            name=data["blueprint"]["name"],
            nodes=data["blueprint"].get("nodes", []),
            edges=data["blueprint"].get("edges", []),
            variables=data["blueprint"].get("variables", {}),
        )
        with self._lock:
            self._graphs[bp.name] = bp
        return bp

    # ── session management ────────────────────────────────────────

    def create_session(
        self, session_id: str, context: dict[str, Any] | None = None
    ) -> SessionSyncAdapter:
        """Create a new session bridge between fleet and xMind."""
        adapter = SessionSyncAdapter(
            session_id=session_id,
            fleet_context=context or {},
        )
        with self._lock:
            self._sessions[session_id] = adapter
        return adapter

    def sync_session_to_xmind(self, session_id: str) -> dict[str, Any]:
        """Push fleet session state to xMind."""
        with self._lock:
            adapter = self._sessions.get(session_id)
            if adapter is None:
                return {"error": f"Session {session_id} not found"}

        xmind = self._load_xmind()
        if xmind is None:
            return {
                "error": "xMind not available",
                "fallback": adapter.to_xmind_payload(),
            }

        try:
            payload = adapter.to_xmind_payload()
            # xMind session binding call
            handle = xmind.SessionManager.bindSession(payload)
            adapter.xmind_session_handle = handle
            self._sessions_synced += 1
            return {"status": "synced", "session_id": session_id}
        except Exception as exc:
            logger.warning("xMind sync failed: %s", exc)
            return {"error": str(exc), "fallback": adapter.to_xmind_payload()}

    def sync_session_from_xmind(self, session_id: str) -> dict[str, Any]:
        """Pull xMind session state back to fleet."""
        with self._lock:
            adapter = self._sessions.get(session_id)
            if adapter is None:
                return {"error": f"Session {session_id} not found"}

        xmind = self._load_xmind()
        if xmind is None or adapter.xmind_session_handle is None:
            return {"error": "xMind not available or session not bound"}

        try:
            payload = xmind.SessionManager.getSessionState(adapter.xmind_session_handle)
            adapter.from_xmind_payload(payload)
            return {
                "status": "synced",
                "session_id": session_id,
                "context": adapter.fleet_context,
            }
        except Exception as exc:
            logger.warning("xMind pull failed: %s", exc)
            return {"error": str(exc)}

    # ── distributed execution ─────────────────────────────────────

    def execute_remote(self, blueprint_name: str, session_id: str) -> dict[str, Any]:
        """Execute a blueprint on a remote xlang node via LRPC."""
        with self._lock:
            blueprint = self._graphs.get(blueprint_name)
            if blueprint is None:
                return {"error": f"Blueprint '{blueprint_name}' not found"}

        xlang = self._load_xlang()
        if xlang is None:
            return {"error": "xlang runtime not available"}

        if not self.lrpc_endpoint:
            return {"error": "No LRPC endpoint configured"}

        try:
            # LRPC remote import pattern
            remote = xlang.importModule(
                f"xmind_srv thru '{self.lrpc_endpoint}' as xmind_srv"
            )
            result = remote.executeGraph(blueprint.to_yaml(), session_id)
            self._graphs_executed += 1
            self._remote_calls += 1
            return {"status": "executed", "result": result}
        except Exception as exc:
            logger.error("Remote execution failed: %s", exc)
            return {"error": str(exc)}

    # ── local execution (fallback) ────────────────────────────────

    def execute_local(
        self, blueprint_name: str, session_id: str, inputs: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute a blueprint locally via Python fallback (no xlang required)."""
        with self._lock:
            blueprint = self._graphs.get(blueprint_name)
            if blueprint is None:
                return {"error": f"Blueprint '{blueprint_name}' not found"}

        # Simple topological execution of the graph
        results: dict[str, Any] = dict(inputs or {})
        node_map = {n["id"]: n for n in blueprint.nodes}

        # Build adjacency list
        adjacency: dict[str, list[str]] = {n["id"]: [] for n in blueprint.nodes}
        for e in blueprint.edges:
            adjacency.setdefault(e["source"], []).append(e["target"])

        # Topological sort (Kahn's algorithm)
        in_degree = {n["id"]: 0 for n in blueprint.nodes}
        for e in blueprint.edges:
            in_degree[e["target"]] = in_degree.get(e["target"], 0) + 1

        queue = [n for n, d in in_degree.items() if d == 0]
        executed: list[str] = []

        while queue:
            node_id = queue.pop(0)
            node = node_map[node_id]
            node_type = node.get("type", "function")

            # Execute node
            if node_type == "agent":
                # Simulate LLM agent execution
                prompt = node.get("prompt_template", "")
                results[node_id] = {
                    "status": "simulated",
                    "prompt": prompt,
                    "output": f"Agent {node_id} response",
                }
            elif node_type == "action":
                # Simulate action execution
                results[node_id] = {
                    "status": "simulated",
                    "action": node.get("action_type", "noop"),
                }
            else:
                # Function pass-through
                input_val = results.get(node.get("input_pins", ["input"])[0], None)
                results[node_id] = {"output": input_val}

            executed.append(node_id)
            for target in adjacency.get(node_id, []):
                in_degree[target] -= 1
                if in_degree[target] == 0:
                    queue.append(target)

        self._graphs_executed += 1
        return {
            "status": "executed",
            "executed_nodes": executed,
            "results": results,
            "blueprint": blueprint_name,
            "session_id": session_id,
        }

    # ── stats ─────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "node_id": self.node_id,
                "graphs_loaded": len(self._graphs),
                "sessions_active": len(self._sessions),
                "graphs_executed": self._graphs_executed,
                "sessions_synced": self._sessions_synced,
                "remote_calls": self._remote_calls,
                "xlang_available": self._xlang is not None,
                "xmind_available": self._xmind is not None,
            }
