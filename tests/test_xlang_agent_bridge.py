"""Tests for XlangAgentBridge.

Covers:
- AgentFlowBlueprint conversion from JSON graph
- YAML serialization / deserialization
- Session creation and sync
- Local execution (topological sort, node execution)
- Blueprint roundtrip
- Stats tracking
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from fleet.xlang_agent_bridge import (
    XlangAgentBridge,
    AgentFlowBlueprint,
    SessionSyncAdapter,
)


class TestAgentFlowBlueprint:
    def test_from_json_graph(self) -> None:
        graph = {
            "nodes": [
                {"id": "n1", "type": "agent", "config": {"model": "gpt-4"}, "prompt": "You are a helpful assistant"},
                {"id": "n2", "type": "action", "action": "rest_api", "endpoint": "https://api.example.com"},
                {"id": "n3", "type": "function", "function": "summarize"},
            ],
            "edges": [
                {"source": "n1", "target": "n2", "relation": "delegates"},
                {"source": "n2", "target": "n3", "source_pin": "output", "target_pin": "input"},
            ],
        }
        bp = AgentFlowBlueprint.from_json_graph(graph, name="test")
        assert bp.name == "test"
        assert len(bp.nodes) == 3
        assert bp.nodes[0]["type"] == "agent"
        assert bp.nodes[1]["type"] == "action"
        assert len(bp.edges) == 2

    def test_to_yaml(self) -> None:
        bp = AgentFlowBlueprint(
            name="test",
            nodes=[{"id": "n1", "type": "agent"}],
            edges=[{"source": "n1", "target": "n2"}],
        )
        yaml_str = bp.to_yaml()
        assert "blueprint" in yaml_str
        assert "test" in yaml_str
        assert "n1" in yaml_str


class TestSessionSyncAdapter:
    def test_to_xmind_payload(self) -> None:
        adapter = SessionSyncAdapter(
            session_id="sess_001",
            fleet_context={
                "history": [{"role": "user", "content": "hello"}],
                "variables": {"topic": "test"},
                "node_id": "Oracle1",
                "agent_id": "agent_42",
            },
        )
        payload = adapter.to_xmind_payload()
        assert payload["session_id"] == "sess_001"
        assert payload["metadata"]["node_id"] == "Oracle1"
        assert len(payload["history"]) == 1

    def test_from_xmind_payload(self) -> None:
        adapter = SessionSyncAdapter(session_id="sess_001")
        payload = {
            "history": [{"role": "assistant", "content": "hi"}],
            "variables": {"topic": "updated"},
        }
        adapter.from_xmind_payload(payload)
        assert adapter.fleet_context["history"][0]["content"] == "hi"
        assert adapter.fleet_context["variables"]["topic"] == "updated"


class TestXlangAgentBridge:
    def test_convert_graph(self) -> None:
        bridge = XlangAgentBridge(node_id="test_node")
        graph = {
            "nodes": [
                {"id": "n1", "type": "agent", "prompt": "Analyze"},
                {"id": "n2", "type": "function", "function": "report"},
            ],
            "edges": [{"source": "n1", "target": "n2"}],
        }
        bp = bridge.convert_graph(graph, name="analysis")
        assert bp.name == "analysis"
        assert "analysis" in bridge._graphs

    def test_save_and_load_blueprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bridge = XlangAgentBridge(node_id="test_node")
            graph = {
                "nodes": [{"id": "n1", "type": "agent"}],
                "edges": [],
            }
            bridge.convert_graph(graph, name="test_bp")

            path = Path(tmp) / "test_bp.yaml"
            bridge.save_blueprint("test_bp", path)
            assert path.exists()

            loaded = bridge.load_blueprint(path)
            assert loaded.name == "test_bp"
            assert len(loaded.nodes) == 1

    def test_create_session(self) -> None:
        bridge = XlangAgentBridge(node_id="test_node")
        adapter = bridge.create_session("sess_001", {"topic": "test"})
        assert adapter.session_id == "sess_001"
        assert bridge._sessions["sess_001"] == adapter

    def test_execute_local(self) -> None:
        bridge = XlangAgentBridge(node_id="test_node")
        graph = {
            "nodes": [
                {"id": "input", "type": "function", "function": "pass_through"},
                {"id": "process", "type": "function", "function": "uppercase"},
            ],
            "edges": [
                {"source": "input", "target": "process", "source_pin": "output", "target_pin": "input"},
            ],
        }
        bridge.convert_graph(graph, name="pipeline")
        result = bridge.execute_local("pipeline", "sess_001", {"input": "hello"})
        assert result["status"] == "executed"
        assert "input" in result["executed_nodes"]
        assert "process" in result["executed_nodes"]

    def test_execute_local_missing_blueprint(self) -> None:
        bridge = XlangAgentBridge(node_id="test_node")
        result = bridge.execute_local("missing", "sess_001")
        assert "error" in result
        assert "not found" in result["error"]

    def test_execute_local_empty_graph(self) -> None:
        bridge = XlangAgentBridge(node_id="test_node")
        graph = {"nodes": [], "edges": []}
        bridge.convert_graph(graph, name="empty")
        result = bridge.execute_local("empty", "sess_001")
        assert result["status"] == "executed"
        assert len(result["executed_nodes"]) == 0

    def test_stats(self) -> None:
        bridge = XlangAgentBridge(node_id="test_node")
        graph = {
            "nodes": [{"id": "n1", "type": "function"}],
            "edges": [],
        }
        bridge.convert_graph(graph, name="test")
        bridge.execute_local("test", "sess_001")

        stats = bridge.stats
        assert stats["node_id"] == "test_node"
        assert stats["graphs_loaded"] == 1
        assert stats["graphs_executed"] == 1
        assert stats["xlang_available"] is False  # not loaded in test
        assert stats["xmind_available"] is False
