"""Tests for logos.a2a_protocol.

Covers: A2AAgentCard, A2ATask, A2AServer, A2AClient, A2AProtocolAdapter.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict

import pytest

from a2a.protocol import (
    A2AAgentCard,
    A2AClient,
    A2AError,
    A2AProtocolAdapter,
    A2AServer,
    A2ATask,
    TaskStatus,
)


# ═══════════════════════════════════════════════════════════
# A2AAgentCard
# ═══════════════════════════════════════════════════════════


class TestA2AAgentCard:
    def test_to_dict_roundtrip(self) -> None:
        card = A2AAgentCard(
            name="TestAgent",
            description="A test agent",
            version="1.0.0",
            url="http://example.com",
            capabilities={"streaming": True},
            skills=[{"id": "test", "name": "Testing"}],
            authentication={"schemes": ["bearer"]},
        )
        d = card.to_dict()
        assert d["name"] == "TestAgent"
        assert d["capabilities"]["streaming"] is True
        assert d["skills"][0]["id"] == "test"

    def test_to_json_canonical(self) -> None:
        card = A2AAgentCard(name="A", description="B", version="1", url="")
        j = card.to_json()
        # Sorted keys: authentication comes before name
        assert '"name":"A"' in j
        # No whitespace
        assert " " not in j
        assert j.index("authentication") < j.index("name")

    def test_from_dict(self) -> None:
        data = {
            "name": "FleetBot",
            "description": "Fleet bot",
            "version": "2.0.0",
            "url": "http://fleet.local",
            "capabilities": {},
            "skills": [],
            "authentication": {},
            "defaultInputContentType": "text/plain",
        }
        card = A2AAgentCard.from_dict(data)
        assert card.name == "FleetBot"
        assert card.default_input_content_type == "text/plain"

    def test_from_agent_identity(self) -> None:
        class FakeIdentity:
            agent_id = "bot_1"
            description = "test bot"
            card = type(
                "Card",
                (object,),
                {
                    "name": "BotCard",
                    "version": "1.2.3",
                    "description": "desc",
                    "url": "http://bot",
                    "capabilities": {"streaming": True},
                    "skills": [],
                    "authentication": {},
                },
            )()

        card = A2AAgentCard.from_agent_identity(FakeIdentity())
        assert card.name == "BotCard"
        assert card.version == "1.2.3"

    def test_from_agent_identity_fallback(self) -> None:
        class MinimalIdentity:
            agent_id = "m_1"

        card = A2AAgentCard.from_agent_identity(MinimalIdentity())
        assert card.name == "m_1"
        assert card.capabilities["streaming"] is True


# ═══════════════════════════════════════════════════════════
# A2ATask
# ═══════════════════════════════════════════════════════════


class TestA2ATask:
    def test_default_status_submitted(self) -> None:
        task = A2ATask(id="t1", session_id="s1")
        assert task.status == TaskStatus.SUBMITTED

    def test_set_status_updates_history(self) -> None:
        task = A2ATask(id="t1", session_id="s1")
        task.set_status(TaskStatus.WORKING)
        assert task.status == TaskStatus.WORKING
        assert len(task.history) == 1
        assert task.history[0]["status"] == "working"

    def test_add_artifact(self) -> None:
        task = A2ATask(id="t1", session_id="s1")
        task.add_artifact({"type": "file", "name": "report.txt"})
        assert len(task.artifact) == 1
        assert task.artifact[0]["name"] == "report.txt"

    def test_to_dict_roundtrip(self) -> None:
        task = A2ATask(id="t1", session_id="s1")
        task.set_status(TaskStatus.COMPLETED)
        task.add_artifact({"data": "ok"})
        d = task.to_dict()
        restored = A2ATask.from_dict(d)
        assert restored.id == "t1"
        assert restored.status == TaskStatus.COMPLETED
        assert restored.artifact[0]["data"] == "ok"


# ═══════════════════════════════════════════════════════════
# A2AServer
# ═══════════════════════════════════════════════════════════


class TestA2AServer:
    def test_handle_agent_card(self) -> None:
        card = A2AAgentCard(name="S", description="D", version="1", url="")
        server = A2AServer(agent_card=card)
        req = {"jsonrpc": "2.0", "id": 1, "method": "agent/cards", "params": {}}
        resp = server.handle_request(req)
        assert "error" not in resp
        assert resp["result"]["name"] == "S"

    def test_handle_task_send(self) -> None:
        card = A2AAgentCard(name="S", description="D", version="1", url="")
        server = A2AServer(agent_card=card)
        req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tasks/send",
            "params": {"metadata": {"repo": "sunset"}},
        }
        resp = server.handle_request(req)
        assert "error" not in resp
        assert resp["result"]["status"] == "working"
        assert resp["result"]["metadata"]["repo"] == "sunset"

    def test_handle_task_status(self) -> None:
        card = A2AAgentCard(name="S", description="D", version="1", url="")
        server = A2AServer(agent_card=card)
        server.handle_request({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tasks/send",
            "params": {"id": "task_1"},
        })
        resp = server.handle_request({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tasks/status",
            "params": {"id": "task_1"},
        })
        assert resp["result"]["id"] == "task_1"

    def test_handle_task_cancel(self) -> None:
        card = A2AAgentCard(name="S", description="D", version="1", url="")
        server = A2AServer(agent_card=card)
        server.handle_request({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tasks/send",
            "params": {"id": "task_x"},
        })
        resp = server.handle_request({
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tasks/cancel",
            "params": {"id": "task_x"},
        })
        assert resp["result"]["status"] == "cancelled"

    def test_handle_unknown_method(self) -> None:
        card = A2AAgentCard(name="S", description="D", version="1", url="")
        server = A2AServer(agent_card=card)
        resp = server.handle_request({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "magic/trick",
            "params": {},
        })
        assert "error" in resp
        assert resp["error"]["code"] == -32601

    def test_task_handler_processing(self) -> None:
        card = A2AAgentCard(name="S", description="D", version="1", url="")

        def handler(task: A2ATask) -> A2ATask:
            task.set_status(TaskStatus.COMPLETED)
            task.add_artifact({"result": "done"})
            return task

        server = A2AServer(agent_card=card, task_handler=handler)
        resp = server.handle_request({
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tasks/send",
            "params": {"id": "th1"},
        })
        # Handler runs synchronously in this implementation
        assert resp["result"]["status"] == "completed"
        assert resp["result"]["artifact"][0]["result"] == "done"

    def test_task_handler_failure(self) -> None:
        card = A2AAgentCard(name="S", description="D", version="1", url="")

        def bad_handler(task: A2ATask) -> A2ATask:
            raise RuntimeError("boom")

        server = A2AServer(agent_card=card, task_handler=bad_handler)
        resp = server.handle_request({
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tasks/send",
            "params": {"id": "th2"},
        })
        assert resp["result"]["status"] == "failed"
        assert "boom" in resp["result"]["metadata"]["error"]

    def test_sse_listener_notification(self) -> None:
        card = A2AAgentCard(name="S", description="D", version="1", url="")
        server = A2AServer(agent_card=card)
        events: list = []
        server.register_sse_listener(lambda et, d: events.append((et, d)))
        server.handle_request({
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tasks/send",
            "params": {"id": "sse_t"},
        })
        assert any(et == "task_update" for et, _ in events)

    def test_get_and_list_tasks(self) -> None:
        card = A2AAgentCard(name="S", description="D", version="1", url="")
        server = A2AServer(agent_card=card)
        server.handle_request({
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tasks/send",
            "params": {"id": "tlist"},
        })
        assert server.get_task("tlist") is not None
        assert "tlist" in server.list_tasks()


# ═══════════════════════════════════════════════════════════
# A2AClient
# ═══════════════════════════════════════════════════════════


class TestA2AClient:
    def test_discover_agent(self) -> None:
        async def mock_transport(url: str, request: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {
                    "name": "RemoteAgent",
                    "description": "Remote",
                    "version": "1.0.0",
                    "url": "http://remote",
                    "capabilities": {},
                    "skills": [],
                    "authentication": {},
                },
            }

        client = A2AClient(transport=mock_transport)
        card = asyncio.run(client.discover_agent("http://remote"))
        assert card.name == "RemoteAgent"

    def test_send_task(self) -> None:
        async def mock_transport(url: str, request: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {
                    "id": "task_123",
                    "sessionId": "sess_456",
                    "status": "working",
                    "artifact": [],
                    "history": [],
                    "metadata": {},
                    "createdAt": time.time(),
                    "updatedAt": time.time(),
                },
            }

        client = A2AClient(transport=mock_transport)
        task = asyncio.run(client.send_task("http://remote", "test", {"key": "val"}))
        assert task.id == "task_123"
        assert task.status == TaskStatus.WORKING

    def test_get_task_status(self) -> None:
        async def mock_transport(url: str, request: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {
                    "id": "task_123",
                    "sessionId": "sess_456",
                    "status": "completed",
                    "artifact": [],
                    "history": [],
                    "metadata": {},
                    "createdAt": time.time(),
                    "updatedAt": time.time(),
                },
            }

        client = A2AClient(transport=mock_transport)
        task = asyncio.run(client.get_task_status("http://remote", "task_123"))
        assert task.status == TaskStatus.COMPLETED

    def test_rpc_call_no_transport(self) -> None:
        client = A2AClient()
        with pytest.raises(A2AError) as exc_info:
            asyncio.run(client._rpc_call("http://x", "m", {}))
        assert exc_info.value.code == -32603


# ═══════════════════════════════════════════════════════════
# A2AProtocolAdapter
# ═══════════════════════════════════════════════════════════


class TestA2AProtocolAdapter:
    def test_register_agent(self) -> None:
        adapter = A2AProtocolAdapter()
        class FakeIdentity:
            agent_id = "agent_1"
            card = type(
                "Card",
                (object,),
                {
                    "name": "A1",
                    "version": "1.0",
                    "description": "D",
                    "url": "",
                    "capabilities": {},
                    "skills": [{"id": "s1"}],
                    "authentication": {},
                },
            )()
        card = adapter.register_agent(FakeIdentity())
        assert card.name == "A1"
        assert "agent_1" in adapter.list_agents()

    def test_get_server_routes(self) -> None:
        adapter = A2AProtocolAdapter()
        class FakeIdentity:
            agent_id = "agent_1"
            card = type(
                "Card",
                (object,),
                {
                    "name": "A1",
                    "version": "1",
                    "description": "D",
                    "url": "",
                    "capabilities": {},
                    "skills": [],
                    "authentication": {},
                },
            )()
        adapter.register_agent(FakeIdentity())
        routes = adapter.get_server_routes()
        assert "/tasks/send" in routes
        assert "/tasks/status" in routes
        assert "/tasks/cancel" in routes
        assert "/agent/cards" in routes

    def test_server_routes_dispatch(self) -> None:
        adapter = A2AProtocolAdapter()
        class FakeIdentity:
            agent_id = "agent_1"
            card = type(
                "Card",
                (object,),
                {
                    "name": "A1",
                    "version": "1",
                    "description": "D",
                    "url": "",
                    "capabilities": {},
                    "skills": [],
                    "authentication": {},
                },
            )()
        adapter.register_agent(FakeIdentity())
        routes = adapter.get_server_routes()
        resp = routes["/tasks/send"]({"agent_id": "agent_1", "id": 1, "params": {"id": "t1"}})
        assert resp["result"]["status"] == "working"

    def test_server_routes_unknown_agent(self) -> None:
        adapter = A2AProtocolAdapter()
        routes = adapter.get_server_routes()
        resp = routes["/tasks/send"]({"agent_id": "ghost", "id": 1, "params": {}})
        assert "error" in resp
        assert resp["error"]["code"] == -32002

    def test_attach_to_fleet_conductor(self) -> None:
        adapter = A2AProtocolAdapter()
        class FakeConductor:
            def orchestrate(self, repo_path: str, tasks: list) -> Dict[str, Any]:
                return {"status": "ok"}
        adapter.attach_to_fleet_conductor(FakeConductor())
        assert "fleet_conductor" in adapter.list_agents()
        card = adapter.get_agent_card("fleet_conductor")
        assert card is not None
        assert card.name == "FleetConductor"
        assert len(card.skills) >= 2

    def test_attach_to_sse_dashboard(self, caplog: Any) -> None:
        import logging
        with caplog.at_level(logging.INFO):
            adapter = A2AProtocolAdapter()
            class FakeIdentity:
                agent_id = "a1"
                card = type("Card", (object,), {
                    "name": "A", "version": "1", "description": "D", "url": "",
                    "capabilities": {}, "skills": [], "authentication": {},
                })()
            adapter.register_agent(FakeIdentity())
            class FakeSSE:
                def publish(self, event_type: str, data: Any) -> None:
                    pass
            adapter.attach_to_sse_dashboard(FakeSSE())
            # Should log success
            assert "A2A events wired" in caplog.text

    def test_attach_to_sse_dashboard_no_publish(self, caplog: Any) -> None:
        adapter = A2AProtocolAdapter()
        class FakeSSE:
            pass
        adapter.attach_to_sse_dashboard(FakeSSE())
        assert "no publish() method" in caplog.text

    def test_stats(self) -> None:
        adapter = A2AProtocolAdapter(node_id="test_node")
        assert adapter.stats()["node_id"] == "test_node"
        assert adapter.stats()["registered_agents"] == 0

    def test_discover_and_send(self) -> None:
        async def mock_transport(url: str, request: Dict[str, Any]) -> Dict[str, Any]:
            if request["method"] == "agent/cards":
                return {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": {
                        "name": "Remote",
                        "description": "R",
                        "version": "1",
                        "url": "",
                        "capabilities": {},
                        "skills": [],
                        "authentication": {},
                    },
                }
            return {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {
                    "id": "t1",
                    "sessionId": "s1",
                    "status": "completed",
                    "artifact": [],
                    "history": [],
                    "metadata": {},
                    "createdAt": time.time(),
                    "updatedAt": time.time(),
                },
            }

        adapter = A2AProtocolAdapter()
        adapter._client = A2AClient(transport=mock_transport)
        task = asyncio.run(adapter.discover_and_send("http://remote", "test", {}))
        assert task.status == TaskStatus.COMPLETED
