"""Tests for logos.a2a_identity — AgentCard, AgentRegistry, AgentIdentity."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from logos.a2a_identity import (
    AgentCard,
    AgentIdentity,
    AgentRegistry,
    NegotiationError,
    TaskHandle,
    TaskState,
    ValidationError,
)

# ── fixtures ────────────────────────────────────────────────

@pytest.fixture
def valid_card_dict() -> Dict[str, Any]:
    return {
        "name": "test-agent",
        "version": "1.0.0",
        "description": "A test agent for unit tests",
        "url": "http://test.local:4047/agent",
        "capabilities": {"streaming": True, "pushNotifications": False},
        "skills": [
            {
                "id": "echo",
                "name": "Echo",
                "description": "Echo back input",
                "tags": ["test"],
                "examples": ["Hello"],
            }
        ],
        "authentication": {"schemes": ["mtls"]},
    }


@pytest.fixture
def valid_card_json(valid_card_dict: Dict[str, Any]) -> str:
    return json.dumps(valid_card_dict)


@pytest.fixture
def tmp_key_dir():
    with tempfile.TemporaryDirectory() as td:
        yield td


# ── 1. AgentCard loads valid JSON ──────────────────────────

def test_agent_card_loads_valid_json(valid_card_json: str) -> None:
    card = AgentCard.from_json(valid_card_json)
    assert card.name == "test-agent"
    assert card.version == "1.0.0"
    assert card.supports_skill("echo")
    assert not card.supports_skill("nonexistent")
    assert card.capability("streaming") is True
    assert card.capability("pushNotifications") is False


# ── 2. AgentCard rejects missing required fields ────────────

def test_agent_card_rejects_missing_required() -> None:
    incomplete = {"name": "x", "version": "1.0.0"}
    with pytest.raises(ValidationError) as exc_info:
        AgentCard.from_dict(incomplete)
    assert "description" in str(exc_info.value)
    assert "capabilities" in str(exc_info.value)
    assert "skills" in str(exc_info.value)


def test_agent_card_rejects_missing_capability_bools() -> None:
    bad = {
        "name": "x",
        "version": "1.0.0",
        "description": "d",
        "capabilities": {"streaming": True},
        "skills": [],
    }
    with pytest.raises(ValidationError) as exc_info:
        AgentCard.from_dict(bad)
    assert "pushNotifications" in str(exc_info.value)


def test_agent_card_rejects_invalid_skill() -> None:
    bad = {
        "name": "x",
        "version": "1.0.0",
        "description": "d",
        "capabilities": {"streaming": True, "pushNotifications": False},
        "skills": [{"id": "bad"}],
    }
    with pytest.raises(ValidationError) as exc_info:
        AgentCard.from_dict(bad)
    assert "tags" in str(exc_info.value)


# ── 3. Registry discovers from nexus ──────────────────────

@pytest.mark.asyncio
async def test_registry_discovers_from_nexus(valid_card_dict: Dict[str, Any]) -> None:
    registry = AgentRegistry()
    fake_card = AgentCard.from_dict(valid_card_dict)

    async def fake_http(url: str):
        return [("test-agent", fake_card)]

    registry._http_get_agent_cards = fake_http  # type: ignore[method-assign]
    cards = await registry.discover("http://nexus.fleet.local:4047")
    assert len(cards) == 1
    assert cards[0].name == "test-agent"
    assert registry.get("test-agent") is not None


# ── 4. Registry negotiate_task returns handle ───────────────

def test_registry_negotiate_task_returns_handle(valid_card_dict: Dict[str, Any]) -> None:
    registry = AgentRegistry()
    card = AgentCard.from_dict(valid_card_dict)
    registry.register("echo-agent", card)

    handle = registry.negotiate_task(
        agent_id="echo-agent",
        task_type="echo",
        payload={"message": "hello"},
    )
    assert isinstance(handle, TaskHandle)
    assert handle.agent_id == "echo-agent"
    assert handle.task_type == "echo"
    assert handle.state == TaskState.PENDING
    assert handle.payload == {"message": "hello"}


def test_registry_negotiate_task_unknown_agent() -> None:
    registry = AgentRegistry()
    with pytest.raises(NegotiationError) as exc_info:
        registry.negotiate_task("ghost", "echo", {})
    assert "Unknown agent" in str(exc_info.value)


def test_registry_negotiate_task_unsupported_skill(valid_card_dict: Dict[str, Any]) -> None:
    registry = AgentRegistry()
    card = AgentCard.from_dict(valid_card_dict)
    registry.register("echo-agent", card)
    with pytest.raises(NegotiationError) as exc_info:
        registry.negotiate_task("echo-agent", "compile", {})
    assert "does not support skill" in str(exc_info.value)


# ── 5. Stream handler receives chunks ───────────────────────

@pytest.mark.asyncio
async def test_stream_handler_receives_chunks(valid_card_dict: Dict[str, Any]) -> None:
    registry = AgentRegistry()
    card = AgentCard.from_dict(valid_card_dict)
    registry.register("stream-agent", card)

    received: list[Any] = []

    def handler(agent_id: str, chunk: Dict[str, Any]) -> None:
        received.append((agent_id, chunk))

    registry.register_stream_handler("stream-agent", handler)

    handle = registry.negotiate_task("stream-agent", "echo", {"x": 1})
    handle.set_state(TaskState.SUBMITTED)

    chunk = {"delta": "hello"}
    registry.dispatch_chunk("stream-agent", chunk)

    assert len(received) == 1
    assert received[0] == ("stream-agent", chunk)
    assert handle.chunks == [chunk]
    assert handle.state == TaskState.STREAMING

    registry.unregister_stream_handler("stream-agent")
    assert not registry.unregister_stream_handler("stream-agent")


# ── 6. Identity signs tasks ─────────────────────────────────

def test_identity_signs_task(tmp_key_dir: str) -> None:
    identity = AgentIdentity("test-agent", base_dir=tmp_key_dir)
    payload = {"task": "echo", "data": [1, 2, 3]}
    sig = identity.sign_task(payload)
    assert isinstance(sig, str)
    assert len(sig) > 0
    # Signature should be deterministic for same payload
    sig2 = identity.sign_task(payload)
    assert sig == sig2
    # Different payload → different signature
    sig3 = identity.sign_task({"task": "echo", "data": [1, 2, 4]})
    assert sig3 != sig


# ── 7. Identity verifies remote task ─────────────────────────

def test_identity_verifies_remote_task(tmp_key_dir: str) -> None:
    identity = AgentIdentity("test-agent", base_dir=tmp_key_dir)
    payload = {"task": "echo", "data": [1, 2, 3]}
    sig = identity.sign_task(payload)
    assert identity.verify_task(payload, sig) is True

    # Tampered payload should fail
    bad_payload = {"task": "echo", "data": [1, 2, 4]}
    assert identity.verify_task(bad_payload, sig) is False

    # Bad signature format should fail
    assert identity.verify_task(payload, "not-a-valid-sig!!!") is False


# ── 8. Registry fallback on nexus down ────────────────────

@pytest.mark.asyncio
async def test_registry_fallback_on_nexus_down(valid_card_dict: Dict[str, Any]) -> None:
    registry = AgentRegistry()
    card = AgentCard.from_dict(valid_card_dict)
    registry.register("local-agent", card)
    # Seed fallback cache explicitly
    registry._fallback_local_cards["local-agent"] = card

    async def failing_http(url: str):
        raise NegotiationError("Connection refused")

    registry._http_get_agent_cards = failing_http  # type: ignore[method-assign]
    cards = await registry.discover("http://nexus.fleet.local:4047")
    assert len(cards) == 1
    assert cards[0].name == "test-agent"


@pytest.mark.asyncio
async def test_registry_fallback_raises_when_no_cache() -> None:
    registry = AgentRegistry()

    async def failing_http(url: str):
        raise NegotiationError("Connection refused")

    registry._http_get_agent_cards = failing_http  # type: ignore[method-assign]
    with pytest.raises(NegotiationError):
        await registry.discover("http://nexus.fleet.local:4047")


# ── 9. AgentCard round-trip serialization ───────────────────

def test_agent_card_round_trip(valid_card_dict: Dict[str, Any]) -> None:
    card = AgentCard.from_dict(valid_card_dict)
    dumped = card.to_dict()
    # Ensure the serialized dict can be parsed back
    card2 = AgentCard.from_dict(dumped)
    assert card2.name == card.name
    assert card2.version == card.version
    assert card2.capabilities == card.capabilities


def test_agent_card_from_file(valid_card_dict: Dict[str, Any], tmp_path: Path) -> None:
    path = tmp_path / "agent.json"
    path.write_text(json.dumps(valid_card_dict))
    card = AgentCard.from_file(path)
    assert card.name == "test-agent"


# ── 10. AgentIdentity key persistence ───────────────────────

def test_identity_key_persistence(tmp_key_dir: str) -> None:
    identity = AgentIdentity("persist-agent", base_dir=tmp_key_dir)
    pem_path = Path(tmp_key_dir) / "persist-agent.pem"
    pub_path = Path(tmp_key_dir) / "persist-agent.pub"
    assert pem_path.exists()
    assert pub_path.exists()
    assert os.stat(pem_path).st_mode & 0o777 == 0o600

    # Re-loading should read the same keys
    identity2 = AgentIdentity("persist-agent", base_dir=tmp_key_dir)
    payload = {"action": "test"}
    sig = identity.sign_task(payload)
    assert identity2.verify_task(payload, sig) is True


# ── 11. TaskHandle chunk delivery ──────────────────────────

@pytest.mark.asyncio
async def test_task_handle_chunk_delivery() -> None:
    handle = TaskHandle(
        task_id="t-1",
        agent_id="a-1",
        task_type="echo",
        payload={},
    )
    handle.deliver_chunk({"part": 1})
    handle.deliver_chunk({"part": 2})
    assert handle.chunks == [{"part": 1}, {"part": 2}]

    chunk = await handle.next_chunk(timeout=0.1)
    assert chunk == {"part": 2}


@pytest.mark.asyncio
async def test_task_handle_next_chunk_timeout() -> None:
    handle = TaskHandle(
        task_id="t-2",
        agent_id="a-2",
        task_type="echo",
        payload={},
    )
    chunk = await handle.next_chunk(timeout=0.01)
    assert chunk is None


# ── 12. Registry load_local_cards ───────────────────────────

def test_registry_load_local_cards(tmp_path: Path) -> None:
    d = tmp_path / "cards"
    d.mkdir()
    card_a = {
        "name": "agent-a",
        "version": "1.0.0",
        "description": "A",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "skills": [],
    }
    card_b = {
        "name": "agent-b",
        "version": "1.0.0",
        "description": "B",
        "capabilities": {"streaming": True, "pushNotifications": False},
        "skills": [],
    }
    (d / "agent_a.json").write_text(json.dumps(card_a))
    (d / "agent_b.json").write_text(json.dumps(card_b))
    (d / "bad.txt").write_text("not json")  # should be ignored

    registry = AgentRegistry()
    count = registry.load_local_cards(d)
    assert count == 2
    assert sorted(registry.list_agents()) == ["agent_a", "agent_b"]
