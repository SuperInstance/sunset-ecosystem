#!/usr/bin/env python3
"""Tests for fleet/i2i_bridge.py."""
import json
import os
import tempfile
from pathlib import Path

import pytest

from fleet.i2i_bridge import (
    AgentIdentity,
    Bottle,
    I2IBridge,
    I2IMessage,
    IndividualLayer,
    InstanceLayer,
    InteractionLayer,
    IronLayer,
    IterationLayer,
)


class TestAgentIdentity:
    def test_creation(self):
        a = AgentIdentity("Oracle1", "lighthouse", "ARM64", ("coordination",))
        assert a.name == "Oracle1"
        assert a.role == "lighthouse"
        assert a.capabilities == ("coordination",)

    def test_frozen(self):
        a = AgentIdentity("FM", "forge", "RTX4050")
        with pytest.raises(AttributeError):
            a.name = "X"


class TestI2IMessage:
    def test_to_json(self):
        sender = AgentIdentity("test", "tester", "generic")
        msg = I2IMessage("instance", sender, None, {"x": 1})
        data = json.loads(msg.to_json())
        assert data["layer"] == "instance"
        assert data["sender"]["name"] == "test"
        assert data["payload"]["x"] == 1


class TestInstanceLayer:
    def test_call(self):
        layer = InstanceLayer()
        sender = AgentIdentity("A", "test", "cpu")
        result = layer.call(sender, "/health", {})
        assert result["status"] == "simulated"
        assert len(layer.history()) == 1


class TestIterationLayer:
    def test_submit_tile(self):
        layer = IterationLayer()
        sender = AgentIdentity("CCC", "designer", "kimi")
        result = layer.submit_tile(sender, "ethos", "Q", "A")
        assert result["status"] == "buffered"
        assert len(layer.history()) == 1


class TestIndividualLayer:
    def test_send_bottle(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Init git repo
            os.system(f"cd {tmp} && git init -q && git config user.email 'test@test' && git config user.name 'Test'")
            layer = IndividualLayer(repo_path=tmp)
            bottle = Bottle(from_agent="CCC", to_agent="FM", subject="Test Bottle", body="Hello", repo_path=tmp)
            commit = layer.send_bottle(bottle)
            assert len(commit) == 40  # git SHA
            assert len(layer.history()) == 1

    def test_read_bottles(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.system(f"cd {tmp} && git init -q && git config user.email 'test@test' && git config user.name 'Test'")
            layer = IndividualLayer(repo_path=tmp)
            bottle = Bottle(from_agent="Oracle1", subject="Read Me", body="Content", repo_path=tmp)
            layer.send_bottle(bottle)
            bottles = layer.read_bottles("Oracle1")
            assert len(bottles) >= 1
            assert "Oracle1" in bottles[0]["from"]


class TestInteractionLayer:
    def test_send(self):
        layer = InteractionLayer()
        sender = AgentIdentity("A", "agent", "cpu")
        result = layer.send(sender, "#fleet-ops", "hello")
        assert result["status"] == "sent"
        assert len(layer.history()) == 1


class TestIronLayer:
    def test_register(self):
        layer = IronLayer()
        agent = AgentIdentity("JC1", "edge", "Jetson", ("tensorrt", "gpu"))
        layer.register_hardware(agent, {"gpu": "Orin", "ram": "8GB"})
        assert "JC1" in layer.get_topology()
        assert len(layer.history()) == 1

    def test_find_by_capability(self):
        layer = IronLayer()
        layer.register_hardware(AgentIdentity("JC1", "edge", "Jetson", ("tensorrt",)), {"gpu": "Orin"})
        layer.register_hardware(AgentIdentity("FM", "forge", "RTX", ()), {"gpu": "RTX4050"})
        found = layer.find_by_capability("tensorrt")
        assert len(found) == 1
        assert found[0].name == "JC1"


class TestI2IBridge:
    def test_init(self):
        bridge = I2IBridge()
        assert bridge.identity.name == "unknown"

    def test_summary(self):
        bridge = I2IBridge()
        bridge.instance.call(AgentIdentity("A", "t", "c"), "/x", {})
        s = bridge.summary()
        assert s["instance"] == 1
        assert s["iteration"] == 0

    def test_full_history(self):
        bridge = I2IBridge()
        bridge.instance.call(AgentIdentity("A", "t", "c"), "/x", {})
        bridge.iteration.submit_tile(AgentIdentity("B", "t", "c"), "d", "q", "a")
        assert len(bridge.full_history()) == 2

    def test_repr(self):
        bridge = I2IBridge()
        assert "I2IBridge" in repr(bridge)
