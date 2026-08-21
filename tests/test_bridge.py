"""Tests for fleet.bridge — Bridge interface and registry."""

import pytest
from fleet.bridge import (
    Bridge,
    BridgeRegistry,
    BridgeStatus,
    BridgeEvent,
    BridgeCompiler,
)


class _FakeBridge(Bridge):
    """Minimal bridge implementation for testing."""

    def __init__(self, node_id: str = "test"):
        super().__init__(node_id=node_id)
        self._connected = False
        self._pushed = []
        self._pulled = []

    def connect(self, host, port):
        self._connected = True
        self._host = host
        self._port = port

    def push(self, data):
        self._pushed.append(data)

    def pull(self, query):
        self._pulled.append(query)
        return {"query": query}

    def disconnect(self):
        self._connected = False

    def health(self):
        return BridgeStatus.CONNECTED if self._connected else BridgeStatus.DISCONNECTED


class TestBridge:
    def test_abstract_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Bridge()

    def test_bridge_lifecycle(self):
        b = _FakeBridge("alpha")
        assert b.node_id == "alpha"
        assert b.health() == BridgeStatus.DISCONNECTED

        b.connect("localhost", 8080)
        assert b.health() == BridgeStatus.CONNECTED
        assert b._host == "localhost"
        assert b._port == 8080

        b.push({"status": "ok"})
        assert len(b._pushed) == 1

        result = b.pull("agents")
        assert result["query"] == "agents"

        b.disconnect()
        assert b.health() == BridgeStatus.DISCONNECTED

    def test_bridge_status_enum(self):
        assert BridgeStatus.DISCONNECTED.name == "DISCONNECTED"
        assert BridgeStatus.CONNECTING.name == "CONNECTING"
        assert BridgeStatus.CONNECTED.name == "CONNECTED"
        assert BridgeStatus.ERROR.name == "ERROR"


class TestBridgeRegistry:
    def test_register_and_create(self):
        reg = BridgeRegistry()
        reg.register("fake", _FakeBridge)
        b = reg.create("fake", node_id="n1")
        assert isinstance(b, _FakeBridge)
        assert b.node_id == "n1"

    def test_create_unknown_raises(self):
        reg = BridgeRegistry()
        with pytest.raises(KeyError):
            reg.create("unknown", node_id="n1")

    def test_list_bridges(self):
        reg = BridgeRegistry()
        reg.register("a", _FakeBridge)
        reg.register("b", _FakeBridge)
        names = reg.list_bridges()
        assert "a" in names
        assert "b" in names

    def test_get_bridge(self):
        reg = BridgeRegistry()
        reg.register("x", _FakeBridge)
        # Create first, then get
        created = reg.create("x", node_id="n1")
        got = reg.get("x", node_id="n1")
        assert got is created
        assert isinstance(got, _FakeBridge)
        assert got.node_id == "n1"

    def test_get_unknown_returns_none(self):
        reg = BridgeRegistry()
        assert reg.get("unknown", node_id="n1") is None

    def test_health_check_all(self):
        reg = BridgeRegistry()
        reg.register("fake", _FakeBridge)
        b = reg.create("fake", node_id="n1")
        b.connect("h", 1)
        events = reg.health_check_all()
        key = "fake::n1"
        assert key in events
        assert events[key].status == BridgeStatus.CONNECTED


class TestBridgeCompiler:
    def test_compile_schema_not_implemented(self):
        reg = BridgeRegistry()
        bc = BridgeCompiler(registry=reg)
        with pytest.raises(NotImplementedError):
            bc.compile_from_schema(
                "TestBridge", {"connect": ["host", "port"], "push": ["data"]}
            )

    def test_compile_returns_bridge_subclass_not_implemented(self):
        reg = BridgeRegistry()
        bc = BridgeCompiler(registry=reg)
        with pytest.raises(NotImplementedError):
            bc.compile_from_schema(
                "Dyn",
                {
                    "connect": ["host"],
                    "push": ["data"],
                    "pull": ["query"],
                    "disconnect": [],
                },
            )
