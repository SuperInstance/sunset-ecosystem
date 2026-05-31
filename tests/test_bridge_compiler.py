"""Tests for fleet.bridge_compiler.

Coverage:
- Schema parsing (minimal JSON schema, OpenAPI-like)
- HTTP bridge generation (connect, push, pull, disconnect)
- gRPC bridge generation (stub)
- FFI bridge generation (ctypes pattern)
- Generated class instantiation and registration
- Health check on generated bridge
- Error handling (bad schema, missing endpoints)
- Schema validation (required fields)
- Multiple endpoint generation
- Edge cases (empty schema, no host, no port)
"""
from __future__ import annotations

import ctypes
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fleet.bridge import Bridge, BridgeRegistry, BridgeStatus, BridgeEvent
from fleet.bridge_compiler import (
    BridgeCompiler,
    BridgeSchema,
    SchemaParser,
    HTTPBridgeGenerator,
    GRPCBridgeGenerator,
    FFIBridgeGenerator,
    EndpointSchema,
    BridgeCompilerError,
    SchemaValidationError,
    UnsupportedProtocolError,
    MissingEndpointError,
)


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def registry() -> BridgeRegistry:
    return BridgeRegistry()


@pytest.fixture
def compiler(registry: BridgeRegistry) -> BridgeCompiler:
    return BridgeCompiler(registry=registry)


@pytest.fixture
def minimal_http_schema() -> dict:
    return {
        "name": "test_http",
        "host": "localhost",
        "port": 8080,
        "protocol": "http",
        "endpoints": {
            "push": "/push",
            "pull": "/pull",
        },
    }


@pytest.fixture
def openapi_like_schema() -> dict:
    return {
        "name": "weather_api",
        "host": "api.weather.com",
        "port": 443,
        "protocol": "http",
        "endpoints": {
            "push": {"path": "/v1/push", "method": "POST"},
            "pull": {"path": "/v1/pull", "method": "GET"},
        },
    }


@pytest.fixture
def grpc_schema() -> dict:
    return {
        "name": "test_grpc",
        "host": "localhost",
        "port": 50051,
        "protocol": "grpc",
        "endpoints": {
            "push": "/push",
            "pull": "/pull",
        },
    }


@pytest.fixture
def ffi_schema(tmp_path: Path) -> dict:
    # Build a tiny mock shared library for FFI tests
    lib_path = tmp_path / "libmock.so"
    c_code = tmp_path / "mock_lib.c"
    c_code.write_text(
        """
        int mock_push(const char* data) { return 0; }
        int mock_pull(const char* query, char* buf, unsigned long len) {
            const char* resp = "{\\"status\\": \"ok\\"}";
            unsigned long i;
            for (i = 0; i < len && resp[i]; i++) buf[i] = resp[i];
            if (i < len) buf[i] = '\\0';
            return 0;
        }
        """
    )
    import subprocess
    subprocess.run(
        ["gcc", "-shared", "-fPIC", "-o", str(lib_path), str(c_code)],
        check=True,
        capture_output=True,
    )
    return {
        "name": "test_ffi",
        "host": "localhost",
        "port": 0,
        "protocol": "ffi",
        "library_path": str(lib_path),
        "endpoints": {
            "push": {"path": "mock_push", "method": "POST"},
            "pull": {"path": "mock_pull", "method": "GET"},
        },
    }


# ── 1. Schema parsing: minimal JSON schema ────────────────────────

def test_schema_parser_minimal(minimal_http_schema: dict) -> None:
    schema = SchemaParser.parse(minimal_http_schema)
    assert isinstance(schema, BridgeSchema)
    assert schema.name == "test_http"
    assert schema.host == "localhost"
    assert schema.port == 8080
    assert schema.protocol == "http"
    assert "push" in schema.endpoints
    assert "pull" in schema.endpoints
    assert schema.endpoints["push"].path == "/push"
    assert schema.endpoints["push"].method == "GET"  # default


# ── 2. Schema parsing: OpenAPI-like ───────────────────────────────

def test_schema_parser_openapi_like(openapi_like_schema: dict) -> None:
    schema = SchemaParser.parse(openapi_like_schema)
    assert schema.name == "weather_api"
    assert schema.endpoints["push"].method == "POST"
    assert schema.endpoints["push"].path == "/v1/push"
    assert schema.endpoints["pull"].method == "GET"
    assert schema.endpoints["pull"].path == "/v1/pull"


# ── 3. OpenAPI spec parsing ─────────────────────────────────────

def test_schema_parser_from_openapi_spec() -> None:
    spec = {
        "servers": [{"url": "https://api.example.com:8443/v1"}],
        "paths": {
            "/items/{id}": {
                "get": {
                    "operationId": "pullItem",
                },
            },
            "/items": {
                "post": {
                    "operationId": "pushItem",
                },
            },
        },
    }
    schema = SchemaParser.parse_openapi(spec, "items_api")
    assert schema.name == "items_api"
    assert schema.host == "api.example.com"
    assert schema.port == 8443
    assert "push" in schema.endpoints
    assert "pull" in schema.endpoints


# ── 4. HTTP bridge generation ─────────────────────────────────────

def test_http_bridge_generation(compiler: BridgeCompiler, minimal_http_schema: dict) -> None:
    bridge_class = compiler.compile_from_dict(minimal_http_schema)
    assert issubclass(bridge_class, Bridge)
    assert bridge_class.__name__ == "TestHttpBridge"


# ── 5. HTTP bridge connect / push / pull / disconnect (mocked) ────

def test_http_bridge_lifecycle(compiler: BridgeCompiler, minimal_http_schema: dict) -> None:
    bridge_class = compiler.compile_from_dict(minimal_http_schema)
    bridge = bridge_class(node_id="node-1")

    # Mock requests.Session so we don't need network
    mock_session = MagicMock()
    mock_response_push = MagicMock()
    mock_response_push.status_code = 201
    mock_response_pull = MagicMock()
    mock_response_pull.status_code = 200
    mock_response_pull.json.return_value = {"data": "ok"}
    mock_response_pull.text = '{"data": "ok"}'

    mock_session.request.side_effect = [mock_response_push, mock_response_pull, mock_response_pull]

    with patch("requests.Session", return_value=mock_session):
        bridge.connect("localhost", 8080)
        assert bridge.status == BridgeStatus.CONNECTED

        assert bridge.push({"key": "value"}) is True
        mock_session.request.assert_called_once()
        args, kwargs = mock_session.request.call_args
        assert kwargs["json"] == {"key": "value"}

        result = bridge.pull({"id": 1})
        assert result == {"data": "ok"}

        bridge.disconnect()
        assert bridge.status == BridgeStatus.DISCONNECTED
        mock_session.close.assert_called_once()


# ── 6. gRPC bridge generation ─────────────────────────────────────

def test_grpc_bridge_generation(compiler: BridgeCompiler, grpc_schema: dict) -> None:
    bridge_class = compiler.compile_from_dict(grpc_schema)
    assert issubclass(bridge_class, Bridge)
    assert bridge_class.__name__ == "TestGrpcBridge"


# ── 7. gRPC bridge stub lifecycle (mocked grpc) ─────────────────

def test_grpc_bridge_lifecycle(compiler: BridgeCompiler, grpc_schema: dict) -> None:
    bridge_class = compiler.compile_from_dict(grpc_schema)
    bridge = bridge_class(node_id="node-1")

    mock_channel = MagicMock()
    with patch.dict(
        sys.modules,
        {
            "grpc": types.ModuleType("grpc"),
        },
    ):
        # Need to inject insecure_channel on the module
        grpc_mod = sys.modules["grpc"]
        grpc_mod.insecure_channel = MagicMock(return_value=mock_channel)
        bridge.connect("localhost", 50051)
        assert bridge.status == BridgeStatus.CONNECTED

        assert bridge.push({"data": "x"}) is True
        assert bridge.pull({"q": "y"}) == {"stub": True, "query": {"q": "y"}}

        bridge.disconnect()
        assert bridge.status == BridgeStatus.DISCONNECTED
        mock_channel.close.assert_called_once()


# ── 8. FFI bridge generation ─────────────────────────────────────

def test_ffi_bridge_generation(compiler: BridgeCompiler, ffi_schema: dict) -> None:
    bridge_class = compiler.compile_from_dict(ffi_schema)
    assert issubclass(bridge_class, Bridge)
    assert bridge_class.__name__ == "TestFfiBridge"


# ── 9. FFI bridge connect / push / pull / disconnect (real .so) ─

def test_ffi_bridge_lifecycle(compiler: BridgeCompiler, ffi_schema: dict) -> None:
    bridge_class = compiler.compile_from_dict(ffi_schema)
    bridge = bridge_class(node_id="node-1")
    bridge.connect("localhost", 0)
    assert bridge.status == BridgeStatus.CONNECTED

    assert bridge.push({"key": "value"}) is True
    result = bridge.pull("query")
    assert result is not None
    assert "status" in result

    bridge.disconnect()
    assert bridge.status == BridgeStatus.DISCONNECTED


# ── 10. Generated class instantiation and registration ────────────

def test_generated_class_registered(compiler: BridgeCompiler, minimal_http_schema: dict) -> None:
    bridge_class = compiler.compile_from_dict(minimal_http_schema)
    instance = compiler.registry.create("test_http", node_id="alpha")
    assert isinstance(instance, bridge_class)
    assert compiler.registry.get("test_http", "alpha") is instance


# ── 11. Health check on generated HTTP bridge ───────────────────

def test_health_check_on_generated_bridge(compiler: BridgeCompiler, minimal_http_schema: dict) -> None:
    bridge_class = compiler.compile_from_dict(minimal_http_schema)
    bridge = bridge_class(node_id="health-node")

    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_session.request.return_value = mock_response

    with patch("requests.Session", return_value=mock_session):
        bridge.connect("localhost", 8080)
        event = bridge.health_check()
        assert isinstance(event, BridgeEvent)
        assert event.bridge_name == "TestHttpBridge"
        assert event.status == BridgeStatus.CONNECTED
        assert event.error is None


# ── 12. Error handling: bad schema (unsupported protocol) ─────

def test_bad_schema_unsupported_protocol() -> None:
    bad = {"name": "x", "host": "h", "port": 1, "protocol": "ftp"}
    with pytest.raises(SchemaValidationError) as exc_info:
        SchemaParser.parse(bad)
    assert "ftp" in str(exc_info.value)
    assert "Unsupported protocol" in str(exc_info.value)


# ── 13. Error handling: missing endpoints ───────────────────────

def test_missing_endpoints(compiler: BridgeCompiler) -> None:
    schema = {
        "name": "no_endpoints",
        "host": "localhost",
        "port": 8080,
        "protocol": "http",
        "endpoints": {},
    }
    parsed = SchemaParser.parse(schema)
    with pytest.raises(MissingEndpointError) as exc_info:
        compiler.compile(parsed)
    assert "push" in str(exc_info.value)
    assert "pull" in str(exc_info.value)


# ── 14. Schema validation: required fields ───────────────────────

def test_schema_validation_required_fields() -> None:
    with pytest.raises(SchemaValidationError) as exc_info:
        SchemaParser.parse({})
    assert "empty" in str(exc_info.value).lower()

    with pytest.raises(SchemaValidationError) as exc_info:
        SchemaParser.parse({"name": "x", "host": "h"})
    assert "port" in str(exc_info.value)
    assert "protocol" in str(exc_info.value)


# ── 15. Schema validation: invalid port ───────────────────────────

def test_schema_validation_invalid_port() -> None:
    bad = {"name": "x", "host": "h", "port": -1, "protocol": "http"}
    with pytest.raises(SchemaValidationError) as exc_info:
        SchemaParser.parse(bad)
    assert "port" in str(exc_info.value)


# ── 16. Multiple endpoint generation ──────────────────────────────

def test_multiple_endpoint_generation(compiler: BridgeCompiler) -> None:
    schema = {
        "name": "multi_endpoint",
        "host": "localhost",
        "port": 8080,
        "protocol": "http",
        "endpoints": {
            "push": {"path": "/push", "method": "POST"},
            "pull": {"path": "/pull", "method": "GET"},
            "health": {"path": "/health", "method": "GET"},
            "delete": {"path": "/delete", "method": "DELETE"},
        },
    }
    bridge_class = compiler.compile_from_dict(schema)
    bridge = bridge_class(node_id="multi")
    assert issubclass(bridge_class, Bridge)
    # push and pull are required; extra endpoints are parsed but not
    # used by the current Bridge ABC — still, schema should have them
    parsed = SchemaParser.parse(schema)
    assert len(parsed.endpoints) == 4


# ── 17. Edge case: empty schema ─────────────────────────────────

def test_empty_schema() -> None:
    with pytest.raises(SchemaValidationError):
        SchemaParser.parse({})
    with pytest.raises(SchemaValidationError):
        SchemaParser.parse(None)  # type: ignore[arg-type]


# ── 18. Edge case: no host ───────────────────────────────────────

def test_no_host() -> None:
    bad = {"name": "x", "port": 8080, "protocol": "http"}
    with pytest.raises(SchemaValidationError) as exc_info:
        SchemaParser.parse(bad)
    assert "host" in str(exc_info.value)


# ── 19. Edge case: no port ──────────────────────────────────────

def test_no_port() -> None:
    bad = {"name": "x", "host": "h", "protocol": "http"}
    with pytest.raises(SchemaValidationError) as exc_info:
        SchemaParser.parse(bad)
    assert "port" in str(exc_info.value)


# ── 20. BridgeCompiler dispatches to correct generator ──────────

def test_compiler_dispatches_http(compiler: BridgeCompiler, minimal_http_schema: dict) -> None:
    bridge_class = compiler.compile_from_dict(minimal_http_schema)
    assert bridge_class.__name__ == "TestHttpBridge"


def test_compiler_dispatches_grpc(compiler: BridgeCompiler, grpc_schema: dict) -> None:
    bridge_class = compiler.compile_from_dict(grpc_schema)
    assert bridge_class.__name__ == "TestGrpcBridge"


def test_compiler_dispatches_ffi(compiler: BridgeCompiler, ffi_schema: dict) -> None:
    bridge_class = compiler.compile_from_dict(ffi_schema)
    assert bridge_class.__name__ == "TestFfiBridge"


# ── 21. Unsupported protocol at compiler level ──────────────────

def test_unsupported_protocol_at_compiler(compiler: BridgeCompiler) -> None:
    schema = BridgeSchema(
        name="bad",
        host="h",
        port=1,
        protocol="ftp",
        endpoints={"push": EndpointSchema(path="/"), "pull": EndpointSchema(path="/")},
    )
    with pytest.raises(UnsupportedProtocolError) as exc_info:
        compiler.compile(schema)
    assert "ftp" in str(exc_info.value)


# ── 22. FFI missing library_path ──────────────────────────────────

def test_ffi_missing_library_path(compiler: BridgeCompiler) -> None:
    schema = BridgeSchema(
        name="bad_ffi",
        host="h",
        port=1,
        protocol="ffi",
        endpoints={"push": EndpointSchema(path="push"), "pull": EndpointSchema(path="pull")},
    )
    with pytest.raises(SchemaValidationError) as exc_info:
        compiler.compile(schema)
    assert "library_path" in str(exc_info.value)


# ── 23. HTTP bridge push failure on disconnected ─────────────────

def test_http_push_when_disconnected(compiler: BridgeCompiler, minimal_http_schema: dict) -> None:
    bridge_class = compiler.compile_from_dict(minimal_http_schema)
    bridge = bridge_class(node_id="disc")
    # Never connected
    assert bridge.push({"x": 1}) is False


# ── 24. HTTP bridge pull failure on disconnected ─────────────────

def test_http_pull_when_disconnected(compiler: BridgeCompiler, minimal_http_schema: dict) -> None:
    bridge_class = compiler.compile_from_dict(minimal_http_schema)
    bridge = bridge_class(node_id="disc")
    # Never connected
    assert bridge.pull("q") is None


# ── 25. HTTP bridge error handling (network failure) ─────────────

def test_http_bridge_network_error(compiler: BridgeCompiler, minimal_http_schema: dict) -> None:
    bridge_class = compiler.compile_from_dict(minimal_http_schema)
    bridge = bridge_class(node_id="err")

    mock_session = MagicMock()
    mock_session.request.side_effect = Exception("connection refused")

    with patch("requests.Session", return_value=mock_session):
        bridge.connect("localhost", 8080)
        assert bridge.push({"x": 1}) is False
        assert bridge.status == BridgeStatus.ERROR


# ── 26. BridgeCompiler list_supported_protocols ─────────────────

def test_list_supported_protocols(compiler: BridgeCompiler) -> None:
    protocols = compiler.list_supported_protocols()
    assert set(protocols) == {"http", "grpc", "ffi"}


# ── 27. Health check on disconnected bridge returns error ───────

def test_health_check_disconnected(compiler: BridgeCompiler, minimal_http_schema: dict) -> None:
    bridge_class = compiler.compile_from_dict(minimal_http_schema)
    bridge = bridge_class(node_id="disc")
    event = bridge.health_check()
    assert event.status == BridgeStatus.ERROR
    assert event.error is not None


# ── 28. SchemaParser endpoint spec as string vs dict ────────────

def test_endpoint_spec_variants() -> None:
    schema = {
        "name": "variants",
        "host": "h",
        "port": 80,
        "protocol": "http",
        "endpoints": {
            "push": "/push",  # string
            "pull": {"path": "/pull", "method": "POST"},  # dict
        },
    }
    parsed = SchemaParser.parse(schema)
    assert parsed.endpoints["push"].path == "/push"
    assert parsed.endpoints["push"].method == "GET"  # default
    assert parsed.endpoints["pull"].path == "/pull"
    assert parsed.endpoints["pull"].method == "POST"


# ── 29. EndpointSchema defaults ─────────────────────────────────

def test_endpoint_schema_defaults() -> None:
    ep = BridgeSchema(name="n", host="h", port=1, protocol="http", endpoints={"x": EndpointSchema(path="/")})
    assert ep.endpoints["x"].method == "GET"
    assert ep.endpoints["x"].headers == {}
    assert ep.endpoints["x"].body_template is None
