"""Bridge — Common interface for all fleet adapters.

Every bridge in the fleet connects Python to a foreign system. This module
extracts the common interface and provides a uniform way to manage bridges.

Usage
-----
    from fleet.bridge import Bridge, BridgeRegistry

    class MyBridge(Bridge):
        def connect(self, host, port): ...
        def push(self, data): ...
        def pull(self, query): ...
        def disconnect(self): ...

    registry = BridgeRegistry()
    registry.register("my_system", MyBridge)
    bridge = registry.create("my_system", node_id="alpha")
    bridge.connect("localhost", 8080)
    bridge.push({"status": "ok"})
    data = bridge.pull("query")
    bridge.disconnect()
"""

from __future__ import annotations

__all__ = [
    "Bridge",
    "BridgeRegistry",
    "BridgeStatus",
    "BridgeEvent",
    "BridgeCompiler",
]

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Type
import logging

logger = logging.getLogger(__name__)


# ── Status ──────────────────────────────────────────────────────


class BridgeStatus(Enum):
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    ERROR = auto()


# ── Event ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class BridgeEvent:
    """Emitted by a bridge during operation."""

    bridge_name: str
    status: BridgeStatus
    latency_ms: float
    payload_size: int
    error: Optional[str] = None


# ── ABC ─────────────────────────────────────────────────────────


class Bridge(ABC):
    """Common interface for all fleet adapters."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._status = BridgeStatus.DISCONNECTED

    @property
    def status(self) -> BridgeStatus:
        return self._status

    @abstractmethod
    def connect(self, host: str, port: int) -> None:
        """Establish connection to the foreign system."""
        ...

    @abstractmethod
    def push(self, data: Any) -> bool:
        """Send data to the foreign system."""
        ...

    @abstractmethod
    def pull(self, query: Any) -> Any:
        """Fetch data from the foreign system."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection."""
        ...

    def health_check(self) -> BridgeEvent:
        """Quick connectivity check."""
        import time

        t0 = time.perf_counter()
        try:
            self.push({"__health_check": True})
            latency = (time.perf_counter() - t0) * 1000
            return BridgeEvent(
                bridge_name=self.__class__.__name__,
                status=BridgeStatus.CONNECTED,
                latency_ms=latency,
                payload_size=0,
            )
        except Exception as e:
            return BridgeEvent(
                bridge_name=self.__class__.__name__,
                status=BridgeStatus.ERROR,
                latency_ms=(time.perf_counter() - t0) * 1000,
                payload_size=0,
                error=str(e),
            )


# ── Registry ────────────────────────────────────────────────────


class BridgeRegistry:
    """Central registry for all fleet bridges."""

    def __init__(self):
        self._bridges: Dict[str, Type[Bridge]] = {}
        self._instances: Dict[str, Bridge] = {}

    def register(self, name: str, bridge_class: Type[Bridge]) -> None:
        """Register a bridge class under a name."""
        self._bridges[name] = bridge_class
        logger.info("Registered bridge: %s", name)

    def create(self, name: str, node_id: str, **kwargs: Any) -> Bridge:
        """Instantiate a registered bridge."""
        if name not in self._bridges:
            raise KeyError(
                f"Bridge '{name}' not registered. Known: {list(self._bridges.keys())}"
            )
        instance = self._bridges[name](node_id=node_id, **kwargs)
        self._instances[f"{name}::{node_id}"] = instance
        return instance

    def get(self, name: str, node_id: str) -> Optional[Bridge]:
        """Get an existing bridge instance."""
        return self._instances.get(f"{name}::{node_id}")

    def list_bridges(self) -> List[str]:
        """List all registered bridge names."""
        return list(self._bridges.keys())

    def health_check_all(self) -> Dict[str, BridgeEvent]:
        """Run health check on all instantiated bridges."""
        return {key: bridge.health_check() for key, bridge in self._instances.items()}


# ── Compiler (stub for auto-generation) ─────────────────────────


class BridgeCompiler:
    """Auto-generate bridge classes from foreign system schemas.

    Stub: full implementation will infer method signatures from
    OpenAPI, protobuf, or Rust trait definitions.
    """

    def __init__(self, registry: BridgeRegistry):
        self.registry = registry

    def compile_from_schema(self, name: str, schema: Dict[str, Any]) -> Type[Bridge]:
        """Generate a bridge class from a schema description.

        Schema format (minimal):
            {
                "host": str,
                "port": int,
                "push_endpoint": str,
                "pull_endpoint": str,
                "protocol": "http" | "grpc" | "ffi",
            }
        """
        # TODO: Generate a dynamic class from the schema
        raise NotImplementedError("Schema compilation not yet implemented")
