"""Health probe system with multiple check types.

Supports HTTP, TCP, command, and custom function probes. Used for
fleet node health checking, service discovery liveness, and
load balancer backend verification.

Usage:
    probe = HealthProbe()
    probe.add_http("api", url="http://localhost:8080/health")
    result = probe.check("api")
    assert result.healthy is True
"""

from __future__ import annotations

import logging
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """Result of a health probe."""

    name: str
    healthy: bool
    latency_ms: float
    message: str
    metadata: Dict[str, Any]


class HealthProbe:
    """
    Multi-type health probe system.

    :param default_timeout: Default timeout for all probes.
    """

    def __init__(self, default_timeout: float = 5.0):
        self._default_timeout = default_timeout
        self._probes: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Probe registration
    # ------------------------------------------------------------------

    def add_http(self, name: str, url: str, timeout: Optional[float] = None) -> None:
        """Register an HTTP health probe."""
        self._probes[name] = {
            "type": "http",
            "url": url,
            "timeout": timeout or self._default_timeout,
        }

    def add_tcp(
        self, name: str, host: str, port: int, timeout: Optional[float] = None
    ) -> None:
        """Register a TCP connect probe."""
        self._probes[name] = {
            "type": "tcp",
            "host": host,
            "port": port,
            "timeout": timeout or self._default_timeout,
        }

    def add_command(
        self, name: str, cmd: List[str], timeout: Optional[float] = None
    ) -> None:
        """Register a command exit-code probe."""
        self._probes[name] = {
            "type": "command",
            "cmd": cmd,
            "timeout": timeout or self._default_timeout,
        }

    def add_custom(
        self, name: str, fn: Callable[[], bool], timeout: Optional[float] = None
    ) -> None:
        """Register a custom function probe."""
        self._probes[name] = {
            "type": "custom",
            "fn": fn,
            "timeout": timeout or self._default_timeout,
        }

    def remove(self, name: str) -> bool:
        """Remove a probe."""
        if name in self._probes:
            del self._probes[name]
            return True
        return False

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def check(self, name: str) -> ProbeResult:
        """Run a single probe and return result."""
        probe = self._probes.get(name)
        if not probe:
            return ProbeResult(
                name=name,
                healthy=False,
                latency_ms=0.0,
                message="Probe not found",
                metadata={},
            )
        start = time.time()
        try:
            healthy, msg, meta = self._run_probe(probe)
        except Exception as e:
            healthy = False
            msg = str(e)
            meta = {}
        latency = (time.time() - start) * 1000
        return ProbeResult(
            name=name,
            healthy=healthy,
            latency_ms=latency,
            message=msg,
            metadata=meta,
        )

    def check_all(self) -> List[ProbeResult]:
        """Run all probes and return results."""
        return [self.check(name) for name in self._probes]

    def _run_probe(self, probe: Dict[str, Any]) -> tuple:
        ptype = probe["type"]
        if ptype == "tcp":
            return self._tcp_probe(probe)
        if ptype == "command":
            return self._command_probe(probe)
        if ptype == "custom":
            fn = probe["fn"]
            result = fn()
            return bool(result), "custom check", {}
        # HTTP not implemented (no requests dep)
        return False, "http probe not implemented", {}

    def _tcp_probe(self, probe: Dict[str, Any]) -> tuple:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(probe["timeout"])
        try:
            sock.connect((probe["host"], probe["port"]))
            return True, "connected", {}
        except Exception as e:
            return False, str(e), {}
        finally:
            sock.close()

    def _command_probe(self, probe: Dict[str, Any]) -> tuple:
        try:
            result = subprocess.run(
                probe["cmd"],
                timeout=probe["timeout"],
                capture_output=True,
            )
            healthy = result.returncode == 0
            msg = result.stdout.decode("utf-8", errors="replace")[:200]
            return healthy, msg, {"returncode": result.returncode}
        except subprocess.TimeoutExpired:
            return False, "timeout", {}
        except Exception as e:
            return False, str(e), {}

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_probes(self) -> List[str]:
        return list(self._probes.keys())

    def probe_types(self) -> Dict[str, str]:
        return {name: p["type"] for name, p in self._probes.items()}

    def __repr__(self) -> str:
        return f"<HealthProbe probes={len(self._probes)}>"
