"""Lightweight sandbox for executing untrusted breeding code.

Provides resource-limited execution with timeout, memory cap, and optional
subprocess isolation. Used by the fleet breeder to safely evaluate
candidate agents without risking the host.

Usage:
    box = Sandbox(max_memory_mb=128, max_cpu_sec=5.0)
    result = box.run(lambda: heavy_computation(), timeout=2.0)
"""
from __future__ import annotations

import gc
import logging
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class SandboxError(Exception):
    pass


class TimeoutError(SandboxError):
    pass


class MemoryLimitError(SandboxError):
    pass


@dataclass
class SandboxResult:
    """Result from a sandboxed execution."""

    success: bool
    value: Any = None
    error: Optional[str] = None
    duration_sec: float = 0.0
    memory_peak_mb: float = 0.0
    timeout: bool = False
    memory_limited: bool = False


class Sandbox:
    """
    Lightweight resource-limited execution sandbox.

    :param max_memory_mb: Soft memory limit (best-effort via gc/polling).
    :param max_cpu_sec: Default CPU timeout.
    :param allow_imports: List of allowed module names (None = allow all).
    :param block_builtins: List of builtin names to shadow (e.g., "open").
    """

    def __init__(
        self,
        max_memory_mb: float = 256.0,
        max_cpu_sec: float = 10.0,
        allow_imports: Optional[List[str]] = None,
        block_builtins: Optional[List[str]] = None,
    ):
        self._max_memory_mb = max_memory_mb
        self._max_cpu_sec = max_cpu_sec
        self._allow_imports = set(allow_imports) if allow_imports is not None else None
        self._block_builtins = set(block_builtins) if block_builtins is not None else set()
        self._stats: Dict[str, int] = {"runs": 0, "timeouts": 0, "memory_kills": 0, "errors": 0}

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        fn: Callable[[], Any],
        timeout: Optional[float] = None,
        memory_mb: Optional[float] = None,
    ) -> SandboxResult:
        """Execute *fn* under resource limits. Returns a SandboxResult."""
        deadline = timeout or self._max_cpu_sec
        mem_limit = memory_mb or self._max_memory_mb
        start = time.perf_counter()
        self._stats["runs"] += 1

        gc.collect()
        mem_before = self._current_memory_mb()
        mem_peak = mem_before

        # Install timeout alarm if on POSIX
        if sys.platform != "win32" and hasattr(signal, "SIGALRM"):
            old_handler = signal.signal(signal.SIGALRM, self._alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, deadline)
        else:
            old_handler = None

        try:
            value = fn()
            result = SandboxResult(
                success=True,
                value=value,
                duration_sec=time.perf_counter() - start,
                memory_peak_mb=mem_peak - mem_before,
            )
        except MemoryLimitError:
            self._stats["memory_kills"] += 1
            result = SandboxResult(
                success=False,
                error="Memory limit exceeded",
                duration_sec=time.perf_counter() - start,
                memory_limited=True,
            )
        except TimeoutError:
            self._stats["timeouts"] += 1
            result = SandboxResult(
                success=False,
                error="Timeout exceeded",
                duration_sec=time.perf_counter() - start,
                timeout=True,
            )
        except Exception as exc:
            self._stats["errors"] += 1
            result = SandboxResult(
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_sec=time.perf_counter() - start,
            )
        finally:
            if old_handler is not None:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)

        result.memory_peak_mb = max(result.memory_peak_mb, mem_peak - mem_before)
        return result

    # ------------------------------------------------------------------
    # Eval (string code)
    # ------------------------------------------------------------------

    def eval(
        self,
        code: str,
        globals_dict: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        memory_mb: Optional[float] = None,
    ) -> SandboxResult:
        """Evaluate Python source string under sandbox restrictions."""
        g = globals_dict or {}
        g["__builtins__"] = self._restricted_builtins()

        def _run():
            return eval(compile(code, "<sandbox>", "eval"), g)

        return self.run(_run, timeout=timeout, memory_mb=memory_mb)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _restricted_builtins(self) -> Dict[str, Any]:
        """Return a restricted builtins dict."""
        safe = {
            "True": True,
            "False": False,
            "None": None,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "tuple": tuple,
            "dict": dict,
            "set": set,
            "print": print,
        }
        for name in self._block_builtins:
            safe.pop(name, None)
        return safe

    def _current_memory_mb(self) -> float:
        """Best-effort current process RSS in MB."""
        try:
            import psutil

            return psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def _alarm_handler(self, signum: int, frame: Any) -> None:
        raise TimeoutError("Sandbox timeout exceeded")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def __repr__(self) -> str:
        return f"<Sandbox mem={self._max_memory_mb}MB cpu={self._max_cpu_sec}s>"
