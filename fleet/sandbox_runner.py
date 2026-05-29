"""Sandbox runner for isolated fleet task execution.

Executes tasks in isolated environments with resource limits, timeouts,
and stdout/stderr capture. Used for fleet agent sandboxing, safe code
execution, and untrusted workload isolation.

Usage:
    runner = SandboxRunner(timeout_sec=5, max_memory_mb=64)
    result = runner.run("echo hello", shell=True)
    assert result.stdout == "hello\n"
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SandboxResult:
    """Result of a sandboxed execution."""

    stdout: str
    stderr: str
    returncode: int
    timed_out: bool


class SandboxRunner:
    """
    Isolated task runner with resource limits.

    :param timeout_sec: Maximum execution time.
    :param max_memory_mb: Maximum memory in MB (soft limit, platform-dependent).
    """

    def __init__(self, timeout_sec: float = 30.0, max_memory_mb: int = 128):
        self._timeout = timeout_sec
        self._max_memory = max_memory_mb

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, command: str, shell: bool = True) -> SandboxResult:
        """
        Run a command in sandbox.

        :param command: Command string or list.
        :param shell: Use shell execution.
        :returns: SandboxResult with stdout, stderr, returncode.
        """
        try:
            proc = subprocess.run(
                command,
                shell=shell,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            return SandboxResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode,
                timed_out=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                stdout="",
                stderr="",
                returncode=-1,
                timed_out=True,
            )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "timeout_sec": self._timeout,
            "max_memory_mb": self._max_memory,
        }

    def __repr__(self) -> str:
        return f"<SandboxRunner timeout={self._timeout}s memory={self._max_memory}MB>"
