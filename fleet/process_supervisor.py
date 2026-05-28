"""process_supervisor.py — Process lifecycle management.

Provides:
1. Start / stop / restart processes
2. Health check polling
3. Auto-restart on failure (with backoff)
4. Graceful shutdown with SIGTERM → SIGKILL
5. Process status and uptime tracking

Usage:
    ps = ProcessSupervisor()
    ps.start("worker-1", cmd=["python", "worker.py"], health_check=lambda: check_port(8080))
    ps.restart("worker-1")
    ps.stop("worker-1")
"""
from __future__ import annotations

__all__ = [
    "ProcessSupervisor",
    "ManagedProcess",
    "ProcessNotFound",
]

import logging
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ProcessNotFound(Exception):
    """Raised when a managed process is not found."""


@dataclass
class ManagedProcess:
    """A managed process record."""
    name: str
    cmd: list[str]
    process: subprocess.Popen | None = None
    health_check: Callable[[], bool] | None = None
    restart_policy: str = "always"  # always, on-failure, never
    max_restarts: int = 5
    restart_window: float = 60.0
    restart_delay: float = 1.0
    status: str = "stopped"
    uptime: float = 0.0
    start_time: float | None = None
    restart_count: int = 0
    last_restart_time: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)


class ProcessSupervisor:
    """Supervise processes with health checks and auto-restart."""

    def __init__(self, poll_interval: float = 5.0) -> None:
        self._poll_interval = poll_interval
        self._processes: dict[str, ManagedProcess] = {}
        self._monitor_thread: threading.Thread | None = None
        self._stop_monitor = threading.Event()
        self._lock = threading.Lock()

    def start(
        self,
        name: str,
        cmd: list[str],
        health_check: Callable[[], bool] | None = None,
        restart_policy: str = "always",
        max_restarts: int = 5,
        restart_delay: float = 1.0,
        restart_window: float = 60.0,
    ) -> ManagedProcess:
        """Start a managed process."""
        proc = ManagedProcess(
            name=name,
            cmd=list(cmd),
            health_check=health_check,
            restart_policy=restart_policy,
            max_restarts=max_restarts,
            restart_delay=restart_delay,
            restart_window=restart_window,
        )
        self._processes[name] = proc
        self._launch(proc)
        self._ensure_monitor()
        return proc

    def _launch(self, proc: ManagedProcess) -> None:
        with proc._lock:
            if proc.process is not None and proc.process.poll() is None:
                logger.warning(f"Process '{proc.name}' already running")
                return
            try:
                proc.process = subprocess.Popen(
                    proc.cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                proc.status = "running"
                proc.start_time = time.time()
                proc.restart_count += 1
                proc.last_restart_time = time.time()
                logger.info(f"Started '{proc.name}' (pid={proc.process.pid})")
            except Exception as e:
                proc.status = "failed"
                logger.error(f"Failed to start '{proc.name}': {e}")

    def stop(self, name: str, timeout: float = 10.0) -> bool:
        """Stop a managed process gracefully."""
        proc = self._processes.get(name)
        if not proc:
            raise ProcessNotFound(f"Process '{name}' not found")

        with proc._lock:
            if proc.process is None:
                proc.status = "stopped"
                return True

            try:
                proc.process.send_signal(signal.SIGTERM)
                proc.process.wait(timeout=timeout)
                proc.status = "stopped"
                proc.process = None
                proc.uptime = 0.0
                return True
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing '{proc.name}'")
                proc.process.kill()
                proc.process.wait()
                proc.status = "stopped"
                proc.process = None
                return True
            except Exception as e:
                logger.error(f"Error stopping '{proc.name}': {e}")
                return False

    def restart(self, name: str) -> ManagedProcess:
        """Restart a managed process."""
        self.stop(name)
        proc = self._processes.get(name)
        if proc:
            self._launch(proc)
        return proc

    def status(self, name: str) -> dict[str, Any] | None:
        """Get process status."""
        proc = self._processes.get(name)
        if not proc:
            return None
        with proc._lock:
            uptime = 0.0
            if proc.start_time:
                uptime = time.time() - proc.start_time
            return {
                "name": proc.name,
                "status": proc.status,
                "pid": proc.process.pid if proc.process else None,
                "uptime": uptime,
                "restart_count": proc.restart_count,
                "cmd": proc.cmd,
            }

    def list_processes(self) -> list[str]:
        """List all managed process names."""
        return list(self._processes.keys())

    def _ensure_monitor(self) -> None:
        """Start the health monitor thread if not running."""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._stop_monitor.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()

    def _monitor_loop(self) -> None:
        """Background loop: health checks and auto-restart."""
        while not self._stop_monitor.is_set():
            for proc in list(self._processes.values()):
                self._check_process(proc)
            self._stop_monitor.wait(self._poll_interval)

    def _check_process(self, proc: ManagedProcess) -> None:
        needs_restart = False
        with proc._lock:
            if proc.status != "running":
                return

            # Check if process exited
            if proc.process is not None and proc.process.poll() is not None:
                proc.status = "exited"
                logger.warning(f"Process '{proc.name}' exited with code {proc.process.returncode}")
                needs_restart = True

            # Health check
            elif proc.health_check is not None:
                try:
                    if not proc.health_check():
                        logger.warning(f"Health check failed for '{proc.name}'")
                        proc.status = "unhealthy"
                        needs_restart = True
                except Exception as e:
                    logger.error(f"Health check error for '{proc.name}': {e}")

        if needs_restart:
            self._maybe_restart(proc)

    def _maybe_restart(self, proc: ManagedProcess) -> None:
        """Restart a process if policy allows."""
        if proc.restart_policy == "never":
            return

        # Rate limit restarts
        if proc.last_restart_time and time.time() - proc.last_restart_time < proc.restart_window:
            if proc.restart_count >= proc.max_restarts:
                logger.error(f"Max restarts reached for '{proc.name}', giving up")
                proc.status = "failed"
                return

        logger.info(f"Restarting '{proc.name}' in {proc.restart_delay}s...")
        time.sleep(proc.restart_delay)
        self._launch(proc)

    def shutdown(self) -> None:
        """Stop all processes and the monitor."""
        self._stop_monitor.set()
        for name in list(self._processes.keys()):
            self.stop(name)
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)

    def stats(self) -> dict[str, Any]:
        """Supervisor statistics."""
        running = sum(1 for p in self._processes.values() if p.status == "running")
        return {
            "total": len(self._processes),
            "running": running,
            "monitor_active": self._monitor_thread is not None and self._monitor_thread.is_alive(),
        }

    def __repr__(self) -> str:
        running = sum(1 for p in self._processes.values() if p.status == "running")
        return f"ProcessSupervisor(total={len(self._processes)}, running={running})"
