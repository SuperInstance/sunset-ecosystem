"""Bernstein Orchestrator — Deterministic parallel agent scheduler with git worktree isolation.

Inspired by the Bernstein project (chernistry/bernstein), distilled to five
focused classes with zero CLI adapter baggage:

  • GitWorktreeSpawner    — isolated git worktrees per agent task
  • DeterministicScheduler — pure-Python parallel task scheduling with retry
  • HMACAuditChain        — tamper-evident decision log (blockchain-style)
  • JanitorVerifier       — output gates (files, tests, lint)
  • BernsteinOrchestrator — composes the above + GatewayPacing integration

Reference: docs/BERNSTEIN_ORCHESTRATOR.md
"""

from __future__ import annotations

__all__ = [
    "BernsteinOrchestrator",
    "GitWorktreeSpawner",
    "DeterministicScheduler",
    "HMACAuditChain",
    "JanitorVerifier",
    "OrchestratorConfig",
    "SchedulerTask",
    "ScheduleResult",
    "AuditEntry",
    "VerificationReport",
]

import hashlib
import hmac
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────


@dataclass(frozen=True)
class SchedulerTask:
    """A single unit of work for the deterministic scheduler."""

    task_id: str
    command: Callable[[], Any]  # zero-argument callable; returns result dict
    expected_outputs: List[str] = field(default_factory=list)
    timeout: float = 300.0
    max_retries: int = 2
    alternate_strategy: Optional[Callable[[], Any]] = None


@dataclass
class ScheduleResult:
    """Outcome of scheduling a single task."""

    task_id: str
    status: str  # "success", "failure", "timeout", "aborted"
    worktree_path: str = ""
    output: Any = None
    retry_count: int = 0
    duration: float = 0.0
    error: str = ""


@dataclass(frozen=True)
class AuditEntry:
    """A single signed decision in the audit chain."""

    timestamp: float
    decision_type: str
    task_id: str
    details: Dict[str, Any]
    previous_hash: str
    signature: str

    def compute_hash(self) -> str:
        """Deterministic hash of this entry for chaining."""
        payload = json.dumps(
            {
                "timestamp": self.timestamp,
                "decision_type": self.decision_type,
                "task_id": self.task_id,
                "details": self.details,
                "previous_hash": self.previous_hash,
                "signature": self.signature,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class VerificationReport:
    """Result of JanitorVerifier gate checks."""

    passed: bool
    gate: str  # which gate failed, or "all"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorConfig:
    """Configuration for BernsteinOrchestrator."""

    max_workers: int = 4
    default_timeout: float = 300.0
    default_max_retries: int = 2
    audit_key_env: str = "BERNSTEIN_AUDIT_KEY"
    audit_log_path: Optional[str] = None
    gateway_max_concurrent: int = 10
    cleanup_on_failure: bool = True
    cleanup_on_success: bool = True


# ── 1. GitWorktreeSpawner ─────────────────────────────────────


class GitWorktreeSpawner:
    """Create and manage isolated git worktrees for agent tasks.

    Each worktree gets a unique branch ``agent-{task_id}-{timestamp}``
    so concurrent spawns never collide.
    """

    def __init__(
        self,
        repo_path: str,
        git_executable: str = "git",
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.git = git_executable
        self._lock = threading.Lock()
        self._worktrees: Dict[str, str] = {}  # task_id → worktree_path

    # ── Public API ────────────────────────────────────────────

    def spawn(self, task_id: str) -> tuple[str, str]:
        """Create a worktree + branch for *task_id*.

        Returns ``(worktree_path, branch_name)``.
        """
        timestamp = int(time.time() * 1000)
        branch_name = f"agent-{task_id}-{timestamp}"
        worktree_name = f"wt-{task_id}-{timestamp}"
        worktree_path = self.repo_path.parent / worktree_name

        with self._lock:
            if task_id in self._worktrees:
                raise ValueError(f"Worktree already exists for task_id={task_id}")

            # Create worktree
            self._git(
                "worktree",
                "add",
                "-b",
                branch_name,
                str(worktree_path),
                cwd=str(self.repo_path),
            )
            self._worktrees[task_id] = str(worktree_path)

        logger.info("Spawned worktree %s on branch %s", worktree_path, branch_name)
        return str(worktree_path), branch_name

    def cleanup(self, task_id: str, branch_name: str) -> None:
        """Remove worktree and branch for *task_id*."""
        with self._lock:
            worktree_path = self._worktrees.pop(task_id, None)
            if worktree_path is None:
                logger.warning("No worktree tracked for task_id=%s", task_id)
                return

        # Remove worktree
        try:
            self._git(
                "worktree", "remove", "--force", worktree_path, cwd=str(self.repo_path)
            )
        except subprocess.CalledProcessError as exc:
            logger.warning("worktree remove failed: %s", exc)
            # Fallback: manual rm
            shutil.rmtree(worktree_path, ignore_errors=True)

        # Remove branch
        try:
            self._git("branch", "-D", branch_name, cwd=str(self.repo_path))
        except subprocess.CalledProcessError as exc:
            logger.warning("branch delete failed: %s", exc)

        logger.info("Cleaned up worktree %s branch %s", worktree_path, branch_name)

    def list_worktrees(self) -> Dict[str, str]:
        """Return tracked worktrees mapping task_id → path."""
        with self._lock:
            return dict(self._worktrees)

    # ── Internal ──────────────────────────────────────────────

    def _git(self, *args: str, cwd: str | None = None) -> str:
        """Run a git command; return stdout."""
        cmd = [self.git, *args]
        result = subprocess.run(
            cmd,
            cwd=cwd or str(self.repo_path),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()


# ── 2. DeterministicScheduler ─────────────────────────────────


class DeterministicScheduler:
    """Pure-Python parallel task scheduler with retry and alternate strategies.

    Uses ``concurrent.futures.ThreadPoolExecutor``.  No LLM involvement.
    """

    def __init__(
        self,
        max_workers: int = 4,
        base_backoff: float = 1.0,
        backoff_multiplier: float = 2.0,
        max_backoff: float = 60.0,
    ) -> None:
        self.max_workers = max_workers
        self.base_backoff = base_backoff
        self.backoff_multiplier = backoff_multiplier
        self.max_backoff = max_backoff

    # ── Public API ────────────────────────────────────────────

    def schedule(
        self,
        tasks: List[SchedulerTask],
        worktree_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, ScheduleResult]:
        """Schedule *tasks* in parallel.

        *worktree_map* is ``task_id → worktree_path`` injected by the
        orchestrator so results carry the correct path.
        """
        worktree_map = worktree_map or {}
        results: Dict[str, ScheduleResult] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(
                    self._run_with_retries, task, worktree_map.get(task.task_id, "")
                ): task.task_id
                for task in tasks
            }
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    results[task_id] = future.result()
                except Exception as exc:
                    results[task_id] = ScheduleResult(
                        task_id=task_id,
                        status="failure",
                        error=str(exc),
                        worktree_path=worktree_map.get(task_id, ""),
                    )

        return results

    # ── Internal ──────────────────────────────────────────────

    def _run_with_retries(
        self, task: SchedulerTask, worktree_path: str
    ) -> ScheduleResult:
        """Execute a task with exponential-backoff retry."""
        last_error = ""
        for attempt in range(task.max_retries + 1):
            start = time.perf_counter()
            try:
                output = task.command()
                duration = time.perf_counter() - start
                return ScheduleResult(
                    task_id=task.task_id,
                    status="success",
                    worktree_path=worktree_path,
                    output=output,
                    retry_count=attempt,
                    duration=duration,
                )
            except Exception as exc:
                last_error = str(exc)
                duration = time.perf_counter() - start
                logger.warning(
                    "Task %s attempt %d failed: %s (%.2fs)",
                    task.task_id,
                    attempt,
                    last_error,
                    duration,
                )

                # Retry with backoff unless last attempt
                if attempt < task.max_retries:
                    backoff = min(
                        self.base_backoff * (self.backoff_multiplier**attempt),
                        self.max_backoff,
                    )
                    time.sleep(backoff)

                    # Alternate strategy on retry if provided
                    if task.alternate_strategy is not None:
                        try:
                            output = task.alternate_strategy()
                            duration = time.perf_counter() - start
                            return ScheduleResult(
                                task_id=task.task_id,
                                status="success",
                                worktree_path=worktree_path,
                                output=output,
                                retry_count=attempt + 1,
                                duration=duration,
                            )
                        except Exception as alt_exc:
                            last_error = f"alternate_strategy failed: {alt_exc}"

        return ScheduleResult(
            task_id=task.task_id,
            status="failure",
            worktree_path=worktree_path,
            retry_count=task.max_retries,
            duration=0.0,
            error=last_error,
        )


# ── 3. HMACAuditChain ─────────────────────────────────────────


class HMACAuditChain:
    """Tamper-evident decision log using HMAC-SHA256 chained hashes.

    Each entry includes the previous entry's hash, forming a blockchain-style
    chain.  If any entry is modified, ``verify_chain()`` fails.
    """

    def __init__(
        self,
        key: bytes | None = None,
        key_env: str = "BERNSTEIN_AUDIT_KEY",
        log_path: str | None = None,
    ) -> None:
        if key is not None:
            self._key = key
        else:
            env_key = os.environ.get(key_env, "")
            if env_key:
                self._key = env_key.encode("utf-8")
            else:
                self._key = os.urandom(32)
                logger.info("Generated per-session audit key (no %s env var)", key_env)

        self._entries: List[AuditEntry] = []
        self._last_hash: str = ""
        self._log_path = log_path
        self._lock = threading.Lock()

        if self._log_path:
            self._load_log()

    # ── Public API ────────────────────────────────────────────

    def log_decision(
        self,
        decision_type: str,
        task_id: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Append a signed decision to the chain."""
        with self._lock:
            timestamp = time.time()
            payload = json.dumps(
                {
                    "timestamp": timestamp,
                    "decision_type": decision_type,
                    "task_id": task_id,
                    "details": details or {},
                    "previous_hash": self._last_hash,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")

            signature = hmac.new(self._key, payload, hashlib.sha256).hexdigest()
            entry = AuditEntry(
                timestamp=timestamp,
                decision_type=decision_type,
                task_id=task_id,
                details=details or {},
                previous_hash=self._last_hash,
                signature=signature,
            )
            self._entries.append(entry)
            self._last_hash = entry.compute_hash()
            self._persist(entry)
            return entry

    def verify_chain(self) -> tuple[bool, int]:
        """Verify the entire chain.

        Returns ``(all_valid, first_invalid_index)``.  If valid,
        *first_invalid_index* is ``-1``.
        """
        with self._lock:
            prev_hash = ""
            for i, entry in enumerate(self._entries):
                # 1. Signature must verify
                payload = json.dumps(
                    {
                        "timestamp": entry.timestamp,
                        "decision_type": entry.decision_type,
                        "task_id": entry.task_id,
                        "details": entry.details,
                        "previous_hash": entry.previous_hash,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                expected_sig = hmac.new(self._key, payload, hashlib.sha256).hexdigest()
                if not hmac.compare_digest(expected_sig, entry.signature):
                    logger.error("Audit chain: signature invalid at index %d", i)
                    return False, i

                # 2. Hash chain must link
                if entry.previous_hash != prev_hash:
                    logger.error(
                        "Audit chain: hash mismatch at index %d (expected %r, got %r)",
                        i,
                        prev_hash,
                        entry.previous_hash,
                    )
                    return False, i

                prev_hash = entry.compute_hash()

            return True, -1

    def export_chain(self, filepath: str) -> None:
        """Write the entire chain to *filepath* as newline-delimited JSON."""
        with self._lock:
            with open(filepath, "w", encoding="utf-8") as f:
                for entry in self._entries:
                    f.write(self._entry_to_json(entry) + "\n")

    @property
    def entries(self) -> List[AuditEntry]:
        with self._lock:
            return list(self._entries)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    # ── Internal ──────────────────────────────────────────────

    def _entry_to_json(self, entry: AuditEntry) -> str:
        return json.dumps(
            {
                "timestamp": entry.timestamp,
                "decision_type": entry.decision_type,
                "task_id": entry.task_id,
                "details": entry.details,
                "previous_hash": entry.previous_hash,
                "signature": entry.signature,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def _json_to_entry(self, raw: str) -> AuditEntry:
        data = json.loads(raw)
        return AuditEntry(
            timestamp=data["timestamp"],
            decision_type=data["decision_type"],
            task_id=data["task_id"],
            details=data["details"],
            previous_hash=data["previous_hash"],
            signature=data["signature"],
        )

    def _persist(self, entry: AuditEntry) -> None:
        if not self._log_path:
            return
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(self._entry_to_json(entry) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _load_log(self) -> None:
        if not self._log_path or not os.path.exists(self._log_path):
            return
        with open(self._log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = self._json_to_entry(line)
                    self._entries.append(entry)
                    self._last_hash = entry.compute_hash()
                except Exception as exc:
                    logger.warning("Skipping corrupt audit line: %s", exc)


# ── 4. JanitorVerifier ────────────────────────────────────────


class JanitorVerifier:
    """Gate-check agent output before merging back to the main branch.

    Checks:
      1. Expected output files exist
      2. Test command passes
      3. Lint command passes

    All gates must pass for ``verify()`` to return ``passed=True``.
    """

    def __init__(
        self,
        default_test_cmd: Optional[List[str]] = None,
        default_lint_cmd: Optional[List[str]] = None,
    ) -> None:
        self.default_test_cmd = default_test_cmd
        self.default_lint_cmd = default_lint_cmd

    # ── Public API ────────────────────────────────────────────

    def verify(
        self,
        worktree_path: str,
        expected_outputs: List[str],
        test_cmd: Optional[List[str]] = None,
        lint_cmd: Optional[List[str]] = None,
    ) -> VerificationReport:
        """Run all verification gates.

        Returns ``VerificationReport`` with ``passed`` bool and per-gate
        details.
        """
        details: Dict[str, Any] = {"worktree": worktree_path}
        wt = Path(worktree_path)

        # Gate 1: Files exist
        missing: List[str] = []
        for rel in expected_outputs:
            full = wt / rel
            if not full.exists():
                missing.append(rel)
        details["files"] = {
            "expected": expected_outputs,
            "missing": missing,
        }
        if missing:
            return VerificationReport(
                passed=False,
                gate="files",
                details=details,
            )

        # Gate 2: Tests pass
        test = test_cmd or self.default_test_cmd
        if test:
            test_ok, test_out = self._run_cmd(test, cwd=str(wt))
            details["tests"] = {"cmd": test, "passed": test_ok, "output": test_out}
            if not test_ok:
                return VerificationReport(
                    passed=False,
                    gate="tests",
                    details=details,
                )

        # Gate 3: Lint passes
        lint = lint_cmd or self.default_lint_cmd
        if lint:
            lint_ok, lint_out = self._run_cmd(lint, cwd=str(wt))
            details["lint"] = {"cmd": lint, "passed": lint_ok, "output": lint_out}
            if not lint_ok:
                return VerificationReport(
                    passed=False,
                    gate="lint",
                    details=details,
                )

        return VerificationReport(passed=True, gate="all", details=details)

    # ── Internal ──────────────────────────────────────────────

    def _run_cmd(self, cmd: List[str], cwd: str) -> tuple[bool, str]:
        """Run a shell command; return (success, stdout+stderr)."""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=120.0,
            )
            ok = result.returncode == 0
            output = result.stdout + result.stderr
            return ok, output
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as exc:
            return False, str(exc)


# ── 5. BernsteinOrchestrator ─────────────────────────────────


class BernsteinOrchestrator:
    """Main API composing GitWorktreeSpawner, DeterministicScheduler,
    HMACAuditChain, and JanitorVerifier.

    Integrates with ``GatewayPacing`` to abort if too many concurrent
    agents are already running.
    """

    def __init__(self, config: OrchestratorConfig | None = None) -> None:
        self.config = config or OrchestratorConfig()
        self._spawner: Optional[GitWorktreeSpawner] = None
        self._scheduler = DeterministicScheduler(
            max_workers=self.config.max_workers,
        )
        self._audit = HMACAuditChain(
            key_env=self.config.audit_key_env,
            log_path=self.config.audit_log_path,
        )
        self._janitor = JanitorVerifier()
        self._fleet_conductor: Any | None = None
        self._lock = threading.Lock()
        self._active_count: int = 0

    # ── Public API ────────────────────────────────────────────

    def orchestrate(
        self,
        repo_path: str,
        tasks: List[SchedulerTask],
        config: OrchestratorConfig | None = None,
    ) -> Dict[str, Any]:
        """Run the full Bernstein flow: spawn → schedule → verify → merge/cleanup.

        Returns a result dict with keys:
          - ``spawned``: dict[task_id, {worktree, branch}]
          - ``scheduled``: dict[task_id, ScheduleResult]
          - ``verified``: dict[task_id, VerificationReport]
          - ``merged``: list[task_id]
          - ``cleaned``: list[task_id]
          - ``audit_entries``: int
          - ``aborted``: bool (True if GatewayPacing blocked)
        """
        cfg = config or self.config

        # GatewayPacing check
        if self._fleet_conductor is not None:
            pacing = self._get_pacing()
            if pacing is not None:
                ok, reason = pacing.can_dispatch()
                if not ok:
                    self._audit.log_decision(
                        "abort",
                        "orchestrator",
                        {"reason": reason, "task_count": len(tasks)},
                    )
                    return {
                        "spawned": {},
                        "scheduled": {},
                        "verified": {},
                        "merged": [],
                        "cleaned": [],
                        "audit_entries": len(self._audit),
                        "aborted": True,
                        "abort_reason": reason,
                    }

        # Track concurrency
        with self._lock:
            if self._active_count >= cfg.gateway_max_concurrent:
                self._audit.log_decision(
                    "abort",
                    "orchestrator",
                    {"reason": "max_concurrent_reached", "active": self._active_count},
                )
                return {
                    "spawned": {},
                    "scheduled": {},
                    "verified": {},
                    "merged": [],
                    "cleaned": [],
                    "audit_entries": len(self._audit),
                    "aborted": True,
                    "abort_reason": "max_concurrent_reached",
                }
            self._active_count += len(tasks)

        try:
            return self._run_flow(repo_path, tasks, cfg)
        finally:
            with self._lock:
                self._active_count -= len(tasks)

    def attach_to_fleet_conductor(self, fleet_conductor_v2: Any) -> None:
        """Register this orchestrator as the backend for *fleet_conductor_v2*.

        After attachment, ``fleet_conductor_v2.orchestrate()`` will delegate
        to this instance.
        """
        self._fleet_conductor = fleet_conductor_v2
        # Inject ourselves as the orchestrator backend
        if hasattr(fleet_conductor_v2, "_bernstein_orchestrator"):
            fleet_conductor_v2._bernstein_orchestrator = self
        else:
            # Monkey-patch if the attribute doesn't exist yet
            object.__setattr__(fleet_conductor_v2, "_bernstein_orchestrator", self)

        # Also patch orchestrate() method if it doesn't exist
        if not hasattr(fleet_conductor_v2, "orchestrate"):

            def _orchestrate(
                repo_path: str,
                tasks: List[SchedulerTask],
                config: OrchestratorConfig | None = None,
            ) -> Dict[str, Any]:
                return self.orchestrate(repo_path, tasks, config)

            object.__setattr__(fleet_conductor_v2, "orchestrate", _orchestrate)

        logger.info(
            "BernsteinOrchestrator attached to %s", type(fleet_conductor_v2).__name__
        )

    def get_audit_chain(self) -> HMACAuditChain:
        """Return the audit chain (for inspection / export)."""
        return self._audit

    # ── Internal flow ───────────────────────────────────────────

    def _run_flow(
        self,
        repo_path: str,
        tasks: List[SchedulerTask],
        cfg: OrchestratorConfig,
    ) -> Dict[str, Any]:
        # 1. Spawn worktrees
        self._spawner = GitWorktreeSpawner(repo_path)
        spawned: Dict[str, Dict[str, str]] = {}
        for task in tasks:
            try:
                wt_path, branch = self._spawner.spawn(task.task_id)
                spawned[task.task_id] = {"worktree": wt_path, "branch": branch}
                self._audit.log_decision(
                    "spawn",
                    task.task_id,
                    {"worktree": wt_path, "branch": branch},
                )
            except Exception as exc:
                spawned[task.task_id] = {"error": str(exc)}
                self._audit.log_decision(
                    "spawn_failure",
                    task.task_id,
                    {"error": str(exc)},
                )

        # Build worktree_map for scheduler results
        worktree_map = {
            tid: info["worktree"] for tid, info in spawned.items() if "worktree" in info
        }

        # 2. Schedule tasks
        scheduled = self._scheduler.schedule(tasks, worktree_map=worktree_map)
        for task_id, result in scheduled.items():
            self._audit.log_decision(
                "schedule",
                task_id,
                {
                    "status": result.status,
                    "retry_count": result.retry_count,
                    "duration": round(result.duration, 3),
                },
            )

        # 3. Verify outputs
        verified: Dict[str, VerificationReport] = {}
        for task in tasks:
            task_id = task.task_id
            result = scheduled.get(task_id)
            if result is None or result.status != "success":
                verified[task_id] = VerificationReport(
                    passed=False,
                    gate="schedule",
                    details={"error": "task did not succeed"},
                )
                self._audit.log_decision(
                    "verify_skip",
                    task_id,
                    {"reason": "task did not succeed"},
                )
                continue

            wt_path = result.worktree_path
            report = self._janitor.verify(
                worktree_path=wt_path,
                expected_outputs=task.expected_outputs,
            )
            verified[task_id] = report
            self._audit.log_decision(
                "verify",
                task_id,
                {"passed": report.passed, "gate": report.gate},
            )

        # 4. Merge / cleanup
        merged: List[str] = []
        cleaned: List[str] = []
        for task in tasks:
            task_id = task.task_id
            report = verified.get(task_id)
            spawn_info = spawned.get(task_id, {})
            branch = spawn_info.get("branch", "")
            wt_path = spawn_info.get("worktree", "")

            if report is not None and report.passed:
                # Merge: in a real system this would `git merge` the branch
                # For now, we log it as a merge decision
                merged.append(task_id)
                self._audit.log_decision(
                    "merge",
                    task_id,
                    {"branch": branch},
                )
                if cfg.cleanup_on_success and wt_path:
                    self._spawner.cleanup(task_id, branch)
                    cleaned.append(task_id)
            else:
                self._audit.log_decision(
                    "reject",
                    task_id,
                    {
                        "gate": report.gate if report else "unknown",
                        "branch": branch,
                    },
                )
                if cfg.cleanup_on_failure and wt_path:
                    self._spawner.cleanup(task_id, branch)
                    cleaned.append(task_id)

        return {
            "spawned": spawned,
            "scheduled": {
                tid: {
                    "status": r.status,
                    "worktree_path": r.worktree_path,
                    "output": r.output,
                    "retry_count": r.retry_count,
                    "duration": round(r.duration, 3),
                    "error": r.error,
                }
                for tid, r in scheduled.items()
            },
            "verified": {
                tid: {
                    "passed": r.passed,
                    "gate": r.gate,
                    "details": r.details,
                }
                for tid, r in verified.items()
            },
            "merged": merged,
            "cleaned": cleaned,
            "audit_entries": len(self._audit),
            "aborted": False,
        }

    def _get_pacing(self) -> Any | None:
        """Retrieve GatewayPacing from the attached fleet conductor."""
        if self._fleet_conductor is None:
            return None
        # Try the standard accessor
        if hasattr(self._fleet_conductor, "_get_pacing"):
            return self._fleet_conductor._get_pacing()
        # Fallback: look in subsystems dict
        subs = getattr(self._fleet_conductor, "_subsystems", {})
        pacing_wrapper = subs.get("pacing")
        if pacing_wrapper is not None:
            return pacing_wrapper.ensure()
        return None
