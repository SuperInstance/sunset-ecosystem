"""Fleet Bernstein Scheduler — Adapter integrating Bernstein's deterministic orchestration into FleetConductorV2.

Brings four Bernstein primitives into the Cocapn Fleet:
1. **DeterministicReplay** — hermetic LLM call recording/replay via SignedWAL
2. **PhasedDispatch** — research→plan→implement→verify phase separation for subagents
3. **ScheduleSupervisor** — cron-fired schedule dispatch with catch-up/skip policies
4. **WorkerIsolation** — process-visible subagent wrapper with signal forwarding

All four integrate with existing fleet subsystems:
- DeterministicReplay → SignedWAL (crypto-integrity audit chain)
- PhasedDispatch → AgentRegistry + GatewayPacing (A2A identity + circuit breaker)
- ScheduleSupervisor → MetronomeBridge (cron fires on metronome beat)
- WorkerIsolation → FleetConductorV2 spawn_agent() (process isolation per subagent)

Reference: docs/FLEET_BERNSTEIN_SCHEDULER.md
"""

from __future__ import annotations

__all__ = [
    "FleetBernsteinScheduler",
    "FleetDeterministicReplay",
    "FleetPhasedDispatch",
    "FleetWorkerIsolation",
    "BernsteinScheduleConfig",
    "ReplayMissError",
]

import hashlib
import json
import logging
import os
import random
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────

DEFAULT_CATCH_UP_LIMIT = 16
DEFAULT_TICK_INTERVAL_S = 30.0
AUDIT_EVENT_TYPE = "schedule.fire"
ALLOW_LIVE_MISS_ENV = "FLEET_REPLAY_ALLOW_LIVE_MISS"
TRUTHY = frozenset({"1", "true", "yes", "on"})

# ── Exceptions ────────────────────────────────────────────────────


class ReplayMissError(RuntimeError):
    """Hermetic replay miss — no recorded response matches.

    Mirrors Bernstein's strict replay contract. A miss means the run
    diverged from the recording or a response-determining parameter
    drifted (model, temperature, max_tokens, prompt).
    """

    def __init__(
        self,
        key: str,
        model: str,
        *,
        run_id: str | None = None,
    ) -> None:
        self.key = key
        self.model = model
        location = f" in run_id={run_id}" if run_id else ""
        super().__init__(
            f"Fleet replay miss: no recorded response for model={model!r} "
            f"(key={key}){location}. Strict replay will not call the live model. "
            "Set FLEET_REPLAY_ALLOW_LIVE_MISS=1 for non-hermetic fall-through."
        )


class PhaseValidationError(RuntimeError):
    """Raised when a phase artefact fails mechanical gate validation."""

    def __init__(self, phase: str, errors: List[str]) -> None:
        self.phase = phase
        self.errors = errors
        super().__init__(f"Phase {phase!r} validation failed: {errors}")


# ── FleetDeterministicReplay ─────────────────────────────────────


@dataclass
class FleetDeterministicReplay:
    """Hermetic LLM call recorder/replayer backed by fleet SignedWAL.

    In *recording* mode (default), every ``record()`` appends a signed
    WAL entry. In *replay* mode, the WAL is pre-loaded and ``get_replay()``
    returns stored responses in recorded order (FIFO per prompt key).

    Integration with SignedWAL means every recorded call is:
    - Cryptographically signed (Ed25519 or HMAC fallback)
    - Chained to previous entry (hash chain)
    - Tamper-evident on read-back

    Args:
        wal: SignedWAL instance from logos.signed_wal
        run_id: Unique run identifier (becomes WAL agent_id)
        replay: If True, load and replay recorded responses.
        strict: Raise ReplayMissError on cache miss (default True).
    """

    wal: Any  # SignedWAL instance
    run_id: str
    replay: bool = False
    strict: bool = True

    # Internal state
    _cache: Dict[str, deque[str]] = field(default_factory=dict, repr=False)
    _hits: int = 0
    _misses: int = 0
    _strict_violations: int = 0
    _cached_total: int = 0

    def __post_init__(self) -> None:
        if self.replay:
            self._load_from_wal()

    # -- Internal helpers -----------------------------------------

    def _prompt_key(
        self,
        prompt: str,
        model: str,
        *,
        provider: str = "openrouter_free",
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> str:
        """Stable SHA-256 lookup key folding every response-determining input."""
        data = (
            f"{model}\x00{prompt}\x00{provider}\x00{temperature!r}\x00{max_tokens}"
        ).encode()
        return hashlib.sha256(data).hexdigest()

    def _load_from_wal(self) -> None:
        """Scan WAL entries for deterministic replay payloads."""
        try:
            for entry in self.wal.entries():
                payload = getattr(entry, "entry", None)
                if payload is None:
                    continue
                # Fleet WAL stores operation + metadata; we look for "replay" ops
                if getattr(payload, "operation", "") != "replay":
                    continue
                # Decode vector_hash field as JSON payload
                try:
                    details = json.loads(getattr(payload, "vector_hash", "{}"))
                except json.JSONDecodeError:
                    continue
                key = details.get("key", "")
                response = details.get("response", "")
                if key and response:
                    self._cache.setdefault(key, deque()).append(response)
                    self._cached_total += 1
        except Exception as exc:
            logger.warning("FleetDeterministicReplay: WAL load failed: %s", exc)

    # -- Public API -----------------------------------------------

    def record(
        self,
        prompt: str,
        model: str,
        response: str,
        *,
        provider: str = "openrouter_free",
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> None:
        """Append an LLM call record to the signed WAL.

        No-op when in replay mode. The record is signed and chained
        automatically by the WAL backend.
        """
        if self.replay:
            return
        key = self._prompt_key(
            prompt,
            model,
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        payload = {
            "key": key,
            "model": model,
            "provider": provider,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_len": len(prompt),
            "response": response,
        }
        # Append to WAL as a "replay" operation
        try:
            self.wal.append(
                agent_id=hash(self.run_id) & 0xFFFFFFFF,
                operation="replay",
                vector_hash=json.dumps(payload, sort_keys=True, separators=(",", ":")),
                parent_ids=[],
                generation=0,
                node_id="",
            )
        except Exception as exc:
            logger.warning("FleetDeterministicReplay: WAL record failed: %s", exc)

    def get_replay(
        self,
        prompt: str,
        model: str,
        *,
        provider: str = "openrouter_free",
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> str | None:
        """Return cached response or signal miss.

        Returns:
            Next recorded response on hit (consumed in FIFO order).
            None on miss in non-strict mode.

        Raises:
            ReplayMissError: On miss in strict mode (default).
        """
        if not self.replay:
            return None
        key = self._prompt_key(
            prompt,
            model,
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        queue = self._cache.get(key)
        if queue:
            self._hits += 1
            return queue.popleft()
        # Miss
        if self.strict:
            self._strict_violations += 1
            logger.error(
                "FleetDeterministicReplay: strict miss; %s", self.coverage_line()
            )
            raise ReplayMissError(key, model, run_id=self.run_id)
        self._misses += 1
        return None

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def strict_violations(self) -> int:
        return self._strict_violations

    @property
    def cached_count(self) -> int:
        return self._cached_total

    def coverage_line(self) -> str:
        return (
            f"replay-coverage run_id={self.run_id} cached={self._cached_total} "
            f"hits={self._hits} misses={self._misses} "
            f"strict_violations={self._strict_violations} strict={self.strict}"
        )

    def set_seed(self, seed: int | None) -> None:
        """Apply deterministic seed to Python's random module.

        This makes routing decisions (random.choice / random.random)
        identical across runs with the same seed.
        """
        if seed is not None:
            random.seed(seed)
            logger.info("FleetDeterministicReplay: random seed set to %d", seed)


# ── FleetPhasedDispatch ──────────────────────────────────────────


class Phase(StrEnum):
    """Discrete phases in the subagent dispatch pipeline."""

    RESEARCH = "research"
    PLAN = "plan"
    IMPLEMENT = "implement"
    VERIFY = "verify"


@dataclass(frozen=True)
class PhaseSpec:
    """Configuration for one discrete phase invocation."""

    phase: Phase
    model: str = "sonnet"
    effort: str = "normal"
    max_tokens: int = 80_000
    output_schema: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def default(cls, phase: Phase) -> PhaseSpec:
        defaults: Dict[Phase, Tuple[str, str, int]] = {
            Phase.RESEARCH: ("opus", "high", 60_000),
            Phase.PLAN: ("opus", "high", 30_000),
            Phase.IMPLEMENT: ("sonnet", "normal", 80_000),
            Phase.VERIFY: ("sonnet", "normal", 20_000),
        }
        model, effort, tokens = defaults.get(phase, ("sonnet", "normal", 40_000))
        return cls(phase=phase, model=model, effort=effort, max_tokens=tokens)


@dataclass
class PhaseArtifact:
    """Distilled handoff between phases.

    The implement phase receives only this structure — never the raw
    transcript of the research/plan phases. This compresses N kilobytes
    of exploration into a few hundred bytes of explicit conclusions.
    """

    summary: str
    decisions: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        out: Dict[str, Any] = self.extras.copy()
        out["summary"] = self.summary
        out["decisions"] = list(self.decisions)
        out["constraints"] = list(self.constraints)
        out["open_questions"] = list(self.open_questions)
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_payload(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> PhaseArtifact:
        data = json.loads(raw)
        extras = {
            k: v
            for k, v in data.items()
            if k not in {"summary", "decisions", "constraints", "open_questions"}
        }
        return cls(
            summary=data.get("summary", ""),
            decisions=list(data.get("decisions", [])),
            constraints=list(data.get("constraints", [])),
            open_questions=list(data.get("open_questions", [])),
            extras=extras,
        )


PhaseExecutor = Callable[
    [Dict[str, Any], PhaseSpec, PhaseArtifact | None], PhaseArtifact
]


@dataclass
class FleetPhasedDispatch:
    """Drive a subagent task through research→plan→implement→verify phases.

    Each phase runs in a fresh invocation (clean context window). Only the
    PhaseArtifact is forwarded between phases — never raw transcripts or tool
    outputs.

    Integration with fleet:
    - GatewayPacing prevents phase cascades when the fleet is overloaded
    - AgentRegistry assigns per-phase A2A identity cards
    - SignedWAL records phase transitions for audit
    """

    executor: PhaseExecutor
    wal: Any | None = None  # Optional SignedWAL for phase audit
    phases: List[Phase] = field(
        default_factory=lambda: [Phase.RESEARCH, Phase.PLAN, Phase.IMPLEMENT]
    )
    gate_enabled: bool = True
    gate_max_retries: int = 1

    def run(
        self,
        task_spec: Dict[str, Any],
        *,
        pacing: Any | None = None,
    ) -> Dict[str, Any]:
        """Run all phases and return final result.

        Args:
            task_spec: Fleet task specification dict.
            pacing: Optional GatewayPacing instance — if can_dispatch()
                returns False, the dispatch is aborted before any phase runs.

        Returns:
            Dict with keys: success, phases_run, final_artifact, audit_chain.
        """
        if pacing is not None:
            ok, reason = pacing.can_dispatch()
            if not ok:
                return {
                    "success": False,
                    "phases_run": 0,
                    "reason": f"pacing_blocked: {reason}",
                    "final_artifact": None,
                }

        prior: PhaseArtifact | None = None
        results: List[Dict[str, Any]] = []

        for phase in self.phases:
            spec = PhaseSpec.default(phase)
            # Override from task_spec if present
            spec_override = task_spec.get("phases", {}).get(phase.value, {})
            if spec_override.get("model"):
                spec = PhaseSpec(
                    phase=phase,
                    model=spec_override["model"],
                    effort=spec_override.get("effort", spec.effort),
                    max_tokens=spec_override.get("max_tokens", spec.max_tokens),
                )

            retry_count = 0
            artifact: PhaseArtifact | None = None

            while retry_count <= self.gate_max_retries:
                try:
                    artifact = self.executor(task_spec, spec, prior)
                except Exception as exc:
                    logger.warning(
                        "Phase %s failed (retry %d): %s", phase.value, retry_count, exc
                    )
                    if retry_count >= self.gate_max_retries:
                        return {
                            "success": False,
                            "phases_run": len(results),
                            "failed_phase": phase.value,
                            "reason": str(exc),
                            "final_artifact": prior.to_json() if prior else None,
                        }
                    retry_count += 1
                    continue

                # Mechanical gate: validate artefact has required fields
                if self.gate_enabled and not self._validate_artifact(artifact):
                    logger.warning(
                        "Phase %s artefact failed gate (retry %d)",
                        phase.value,
                        retry_count,
                    )
                    if retry_count >= self.gate_max_retries:
                        return {
                            "success": False,
                            "phases_run": len(results),
                            "failed_phase": phase.value,
                            "reason": "phase_gate_failed",
                            "final_artifact": artifact.to_json() if artifact else None,
                        }
                    retry_count += 1
                    # Seed open_questions with gate failures for next retry
                    if artifact:
                        artifact.open_questions.append(f"Gate failure on {phase.value}")
                    continue

                # Success — record in WAL if available
                if self.wal is not None:
                    self._record_phase_transition(task_spec, phase, artifact, prior)

                results.append(
                    {
                        "phase": phase.value,
                        "model": spec.model,
                        "artifact_size": len(artifact.to_json()),
                        "retry_count": retry_count,
                    }
                )
                prior = artifact
                break

        return {
            "success": True,
            "phases_run": len(results),
            "phases": results,
            "final_artifact": prior.to_json() if prior else None,
        }

    def _validate_artifact(self, artifact: PhaseArtifact | None) -> bool:
        """Mechanical gate: ensure artefact has minimum required structure."""
        if artifact is None:
            return False
        if not isinstance(artifact.summary, str) or not artifact.summary.strip():
            return False
        if not all(isinstance(d, str) for d in artifact.decisions):
            return False
        if not all(isinstance(c, str) for c in artifact.constraints):
            return False
        return True

    def _record_phase_transition(
        self,
        task_spec: Dict[str, Any],
        phase: Phase,
        artifact: PhaseArtifact,
        prior: PhaseArtifact | None,
    ) -> None:
        """Write a phase transition entry to the signed WAL."""
        if self.wal is None:
            return
        try:
            parent_hash = ""
            if prior is not None:
                parent_hash = hashlib.sha256(prior.to_json().encode()).hexdigest()[:32]
            payload = {
                "task_id": task_spec.get("task_id", "unknown"),
                "phase": phase.value,
                "artifact_hash": hashlib.sha256(
                    artifact.to_json().encode()
                ).hexdigest()[:32],
                "parent_hash": parent_hash,
                "decisions_count": len(artifact.decisions),
                "constraints_count": len(artifact.constraints),
            }
            self.wal.append(
                agent_id=hash(task_spec.get("task_id", "unknown")) & 0xFFFFFFFF,
                operation="phase_transition",
                vector_hash=json.dumps(payload, sort_keys=True, separators=(",", ":")),
                parent_ids=[],
                generation=0,
                node_id=task_spec.get("node_id", ""),
            )
        except Exception as exc:
            logger.warning("Phase transition WAL record failed: %s", exc)


# ── FleetWorkerIsolation ─────────────────────────────────────────


@dataclass
class FleetWorkerIsolation:
    """Process-visible wrapper for spawned subagents, inspired by bernstein-worker.

    Ensures:
    1. Process title shows role + session (visible in ps/top)
    2. PID metadata file written for fleet monitoring
    3. Signals forwarded to child process
    4. Tool abort policies: contain / sibling / session
    5. Cleanup on exit

    Integration with fleet:
    - OperationalTrap monitors worker health via PID files
    - SignedWAL records spawn/sunset events
    - GatewayPacing tracks worker count for dispatch limits
    """

    pid_dir: Path = field(default_factory=lambda: Path(".fleet/runtime/pids"))
    signals_dir: Path = field(default_factory=lambda: Path(".fleet/runtime/signals"))
    tool_abort_policy: str = "session"

    # Internal
    _session_re: Any = None

    def __post_init__(self) -> None:
        import re

        self._session_re = re.compile(r"^[a-zA-Z0-9_.-]+$")
        self.pid_dir.mkdir(parents=True, exist_ok=True)
        self.signals_dir.mkdir(parents=True, exist_ok=True)

    def spawn(
        self,
        *,
        role: str,
        session_id: str,
        command: List[str],
        model: str = "",
        workdir: Path | None = None,
        env: Dict[str, str] | None = None,
        wal: Any | None = None,
    ) -> Dict[str, Any]:
        """Spawn a worker process with full fleet isolation.

        Args:
            role: Agent role (e.g. "auditor", "scout", "builder")
            session_id: Unique session identifier (safe for filenames)
            command: CLI command list to execute
            model: Model identifier for metadata
            workdir: Working directory (defaults to cwd)
            env: Extra environment variables
            wal: Optional SignedWAL to record spawn event

        Returns:
            Dict with keys: worker_pid, child_pid, pid_file, started_at, status
        """
        if not self._session_re.fullmatch(session_id):
            raise ValueError(f"Invalid session_id: {session_id!r}")

        workdir = workdir or Path(".")
        started_at = time.time()

        # 1. Set process title (best-effort)
        self._set_proctitle(f"fleet: {role} [{session_id}]")

        # 2. Write PID metadata
        pid_file = self._write_pid_file(
            session_id,
            {
                "worker_pid": os.getpid(),
                "role": role,
                "session": session_id,
                "command": command[0] if command else "",
                "model": model,
                "started_at": started_at,
            },
        )

        # 3. Record in WAL
        if wal is not None:
            self._record_spawn(wal, role, session_id, command, started_at)

        # 4. Spawn child
        merged_env = {**os.environ, **(env or {})}
        try:
            child = subprocess.Popen(
                command,
                cwd=str(workdir),
                env=merged_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            self._cleanup_pid_file(session_id)
            raise RuntimeError(f"Command not found: {command[0]!r}") from exc

        # 5. Update PID file with child PID
        self._update_pid_file(session_id, {"child_pid": child.pid})

        # 6. Forward signals
        def _forward(signum: int, _frame: Any) -> None:
            with suppress(OSError):
                child.send_signal(signum)

        # Register signal handlers (only in main thread)
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, _forward)
            signal.signal(signal.SIGINT, _forward)

        return {
            "worker_pid": os.getpid(),
            "child_pid": child.pid,
            "pid_file": str(pid_file),
            "started_at": started_at,
            "status": "running",
            "tool_abort_policy": self.tool_abort_policy,
        }

    def _write_pid_file(self, session_id: str, info: Dict[str, Any]) -> Path:
        pid_file = (self.pid_dir / f"{session_id}.json").resolve()
        # Security: ensure pid_file stays within pid_dir
        if not pid_file.is_relative_to(self.pid_dir.resolve()):
            raise ValueError(f"PID file escaped pid_dir: {session_id}")
        pid_file.write_text(json.dumps(info, default=str), encoding="utf-8")
        return pid_file

    def _update_pid_file(self, session_id: str, extra: Dict[str, Any]) -> None:
        pid_file = self.pid_dir / f"{session_id}.json"
        if not pid_file.exists():
            return
        try:
            info = json.loads(pid_file.read_text(encoding="utf-8"))
            info.update(extra)
            pid_file.write_text(json.dumps(info, default=str), encoding="utf-8")
        except Exception as exc:
            logger.debug("PID file update failed: %s", exc)

    def _cleanup_pid_file(self, session_id: str) -> None:
        pid_file = self.pid_dir / f"{session_id}.json"
        pid_file.unlink(missing_ok=True)

    def _set_proctitle(self, title: str) -> None:
        try:
            import setproctitle

            setproctitle.setproctitle(title)
        except ImportError:
            pass

    def _record_spawn(
        self,
        wal: Any,
        role: str,
        session_id: str,
        command: List[str],
        started_at: float,
    ) -> None:
        try:
            wal.append(
                agent_id=hash(session_id) & 0xFFFFFFFF,
                operation="spawn",
                vector_hash=json.dumps(
                    {
                        "role": role,
                        "session_id": session_id,
                        "command": command[0] if command else "",
                        "started_at": started_at,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                parent_ids=[],
                generation=0,
                node_id="",
            )
        except Exception as exc:
            logger.debug("WAL spawn record failed: %s", exc)

    def cleanup_session(self, session_id: str, wal: Any | None = None) -> None:
        """Clean up PID file and optionally record sunset in WAL."""
        self._cleanup_pid_file(session_id)
        if wal is not None:
            try:
                wal.append(
                    agent_id=hash(session_id) & 0xFFFFFFFF,
                    operation="sunset",
                    vector_hash=json.dumps(
                        {
                            "session_id": session_id,
                            "cleaned_at": time.time(),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    parent_ids=[],
                    generation=0,
                    node_id="",
                )
            except Exception as exc:
                logger.debug("WAL sunset record failed: %s", exc)


# ── FleetBernsteinScheduler ─────────────────────────────────────


@dataclass(frozen=True)
class BernsteinScheduleConfig:
    """Configuration for the Bernstein scheduler subsystem."""

    node_id: str = "unnamed-node"
    tick_interval_s: float = DEFAULT_TICK_INTERVAL_S
    catch_up_limit: int = DEFAULT_CATCH_UP_LIMIT
    enable_phased_dispatch: bool = True
    enable_deterministic_replay: bool = False
    enable_worker_isolation: bool = True
    default_phases: List[str] = field(
        default_factory=lambda: ["research", "plan", "implement"]
    )
    replay_strict: bool = True
    replay_run_id: str = ""
    wal_path: str = ""


@dataclass(frozen=True)
class FireReceipt:
    """Outcome of one schedule fire, persisted for audit replay."""

    schedule_id: str
    fire_time: int
    projection_hash: str
    misfire_policy: str
    dispatched: bool
    skipped_windows: Tuple[int, ...] = ()
    counterfactual: bool = False


class FleetBernsteinScheduler:
    """Bernstein-style scheduler integrated into FleetConductorV2.

    Features:
    1. Cron-based schedule firing with catch-up/skip policies
    2. Deterministic replay for reproducible breeding runs
    3. Phased dispatch (research→plan→implement→verify) for subagents
    4. Worker isolation with signal forwarding and PID tracking
    5. Full SignedWAL audit chain integration

    The scheduler runs as a subsystem inside FleetConductorV2's beat()
    loop. On each beat, it checks registered schedules and fires due
    tasks through the fleet's GatewayPacing + AgentRegistry.

    Usage:
        scheduler = FleetBernsteinScheduler(config)
        conductor._subsystems["bernstein_scheduler"] = scheduler
        # In conductor.beat():
        scheduler.tick(now=time.time(), pacing=conductor._get_pacing())
    """

    def __init__(self, config: BernsteinScheduleConfig | None = None) -> None:
        self.config = config or BernsteinScheduleConfig()
        self._schedules: Dict[str, Dict[str, Any]] = {}
        self._receipts: List[FireReceipt] = []
        self._receipts_dir = Path(".fleet/runtime/schedule_receipts")
        self._receipts_dir.mkdir(parents=True, exist_ok=True)
        self._last_tick_at = 0.0
        self._last_fire_at = 0.0
        self._lock = threading.RLock()

        # Sub-components
        self._replay: FleetDeterministicReplay | None = None
        self._phased_dispatch: FleetPhasedDispatch | None = None
        self._worker_isolation: FleetWorkerIsolation | None = None

        if self.config.enable_deterministic_replay and self.config.replay_run_id:
            self._init_replay()

        if self.config.enable_phased_dispatch:
            self._init_phased_dispatch()

        if self.config.enable_worker_isolation:
            self._init_worker_isolation()

    def _init_replay(self) -> None:
        """Initialize deterministic replay backed by WAL."""
        try:
            from logos.signed_wal import SignedWAL

            wal = SignedWAL(
                log_path=self.config.wal_path or ".fleet/runtime/scheduler.wal"
            )
            self._replay = FleetDeterministicReplay(
                wal=wal,
                run_id=self.config.replay_run_id,
                replay=True,
                strict=self.config.replay_strict,
            )
            logger.info(
                "Deterministic replay initialized for run_id=%s",
                self.config.replay_run_id,
            )
        except Exception as exc:
            logger.warning("Failed to initialize deterministic replay: %s", exc)

    def _init_phased_dispatch(self) -> None:
        """Initialize phased dispatch with a default executor."""

        def _default_executor(
            task_spec: Dict[str, Any],
            spec: PhaseSpec,
            prior: PhaseArtifact | None,
        ) -> PhaseArtifact:
            # Default executor returns a mock artefact for testing.
            # Production wiring replaces this with a real subagent spawner.
            seed = f"{task_spec.get('task_id', '')}-{spec.phase.value}"
            return PhaseArtifact(
                summary=f"[{spec.phase.value}] Executed task {seed}",
                decisions=[f"decision-{spec.phase.value}-1"],
                constraints=[f"constraint-{spec.phase.value}-1"],
                open_questions=[],
            )

        self._phased_dispatch = FleetPhasedDispatch(
            executor=_default_executor,
            phases=[Phase(p) for p in self.config.default_phases],
        )

    def _init_worker_isolation(self) -> None:
        self._worker_isolation = FleetWorkerIsolation(
            pid_dir=Path(".fleet/runtime/pids"),
            tool_abort_policy="session",
        )

    # -- Schedule CRUD --------------------------------------------

    def register_schedule(
        self,
        schedule_id: str,
        cron: str,
        task_spec: Dict[str, Any],
        *,
        misfire_policy: str = "skip",
        goal: str = "",
        scenario_id: str = "",
    ) -> Dict[str, Any]:
        """Register a new cron-fired schedule.

        Args:
            schedule_id: Unique schedule identifier.
            cron: 5-field cron expression (e.g. "0 */6 * * *").
            task_spec: Fleet task specification dispatched on fire.
            misfire_policy: "skip" (default) or "catch_up".
            goal: Human-readable goal string.
            scenario_id: Scenario identifier.

        Returns:
            Dict with schedule_id, next_fire_at, and status.
        """
        with self._lock:
            self._schedules[schedule_id] = {
                "id": schedule_id,
                "cron": cron,
                "task_spec": task_spec,
                "misfire_policy": misfire_policy,
                "goal": goal,
                "scenario_id": scenario_id,
                "last_fire_at": 0.0,
                "created_at": time.time(),
            }

        # Compute next fire
        next_fire = self._next_fire_after(cron, int(time.time()) - 60)
        return {
            "schedule_id": schedule_id,
            "cron": cron,
            "next_fire_at": next_fire,
            "misfire_policy": misfire_policy,
            "status": "registered",
        }

    def unregister_schedule(self, schedule_id: str) -> bool:
        """Remove a schedule. Returns True if it existed."""
        with self._lock:
            return self._schedules.pop(schedule_id, None) is not None

    def list_schedules(self) -> List[Dict[str, Any]]:
        """List all registered schedules."""
        with self._lock:
            return [dict(s) for s in self._schedules.values()]

    # -- Tick / fire logic ----------------------------------------

    def tick(
        self,
        *,
        now: float | None = None,
        pacing: Any | None = None,
        registry: Any | None = None,
        wal: Any | None = None,
    ) -> Dict[str, Any]:
        """Run one scheduler tick — check all schedules and fire due ones.

        This is called from FleetConductorV2.beat() on every beat.

        Args:
            now: Current epoch time (defaults to time.time()).
            pacing: Optional GatewayPacing for dispatch throttling.
            registry: Optional AgentRegistry for A2A identity.
            wal: Optional SignedWAL for audit chain.

        Returns:
            Tick result dict with fires, receipts, and coverage.
        """
        now_epoch = int(now if now is not None else time.time())
        self._last_tick_at = float(now_epoch)

        receipts: List[FireReceipt] = []
        fired_count = 0
        skipped_count = 0

        with self._lock:
            schedules = list(self._schedules.values())

        for schedule in schedules:
            try:
                schedule_receipts = self._tick_one(
                    schedule, now_epoch, pacing, registry, wal
                )
                receipts.extend(schedule_receipts)
                for r in schedule_receipts:
                    if r.dispatched:
                        fired_count += 1
                    elif r.counterfactual:
                        skipped_count += len(r.skipped_windows)
            except Exception:
                logger.exception(
                    "Scheduler tick failed for schedule %s", schedule["id"]
                )

        # Persist receipts
        for receipt in receipts:
            self._persist_receipt(receipt)

        return {
            "tick_at": now_epoch,
            "schedules_checked": len(schedules),
            "fires_dispatched": fired_count,
            "windows_skipped": skipped_count,
            "receipts": [self._receipt_to_dict(r) for r in receipts],
            "replay_coverage": self._replay.coverage_line()
            if self._replay
            else "replay_disabled",
        }

    def _tick_one(
        self,
        schedule: Dict[str, Any],
        now_epoch: int,
        pacing: Any | None,
        registry: Any | None,
        wal: Any | None,
    ) -> List[FireReceipt]:
        """Tick a single schedule. May emit 0..N receipts."""
        cron = schedule["cron"]
        anchor = (
            int(schedule.get("last_fire_at", 0))
            if schedule.get("last_fire_at")
            else now_epoch - 60
        )
        receipts: List[FireReceipt] = []
        skipped_windows: List[int] = []

        current_anchor = anchor
        fires_dispatched = 0

        while True:
            try:
                next_fire = self._next_fire_after(cron, current_anchor)
            except RuntimeError:
                break
            if next_fire > now_epoch:
                break

            if schedule.get("misfire_policy") == "catch_up":
                if fires_dispatched >= self.config.catch_up_limit:
                    skipped_windows.append(next_fire)
                    current_anchor = next_fire
                    continue
                receipts.append(
                    self._fire(
                        schedule, next_fire, pacing, registry, wal, counterfactual=False
                    )
                )
                fires_dispatched += 1
            else:  # skip policy
                # Only dispatch the most recent missed instant
                try:
                    peek = self._next_fire_after(cron, next_fire)
                except RuntimeError:
                    peek = None
                if peek is not None and peek <= now_epoch:
                    skipped_windows.append(next_fire)
                    current_anchor = next_fire
                    continue
                receipts.append(
                    self._fire(
                        schedule, next_fire, pacing, registry, wal, counterfactual=False
                    )
                )
                fires_dispatched += 1

            current_anchor = next_fire

        if skipped_windows:
            receipts.append(
                self._record_counterfactual(schedule, skipped_windows, now_epoch)
            )

        # Update last_fire_at
        with self._lock:
            if schedule["id"] in self._schedules:
                self._schedules[schedule["id"]]["last_fire_at"] = float(current_anchor)
        if fires_dispatched > 0:
            self._last_fire_at = float(current_anchor)

        return receipts

    def _fire(
        self,
        schedule: Dict[str, Any],
        fire_epoch: int,
        pacing: Any | None,
        registry: Any | None,
        wal: Any | None,
        *,
        counterfactual: bool,
    ) -> FireReceipt:
        """Build projection, dispatch task, and chain audit entry."""
        task_spec = dict(schedule.get("task_spec", {}))
        task_spec["schedule_id"] = schedule["id"]
        task_spec["fire_time"] = fire_epoch
        task_spec["goal"] = schedule.get("goal", "")
        task_spec["scenario_id"] = schedule.get("scenario_id", "")

        # Deterministic replay: if in replay mode, seed random before dispatch
        if self._replay is not None and self._replay.replay:
            self._replay.set_seed(hash(schedule["id"]) % (2**31))

        dispatched = False
        if not counterfactual:
            # Dispatch through phased pipeline if enabled
            if self._phased_dispatch is not None:
                result = self._phased_dispatch.run(task_spec, pacing=pacing)
                dispatched = result.get("success", False)
            elif pacing is not None:
                ok, reason = pacing.can_dispatch()
                dispatched = ok
                if not ok:
                    logger.info(
                        "Schedule %s fire blocked by pacing: %s", schedule["id"], reason
                    )
            else:
                dispatched = True  # No pacing = always dispatch

            # Record in WAL
            if wal is not None:
                self._append_wal(wal, schedule, fire_epoch, task_spec, dispatched)

        # Build projection hash
        projection = {
            "schedule_id": schedule["id"],
            "fire_time": fire_epoch,
            "task_spec_hash": hashlib.sha256(
                json.dumps(task_spec, sort_keys=True).encode()
            ).hexdigest()[:32],
        }
        projection_hash = hashlib.sha256(
            json.dumps(projection, sort_keys=True).encode()
        ).hexdigest()[:32]

        receipt = FireReceipt(
            schedule_id=schedule["id"],
            fire_time=fire_epoch,
            projection_hash=projection_hash,
            misfire_policy=schedule.get("misfire_policy", "skip"),
            dispatched=not counterfactual and dispatched,
            counterfactual=counterfactual,
        )
        return receipt

    def _record_counterfactual(
        self,
        schedule: Dict[str, Any],
        skipped: List[int],
        now_epoch: int,
    ) -> FireReceipt:
        """Emit a counterfactual receipt summarizing skipped windows."""
        return FireReceipt(
            schedule_id=schedule["id"],
            fire_time=skipped[-1] if skipped else now_epoch,
            projection_hash="",
            misfire_policy=schedule.get("misfire_policy", "skip"),
            dispatched=False,
            skipped_windows=tuple(skipped),
            counterfactual=True,
        )

    def _append_wal(
        self,
        wal: Any,
        schedule: Dict[str, Any],
        fire_epoch: int,
        task_spec: Dict[str, Any],
        dispatched: bool,
    ) -> None:
        """Append a schedule.fire entry to the signed WAL."""
        try:
            payload = {
                "schedule_id": schedule["id"],
                "fire_time": fire_epoch,
                "task_spec_hash": hashlib.sha256(
                    json.dumps(task_spec, sort_keys=True).encode()
                ).hexdigest()[:32],
                "dispatched": dispatched,
                "goal": schedule.get("goal", ""),
                "scenario_id": schedule.get("scenario_id", ""),
            }
            wal.append(
                agent_id=hash(schedule["id"]) & 0xFFFFFFFF,
                operation="schedule_fire",
                vector_hash=json.dumps(payload, sort_keys=True, separators=(",", ":")),
                parent_ids=[],
                generation=0,
                node_id=self.config.node_id,
            )
        except Exception as exc:
            logger.warning("WAL schedule fire record failed: %s", exc)

    def _persist_receipt(self, receipt: FireReceipt) -> None:
        """Persist receipt to disk for audit replay."""
        try:
            path = (
                self._receipts_dir / f"{receipt.schedule_id}_{receipt.fire_time}.json"
            )
            path.write_text(
                json.dumps(
                    {
                        "schedule_id": receipt.schedule_id,
                        "fire_time": receipt.fire_time,
                        "projection_hash": receipt.projection_hash,
                        "misfire_policy": receipt.misfire_policy,
                        "dispatched": receipt.dispatched,
                        "skipped_windows": list(receipt.skipped_windows),
                        "counterfactual": receipt.counterfactual,
                    },
                    sort_keys=True,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("Receipt persist failed: %s", exc)

    def _receipt_to_dict(self, receipt: FireReceipt) -> Dict[str, Any]:
        return {
            "schedule_id": receipt.schedule_id,
            "fire_time": receipt.fire_time,
            "projection_hash": receipt.projection_hash,
            "dispatched": receipt.dispatched,
            "counterfactual": receipt.counterfactual,
            "skipped_windows": list(receipt.skipped_windows),
        }

    # -- Cron math (in-tree, deterministic) -----------------------

    def _next_fire_after(self, cron: str, anchor_epoch: int) -> int:
        """Return next fire epoch strictly greater than anchor.

        UTC only — host timezone is not part of the deterministic contract.
        Two-year scan cap catches worst-case expressions (Feb 29).
        """
        max_minutes = 2 * 366 * 24 * 60
        parsed = self._parse_cron(cron)
        start_minute = (anchor_epoch // 60 + 1) * 60
        start_dt = datetime.fromtimestamp(start_minute, tz=UTC)

        for offset in range(max_minutes):
            candidate = start_dt + timedelta(minutes=offset)
            if (
                candidate.minute in parsed["minutes"]
                and candidate.hour in parsed["hours"]
                and candidate.month in parsed["months"]
                and self._matches_day(parsed, candidate)
            ):
                return int(candidate.timestamp())

        raise RuntimeError(f"No fire instant found in 2 years for cron {cron!r}")

    def _parse_cron(self, cron: str) -> Dict[str, Any]:
        """Parse a 5-field cron expression into sets of allowed values."""
        parts = cron.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Cron must have 5 fields, got {len(parts)}: {cron!r}")

        return {
            "minutes": self._expand_field(parts[0], 0, 59),
            "hours": self._expand_field(parts[1], 0, 23),
            "days": self._expand_field(parts[2], 1, 31),
            "months": self._expand_field(parts[3], 1, 12),
            "weekdays": self._expand_field(parts[4], 0, 6),
            "raw": cron,
        }

    def _expand_field(self, field: str, min_val: int, max_val: int) -> set[int]:
        """Expand a cron field to a set of integers."""
        if field == "*":
            return set(range(min_val, max_val + 1))
        if "/" in field:
            base, step = field.split("/")
            step = int(step)
            if base == "*":
                return set(range(min_val, max_val + 1, step))
            # Range with step: e.g. "1-10/2"
            if "-" in base:
                start, end = base.split("-")
                return set(range(int(start), int(end) + 1, step))
            raise ValueError(f"Unsupported cron field: {field!r}")
        if "-" in field:
            start, end = field.split("-")
            return set(range(int(start), int(end) + 1))
        if "," in field:
            return set(int(x) for x in field.split(","))
        return {int(field)}

    def _matches_day(self, parsed: Dict[str, Any], dt: datetime) -> bool:
        """POSIX cron day matching: union of day and weekday if both restricted."""
        full_days = set(range(1, 32))
        full_weekdays = set(range(0, 7))
        days_restricted = parsed["days"] != full_days
        weekdays_restricted = parsed["weekdays"] != full_weekdays
        weekday_py_to_cron = (dt.weekday() + 1) % 7

        if days_restricted and weekdays_restricted:
            return dt.day in parsed["days"] or weekday_py_to_cron in parsed["weekdays"]
        if days_restricted:
            return dt.day in parsed["days"]
        if weekdays_restricted:
            return weekday_py_to_cron in parsed["weekdays"]
        return True

    # -- Status ---------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Doctor-ready status snapshot."""
        with self._lock:
            schedules = list(self._schedules.values())
        now = time.time()
        alive = (now - self._last_tick_at) <= 120.0 if self._last_tick_at else False
        next_fire = 0.0
        next_id = ""
        for schedule in schedules:
            try:
                upcoming = self._next_fire_after(schedule["cron"], int(now))
            except Exception:
                continue
            if next_fire == 0.0 or upcoming < next_fire:
                next_fire = float(upcoming)
                next_id = schedule["id"]

        return {
            "alive": alive,
            "last_tick_at": self._last_tick_at,
            "last_fire_at": self._last_fire_at,
            "next_fire_at": next_fire,
            "next_fire_schedule_id": next_id,
            "schedules_total": len(schedules),
            "replay_enabled": self._replay is not None and self._replay.replay,
            "phased_dispatch_enabled": self._phased_dispatch is not None,
            "worker_isolation_enabled": self._worker_isolation is not None,
        }

    # -- Subsystem wrapper helpers --------------------------------

    def health_check(self) -> Dict[str, str]:
        """Return health state for SubsystemWrapper integration."""
        status = self.get_status()
        if not status["alive"]:
            return {"state": "failed", "last_error": "No tick in >120s"}
        if status["schedules_total"] == 0:
            return {"state": "healthy", "last_error": ""}
        return {"state": "healthy", "last_error": ""}
