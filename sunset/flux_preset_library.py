"""FLUX Preset Library — reusable constraint presets for breeding decisions.

Rather than hand-crafting constraints for every breeding run, the fleet
maintains a library of presets. Each preset is a named bundle of
constraint callables, metadata, and required opcodes. Presets are
validated against the OpcodeCapabilityIndex so that only PYTHON_SAFE
opcodes are used from Python.

Architecture
------------
``FluxPreset`` — a single preset (name, description, category,
constraints, required opcodes, python_safe flag).

``FluxPresetLibrary`` — the registry. Loads all presets, provides
lookup, filtering, suggestion, and application.

Integration
-----------
* ``BreederDaemonV2.attach_flux_gating()`` calls
  ``FluxPresetLibrary.suggest_preset_for_task()`` to pick a preset.
* ``OperationalTrap`` uses ``ThermalCeiling`` and ``AgentLiveness``
  presets for fleet health gating.
"""

from __future__ import annotations

__all__ = [
    "FluxPreset",
    "FluxPresetLibrary",
    "PRESET_CATEGORIES",
]

import hashlib
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from logos.opcode_capability_index import OpcodeCapabilityIndex, OpcodeStatus


# ── categories ────────────────────────────────────────────


class PresetCategory(Enum):
    """Taxonomy for preset families."""

    BREEDING = auto()
    THERMAL = auto()
    LIVENESS = auto()
    CRYPTO = auto()
    SYNC = auto()
    BATCHING = auto()
    MEMORY = auto()
    DIVERSITY = auto()


PRESET_CATEGORIES = {c.name.lower(): c for c in PresetCategory}


# ── data structures ───────────────────────────────────────


@dataclass(frozen=True)
class FluxPreset:
    """A reusable FLUX constraint preset.

    Args:
        name: Unique preset identifier (PascalCase).
        description: Human-readable summary.
        category: PresetCategory enum value.
        constraints: List of callable(ctx: dict) -> dict. Each callable
            receives a context dict and returns a result dict with at
            least ``passed`` (bool) and ``severity`` (str) keys.
        required_opcodes: List of opcode names used by this preset.
        python_safe: ``True`` iff every required opcode is
            PYTHON_SAFE in the canonical index.
    """

    name: str
    description: str
    category: PresetCategory
    constraints: Tuple[Callable[[Dict[str, Any]], Dict[str, Any]], ...] = field(
        repr=False,
    )
    required_opcodes: Tuple[str, ...]
    python_safe: bool = True


# ── constraint callables (PYTHON_SAFE only) ───────────────

# These are the raw constraint functions that presets compose.
# Every function is pure-Python and uses only opcodes that are
# PYTHON_SAFE in the canonical OpcodeCapabilityIndex.


def _weight_bounds_check(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """RangeCheck + Validate — weight norm must be within bounds."""
    weights = ctx.get("weights")
    bounds = ctx.get("weight_bounds", (0.0, 10.0))
    if weights is None:
        return {"passed": True, "severity": "info", "detail": "no weights"}
    try:
        import numpy as np

        w_norm = float(np.linalg.norm(weights))
    except ImportError:
        w_norm = float(weights) if isinstance(weights, (int, float)) else 0.0
    w_min, w_max = bounds
    passed = w_min <= w_norm <= w_max
    return {
        "passed": passed,
        "severity": "critical" if not passed else "info",
        "detail": {"norm": w_norm, "bounds": [w_min, w_max]},
        "opcode_trace": ["RangeCheck", "Validate"],
    }


def _chaos_limit_check(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """RangeCheck + ClassifySeverity — chaos must be below limit."""
    chaos = ctx.get("chaos")
    limit = ctx.get("chaos_limit", 1.0)
    if chaos is None:
        return {"passed": True, "severity": "info", "detail": "no chaos"}
    passed = chaos <= limit
    return {
        "passed": passed,
        "severity": "warning" if not passed else "info",
        "detail": {"chaos": chaos, "limit": limit},
        "opcode_trace": ["RangeCheck", "ClassifySeverity"],
    }


def _thermal_budget_check(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """RangeCheck + ClassifySeverity — fleet thermal headroom."""
    thermal = ctx.get("thermal_headroom")
    limit = ctx.get("thermal_budget_limit", 0.95)
    if thermal is None:
        return {"passed": True, "severity": "info", "detail": "no thermal"}
    passed = thermal <= limit
    return {
        "passed": passed,
        "severity": "warning" if not passed else "info",
        "detail": {"thermal_headroom": thermal, "limit": limit},
        "opcode_trace": ["RangeCheck", "ClassifySeverity"],
    }


def _batch_size_limit_check(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Saturate + Validate — batch size must not exceed ceiling."""
    batch_size = ctx.get("batch_size", 0)
    max_batch = ctx.get("max_batch_size", 64)
    passed = batch_size <= max_batch
    return {
        "passed": passed,
        "severity": "critical" if not passed else "info",
        "detail": {"batch_size": batch_size, "max_batch_size": max_batch},
        "opcode_trace": ["Saturate", "Validate"],
    }


def _rate_limit_check(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Min / Max / Sub — requests per second within window."""
    rps = ctx.get("requests_per_second", 0.0)
    max_rps = ctx.get("max_rps", 1000.0)
    passed = rps <= max_rps
    return {
        "passed": passed,
        "severity": "warning" if not passed else "info",
        "detail": {"rps": rps, "max_rps": max_rps},
        "opcode_trace": ["Min", "Max", "Sub"],
    }


def _memory_budget_check(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """RangeCheck — per-agent memory must be under cap."""
    memory_mb = ctx.get("memory_mb", 0)
    cap_mb = ctx.get("memory_cap_mb", 1024)
    passed = memory_mb <= cap_mb
    return {
        "passed": passed,
        "severity": "critical" if not passed else "info",
        "detail": {"memory_mb": memory_mb, "cap_mb": cap_mb},
        "opcode_trace": ["RangeCheck"],
    }


def _diversity_floor_check(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Min / Abs — population diversity must stay above floor."""
    diversity = ctx.get("diversity_score", 1.0)
    floor = ctx.get("diversity_floor", 0.1)
    passed = diversity >= floor
    return {
        "passed": passed,
        "severity": "warning" if not passed else "info",
        "detail": {"diversity": diversity, "floor": floor},
        "opcode_trace": ["Min", "Abs"],
    }


def _thermal_ceiling_hard_check(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Validate — hard ceiling; any breach is critical."""
    thermal = ctx.get("thermal_headroom")
    limit = ctx.get("thermal_ceiling", 0.99)
    if thermal is None:
        return {"passed": True, "severity": "info", "detail": "no thermal"}
    passed = thermal < limit  # strict < for hard ceiling
    return {
        "passed": passed,
        "severity": "critical" if not passed else "info",
        "detail": {"thermal_headroom": thermal, "ceiling": limit},
        "opcode_trace": ["Validate"],
    }


def _heartbeat_timeout_check(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Sub / Validate — last heartbeat must be within timeout."""
    last_beat = ctx.get("last_heartbeat", 0.0)
    timeout = ctx.get("heartbeat_timeout_seconds", 30.0)
    now = ctx.get("now", time.time())
    elapsed = now - last_beat
    passed = elapsed <= timeout
    return {
        "passed": passed,
        "severity": "critical" if not passed else "info",
        "detail": {"elapsed": elapsed, "timeout": timeout},
        "opcode_trace": ["Sub", "Validate"],
    }


def _crash_detection_check(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """EmitEvent — flag if consecutive failures exceed threshold."""
    failures = ctx.get("consecutive_failures", 0)
    threshold = ctx.get("crash_threshold", 3)
    passed = failures < threshold
    return {
        "passed": passed,
        "severity": "critical" if not passed else "info",
        "detail": {"failures": failures, "threshold": threshold},
        "opcode_trace": ["EmitEvent"],
    }


def _mesh_gossip_consistency_check(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Add / Sub / Validate — compare local vs gossiped vector hash."""
    local_hash = ctx.get("local_hash")
    gossip_hash = ctx.get("gossip_hash")
    if local_hash is None or gossip_hash is None:
        return {"passed": True, "severity": "info", "detail": "missing hash"}
    passed = local_hash == gossip_hash
    return {
        "passed": passed,
        "severity": "warning" if not passed else "info",
        "detail": {
            "local": local_hash[:8] if isinstance(local_hash, str) else local_hash,
            "gossip": gossip_hash[:8] if isinstance(gossip_hash, str) else gossip_hash,
        },
        "opcode_trace": ["Sub", "Validate"],
    }


def _signature_verification_check(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Pure-Python signature verification (no Rust Prove opcode).

    Uses hashlib SHA-256 as a PYTHON_SAFE stand-in for HashCommit.
    In production with the Rust FFI, this preset would upgrade to the
    real Prove + HashCommit opcodes.
    """
    payload = ctx.get("payload")
    signature = ctx.get("signature")
    if payload is None or signature is None:
        return {"passed": True, "severity": "info", "detail": "no payload"}
    expected = hashlib.sha256(str(payload).encode()).hexdigest()[:16]
    passed = signature == expected
    return {
        "passed": passed,
        "severity": "critical" if not passed else "info",
        "detail": {
            "expected_prefix": expected,
            "received_prefix": signature[:16]
            if isinstance(signature, str)
            else signature,
        },
        "opcode_trace": ["EmitEvent"],  # PYTHON_SAFE logging only
    }


def _hash_commitment_check(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Pure-Python hash commitment (PYTHON_SAFE stand-in for HashCommit)."""
    state = ctx.get("state")
    commitment = ctx.get("commitment")
    if state is None or commitment is None:
        return {"passed": True, "severity": "info", "detail": "no state"}
    computed = hashlib.sha256(str(state).encode()).hexdigest()
    passed = computed == commitment
    return {
        "passed": passed,
        "severity": "critical" if not passed else "info",
        "detail": {"computed": computed[:16], "commitment": commitment[:16]},
        "opcode_trace": ["EmitEvent"],
    }


# ── default presets ───────────────────────────────────────

_DEF_PRESETS: List[FluxPreset] = [
    # 1. RangeCheck — weight bounds, chaos limits, thermal budget
    FluxPreset(
        name="RangeCheck",
        description="Weight bounds, chaos limits, and thermal budget range checks.",
        category=PresetCategory.BREEDING,
        constraints=(_weight_bounds_check, _chaos_limit_check, _thermal_budget_check),
        required_opcodes=("RangeCheck", "Validate", "ClassifySeverity"),
        python_safe=True,
    ),
    # 2. ProveAndHashCommit — signature verification + hash commitment
    FluxPreset(
        name="ProveAndHashCommit",
        description="Signature verification and hash commitment for provenance. Pure-Python fallbacks (EmitEvent) until Rust Prove/HashCommit FFI is ready.",
        category=PresetCategory.CRYPTO,
        constraints=(_signature_verification_check, _hash_commitment_check),
        required_opcodes=(
            "EmitEvent",
        ),  # PYTHON_SAFE only; Prove/HashCommit are RUST_ONLY
        python_safe=True,
    ),
    # 3. StreamBatch — batch size limits, rate limiting
    FluxPreset(
        name="StreamBatch",
        description="Batch size ceiling and request-per-second rate limiting.",
        category=PresetCategory.BATCHING,
        constraints=(_batch_size_limit_check, _rate_limit_check),
        required_opcodes=("Saturate", "Validate", "Min", "Max", "Sub"),
        python_safe=True,
    ),
    # 4. MemoryBudget — per-agent memory cap
    FluxPreset(
        name="MemoryBudget",
        description="Per-agent memory consumption must stay under a hard cap.",
        category=PresetCategory.MEMORY,
        constraints=(_memory_budget_check,),
        required_opcodes=("RangeCheck",),
        python_safe=True,
    ),
    # 5. DiversityFloor — minimum population diversity score
    FluxPreset(
        name="DiversityFloor",
        description="Population diversity score must stay above a minimum floor.",
        category=PresetCategory.DIVERSITY,
        constraints=(_diversity_floor_check,),
        required_opcodes=("Min", "Abs"),
        python_safe=True,
    ),
    # 6. ThermalCeiling — thermal budget hard cap
    FluxPreset(
        name="ThermalCeiling",
        description="Hard thermal ceiling. Any breach is critical and blocks breeding.",
        category=PresetCategory.THERMAL,
        constraints=(_thermal_ceiling_hard_check,),
        required_opcodes=("Validate",),
        python_safe=True,
    ),
    # 7. AgentLiveness — heartbeat timeout, crash detection
    FluxPreset(
        name="AgentLiveness",
        description="Agent must heartbeat within timeout and must not exceed consecutive failure threshold.",
        category=PresetCategory.LIVENESS,
        constraints=(_heartbeat_timeout_check, _crash_detection_check),
        required_opcodes=("Sub", "Validate", "EmitEvent"),
        python_safe=True,
    ),
    # 8. CrossNodeSync — mesh gossip consistency check
    FluxPreset(
        name="CrossNodeSync",
        description="Local state hash must match gossiped state hash from peers.",
        category=PresetCategory.SYNC,
        constraints=(_mesh_gossip_consistency_check,),
        required_opcodes=("Sub", "Validate"),
        python_safe=True,
    ),
    # 9. BreedingStandard — composite of RangeCheck + DiversityFloor
    FluxPreset(
        name="BreedingStandard",
        description="Default breeding gate: weights, chaos, thermal, and diversity.",
        category=PresetCategory.BREEDING,
        constraints=(
            _weight_bounds_check,
            _chaos_limit_check,
            _thermal_budget_check,
            _diversity_floor_check,
        ),
        required_opcodes=("RangeCheck", "Validate", "ClassifySeverity", "Min", "Abs"),
        python_safe=True,
    ),
    # 10. FleetHealth — composite of ThermalCeiling + AgentLiveness
    FluxPreset(
        name="FleetHealth",
        description="Fleet-wide health gate: thermal ceiling and agent liveness.",
        category=PresetCategory.THERMAL,
        constraints=(
            _thermal_ceiling_hard_check,
            _heartbeat_timeout_check,
            _crash_detection_check,
        ),
        required_opcodes=("Validate", "Sub", "EmitEvent"),
        python_safe=True,
    ),
]


# ── preset library ──────────────────────────────────────────


class FluxPresetLibrary:
    """Registry of reusable FLUX constraint presets.

    Usage::

        lib = FluxPresetLibrary()
        preset = lib.get_preset("RangeCheck")
        results = lib.apply_preset("RangeCheck", ctx={"weights": 5.0, "chaos": 0.3})
        name = lib.suggest_preset_for_task("breed new agents with weight checks")
    """

    def __init__(
        self,
        presets: Optional[List[FluxPreset]] = None,
        index: Optional[OpcodeCapabilityIndex] = None,
    ) -> None:
        self._by_name: Dict[str, FluxPreset] = {}
        self._index = index or OpcodeCapabilityIndex()
        src = presets if presets is not None else list(_DEF_PRESETS)
        for p in src:
            self._validate_and_register(p)

    # ── internal ────────────────────────────────────────────

    def _validate_and_register(self, preset: FluxPreset) -> None:
        # Cross-check required opcodes against the capability index.
        all_safe = True
        for opcode in preset.required_opcodes:
            if not self._index.can_use_from_python(opcode):
                all_safe = False
                break
        # If a preset claims python_safe but the index disagrees, warn
        # by mutating the effective flag (but we keep the frozen dataclass).
        effective_safe = preset.python_safe and all_safe
        if effective_safe != preset.python_safe:
            # Re-create with corrected flag — dataclass is frozen
            preset = FluxPreset(
                name=preset.name,
                description=preset.description,
                category=preset.category,
                constraints=preset.constraints,
                required_opcodes=preset.required_opcodes,
                python_safe=effective_safe,
            )
        self._by_name[preset.name] = preset

    # ── public API ──────────────────────────────────────────

    def get_preset(self, name: str) -> FluxPreset:
        """Return a preset by exact name. Raises KeyError if missing."""
        preset = self._by_name.get(name)
        if preset is None:
            raise KeyError(
                f"No preset named '{name}'. Available: {list(self._by_name.keys())}"
            )
        return preset

    def list_presets(
        self,
        category: Optional[str] = None,
        python_safe_only: bool = False,
    ) -> List[FluxPreset]:
        """List presets, optionally filtered by category and python_safe flag.

        Args:
            category: One of the ``PresetCategory`` names (e.g. "breeding",
                "thermal", "liveness"). Case-insensitive.
            python_safe_only: If ``True``, exclude presets whose
                ``python_safe`` flag is ``False``.
        """
        result: List[FluxPreset] = []
        if category is not None:
            cat_key = category.lower()
            cat_enum = PRESET_CATEGORIES.get(cat_key)
            if cat_enum is None:
                return []
        else:
            cat_enum = None
        for preset in self._by_name.values():
            if cat_enum is not None and preset.category != cat_enum:
                continue
            if python_safe_only and not preset.python_safe:
                continue
            result.append(preset)
        return sorted(result, key=lambda p: p.name)

    def apply_preset(
        self,
        preset_name: str,
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Run all constraints of a preset against *context*.

        Returns a list of result dicts, one per constraint, each containing
        at least ``passed`` (bool) and ``severity`` (str).
        """
        preset = self.get_preset(preset_name)
        results: List[Dict[str, Any]] = []
        for constraint in preset.constraints:
            result = constraint(context)
            results.append(result)
        return results

    def suggest_preset_for_task(self, task_description: str) -> str:
        """Return the best-matching preset name for a task description.

        Matching is keyword-based. Falls back to ``BreedingStandard``
        when no strong signal is detected.
        """
        desc = task_description.lower()
        scores: Dict[str, int] = {}

        # Keyword maps
        keywords: Dict[str, List[str]] = {
            "RangeCheck": ["range", "bound", "weight", "norm", "check"],
            "ProveAndHashCommit": [
                "prove",
                "hash",
                "commit",
                "signature",
                "verify",
                "crypto",
            ],
            "StreamBatch": ["batch", "stream", "rate", "limit", "rps", "throughput"],
            "MemoryBudget": ["memory", "ram", "heap", "budget", "cap"],
            "DiversityFloor": ["diversity", "variety", "floor", "population"],
            "ThermalCeiling": ["thermal", "heat", "temperature", "ceiling", "throttle"],
            "AgentLiveness": [
                "liveness",
                "heartbeat",
                "crash",
                "alive",
                "dead",
                "health",
            ],
            "CrossNodeSync": ["sync", "gossip", "mesh", "consistency", "node", "peer"],
            "BreedingStandard": [
                "breed",
                "breeding",
                "spawn",
                "new agent",
                "tournament",
            ],
            "FleetHealth": ["fleet", "health", "system", "overall", "status"],
        }

        for preset_name, kws in keywords.items():
            score = sum(1 for kw in kws if kw in desc)
            if score:
                scores[preset_name] = score

        if not scores:
            return "BreedingStandard"

        best = max(scores, key=lambda k: scores[k])
        return best

    @property
    def preset_count(self) -> int:
        return len(self._by_name)

    @property
    def names(self) -> List[str]:
        return sorted(self._by_name.keys())

    def __repr__(self) -> str:
        safe_count = sum(1 for p in self._by_name.values() if p.python_safe)
        return (
            f"FluxPresetLibrary(presets={self.preset_count}, python_safe={safe_count})"
        )
