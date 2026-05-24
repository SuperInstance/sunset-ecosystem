"""FLUX Opcode Capability Index — prevents "compile and crash" failures.

Tracks which of the 58 Rust FLUX-C v3 opcodes can safely be used from
Python today.  Path A (constraint library) is implemented via
``swarm/flux_gating.py``.  Path B (full VM compiler) is blocked on FM
decision.  This index is the ground truth for integration work.

References
----------
* Rust source: ``flux-vm-v3-temp/src/opcode.rs`` — canonical opcode enum
* Path A: ``swarm/flux_gating.py`` — Python fallback implementation
* Audit: ``flux-vm-v3-temp/FLUX-VM-V3-AUDIT.md``
"""
from __future__ import annotations

__all__ = [
    "FluxOpcode",
    "OpcodeCapabilityIndex",
    "OpcodeStatus",
]

import json
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional


class OpcodeStatus(Enum):
    """Lifecycle status for a single opcode."""

    UNTESTED = auto()
    PYTHON_SAFE = auto()   # Has a working Python fallback
    RUST_ONLY = auto()     # Requires the Rust VM; no Python equivalent
    DEPRECATED = auto()    # Do not use in new code
    PLANNED = auto()       # Python fallback is on the roadmap


@dataclass(frozen=True)
class FluxOpcode:
    """Metadata for a single FLUX-C v3 opcode."""

    name: str
    opcode_number: int
    category: str
    description: str
    status: OpcodeStatus
    effort_estimate: str  # "trivial", "low", "medium", "high", "blocked"
    path_a_equivalent: Optional[str] = None  # Python fallback function name


# ── canonical opcode list (extracted from flux-vm-v3-temp/src/opcode.rs) ──

_OPCODES: List[FluxOpcode] = [
    # ── Stack (8) ──
    FluxOpcode(
        "Push", 0x01, "stack",
        "Push a 32-bit immediate onto the operand stack.",
        OpcodeStatus.PYTHON_SAFE, "trivial", None,
    ),
    FluxOpcode(
        "Pop", 0x02, "stack",
        "Pop the top value from the operand stack.",
        OpcodeStatus.PYTHON_SAFE, "trivial", None,
    ),
    FluxOpcode(
        "Dup", 0x03, "stack",
        "Duplicate the top stack element.",
        OpcodeStatus.PYTHON_SAFE, "trivial", None,
    ),
    FluxOpcode(
        "Swap", 0x04, "stack",
        "Swap the top two stack elements.",
        OpcodeStatus.PYTHON_SAFE, "trivial", None,
    ),
    FluxOpcode(
        "Over", 0x05, "stack",
        "Copy the second stack element to the top.",
        OpcodeStatus.PYTHON_SAFE, "trivial", None,
    ),
    FluxOpcode(
        "Drop", 0x06, "stack",
        "Drop the top stack element (discard).",
        OpcodeStatus.PYTHON_SAFE, "trivial", None,
    ),
    FluxOpcode(
        "LoadConst", 0x07, "stack",
        "Load a 32-bit constant onto the stack.",
        OpcodeStatus.PYTHON_SAFE, "trivial", None,
    ),
    FluxOpcode(
        "Nop", 0x08, "stack",
        "No operation.",
        OpcodeStatus.PYTHON_SAFE, "trivial", None,
    ),

    # ── Arithmetic (8) ──
    FluxOpcode(
        "Add", 0x09, "arithmetic",
        "Pop two values, push their sum.",
        OpcodeStatus.PYTHON_SAFE, "trivial", None,
    ),
    FluxOpcode(
        "Sub", 0x0a, "arithmetic",
        "Pop two values, push their difference.",
        OpcodeStatus.PYTHON_SAFE, "trivial", None,
    ),
    FluxOpcode(
        "Mul", 0x0b, "arithmetic",
        "Pop two values, push their product.",
        OpcodeStatus.PYTHON_SAFE, "trivial", None,
    ),
    FluxOpcode(
        "Div", 0x0c, "arithmetic",
        "Pop two values, push integer quotient (i32).  Guards i32::MIN / -1.",
        OpcodeStatus.PYTHON_SAFE, "trivial", None,
    ),
    FluxOpcode(
        "Saturate", 0x0d, "arithmetic",
        "Clamp top-of-stack to a signed range.",
        OpcodeStatus.PYTHON_SAFE, "low", None,
    ),
    FluxOpcode(
        "Min", 0x0e, "arithmetic",
        "Pop two values, push the smaller.",
        OpcodeStatus.PYTHON_SAFE, "trivial", None,
    ),
    FluxOpcode(
        "Max", 0x0f, "arithmetic",
        "Pop two values, push the larger.",
        OpcodeStatus.PYTHON_SAFE, "trivial", None,
    ),
    FluxOpcode(
        "Abs", 0x10, "arithmetic",
        "Replace top-of-stack with its absolute value.  Guards i32::MIN.",
        OpcodeStatus.PYTHON_SAFE, "trivial", None,
    ),

    # ── Register / Memory (4) ──
    FluxOpcode(
        "LoadReg", 0x11, "memory",
        "Load a value from a numbered register (1-byte immediate).",
        OpcodeStatus.RUST_ONLY, "medium", None,
    ),
    FluxOpcode(
        "StoreReg", 0x12, "memory",
        "Store top-of-stack into a numbered register (1-byte immediate).",
        OpcodeStatus.RUST_ONLY, "medium", None,
    ),
    FluxOpcode(
        "LoadRegVec", 0x13, "memory",
        "Load a 4-lane vector register onto the stack.",
        OpcodeStatus.RUST_ONLY, "high", None,
    ),
    FluxOpcode(
        "StoreRegVec", 0x14, "memory",
        "Store top-of-stack into a 4-lane vector register.",
        OpcodeStatus.RUST_ONLY, "high", None,
    ),

    # ── Constraint (10) ──
    FluxOpcode(
        "RangeCheck", 0x15, "control",
        "Assert that top-of-stack lies within pre-set bounds.",
        OpcodeStatus.PYTHON_SAFE, "low", "check_candidate",
    ),
    FluxOpcode(
        "BatchCheck", 0x16, "control",
        "Run a batched range check over N stack entries.",
        OpcodeStatus.RUST_ONLY, "medium", "check_batch",
    ),
    FluxOpcode(
        "AccumulateMask", 0x17, "control",
        "Fold batch results into a bitmask.",
        OpcodeStatus.RUST_ONLY, "medium", "check_batch",
    ),
    FluxOpcode(
        "ClassifySeverity", 0x18, "control",
        "Tag a violation with INFO / WARNING / CRITICAL.",
        OpcodeStatus.PYTHON_SAFE, "low", "check_candidate",
    ),
    FluxOpcode(
        "Prove", 0x19, "crypto",
        "Generate a proof certificate for the top-of-stack value.",
        OpcodeStatus.RUST_ONLY, "high", None,
    ),
    FluxOpcode(
        "QueryBackward", 0x1a, "crypto",
        "Query the proof chain at a given depth.",
        OpcodeStatus.RUST_ONLY, "high", None,
    ),
    FluxOpcode(
        "Simplify", 0x1b, "control",
        "Simplify a constraint expression (algebraic reduction).",
        OpcodeStatus.RUST_ONLY, "high", None,
    ),
    FluxOpcode(
        "Validate", 0x1c, "control",
        "Hard assertion — fault if top-of-stack is zero.",
        OpcodeStatus.PYTHON_SAFE, "low", "check_candidate",
    ),
    FluxOpcode(
        "HashCommit", 0x1d, "crypto",
        "Anchor the current proof state in the SHA-256 chain.",
        OpcodeStatus.RUST_ONLY, "high", None,
    ),
    FluxOpcode(
        "Seal", 0x1e, "crypto",
        "Finalize and seal the current proof certificate.",
        OpcodeStatus.RUST_ONLY, "high", None,
    ),

    # ── Vector / SIMD (6) ──
    FluxOpcode(
        "VecLoad", 0x1f, "memory",
        "Load an 8-lane i8 vector from memory into a SIMD register.",
        OpcodeStatus.RUST_ONLY, "high", None,
    ),
    FluxOpcode(
        "VecStore", 0x20, "memory",
        "Store an 8-lane i8 vector from a SIMD register to memory.",
        OpcodeStatus.RUST_ONLY, "high", None,
    ),
    FluxOpcode(
        "VecRangeCheck", 0x21, "control",
        "SIMD range check across all 8 lanes.",
        OpcodeStatus.RUST_ONLY, "high", None,
    ),
    FluxOpcode(
        "VecMaskMerge", 0x22, "control",
        "Merge two SIMD lane masks.",
        OpcodeStatus.RUST_ONLY, "high", None,
    ),
    FluxOpcode(
        "VecReduce", 0x23, "memory",
        "Horizontal reduction (sum) of a SIMD vector.",
        OpcodeStatus.RUST_ONLY, "medium", None,
    ),
    FluxOpcode(
        "VecGather", 0x24, "memory",
        "Gather scattered vector elements by index.",
        OpcodeStatus.RUST_ONLY, "high", None,
    ),

    # ── Control (6) ──
    FluxOpcode(
        "FwdJump", 0x25, "control",
        "Unconditional forward jump (2-byte offset).",
        OpcodeStatus.PYTHON_SAFE, "low", None,
    ),
    FluxOpcode(
        "CondJump", 0x26, "control",
        "Conditional forward jump if top-of-stack is zero.",
        OpcodeStatus.PYTHON_SAFE, "low", None,
    ),
    FluxOpcode(
        "CallBounded", 0x27, "control",
        "Call a subroutine with a synthetic cycle bound (default 4096).",
        OpcodeStatus.RUST_ONLY, "medium", None,
    ),
    FluxOpcode(
        "Ret", 0x28, "control",
        "Return from subroutine.",
        OpcodeStatus.PYTHON_SAFE, "low", None,
    ),
    FluxOpcode(
        "Halt", 0x29, "control",
        "Stop execution and return success.",
        OpcodeStatus.PYTHON_SAFE, "trivial", None,
    ),
    FluxOpcode(
        "Checkpoint", 0x2a, "control",
        "Push current VM state to the checkpoint stack.",
        OpcodeStatus.RUST_ONLY, "medium", None,
    ),

    # ── Effects (4) ──
    FluxOpcode(
        "SetHandler", 0x2b, "io",
        "Install an effect handler for the current block.",
        OpcodeStatus.RUST_ONLY, "high", None,
    ),
    FluxOpcode(
        "EmitEvent", 0x2c, "io",
        "Emit a structured event to the effects log.",
        OpcodeStatus.PYTHON_SAFE, "low", "record_violation",
    ),
    FluxOpcode(
        "Rollback", 0x2d, "io",
        "Restore the last checkpointed state.",
        OpcodeStatus.RUST_ONLY, "high", None,
    ),
    FluxOpcode(
        "GetResult", 0x2e, "io",
        "Retrieve the current execution result handle.",
        OpcodeStatus.RUST_ONLY, "medium", None,
    ),

    # ── Parallel (4) ──
    FluxOpcode(
        "ParDispatch", 0x2f, "io",
        "Dispatch constraint checks across Rayon worker threads.",
        OpcodeStatus.RUST_ONLY, "high", None,
    ),
    FluxOpcode(
        "ParMerge", 0x30, "io",
        "Merge results from a parallel dispatch.",
        OpcodeStatus.RUST_ONLY, "high", None,
    ),
    FluxOpcode(
        "ParBarrier", 0x31, "io",
        "Synchronize all parallel workers at a barrier.",
        OpcodeStatus.RUST_ONLY, "high", None,
    ),
    FluxOpcode(
        "ParReduce", 0x32, "io",
        "Reduce partial results from parallel workers.",
        OpcodeStatus.RUST_ONLY, "high", None,
    ),

    # ── Provenance (4) ──
    FluxOpcode(
        "SnapRecord", 0x33, "crypto",
        "Record a snapshot entry in the provenance ring buffer.",
        OpcodeStatus.RUST_ONLY, "medium", None,
    ),
    FluxOpcode(
        "SnapQuery", 0x34, "crypto",
        "Query the provenance ring buffer by index.",
        OpcodeStatus.RUST_ONLY, "medium", None,
    ),
    FluxOpcode(
        "SnapHash", 0x35, "crypto",
        "Hash the current snapshot for integrity.",
        OpcodeStatus.RUST_ONLY, "medium", None,
    ),
    FluxOpcode(
        "SnapVerify", 0x36, "crypto",
        "Verify a snapshot hash against the chain (currently a stub).",
        OpcodeStatus.RUST_ONLY, "low", None,
    ),

    # ── Streaming (4) ──
    FluxOpcode(
        "StreamOpen", 0x37, "io",
        "Open a streaming constraint buffer.",
        OpcodeStatus.RUST_ONLY, "medium", None,
    ),
    FluxOpcode(
        "StreamCheck", 0x38, "io",
        "Check the next element in a stream against constraints.",
        OpcodeStatus.RUST_ONLY, "medium", None,
    ),
    FluxOpcode(
        "StreamBatch", 0x39, "io",
        "Process a batch of stream elements.",
        OpcodeStatus.RUST_ONLY, "medium", None,
    ),
    FluxOpcode(
        "StreamClose", 0x3a, "io",
        "Close the streaming buffer and flush results.",
        OpcodeStatus.RUST_ONLY, "medium", None,
    ),
]


class OpcodeCapabilityIndex:
    """Registry that tracks Python safety for every FLUX-C v3 opcode.

    The index is the single source of truth for "can I use this opcode
    from Python?".  Integration code should call
    ``can_use_from_python()`` before emitting or compiling any opcode.

    Persistence
    -----------
    Status updates can be saved to / loaded from a JSON file so that
    manual testing results survive restarts.
    """

    def __init__(self, opcodes: Optional[List[FluxOpcode]] = None) -> None:
        self._by_name: Dict[str, FluxOpcode] = {}
        self._by_number: Dict[int, FluxOpcode] = {}
        self._overrides: Dict[str, OpcodeStatus] = {}

        src = opcodes if opcodes is not None else list(_OPCODES)
        for op in src:
            self._by_name[op.name] = op
            self._by_number[op.opcode_number] = op

    # ── queries ───────────────────────────────────────────────

    @property
    def total_opcodes(self) -> int:
        return len(self._by_name)

    def get(self, opcode: str | int) -> Optional[FluxOpcode]:
        """Lookup by name (str) or opcode number (int)."""
        if isinstance(opcode, str):
            return self._by_name.get(opcode)
        return self._by_number.get(opcode)

    def can_use_from_python(self, opcode: str | int) -> bool:
        """Return ``True`` if the opcode has a working Python fallback."""
        op = self.get(opcode)
        if op is None:
            return False
        status = self._overrides.get(op.name, op.status)
        return status == OpcodeStatus.PYTHON_SAFE

    def get_safe_opcodes(self, category: Optional[str] = None) -> List[FluxOpcode]:
        """Return all PYTHON_SAFE opcodes, optionally filtered by category."""
        result: List[FluxOpcode] = []
        for op in self._by_name.values():
            status = self._overrides.get(op.name, op.status)
            if status != OpcodeStatus.PYTHON_SAFE:
                continue
            if category is not None and op.category != category:
                continue
            result.append(op)
        return sorted(result, key=lambda o: o.opcode_number)

    def get_rust_only_opcodes(self, category: Optional[str] = None) -> List[FluxOpcode]:
        """Return all RUST_ONLY opcodes, optionally filtered by category."""
        result: List[FluxOpcode] = []
        for op in self._by_name.values():
            status = self._overrides.get(op.name, op.status)
            if status != OpcodeStatus.RUST_ONLY:
                continue
            if category is not None and op.category != category:
                continue
            result.append(op)
        return sorted(result, key=lambda o: o.opcode_number)

    def get_gap_report(self) -> List[Dict[str, Any]]:
        """Return a structured report of every RUST_ONLY opcode.

        Each entry contains the opcode metadata plus an effort estimate
        (trivial / low / medium / high / blocked) for implementing a
        Python fallback.
        """
        gaps: List[Dict[str, Any]] = []
        for op in self.get_rust_only_opcodes():
            gaps.append({
                "name": op.name,
                "opcode_number": f"0x{op.opcode_number:02x}",
                "category": op.category,
                "description": op.description,
                "effort_estimate": op.effort_estimate,
                "path_a_equivalent": op.path_a_equivalent,
                "reason": (
                    "Requires Rust VM runtime (SIMD / effects / parallel / provenance)"
                    if op.effort_estimate in ("high", "blocked")
                    else "Needs Python fallback implementation"
                ),
            })
        return gaps

    def suggest_path_a_equivalent(self, opcode: str | int) -> Optional[str]:
        """Suggest the Python fallback function name for an opcode.

        Returns ``None`` when no Path A equivalent exists (the opcode is
        either trivially native in Python or truly Rust-only).
        """
        op = self.get(opcode)
        if op is None:
            return None
        return op.path_a_equivalent

    def count_by_status(self) -> Dict[str, int]:
        """Tally opcodes by their effective status."""
        counts: Dict[str, int] = {}
        for op in self._by_name.values():
            status = self._overrides.get(op.name, op.status)
            key = status.name
            counts[key] = counts.get(key, 0) + 1
        return counts

    def count_by_category(self) -> Dict[str, int]:
        """Tally opcodes by category."""
        counts: Dict[str, int] = {}
        for op in self._by_name.values():
            counts[op.category] = counts.get(op.category, 0) + 1
        return counts

    # ── mutations ─────────────────────────────────────────────

    def update_status(self, opcode: str | int, status: OpcodeStatus) -> None:
        """Override the canonical status for an opcode.

        Overrides are stored in-memory and can be persisted via
        ``save()``.  This is the mechanism for recording manual
        test results (e.g. an UNTESTED opcode is verified
        PYTHON_SAFE after writing a fallback).
        """
        op = self.get(opcode)
        if op is None:
            raise KeyError(f"Unknown opcode: {opcode}")
        self._overrides[op.name] = status

    def reset_status(self, opcode: str | int) -> None:
        """Remove an override, reverting to the canonical status."""
        op = self.get(opcode)
        if op is None:
            raise KeyError(f"Unknown opcode: {opcode}")
        self._overrides.pop(op.name, None)

    # ── persistence ───────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Write the current index (including overrides) to JSON."""
        payload = {
            "version": 1,
            "total_opcodes": self.total_opcodes,
            "canonical": [
                {
                    "name": op.name,
                    "opcode_number": op.opcode_number,
                    "category": op.category,
                    "description": op.description,
                    "status": op.status.name,
                    "effort_estimate": op.effort_estimate,
                    "path_a_equivalent": op.path_a_equivalent,
                }
                for op in sorted(self._by_name.values(), key=lambda o: o.opcode_number)
            ],
            "overrides": {
                name: status.name for name, status in self._overrides.items()
            },
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "OpcodeCapabilityIndex":
        """Load an index from a JSON file produced by ``save()``."""
        data = json.loads(Path(path).read_text())
        opcodes = [
            FluxOpcode(
                name=op["name"],
                opcode_number=op["opcode_number"],
                category=op["category"],
                description=op["description"],
                status=OpcodeStatus[op["status"]],
                effort_estimate=op["effort_estimate"],
                path_a_equivalent=op.get("path_a_equivalent"),
            )
            for op in data["canonical"]
        ]
        idx = cls(opcodes)
        for name, status_name in data.get("overrides", {}).items():
            idx._overrides[name] = OpcodeStatus[status_name]
        return idx

    # ── repr ──────────────────────────────────────────────────

    def __repr__(self) -> str:
        counts = self.count_by_status()
        safe = counts.get("PYTHON_SAFE", 0)
        rust = counts.get("RUST_ONLY", 0)
        return (
            f"OpcodeCapabilityIndex("
            f"total={self.total_opcodes}, "
            f"PYTHON_SAFE={safe}, "
            f"RUST_ONLY={rust})"
        )
