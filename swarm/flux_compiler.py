"""Minimal Python → FLUX bytecode compiler (Path B prototype).

Translates breeding constraint expressions into FLUX VM v3 bytecode.
Focus: arithmetic opcodes, comparison opcodes, and branch opcodes.

The compiler targets the PYTHON_SAFE opcode subset from
``logos.opcode_capability_index`` so that emitted bytecode can be
executed by a Python fallback VM or forwarded to the Rust VM once the
FFI lifecycle functions are wired.

Bytecode format (little-endian)
-------------------------------
| Opcode | Name        | Operands                         | Size |
|--------|-------------|----------------------------------|------|
| 0x01   | Push        | 4-byte f32 immediate             | 5    |
| 0x02   | Pop         | —                                | 1    |
| 0x03   | Dup         | —                                | 1    |
| 0x07   | LoadConst   | 1-byte constant-pool index       | 2    |
| 0x09   | Add         | —                                | 1    |
| 0x0a   | Sub         | —                                | 1    |
| 0x0b   | Mul         | —                                | 1    |
| 0x0c   | Div         | —                                | 1    |
| 0x0d   | Saturate    | 4-byte f32 min + 4-byte f32 max  | 9    |
| 0x0e   | Min         | —                                | 1    |
| 0x0f   | Max         | —                                | 1    |
| 0x10   | Abs         | —                                | 1    |
| 0x15   | RangeCheck  | 4-byte f32 min + 4-byte f32 max  | 9    |
| 0x18   | ClassifySeverity | 1-byte severity (0=INFO,1=WARN,2=CRIT) | 2 |
| 0x1c   | Validate    | —                                | 1    |
| 0x25   | FwdJump     | 2-byte u16 offset (from next instr)| 3    |
| 0x26   | CondJump    | 2-byte u16 offset (from next instr)| 3    |
| 0x29   | Halt        | —                                | 1    |

A simple constraint like ``weight ∈ [w_min, w_max]`` can be compiled
in two ways:

1. **High-level** (one RangeCheck opcode)::

       Push weight
       RangeCheck w_min w_max   ; pushes 1.0 (pass) or 0.0 (fail)
       Validate                 ; traps if TOS == 0
       Halt

2. **Low-level** (arithmetic + branch — the Path B focus)::

       Push weight
       Push w_min
       Sub                      ; TOS = weight - w_min
       CondJump fail            ; jump if TOS <= 0  (weight < min)

       Push weight
       Push w_max
       Sub                      ; TOS = weight - w_max
       CondJump fail            ; jump if TOS > 0   (weight > max)

       Push 1.0                 ; pass
       Halt

     fail:
       Push 0.0                 ; fail
       ClassifySeverity CRITICAL
       Validate                 ; hard trap
       Halt
"""

from __future__ import annotations

__all__ = [
    "FluxOpcode",
    "BytecodeEmitter",
    "FluxCompiler",
    "compile_constraint",
    "PYTHON_SAFE_OPCODES",
]

import struct
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# ── opcode numbers (from flux_compat/opcode_map.py V3_OPCODE_BYTES) ──


class FluxOpcode(IntEnum):
    """FLUX VM v3 opcode byte values."""

    # Stack
    Push = 0x01
    Pop = 0x02
    Dup = 0x03
    Swap = 0x04
    Over = 0x05
    Drop = 0x06
    LoadConst = 0x07
    Nop = 0x08

    # Arithmetic
    Add = 0x09
    Sub = 0x0A
    Mul = 0x0B
    Div = 0x0C
    Saturate = 0x0D
    Min = 0x0E
    Max = 0x0F
    Abs = 0x10

    # Register / Memory
    LoadReg = 0x11
    StoreReg = 0x12
    LoadRegVec = 0x13
    StoreRegVec = 0x14

    # Constraint
    RangeCheck = 0x15
    BatchCheck = 0x16
    AccumulateMask = 0x17
    ClassifySeverity = 0x18
    Prove = 0x19
    QueryBackward = 0x1A
    Simplify = 0x1B
    Validate = 0x1C
    HashCommit = 0x1D
    Seal = 0x1E

    # Vector / SIMD
    VecLoad = 0x1F
    VecStore = 0x20
    VecRangeCheck = 0x21
    VecMaskMerge = 0x22
    VecReduce = 0x23
    VecGather = 0x24

    # Control
    FwdJump = 0x25
    CondJump = 0x26
    CallBounded = 0x27
    Ret = 0x28
    Halt = 0x29
    Checkpoint = 0x2A

    # Effects
    SetHandler = 0x2B
    EmitEvent = 0x2C
    Rollback = 0x2D
    GetResult = 0x2E

    # Parallel
    ParDispatch = 0x2F
    ParMerge = 0x30
    ParBarrier = 0x31
    ParReduce = 0x32

    # Provenance
    SnapRecord = 0x33
    SnapQuery = 0x34
    SnapHash = 0x35
    SnapVerify = 0x36

    # Streaming
    StreamOpen = 0x37
    StreamCheck = 0x38
    StreamBatch = 0x39
    StreamClose = 0x3A


# Opcodes that have a Python-safe fallback (from OpcodeCapabilityIndex)
PYTHON_SAFE_OPCODES: set[str] = {
    "Push",
    "Pop",
    "Dup",
    "Swap",
    "Over",
    "Drop",
    "LoadConst",
    "Nop",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Saturate",
    "Min",
    "Max",
    "Abs",
    "RangeCheck",
    "ClassifySeverity",
    "Validate",
    "FwdJump",
    "CondJump",
    "Ret",
    "Halt",
    "EmitEvent",
}


# ── AST nodes for constraint expressions ──


@dataclass(frozen=True)
class Const:
    """Float constant."""

    value: float


@dataclass(frozen=True)
class Var:
    """Named variable (resolved to a constant pool slot at compile time)."""

    name: str


@dataclass(frozen=True)
class BinOp:
    """Binary arithmetic operation."""

    op: str  # "Add", "Sub", "Mul", "Div", "Min", "Max"
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class UnaryOp:
    """Unary arithmetic operation."""

    op: str  # "Abs"
    operand: "Expr"


@dataclass(frozen=True)
class RangeCheckNode:
    """High-level range-check node."""

    expr: "Expr"
    lo: float
    hi: float


@dataclass(frozen=True)
class IfNode:
    """Conditional branch: if cond then then_expr else else_expr."""

    cond: "CmpOp"
    then_expr: "Expr"
    else_expr: "Expr"


@dataclass(frozen=True)
class CmpOp:
    """Comparison: left op right, where op is one of LT, LE, GT, GE, EQ."""

    op: str  # "LT", "LE", "GT", "GE", "EQ"
    left: "Expr"
    right: "Expr"


Expr = Union[Const, Var, BinOp, UnaryOp, RangeCheckNode, IfNode]


# ── BytecodeEmitter ──


class BytecodeEmitter:
    """Builds a FLUX bytecode sequence with support for back-patching jumps.

    Usage::

        e = BytecodeEmitter()
        e.push(5.0)
        e.op(FluxOpcode.Halt)
        bc = e.to_bytes()
    """

    def __init__(self) -> None:
        self._code: bytearray = bytearray()
        self._const_pool: List[float] = []
        self._labels: Dict[str, int] = {}  # label name → byte offset
        self._backpatches: List[Tuple[str, int]] = []  # (label, offset-of-jump-operand)
        self.var_slots: Dict[str, int] = {}  # variable name → constant pool index

    # ── raw bytes ──

    def _emit_u8(self, val: int) -> None:
        self._code.append(val & 0xFF)

    def _emit_u16(self, val: int) -> None:
        self._code.extend(struct.pack("<H", val & 0xFFFF))

    def _emit_f32(self, val: float) -> None:
        self._code.extend(struct.pack("<f", val))

    # ── high-level emitters ──

    def op(self, opcode: FluxOpcode) -> "BytecodeEmitter":
        """Emit a no-operand opcode."""
        self._emit_u8(opcode)
        return self

    def push(self, value: float) -> "BytecodeEmitter":
        """Emit ``Push <f32>``."""
        self._emit_u8(FluxOpcode.Push)
        self._emit_f32(value)
        return self

    def load_const(self, index: int) -> "BytecodeEmitter":
        """Emit ``LoadConst <u8>``."""
        if not (0 <= index <= 255):
            raise ValueError(f"Constant pool index {index} out of u8 range")
        self._emit_u8(FluxOpcode.LoadConst)
        self._emit_u8(index)
        return self

    def fwd_jump(self, label: str) -> "BytecodeEmitter":
        """Emit ``FwdJump <u16>`` with a back-patch slot for *label*."""
        self._emit_u8(FluxOpcode.FwdJump)
        patch_offset = len(self._code)
        self._emit_u16(0xFFFF)  # placeholder
        self._backpatches.append((label, patch_offset))
        return self

    def cond_jump(self, label: str) -> "BytecodeEmitter":
        """Emit ``CondJump <u16>`` with a back-patch slot for *label*."""
        self._emit_u8(FluxOpcode.CondJump)
        patch_offset = len(self._code)
        self._emit_u16(0xFFFF)  # placeholder
        self._backpatches.append((label, patch_offset))
        return self

    def range_check(self, lo: float, hi: float) -> "BytecodeEmitter":
        """Emit ``RangeCheck <f32 min> <f32 max>``."""
        self._emit_u8(FluxOpcode.RangeCheck)
        self._emit_f32(lo)
        self._emit_f32(hi)
        return self

    def classify_severity(self, severity: int) -> "BytecodeEmitter":
        """Emit ``ClassifySeverity <u8>``.

        Args:
            severity: 0=INFO, 1=WARNING, 2=CRITICAL
        """
        if not (0 <= severity <= 2):
            raise ValueError(f"Severity {severity} must be 0–2")
        self._emit_u8(FluxOpcode.ClassifySeverity)
        self._emit_u8(severity)
        return self

    # ── labels ──

    def label(self, name: str) -> "BytecodeEmitter":
        """Mark the current position as *name* and resolve pending back-patches."""
        offset = len(self._code)
        self._labels[name] = offset
        # Resolve any back-patches that target this label
        unresolved: List[Tuple[str, int]] = []
        for lbl, patch_offset in self._backpatches:
            if lbl == name:
                jump_from = patch_offset + 2  # after the u16 operand
                jump_delta = offset - jump_from
                if not (0 <= jump_delta <= 0xFFFF):
                    raise ValueError(
                        f"Jump delta {jump_delta} to label '{name}' exceeds u16"
                    )
                self._code[patch_offset : patch_offset + 2] = struct.pack(
                    "<H", jump_delta
                )
            else:
                unresolved.append((lbl, patch_offset))
        self._backpatches = unresolved
        return self

    # ── constant pool ──

    def add_const(self, value: float) -> int:
        """Add a float to the constant pool and return its index."""
        try:
            return self._const_pool.index(value)
        except ValueError:
            idx = len(self._const_pool)
            if idx > 255:
                raise ValueError("Constant pool overflow (>255 entries)")
            self._const_pool.append(value)
            return idx

    # ── finalisation ──

    def to_bytes(self) -> bytes:
        """Return the fully resolved bytecode.

        Raises if any back-patches are still unresolved.
        """
        if self._backpatches:
            labels = ", ".join(lbl for lbl, _ in self._backpatches)
            raise ValueError(f"Unresolved labels: {labels}")
        return bytes(self._code)

    @property
    def const_pool(self) -> List[float]:
        return list(self._const_pool)

    @property
    def size(self) -> int:
        return len(self._code)

    def disassemble(self) -> List[str]:
        """Pretty-print the bytecode for debugging."""
        out: List[str] = []
        i = 0
        code = self._code
        while i < len(code):
            op = code[i]
            name = (
                FluxOpcode(op).name
                if op in [o.value for o in FluxOpcode]
                else f"0x{op:02x}"
            )
            if op == FluxOpcode.Push:
                val = struct.unpack("<f", code[i + 1 : i + 5])[0]
                out.append(f"{i:04d}  Push    {val}")
                i += 5
            elif op == FluxOpcode.LoadConst:
                out.append(f"{i:04d}  LoadConst  {code[i + 1]}")
                i += 2
            elif op in (FluxOpcode.FwdJump, FluxOpcode.CondJump):
                off = struct.unpack("<H", code[i + 1 : i + 3])[0]
                out.append(f"{i:04d}  {name}    +{off}")
                i += 3
            elif op == FluxOpcode.RangeCheck:
                lo = struct.unpack("<f", code[i + 1 : i + 5])[0]
                hi = struct.unpack("<f", code[i + 5 : i + 9])[0]
                out.append(f"{i:04d}  RangeCheck  [{lo}, {hi}]")
                i += 9
            elif op == FluxOpcode.ClassifySeverity:
                out.append(f"{i:04d}  ClassifySeverity  {code[i + 1]}")
                i += 2
            elif op == FluxOpcode.Saturate:
                lo = struct.unpack("<f", code[i + 1 : i + 5])[0]
                hi = struct.unpack("<f", code[i + 5 : i + 9])[0]
                out.append(f"{i:04d}  Saturate  [{lo}, {hi}]")
                i += 9
            else:
                out.append(f"{i:04d}  {name}")
                i += 1
        return out


# ── FluxCompiler ──


class FluxCompiler:
    """Compile constraint AST expressions into FLUX bytecode.

    Args:
        prefer_range_check: If ``True`` (default), the compiler emits
            the single ``RangeCheck`` opcode for range constraints.
            If ``False``, it expands the check into arithmetic + branch
            (the low-level Path B pattern).
    """

    def __init__(
        self,
        prefer_range_check: bool = True,
        var_defaults: dict[str, float] | None = None,
    ) -> None:
        self.prefer_range_check = prefer_range_check
        self.var_defaults = var_defaults or {}

    def _fresh_label(self, prefix: str = "lbl") -> str:
        """Generate a unique label name."""
        idx = getattr(self, "_label_counter", 0)
        self._label_counter = idx + 1
        return f"__{prefix}_{idx}"

    def _compile_cmpop_as_expr(self, node: CmpOp, emitter: BytecodeEmitter) -> None:
        """Compile a CmpOp as a value-producing expression (1.0 or 0.0)."""
        true_label = self._fresh_label("cmp_true")
        end_label = self._fresh_label("cmp_end")

        if node.op == "LE":
            self.compile_expr(node.left, emitter)
            self.compile_expr(node.right, emitter)
            emitter.op(FluxOpcode.Sub)
            emitter.cond_jump(true_label)
        elif node.op == "LT":
            self.compile_expr(node.right, emitter)
            self.compile_expr(node.left, emitter)
            emitter.op(FluxOpcode.Sub)
            false_label = self._fresh_label("cmp_false")
            emitter.cond_jump(false_label)
            emitter.push(1.0)
            emitter.fwd_jump(end_label)
            emitter.label(false_label)
            emitter.push(0.0)
            emitter.label(end_label)
            return
        elif node.op == "GE":
            self.compile_expr(node.right, emitter)
            self.compile_expr(node.left, emitter)
            emitter.op(FluxOpcode.Sub)
            emitter.cond_jump(true_label)
        elif node.op == "GT":
            self.compile_expr(node.left, emitter)
            self.compile_expr(node.right, emitter)
            emitter.op(FluxOpcode.Sub)
            false_label = self._fresh_label("cmp_false")
            emitter.cond_jump(false_label)
            emitter.push(1.0)
            emitter.fwd_jump(end_label)
            emitter.label(false_label)
            emitter.push(0.0)
            emitter.label(end_label)
            return
        elif node.op == "EQ":
            self.compile_expr(node.left, emitter)
            self.compile_expr(node.right, emitter)
            emitter.op(FluxOpcode.Sub)
            emitter.op(FluxOpcode.Abs)
            emitter.push(1e-6)
            emitter.op(FluxOpcode.Sub)
            emitter.cond_jump(true_label)
        elif node.op == "NE":
            self.compile_expr(node.left, emitter)
            self.compile_expr(node.right, emitter)
            emitter.op(FluxOpcode.Sub)
            emitter.op(FluxOpcode.Abs)
            emitter.push(1e-6)
            emitter.op(FluxOpcode.Sub)
            # For NE we want TRUE when |diff| > epsilon
            # CondJump fires when |diff| - epsilon <= 0
            # So jump to FALSE on |diff| <= epsilon, fall through to TRUE
            false_label = self._fresh_label("cmp_false")
            emitter.cond_jump(false_label)
            emitter.push(1.0)
            emitter.fwd_jump(end_label)
            emitter.label(false_label)
            emitter.push(0.0)
            emitter.label(end_label)
            return

        emitter.push(0.0)
        emitter.fwd_jump(end_label)
        emitter.label(true_label)
        emitter.push(1.0)
        emitter.label(end_label)

    def compile_expr(
        self, expr: Expr, emitter: BytecodeEmitter, with_validate: bool = True
    ) -> None:
        """Compile an expression into *emitter*."""
        if isinstance(expr, Const):
            emitter.push(expr.value)
        elif isinstance(expr, Var):
            idx = emitter.add_const(self._resolve_var(expr.name))
            emitter.var_slots[expr.name] = idx
            emitter.load_const(idx)
        elif isinstance(expr, BinOp):
            self.compile_expr(expr.left, emitter, with_validate)
            self.compile_expr(expr.right, emitter, with_validate)
            opcode = getattr(FluxOpcode, expr.op, None)
            if opcode is None or expr.op not in PYTHON_SAFE_OPCODES:
                raise ValueError(f"Unsupported or unsafe binary op: {expr.op}")
            emitter.op(opcode)
        elif isinstance(expr, UnaryOp):
            self.compile_expr(expr.operand, emitter, with_validate)
            opcode = getattr(FluxOpcode, expr.op, None)
            if opcode is None or expr.op not in PYTHON_SAFE_OPCODES:
                raise ValueError(f"Unsupported or unsafe unary op: {expr.op}")
            emitter.op(opcode)
        elif isinstance(expr, CmpOp):
            self._compile_cmpop_as_expr(expr, emitter)
        elif isinstance(expr, RangeCheckNode):
            if self.prefer_range_check:
                self.compile_expr(expr.expr, emitter, with_validate)
                emitter.range_check(expr.lo, expr.hi)
            else:
                self._compile_range_check_arithmetic(expr, emitter, with_validate)
        elif isinstance(expr, IfNode):
            self._compile_if(expr, emitter)
        else:
            raise TypeError(f"Unknown expression type: {type(expr).__name__}")

    def _compile_range_check_arithmetic(
        self, node: RangeCheckNode, emitter: BytecodeEmitter, with_validate: bool = True
    ) -> None:
        """Compile a range check using only arithmetic + branch opcodes.

        Pseudocode::

            tmp = expr - lo
            if tmp < 0: goto fail
            tmp = expr - hi
            if tmp > 0: goto fail
            push 1.0
            halt
          fail:
            push 0.0
            classify_severity CRITICAL
            validate
            halt
        """
        expr = node.expr
        lo = node.lo
        hi = node.hi

        # Check expr >= lo  →  expr - lo >= 0  →  jump if negative
        self.compile_expr(expr, emitter, with_validate)
        emitter.push(lo)
        emitter.op(FluxOpcode.Sub)  # TOS = expr - lo
        emitter.cond_jump("__fail_lo")  # jump if TOS <= 0 (weight < min)

        # Check expr <= hi  →  expr - hi <= 0  →  jump if positive
        self.compile_expr(expr, emitter, with_validate)
        emitter.push(hi)
        emitter.op(FluxOpcode.Sub)  # TOS = expr - hi
        # For CondJump we need TOS == 0 to jump. To detect "> 0" we
        # need a trick: negate then check.
        emitter.push(0.0)
        emitter.op(FluxOpcode.Swap)
        emitter.op(FluxOpcode.Sub)  # TOS = 0 - (expr - hi) = hi - expr
        emitter.cond_jump("__fail_hi")  # jump if hi - expr <= 0 (i.e. expr > hi)

        # Pass path
        emitter.push(1.0)
        emitter.op(FluxOpcode.Halt)

        # Fail path (low violation)
        emitter.label("__fail_lo")
        emitter.push(0.0)
        emitter.classify_severity(2)  # CRITICAL
        if with_validate:
            emitter.op(FluxOpcode.Validate)
        emitter.op(FluxOpcode.Halt)

        # Fail path (high violation)
        emitter.label("__fail_hi")
        emitter.push(0.0)
        emitter.classify_severity(2)  # CRITICAL
        if with_validate:
            emitter.op(FluxOpcode.Validate)
        emitter.op(FluxOpcode.Halt)

    def _compile_if(self, node: IfNode, emitter: BytecodeEmitter) -> None:
        """Compile an IfNode using CondJump.

        For the comparison we compile ``left - right`` and use the sign.
        CondJump semantics: jump if TOS <= 0 (treating the value as a
        signed comparison result).
        """
        cond = node.cond
        # Compile left - right
        self.compile_expr(cond.left, emitter)
        self.compile_expr(cond.right, emitter)
        emitter.op(FluxOpcode.Sub)

        # CondJump to else block if condition is false
        else_label = "__else"
        end_label = "__endif"
        then_label = "__then"

        # For LE: left <= right  →  left - right <= 0  →  jump if > 0 to else
        # For GE: left >= right  →  left - right >= 0  →  jump if < 0 to else
        # We use the generic CondJump (jump if TOS <= 0) and arrange the branches
        # by swapping the then/else blocks or inverting the subtraction order.

        if cond.op == "LE":
            # left <= right  →  left - right <= 0  →  jump to THEN if TOS <= 0
            emitter.cond_jump(then_label)  # jump to then if TOS <= 0 (condition TRUE)
            self.compile_expr(node.else_expr, emitter)  # fall-through: TOS > 0 (FALSE)
            emitter.fwd_jump(end_label)
            emitter.label(then_label)
            self.compile_expr(node.then_expr, emitter)
            emitter.label(end_label)
        elif cond.op == "GE":
            # left >= right  →  right - left <= 0  →  jump to THEN if TOS <= 0
            self.compile_expr(cond.right, emitter)
            self.compile_expr(cond.left, emitter)
            emitter.op(FluxOpcode.Sub)
            emitter.cond_jump(then_label)
            self.compile_expr(node.else_expr, emitter)
            emitter.fwd_jump(end_label)
            emitter.label(then_label)
            self.compile_expr(node.then_expr, emitter)
            emitter.label(end_label)
        elif cond.op == "LT":
            # left < right  →  left - right < 0
            # Approximate: use LE pattern with swapped blocks
            self.compile_expr(cond.right, emitter)
            self.compile_expr(cond.left, emitter)
            emitter.op(FluxOpcode.Sub)
            emitter.cond_jump(then_label)
            self.compile_expr(node.else_expr, emitter)
            emitter.fwd_jump(end_label)
            emitter.label(then_label)
            self.compile_expr(node.then_expr, emitter)
            emitter.label(end_label)
        elif cond.op == "GT":
            # left > right  →  right - left < 0
            self.compile_expr(cond.right, emitter)
            self.compile_expr(cond.left, emitter)
            emitter.op(FluxOpcode.Sub)
            emitter.cond_jump(then_label)
            self.compile_expr(node.else_expr, emitter)
            emitter.fwd_jump(end_label)
            emitter.label(then_label)
            self.compile_expr(node.then_expr, emitter)
            emitter.label(end_label)
        elif cond.op == "EQ":
            # left == right  →  left - right == 0
            self.compile_expr(cond.left, emitter)
            self.compile_expr(cond.right, emitter)
            emitter.op(FluxOpcode.Sub)
            emitter.op(FluxOpcode.Abs)
            emitter.push(1e-6)
            emitter.op(FluxOpcode.Sub)
            emitter.cond_jump(then_label)  # jump if |diff| <= epsilon
            self.compile_expr(node.else_expr, emitter)
            emitter.fwd_jump(end_label)
            emitter.label(then_label)
            self.compile_expr(node.then_expr, emitter)
            emitter.label(end_label)
        else:
            raise ValueError(f"Unsupported comparison op: {cond.op}")

    def _resolve_var(self, name: str) -> float:
        """Resolve a variable name to a float value.

        In a real compiler this would look up from a symbol table or
        runtime environment.  For the prototype we use a default map.
        """
        if name in self.var_defaults:
            return self.var_defaults[name]
        defaults: Dict[str, float] = {
            "weight": 5.0,
            "chaos": 0.3,
            "thermal": 0.8,
            "w_min": 0.0,
            "w_max": 10.0,
            "c_limit": 1.0,
            "t_limit": 0.95,
        }
        return defaults.get(name, 0.0)

    def compile_constraint(
        self,
        expr: Expr,
        *,
        with_validate: bool = True,
        with_halt: bool = True,
    ) -> BytecodeEmitter:
        """Compile a constraint expression into a self-contained bytecode sequence.

        Returns a :class:`BytecodeEmitter` that has already been
        finalised (all labels resolved).
        """
        emitter = BytecodeEmitter()
        self.compile_expr(expr, emitter, with_validate)
        if with_validate:
            emitter.op(FluxOpcode.Validate)
        if with_halt:
            emitter.op(FluxOpcode.Halt)
        return emitter


# ── Convenience API ──


def compile_constraint(
    expr: Expr,
    *,
    prefer_range_check: bool = True,
    with_validate: bool = True,
    with_halt: bool = True,
    var_defaults: dict[str, float] | None = None,
) -> Tuple[bytes, List[float], List[str]]:
    """Compile a constraint and return (bytecode, constant_pool, disassembly).

    Example::

        bc, pool, asm = compile_constraint(
            RangeCheckNode(Var("weight"), 0.0, 10.0),
            prefer_range_check=False,
        )
    """
    compiler = FluxCompiler(
        prefer_range_check=prefer_range_check, var_defaults=var_defaults
    )
    emitter = compiler.compile_constraint(
        expr, with_validate=with_validate, with_halt=with_halt
    )
    return emitter.to_bytes(), emitter.const_pool, emitter.disassemble()
