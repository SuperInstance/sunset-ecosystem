"""Tests for ``swarm/flux_compiler.py`` — minimal FLUX bytecode compiler.

Run: ``python3 -m pytest tests/test_flux_compiler.py -x --tb=short``
"""

from __future__ import annotations

import struct
from typing import List, Tuple

import pytest

from swarm.flux_compiler import (
    BinOp,
    BytecodeEmitter,
    CmpOp,
    Const,
    FluxCompiler,
    FluxOpcode,
    IfNode,
    RangeCheckNode,
    UnaryOp,
    Var,
    compile_constraint,
)


# ── helpers ───────────────────────────────────────────────

def decode_push(bc: bytes, offset: int) -> Tuple[float, int]:
    """Decode a Push instruction at *offset*.  Returns (value, next_offset)."""
    assert bc[offset] == FluxOpcode.Push
    val = struct.unpack("<f", bc[offset + 1 : offset + 5])[0]
    return val, offset + 5


def decode_u16(bc: bytes, offset: int) -> int:
    return struct.unpack("<H", bc[offset : offset + 2])[0]


# ═══════════════════════════════════════════════════════════
# 1. BytecodeEmitter basics
# ═══════════════════════════════════════════════════════════

class TestBytecodeEmitter:
    def test_push_sequence(self):
        e = BytecodeEmitter()
        e.push(1.0).push(2.5).op(FluxOpcode.Add)
        bc = e.to_bytes()
        assert len(bc) == 5 + 5 + 1  # two Push + Add

        v1, off1 = decode_push(bc, 0)
        v2, off2 = decode_push(bc, off1)
        assert v1 == pytest.approx(1.0)
        assert v2 == pytest.approx(2.5)
        assert bc[off2] == FluxOpcode.Add

    def test_load_const(self):
        e = BytecodeEmitter()
        idx = e.add_const(3.14)
        e.load_const(idx)
        bc = e.to_bytes()
        assert bc[0] == FluxOpcode.LoadConst
        assert bc[1] == idx

    def test_label_backpatch_fwd_jump(self):
        e = BytecodeEmitter()
        e.push(1.0)
        e.fwd_jump("end")
        e.push(2.0)  # skipped
        e.label("end")
        e.push(3.0)
        e.op(FluxOpcode.Halt)
        bc = e.to_bytes()

        # Jump instruction layout: FwdJump + u16 offset
        jump_opcode_pos = 5  # after first Push
        assert bc[jump_opcode_pos] == FluxOpcode.FwdJump
        offset = decode_u16(bc, jump_opcode_pos + 1)
        # offset should be the bytes from end of jump operand to label "end"
        # jump operand ends at pos 5 + 3 = 8, label is at 8 (after Push 2.0)
        # Actually: Push(1.0) [0..4], FwdJump [5..7], Push(2.0) [8..12], label at 13
        # Wait, let me recalculate:
        # 0-4: Push 1.0 (5 bytes)
        # 5: FwdJump opcode
        # 6-7: u16 offset
        # 8-12: Push 2.0 (5 bytes) — this is the instruction to skip
        # 13-17: Push 3.0 (5 bytes) — label "end" is at offset 13
        # jump operand ends at byte 8 (index 8). offset = 13 - 8 = 5.
        assert offset == 5

    def test_cond_jump_backpatch(self):
        e = BytecodeEmitter()
        e.push(0.0)
        e.cond_jump("zero")
        e.push(1.0)
        e.label("zero")
        e.op(FluxOpcode.Halt)
        bc = e.to_bytes()
        assert len(bc) > 0
        # Verify no unresolved labels
        assert e._backpatches == []

    def test_unresolved_label_raises(self):
        e = BytecodeEmitter()
        e.fwd_jump("missing")
        with pytest.raises(ValueError, match="Unresolved labels"):
            e.to_bytes()

    def test_range_check_instruction(self):
        e = BytecodeEmitter()
        e.push(5.0)
        e.range_check(0.0, 10.0)
        e.op(FluxOpcode.Halt)
        bc = e.to_bytes()
        # Push(5) + RangeCheck(0,10) + Halt
        assert len(bc) == 5 + 9 + 1
        assert bc[5] == FluxOpcode.RangeCheck
        lo = struct.unpack("<f", bc[6:10])[0]
        hi = struct.unpack("<f", bc[10:14])[0]
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(10.0)

    def test_classify_severity(self):
        e = BytecodeEmitter()
        e.classify_severity(2)  # CRITICAL
        bc = e.to_bytes()
        assert bc[0] == FluxOpcode.ClassifySeverity
        assert bc[1] == 2

    def test_disassemble_output(self):
        e = BytecodeEmitter()
        e.push(3.14).op(FluxOpcode.Dup).op(FluxOpcode.Halt)
        asm = e.disassemble()
        assert "Push" in asm[0]
        assert "Dup" in asm[1]
        assert "Halt" in asm[2]


# ═══════════════════════════════════════════════════════════
# 2. FluxCompiler — expression compilation
# ═══════════════════════════════════════════════════════════

class TestFluxCompilerExpressions:
    def test_compile_const(self):
        bc, pool, asm = compile_constraint(Const(42.0), with_validate=False)
        assert len(bc) == 5 + 1  # Push + Halt
        val, _ = decode_push(bc, 0)
        assert val == pytest.approx(42.0)

    def test_compile_var(self):
        bc, pool, asm = compile_constraint(Var("weight"), with_validate=False)
        assert bc[0] == FluxOpcode.LoadConst
        # weight default is 5.0
        assert pool[bc[1]] == pytest.approx(5.0)

    def test_compile_binop_add(self):
        expr = BinOp("Add", Const(1.0), Const(2.0))
        bc, pool, asm = compile_constraint(expr, with_validate=False)
        v1, off1 = decode_push(bc, 0)
        v2, off2 = decode_push(bc, off1)
        assert v1 == pytest.approx(1.0)
        assert v2 == pytest.approx(2.0)
        assert bc[off2] == FluxOpcode.Add

    def test_compile_binop_sub(self):
        expr = BinOp("Sub", Const(5.0), Const(3.0))
        bc, pool, asm = compile_constraint(expr, with_validate=False)
        _, off = decode_push(bc, 0)
        _, off = decode_push(bc, off)
        assert bc[off] == FluxOpcode.Sub

    def test_compile_binop_mul_div(self):
        expr = BinOp("Div", BinOp("Mul", Const(2.0), Const(3.0)), Const(4.0))
        bc, pool, asm = compile_constraint(expr, with_validate=False)
        assert any(b == FluxOpcode.Mul for b in bc)
        assert any(b == FluxOpcode.Div for b in bc)

    def test_compile_binop_min_max(self):
        expr = BinOp("Min", Const(5.0), Const(3.0))
        bc, pool, asm = compile_constraint(expr, with_validate=False)
        _, off = decode_push(bc, 0)
        _, off = decode_push(bc, off)
        assert bc[off] == FluxOpcode.Min

    def test_compile_unary_abs(self):
        expr = UnaryOp("Abs", Const(-7.0))
        bc, pool, asm = compile_constraint(expr, with_validate=False)
        _, off = decode_push(bc, 0)
        assert bc[off] == FluxOpcode.Abs

    def test_compile_nested_arithmetic(self):
        expr = BinOp(
            "Add",
            BinOp("Mul", Const(2.0), Const(3.0)),
            UnaryOp("Abs", Const(-4.0)),
        )
        bc, pool, asm = compile_constraint(expr, with_validate=False)
        # Just verify it compiles and halts
        assert bc[-1] == FluxOpcode.Halt

    def test_unsupported_op_raises(self):
        compiler = FluxCompiler()
        emitter = BytecodeEmitter()
        with pytest.raises(ValueError, match="Unsupported"):
            compiler.compile_expr(BinOp("Pow", Const(2.0), Const(3.0)), emitter)


# ═══════════════════════════════════════════════════════════
# 3. FluxCompiler — range check (high-level vs low-level)
# ═══════════════════════════════════════════════════════════

class TestFluxCompilerRangeCheck:
    def test_range_check_high_level(self):
        expr = RangeCheckNode(Var("weight"), 0.0, 10.0)
        bc, pool, asm = compile_constraint(expr, prefer_range_check=True)
        # Should be: Push(weight) + RangeCheck + Validate + Halt
        assert bc[0] == FluxOpcode.Push or bc[0] == FluxOpcode.LoadConst
        assert FluxOpcode.RangeCheck in bc
        assert bc[-2] == FluxOpcode.Validate
        assert bc[-1] == FluxOpcode.Halt

    def test_range_check_low_level(self):
        expr = RangeCheckNode(Var("weight"), 0.0, 10.0)
        bc, pool, asm = compile_constraint(
            expr, prefer_range_check=False, with_validate=True, with_halt=False
        )
        # Should contain arithmetic + CondJump branches
        assert FluxOpcode.Sub in bc
        assert FluxOpcode.CondJump in bc
        assert FluxOpcode.ClassifySeverity in bc
        assert FluxOpcode.Validate in bc
        assert FluxOpcode.Halt in bc
        # Check that labels were resolved (no 0xFFFF placeholders)
        # 0xFFFF in little-endian is [0xFF, 0xFF]
        assert b"\xff\xff" not in bc

    def test_range_check_low_level_disasm_readable(self):
        expr = RangeCheckNode(Var("weight"), 0.0, 10.0)
        bc, pool, asm = compile_constraint(
            expr, prefer_range_check=False, with_validate=False, with_halt=False
        )
        assert any("CondJump" in line for line in asm)
        assert any("Sub" in line for line in asm)
        assert any("ClassifySeverity" in line for line in asm)

    def test_range_check_with_var_boundaries(self):
        expr = RangeCheckNode(Var("weight"), 2.0, 8.0)
        bc, pool, asm = compile_constraint(expr, prefer_range_check=True)
        assert bc[-1] == FluxOpcode.Halt


# ═══════════════════════════════════════════════════════════
# 4. FluxCompiler — conditional branches
# ═══════════════════════════════════════════════════════════

class TestFluxCompilerBranches:
    def test_if_le_branch(self):
        # if weight <= 5.0 then 1.0 else 0.0
        expr = IfNode(
            CmpOp("LE", Var("weight"), Const(5.0)),
            Const(1.0),
            Const(0.0),
        )
        bc, pool, asm = compile_constraint(expr, with_validate=False)
        assert FluxOpcode.CondJump in bc
        assert FluxOpcode.FwdJump in bc
        assert b"\xff\xff" not in bc  # all labels resolved

    def test_if_ge_branch(self):
        expr = IfNode(
            CmpOp("GE", Var("weight"), Const(3.0)),
            Const(1.0),
            Const(0.0),
        )
        bc, pool, asm = compile_constraint(expr, with_validate=False)
        assert FluxOpcode.CondJump in bc
        assert b"\xff\xff" not in bc

    def test_if_lt_branch(self):
        expr = IfNode(
            CmpOp("LT", Var("weight"), Const(7.0)),
            Const(1.0),
            Const(0.0),
        )
        bc, pool, asm = compile_constraint(expr, with_validate=False)
        assert FluxOpcode.CondJump in bc
        assert b"\xff\xff" not in bc

    def test_if_gt_branch(self):
        expr = IfNode(
            CmpOp("GT", Var("weight"), Const(2.0)),
            Const(1.0),
            Const(0.0),
        )
        bc, pool, asm = compile_constraint(expr, with_validate=False)
        assert FluxOpcode.CondJump in bc
        assert b"\xff\xff" not in bc

    def test_if_eq_branch(self):
        expr = IfNode(
            CmpOp("EQ", Var("weight"), Const(5.0)),
            Const(1.0),
            Const(0.0),
        )
        bc, pool, asm = compile_constraint(expr, with_validate=False)
        assert FluxOpcode.Abs in bc  # abs-based approximation
        assert b"\xff\xff" not in bc


# ═══════════════════════════════════════════════════════════
# 5. Mini VM interpreter — actually *run* the bytecode
# ═══════════════════════════════════════════════════════════

class MiniFluxVM:
    """Tiny Python interpreter for the PYTHON_SAFE FLUX opcode subset.

    Executes bytecode and returns the top-of-stack value (or raises
    FluxTrap on Validate with 0).
    """

    class FluxTrap(Exception):
        pass

    def __init__(self, const_pool: List[float]) -> None:
        self.const_pool = const_pool

    def run(self, bc: bytes) -> float:
        stack: List[float] = []
        i = 0
        while i < len(bc):
            op = bc[i]
            if op == FluxOpcode.Push:
                val = struct.unpack("<f", bc[i + 1 : i + 5])[0]
                stack.append(val)
                i += 5
            elif op == FluxOpcode.Pop:
                stack.pop()
                i += 1
            elif op == FluxOpcode.Dup:
                stack.append(stack[-1])
                i += 1
            elif op == FluxOpcode.Swap:
                a, b = stack.pop(), stack.pop()
                stack.extend([a, b])
                i += 1
            elif op == FluxOpcode.LoadConst:
                idx = bc[i + 1]
                stack.append(self.const_pool[idx])
                i += 2
            elif op == FluxOpcode.Add:
                b, a = stack.pop(), stack.pop()
                stack.append(a + b)
                i += 1
            elif op == FluxOpcode.Sub:
                b, a = stack.pop(), stack.pop()
                stack.append(a - b)
                i += 1
            elif op == FluxOpcode.Mul:
                b, a = stack.pop(), stack.pop()
                stack.append(a * b)
                i += 1
            elif op == FluxOpcode.Div:
                b, a = stack.pop(), stack.pop()
                stack.append(a / b if b != 0 else float("inf"))
                i += 1
            elif op == FluxOpcode.Min:
                b, a = stack.pop(), stack.pop()
                stack.append(min(a, b))
                i += 1
            elif op == FluxOpcode.Max:
                b, a = stack.pop(), stack.pop()
                stack.append(max(a, b))
                i += 1
            elif op == FluxOpcode.Abs:
                stack.append(abs(stack.pop()))
                i += 1
            elif op == FluxOpcode.Saturate:
                lo = struct.unpack("<f", bc[i + 1 : i + 5])[0]
                hi = struct.unpack("<f", bc[i + 5 : i + 9])[0]
                val = stack.pop()
                stack.append(max(lo, min(hi, val)))
                i += 9
            elif op == FluxOpcode.RangeCheck:
                lo = struct.unpack("<f", bc[i + 1 : i + 5])[0]
                hi = struct.unpack("<f", bc[i + 5 : i + 9])[0]
                val = stack.pop()
                stack.append(1.0 if lo <= val <= hi else 0.0)
                i += 9
            elif op == FluxOpcode.ClassifySeverity:
                sev = bc[i + 1]
                # Just pop and push back for stack balance in this mini VM
                val = stack.pop()
                stack.append(val)
                i += 2
            elif op == FluxOpcode.Validate:
                val = stack.pop()
                if val == 0.0:
                    raise self.FluxTrap("Validate failed")
                stack.append(val)
                i += 1
            elif op == FluxOpcode.FwdJump:
                off = struct.unpack("<H", bc[i + 1 : i + 3])[0]
                i += 3 + off
            elif op == FluxOpcode.CondJump:
                off = struct.unpack("<H", bc[i + 1 : i + 3])[0]
                val = stack.pop()
                if val <= 0:
                    i += 3 + off
                else:
                    i += 3
            elif op == FluxOpcode.Halt:
                break
            elif op == FluxOpcode.Nop:
                i += 1
            else:
                raise ValueError(f"Unhandled opcode 0x{op:02x} at {i}")
        return stack[-1] if stack else 0.0


class TestMiniVMExecution:
    """Verify that emitted bytecode actually executes correctly."""

    def _run(self, bc: bytes, pool: List[float]) -> float:
        return MiniFluxVM(pool).run(bc)

    def test_execute_add(self):
        bc, pool, _ = compile_constraint(
            BinOp("Add", Const(3.0), Const(4.0)), with_validate=False
        )
        assert self._run(bc, pool) == pytest.approx(7.0)

    def test_execute_sub(self):
        bc, pool, _ = compile_constraint(
            BinOp("Sub", Const(10.0), Const(3.0)), with_validate=False
        )
        assert self._run(bc, pool) == pytest.approx(7.0)

    def test_execute_mul_div(self):
        bc, pool, _ = compile_constraint(
            BinOp("Div", BinOp("Mul", Const(2.0), Const(3.0)), Const(4.0)),
            with_validate=False,
        )
        assert self._run(bc, pool) == pytest.approx(1.5)

    def test_execute_min_max(self):
        bc, pool, _ = compile_constraint(
            BinOp("Min", Const(5.0), Const(3.0)), with_validate=False
        )
        assert self._run(bc, pool) == pytest.approx(3.0)

    def test_execute_abs(self):
        bc, pool, _ = compile_constraint(
            UnaryOp("Abs", Const(-7.0)), with_validate=False
        )
        assert self._run(bc, pool) == pytest.approx(7.0)

    def test_execute_range_check_pass(self):
        # weight=5.0 in [0.0, 10.0] → pass (1.0)
        bc, pool, _ = compile_constraint(
            RangeCheckNode(Var("weight"), 0.0, 10.0),
            prefer_range_check=True,
            with_validate=False,
        )
        assert self._run(bc, pool) == pytest.approx(1.0)

    def test_execute_range_check_fail(self):
        # weight=5.0, but range [10.0, 20.0] → fail (0.0)
        # We need to override the variable value. Use Const instead.
        bc, pool, _ = compile_constraint(
            RangeCheckNode(Const(5.0), 10.0, 20.0),
            prefer_range_check=True,
            with_validate=False,
        )
        assert self._run(bc, pool) == pytest.approx(0.0)

    def test_execute_validate_pass(self):
        bc, pool, _ = compile_constraint(
            RangeCheckNode(Const(5.0), 0.0, 10.0), prefer_range_check=True
        )
        # Validate is included, so it should not trap
        assert self._run(bc, pool) == pytest.approx(1.0)

    def test_execute_validate_fail(self):
        bc, pool, _ = compile_constraint(
            RangeCheckNode(Const(15.0), 0.0, 10.0), prefer_range_check=True
        )
        with pytest.raises(MiniFluxVM.FluxTrap):
            self._run(bc, pool)

    def test_execute_if_le_true(self):
        # if 3.0 <= 5.0 then 42.0 else 0.0 → 42.0
        expr = IfNode(CmpOp("LE", Const(3.0), Const(5.0)), Const(42.0), Const(0.0))
        bc, pool, _ = compile_constraint(expr, with_validate=False)
        assert self._run(bc, pool) == pytest.approx(42.0)

    def test_execute_if_le_false(self):
        # if 7.0 <= 5.0 then 42.0 else 99.0 → 99.0
        expr = IfNode(CmpOp("LE", Const(7.0), Const(5.0)), Const(42.0), Const(99.0))
        bc, pool, _ = compile_constraint(expr, with_validate=False)
        assert self._run(bc, pool) == pytest.approx(99.0)

    def test_execute_if_ge_true(self):
        expr = IfNode(CmpOp("GE", Const(7.0), Const(5.0)), Const(1.0), Const(0.0))
        bc, pool, _ = compile_constraint(expr, with_validate=False)
        assert self._run(bc, pool) == pytest.approx(1.0)

    def test_execute_if_ge_false(self):
        expr = IfNode(CmpOp("GE", Const(2.0), Const(5.0)), Const(1.0), Const(0.0))
        bc, pool, _ = compile_constraint(expr, with_validate=False)
        assert self._run(bc, pool) == pytest.approx(0.0)

    def test_execute_low_level_range_pass(self):
        # weight=5.0, range [0.0, 10.0] using arithmetic branches
        bc, pool, _ = compile_constraint(
            RangeCheckNode(Const(5.0), 0.0, 10.0),
            prefer_range_check=False,
            with_validate=False,
        )
        assert self._run(bc, pool) == pytest.approx(1.0)

    def test_execute_low_level_range_fail_low(self):
        # weight=-1.0 < min 0.0 → should hit fail path
        bc, pool, _ = compile_constraint(
            RangeCheckNode(Const(-1.0), 0.0, 10.0),
            prefer_range_check=False,
            with_validate=False,
        )
        # The fail path pushes 0.0 then Validate+Halt
        # But Validate is not included (with_validate=False), so we get 0.0
        assert self._run(bc, pool) == pytest.approx(0.0)

    def test_execute_low_level_range_fail_high(self):
        # weight=15.0 > max 10.0 → should hit fail path
        bc, pool, _ = compile_constraint(
            RangeCheckNode(Const(15.0), 0.0, 10.0),
            prefer_range_check=False,
            with_validate=False,
        )
        assert self._run(bc, pool) == pytest.approx(0.0)

    def test_execute_low_level_range_validate_traps(self):
        bc, pool, _ = compile_constraint(
            RangeCheckNode(Const(15.0), 0.0, 10.0),
            prefer_range_check=False,
            with_validate=True,
        )
        with pytest.raises(MiniFluxVM.FluxTrap):
            self._run(bc, pool)

    def test_execute_saturate(self):
        # Saturate 15.0 to [0.0, 10.0] → 10.0
        emitter = BytecodeEmitter()
        emitter.push(15.0)
        emitter.op(FluxOpcode.Saturate)
        # Manually embed bounds for Saturate
        emitter._emit_f32(0.0)
        emitter._emit_f32(10.0)
        emitter.op(FluxOpcode.Halt)
        bc = emitter.to_bytes()
        assert MiniFluxVM([]).run(bc) == pytest.approx(10.0)


# ═══════════════════════════════════════════════════════════
# 6. Integration — breeding-constraint examples
# ═══════════════════════════════════════════════════════════

def test_compile_weight_bounds_constraint():
    """Realistic constraint: weight must be in [w_min, w_max]."""
    expr = RangeCheckNode(Var("weight"), 0.0, 10.0)
    bc, pool, asm = compile_constraint(expr, prefer_range_check=False)
    assert len(bc) > 0
    assert b"\xff\xff" not in bc
    # Should be executable
    vm = MiniFluxVM(pool)
    # weight defaults to 5.0 → pass
    assert vm.run(bc) == pytest.approx(1.0)


def test_compile_chaos_limit_constraint():
    """chaos <= c_limit."""
    expr = IfNode(
        CmpOp("LE", Var("chaos"), Var("c_limit")),
        Const(1.0),
        Const(0.0),
    )
    bc, pool, asm = compile_constraint(expr, with_validate=False)
    vm = MiniFluxVM(pool)
    # chaos default 0.3, c_limit 1.0 → true
    assert vm.run(bc) == pytest.approx(1.0)


def test_compile_thermal_budget_constraint():
    """thermal <= t_limit."""
    expr = IfNode(
        CmpOp("LE", Var("thermal"), Var("t_limit")),
        Const(1.0),
        Const(0.0),
    )
    bc, pool, asm = compile_constraint(expr, with_validate=False)
    vm = MiniFluxVM(pool)
    # thermal default 0.8, t_limit 0.95 → true
    assert vm.run(bc) == pytest.approx(1.0)


def test_compile_combined_constraint():
    """Combined: weight in bounds AND chaos <= limit.

    We model this as nested IfNodes.
    """
    expr = IfNode(
        CmpOp("LE", Var("chaos"), Var("c_limit")),
        RangeCheckNode(Var("weight"), 0.0, 10.0),
        Const(0.0),
    )
    bc, pool, asm = compile_constraint(expr, with_validate=False, prefer_range_check=True)
    vm = MiniFluxVM(pool)
    assert vm.run(bc) == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════
# 7. Edge cases
# ═══════════════════════════════════════════════════════════

def test_empty_expr_with_halt():
    bc, pool, asm = compile_constraint(Const(0.0), with_validate=False)
    assert bc[-1] == FluxOpcode.Halt


def test_div_by_zero_handled_in_vm():
    bc, pool, _ = compile_constraint(
        BinOp("Div", Const(1.0), Const(0.0)), with_validate=False
    )
    vm = MiniFluxVM(pool)
    assert vm.run(bc) == float("inf")


def test_deeply_nested_expr():
    expr = BinOp(
        "Add",
        BinOp(
            "Add",
            BinOp("Add", Const(1.0), Const(2.0)),
            BinOp("Add", Const(3.0), Const(4.0)),
        ),
        BinOp(
            "Add",
            BinOp("Add", Const(5.0), Const(6.0)),
            BinOp("Add", Const(7.0), Const(8.0)),
        ),
    )
    bc, pool, _ = compile_constraint(expr, with_validate=False)
    vm = MiniFluxVM(pool)
    assert vm.run(bc) == pytest.approx(36.0)


def test_constant_pool_deduplication():
    emitter = BytecodeEmitter()
    idx1 = emitter.add_const(3.14)
    idx2 = emitter.add_const(3.14)
    idx3 = emitter.add_const(2.71)
    assert idx1 == idx2
    assert idx3 != idx1
    assert len(emitter.const_pool) == 2
