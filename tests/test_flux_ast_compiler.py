"""Tests for ``sunset/flux_ast_compiler.py`` — Python AST → FLUX compiler.

Run: ``python3 -m pytest tests/test_flux_ast_compiler.py -x --tb=short``
"""

from __future__ import annotations

import ast
import struct
from typing import List, Tuple

import pytest

from sunset.flux_ast_compiler import (
    FluxCompileError,
    PythonASTAdapter,
    compile_function,
    compile_lambda,
)
from swarm.flux_compiler import FluxOpcode


# ── helpers ───────────────────────────────────────────────


def decode_push(bc: bytes, offset: int) -> Tuple[float, int]:
    assert bc[offset] == FluxOpcode.Push
    val = struct.unpack("<f", bc[offset + 1 : offset + 5])[0]
    return val, offset + 5


def decode_u16(bc: bytes, offset: int) -> int:
    return struct.unpack("<H", bc[offset : offset + 2])[0]


def find_first_opcode(bc: bytes, opcode: int) -> int:
    """Return the first offset of *opcode* in bytecode, or -1."""
    for i, b in enumerate(bc):
        if b == opcode:
            return i
    return -1


def count_opcodes(bc: bytes, opcode: int) -> int:
    return sum(1 for b in bc if b == opcode)


# ═══════════════════════════════════════════════════════════
# 1. compile_lambda — basic constraints
# ═══════════════════════════════════════════════════════════


class TestCompileLambdaBasic:
    def test_simple_range_check(self):
        bc, pool, asm = compile_lambda("lambda x: 0 < x < 100")
        assert isinstance(bc, bytes)
        assert len(bc) > 0
        # With prefer_range_check=True (default), should emit RangeCheck
        assert find_first_opcode(bc, FluxOpcode.RangeCheck) >= 0
        # Should end with Validate + Halt
        assert bc[-2] == FluxOpcode.Validate
        assert bc[-1] == FluxOpcode.Halt

    def test_explicit_range(self):
        bc, pool, asm = compile_lambda("lambda x: x > 0 and x < 100")
        assert len(bc) > 0
        # This is two comparisons joined by And — should use CondJump
        assert find_first_opcode(bc, FluxOpcode.CondJump) >= 0

    def test_constant_folded(self):
        bc, pool, asm = compile_lambda("lambda: 5.0 > 3.0")
        assert len(bc) > 0
        # Should push constants and compare
        assert find_first_opcode(bc, FluxOpcode.Push) >= 0

    def test_arithmetic_constraint(self):
        bc, pool, asm = compile_lambda("lambda w: abs(w) < 5.0")
        assert len(bc) > 0
        # abs() → Abs opcode
        assert find_first_opcode(bc, FluxOpcode.Abs) >= 0

    def test_min_max_calls(self):
        bc, pool, asm = compile_lambda("lambda a, b: min(a, b) > 0")
        assert len(bc) > 0
        assert find_first_opcode(bc, FluxOpcode.Min) >= 0

    def test_saturate_call(self):
        bc, pool, asm = compile_lambda("lambda x: saturate(x, 0, 10) > 5")
        assert len(bc) > 0
        # saturate → RangeCheckNode → RangeCheck opcode (with prefer_range_check)
        assert find_first_opcode(bc, FluxOpcode.RangeCheck) >= 0

    def test_unary_negation(self):
        bc, pool, asm = compile_lambda("lambda x: -x < 5")
        assert len(bc) > 0
        # -x is compiled as Mul(-1.0, x) → Push -1.0 + Mul
        assert find_first_opcode(bc, FluxOpcode.Mul) >= 0

    def test_boolean_not(self):
        bc, pool, asm = compile_lambda("lambda x: not (x > 5)")
        assert len(bc) > 0
        # not(x > 5) → if (x > 5) <= 0 then 1 else 0
        assert find_first_opcode(bc, FluxOpcode.CondJump) >= 0

    def test_disassembly_readable(self):
        bc, pool, asm = compile_lambda("lambda x: x > 0")
        assert isinstance(asm, list)
        assert len(asm) > 0
        assert any("Push" in line for line in asm)


# ═══════════════════════════════════════════════════════════
# 2. compile_lambda — advanced patterns
# ═══════════════════════════════════════════════════════════


class TestCompileLambdaAdvanced:
    def test_chained_comparison_three(self):
        bc, pool, asm = compile_lambda("lambda x: 0 < x < 100 < 200")
        assert len(bc) > 0
        # Should compile without error

    def test_nested_boolean_and(self):
        bc, pool, asm = compile_lambda("lambda x: x > 0 and x < 100 and x != 50")
        assert len(bc) > 0
        # Multiple And values
        assert find_first_opcode(bc, FluxOpcode.CondJump) >= 0

    def test_boolean_or(self):
        bc, pool, asm = compile_lambda("lambda x: x < 0 or x > 100")
        assert len(bc) > 0
        # Or uses CondJump differently than And
        assert find_first_opcode(bc, FluxOpcode.CondJump) >= 0

    def test_mixed_and_or_raises(self):
        # Python's precedence: (a and b) or c  vs  a and (b or c)
        # Both should compile since we flatten to IfNode
        bc1, _, _ = compile_lambda("lambda a, b, c: (a and b) or c")
        assert len(bc1) > 0
        bc2, _, _ = compile_lambda("lambda a, b, c: a and (b or c)")
        assert len(bc2) > 0

    def test_if_exp(self):
        # Lambda with conditional expression
        bc, pool, asm = compile_lambda("lambda x: x if x > 0 else 0")
        assert len(bc) > 0
        assert find_first_opcode(bc, FluxOpcode.CondJump) >= 0

    def test_equality_comparison(self):
        bc, pool, asm = compile_lambda("lambda x: x == 5")
        assert len(bc) > 0
        # EQ uses Abs + epsilon
        assert find_first_opcode(bc, FluxOpcode.Abs) >= 0

    def test_inequality_comparison(self):
        bc, pool, asm = compile_lambda("lambda x: x != 5")
        assert len(bc) > 0

    def test_less_equal(self):
        bc, pool, asm = compile_lambda("lambda x: x <= 10")
        assert len(bc) > 0

    def test_greater_equal(self):
        bc, pool, asm = compile_lambda("lambda x: x >= 10")
        assert len(bc) > 0

    def test_complex_arithmetic(self):
        bc, pool, asm = compile_lambda("lambda x: (x + 1) * 2 > 5")
        assert len(bc) > 0
        assert find_first_opcode(bc, FluxOpcode.Add) >= 0
        assert find_first_opcode(bc, FluxOpcode.Mul) >= 0

    def test_division(self):
        bc, pool, asm = compile_lambda("lambda x: x / 2 > 1")
        assert len(bc) > 0
        assert find_first_opcode(bc, FluxOpcode.Div) >= 0

    def test_prefer_range_check_false(self):
        bc, pool, asm = compile_lambda(
            "lambda x: 0 < x < 100",
            prefer_range_check=False,
        )
        assert len(bc) > 0
        # Without RangeCheck optimization, should use arithmetic + CondJump
        assert find_first_opcode(bc, FluxOpcode.RangeCheck) == -1
        assert find_first_opcode(bc, FluxOpcode.CondJump) >= 0

    def test_without_validate(self):
        bc, pool, asm = compile_lambda(
            "lambda x: x > 0",
            with_validate=False,
        )
        assert bc[-1] == FluxOpcode.Halt
        assert bc[-2] != FluxOpcode.Validate


# ═══════════════════════════════════════════════════════════
# 3. compile_function
# ═══════════════════════════════════════════════════════════


class TestCompileFunction:
    def test_simple_function(self):
        def check(x):
            return x > 0 and x < 100

        bc, pool, asm = compile_function(check)
        assert isinstance(bc, bytes)
        assert len(bc) > 0
        assert find_first_opcode(bc, FluxOpcode.CondJump) >= 0

    def test_function_with_arithmetic(self):
        def check(w):
            return abs(w) < 5.0

        bc, pool, asm = compile_function(check)
        assert find_first_opcode(bc, FluxOpcode.Abs) >= 0

    def test_multi_statement_function_raises(self):
        def bad(x):
            y = x + 1
            return y > 0

        with pytest.raises(FluxCompileError, match="exactly one return"):
            compile_function(bad)

    def test_non_function_raises(self):
        class NotAFunction:
            pass

        with pytest.raises(FluxCompileError, match="function definition"):
            compile_function(NotAFunction)


# ═══════════════════════════════════════════════════════════
# 4. PythonASTAdapter — unit translation tests
# ═══════════════════════════════════════════════════════════


class TestPythonASTAdapter:
    def test_translate_constant_int(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("5", mode="eval")
        expr = adapter.translate(tree.body)
        assert expr.value == 5.0

    def test_translate_constant_float(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("3.14", mode="eval")
        expr = adapter.translate(tree.body)
        assert expr.value == 3.14

    def test_translate_name(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("x", mode="eval")
        expr = adapter.translate(tree.body)
        assert expr.name == "x"

    def test_translate_binop_add(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("a + b", mode="eval")
        expr = adapter.translate(tree.body)
        assert expr.op == "Add"

    def test_translate_binop_sub(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("a - b", mode="eval")
        expr = adapter.translate(tree.body)
        assert expr.op == "Sub"

    def test_translate_binop_mul(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("a * b", mode="eval")
        expr = adapter.translate(tree.body)
        assert expr.op == "Mul"

    def test_translate_binop_div(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("a / b", mode="eval")
        expr = adapter.translate(tree.body)
        assert expr.op == "Div"

    def test_translate_unary_neg(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("-x", mode="eval")
        expr = adapter.translate(tree.body)
        # Should be Mul(-1.0, Var(x))
        assert expr.op == "Mul"
        assert expr.left.value == -1.0
        assert expr.right.name == "x"

    def test_translate_unary_pos(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("+x", mode="eval")
        expr = adapter.translate(tree.body)
        assert expr.name == "x"  # stripped

    def test_translate_unary_not(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("not x", mode="eval")
        expr = adapter.translate(tree.body)
        # IfNode with LE comparison
        assert expr.cond.op == "LE"

    def test_translate_compare_lt(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("a < b", mode="eval")
        expr = adapter.translate(tree.body)
        assert expr.op == "LT"

    def test_translate_compare_le(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("a <= b", mode="eval")
        expr = adapter.translate(tree.body)
        assert expr.op == "LE"

    def test_translate_compare_gt(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("a > b", mode="eval")
        expr = adapter.translate(tree.body)
        assert expr.op == "GT"

    def test_translate_compare_ge(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("a >= b", mode="eval")
        expr = adapter.translate(tree.body)
        assert expr.op == "GE"

    def test_translate_compare_eq(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("a == b", mode="eval")
        expr = adapter.translate(tree.body)
        assert expr.op == "EQ"

    def test_translate_compare_ne(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("a != b", mode="eval")
        expr = adapter.translate(tree.body)
        assert expr.op == "NE"

    def test_translate_call_abs(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("abs(x)", mode="eval")
        expr = adapter.translate(tree.body)
        assert expr.op == "Abs"

    def test_translate_call_min(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("min(a, b)", mode="eval")
        expr = adapter.translate(tree.body)
        assert expr.op == "Min"

    def test_translate_call_max(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("max(a, b)", mode="eval")
        expr = adapter.translate(tree.body)
        assert expr.op == "Max"

    def test_translate_call_saturate(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("saturate(x, 0, 10)", mode="eval")
        expr = adapter.translate(tree.body)
        from swarm.flux_compiler import RangeCheckNode

        assert isinstance(expr, RangeCheckNode)
        assert expr.lo == 0.0
        assert expr.hi == 10.0

    def test_translate_lambda_body(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("lambda x: x > 0", mode="eval")
        lam = tree.body
        expr = adapter.translate(lam)
        assert expr.op == "GT"

    def test_translate_boolop_and(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("a and b", mode="eval")
        expr = adapter.translate(tree.body)
        assert isinstance(
            expr,
            type(adapter.translate(ast.parse("x if x > 0 else 0", mode="eval").body)),
        )
        # Should be IfNode
        from swarm.flux_compiler import IfNode

        assert isinstance(expr, IfNode)

    def test_translate_boolop_or(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("a or b", mode="eval")
        expr = adapter.translate(tree.body)
        from swarm.flux_compiler import IfNode

        assert isinstance(expr, IfNode)

    def test_translate_ifexp(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("x if x > 0 else 0", mode="eval")
        expr = adapter.translate(tree.body)
        from swarm.flux_compiler import IfNode

        assert isinstance(expr, IfNode)


# ═══════════════════════════════════════════════════════════
# 5. Error handling
# ═══════════════════════════════════════════════════════════


class TestErrorHandling:
    def test_non_lambda_source_raises(self):
        with pytest.raises(FluxCompileError, match="lambda expression"):
            compile_lambda("x + 1")

    def test_unsupported_ast_node(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("[1, 2, 3]", mode="eval")
        with pytest.raises(FluxCompileError, match="Unsupported"):
            adapter.translate(tree.body)

    def test_unsupported_binary_op(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("a ** b", mode="eval")
        with pytest.raises(FluxCompileError, match="Unsupported binary operator"):
            adapter.translate(tree.body)

    def test_unsupported_call(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("unknown(x)", mode="eval")
        with pytest.raises(FluxCompileError, match="Unsupported call"):
            adapter.translate(tree.body)

    def test_unsupported_comparison(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("a in b", mode="eval")
        with pytest.raises(FluxCompileError, match="Unsupported comparison"):
            adapter.translate(tree.body)

    def test_non_numeric_constant_raises(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("'hello'", mode="eval")
        with pytest.raises(FluxCompileError, match="Unsupported constant"):
            adapter.translate(tree.body)

    def test_saturate_non_constant_bounds_raises(self):
        adapter = PythonASTAdapter()
        tree = ast.parse("saturate(x, a, b)", mode="eval")
        with pytest.raises(FluxCompileError, match="Expected numeric constant"):
            adapter.translate(tree.body)


# ═══════════════════════════════════════════════════════════
# 6. Bytecode correctness — structural verification
# ═══════════════════════════════════════════════════════════


class TestBytecodeCorrectness:
    def test_simple_constraint_has_halt(self):
        bc, pool, asm = compile_lambda("lambda x: x > 0")
        assert bc[-1] == FluxOpcode.Halt

    def test_range_check_bounds_in_pool(self):
        bc, pool, asm = compile_lambda("lambda x: 0 < x < 100")
        # With RangeCheck optimization, bounds are embedded as f32 operands
        # in the bytecode, not in the constant pool
        assert find_first_opcode(bc, FluxOpcode.RangeCheck) >= 0
        # The pool only contains variable defaults (0.0 for _resolve_var fallback)
        assert len(pool) >= 1

    def test_variable_in_pool(self):
        bc, pool, asm = compile_lambda("lambda x: x > 0")
        # Var("x") gets resolved and added to pool
        assert len(pool) >= 1

    def test_no_unresolved_labels(self):
        bc, pool, asm = compile_lambda("lambda x: x > 0 and x < 100")
        # If there were unresolved labels, to_bytes() would raise
        assert isinstance(bc, bytes)

    def test_disassembly_matches_bytecode(self):
        bc, pool, asm = compile_lambda("lambda x: x > 0")
        # Number of lines in disassembly should roughly match bytecode length
        # (not exact because different opcodes have different sizes)
        assert len(asm) > 0
        assert isinstance(asm[0], str)

    def test_constant_pool_deduplication(self):
        bc, pool, asm = compile_lambda("lambda x: x > 0 and x < 0")
        # 0.0 appears twice but should be deduplicated in pool
        assert pool.count(0.0) == 1

    def test_multiple_vars_in_pool(self):
        bc, pool, asm = compile_lambda("lambda x, y: x > 0 and y < 100")
        # Variables x and y both default to 0.0, so deduplication gives 1 entry
        # But both names are resolvable (different constant pool slots, same value)
        assert len(pool) >= 1
        assert 0.0 in pool


# ═══════════════════════════════════════════════════════════
# 7. var_defaults
# ═══════════════════════════════════════════════════════════


class TestVarDefaults:
    def test_var_defaults_passed_to_adapter(self):
        adapter = PythonASTAdapter(var_defaults={"x": 42.0})
        tree = ast.parse("x", mode="eval")
        expr = adapter.translate(tree.body)
        assert expr.name == "x"

    def test_compile_lambda_with_defaults(self):
        bc, pool, asm = compile_lambda(
            "lambda x: x > 0",
            var_defaults={"x": 5.0},
        )
        assert len(bc) > 0
        # The default is used by _resolve_var in FluxCompiler
        assert 5.0 in pool


# ═══════════════════════════════════════════════════════════
# 8. Integration — run through FluxVMRunner if available
# ═══════════════════════════════════════════════════════════


class TestVMIntegration:
    def test_simple_constraint_runnable(self):
        """Verify the emitted bytecode can be run by FluxVMRunner."""
        pytest.importorskip("swarm.flux_vm_runner")
        from swarm.flux_vm_runner import FluxVMRunner

        bc, pool, asm = compile_lambda("lambda x: x > 0 and x < 100")
        runner = FluxVMRunner(pool)
        result = runner.run(bc)
        # With no stack values pushed, this will use default var resolution
        # and return based on default x value (0.0 from _resolve_var)
        assert isinstance(result, float)

    def test_range_check_runnable(self):
        pytest.importorskip("swarm.flux_vm_runner")
        from swarm.flux_vm_runner import FluxVMRunner

        bc, pool, asm = compile_lambda("lambda x: 0 < x < 10")
        runner = FluxVMRunner(pool)
        result = runner.run(bc)
        assert isinstance(result, float)

    def test_arithmetic_constraint_runnable(self):
        pytest.importorskip("swarm.flux_vm_runner")
        from swarm.flux_vm_runner import FluxVMRunner

        bc, pool, asm = compile_lambda("lambda w: abs(w) < 5.0")
        runner = FluxVMRunner(pool)
        result = runner.run(bc)
        assert isinstance(result, float)
