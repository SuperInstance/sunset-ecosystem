"""Python ast → FLUX bytecode compiler (Forge-Flux Bridge).

The missing link: takes real Python expressions (lambda, function AST)
and produces FLUX VM bytecode via the existing FluxCompiler infrastructure.
"""

from __future__ import annotations

import ast
import inspect
from typing import Callable, List, Tuple

from swarm.flux_compiler import (
    FluxCompiler,
    Const,
    Var,
    BinOp,
    UnaryOp,
    RangeCheckNode,
    IfNode,
    CmpOp,
    Expr,
)

__all__ = [
    "FluxCompileError",
    "PythonASTAdapter",
    "compile_lambda",
    "compile_function",
]


class FluxCompileError(Exception):
    """Raised when a Python construct cannot be compiled to FLUX."""
    pass


class PythonASTAdapter:
    """Translate Python ast.AST into FluxCompiler's internal AST."""

    def __init__(self, var_defaults: dict[str, float] | None = None):
        self.var_defaults = var_defaults or {}

    # ── public entry ──

    def translate(self, node: ast.AST) -> Expr:
        """Translate a Python AST node to a FluxCompiler Expr."""
        if isinstance(node, ast.Constant):
            return self._constant(node)
        elif isinstance(node, ast.Name):
            return self._name(node)
        elif isinstance(node, ast.BinOp):
            return self._binop(node)
        elif isinstance(node, ast.UnaryOp):
            return self._unaryop(node)
        elif isinstance(node, ast.Compare):
            return self._compare(node)
        elif isinstance(node, ast.BoolOp):
            return self._boolop(node)
        elif isinstance(node, ast.IfExp):
            return self._ifexp(node)
        elif isinstance(node, ast.Call):
            return self._call(node)
        elif isinstance(node, ast.Lambda):
            return self.translate(node.body)
        else:
            raise FluxCompileError(
                f"Unsupported Python AST node: {type(node).__name__}"
            )

    # ── leaf nodes ──

    def _constant(self, node: ast.Constant) -> Expr:
        if isinstance(node.value, (int, float)):
            return Const(float(node.value))
        raise FluxCompileError(
            f"Unsupported constant type: {type(node.value).__name__}"
        )

    def _name(self, node: ast.Name) -> Expr:
        return Var(node.id)

    # ── arithmetic ──

    def _binop(self, node: ast.BinOp) -> Expr:
        op_map = {
            ast.Add: "Add",
            ast.Sub: "Sub",
            ast.Mult: "Mul",
            ast.Div: "Div",
            ast.Mod: "Mod",
        }
        op = op_map.get(type(node.op))
        if op is None:
            raise FluxCompileError(
                f"Unsupported binary operator: {type(node.op).__name__}"
            )
        return BinOp(op, self.translate(node.left), self.translate(node.right))

    def _unaryop(self, node: ast.UnaryOp) -> Expr:
        if isinstance(node.op, ast.USub):
            return BinOp("Mul", Const(-1.0), self.translate(node.operand))
        elif isinstance(node.op, ast.UAdd):
            return self.translate(node.operand)
        elif isinstance(node.op, ast.Not):
            return IfNode(
                CmpOp("LE", self.translate(node.operand), Const(0.0)),
                Const(1.0),
                Const(0.0),
            )
        else:
            raise FluxCompileError(
                f"Unsupported unary operator: {type(node.op).__name__}"
            )

    # ── comparisons ──

    def _compare(self, node: ast.Compare) -> Expr:
        # Detect simple range pattern: lo < x < hi  →  RangeCheckNode
        if len(node.ops) == 2:
            op0, op1 = node.ops
            lo_expr = node.left
            mid_expr = node.comparators[0]
            hi_expr = node.comparators[1]
            lo_val = self._try_numeric(lo_expr)
            hi_val = self._try_numeric(hi_expr)
            var_expr = self._try_name(mid_expr)
            # Pattern: constant < name < constant
            if var_expr and lo_val is not None and hi_val is not None:
                if isinstance(op0, (ast.Lt, ast.LtE)) and isinstance(op1, (ast.Lt, ast.LtE)):
                    return RangeCheckNode(var_expr, lo_val, hi_val)
                elif isinstance(op0, (ast.Gt, ast.GtE)) and isinstance(op1, (ast.Gt, ast.GtE)):
                    # Reverse: constant > name > constant → same range
                    return RangeCheckNode(var_expr, hi_val, lo_val)
            # Also check: name > constant > constant (unusual but valid)
            var_expr2 = self._try_name(lo_expr)
            lo_val2 = self._try_numeric(mid_expr)
            hi_val2 = self._try_numeric(hi_expr)
            if var_expr2 and lo_val2 is not None and hi_val2 is not None:
                if isinstance(op0, (ast.Gt, ast.GtE)) and isinstance(op1, (ast.Gt, ast.GtE)):
                    return RangeCheckNode(var_expr2, hi_val2, lo_val2)

        # Flatten chain: a < b < c → a < b and b < c
        if len(node.ops) == 1:
            return self._single_compare(node.ops[0], node.left, node.comparators[0])

        # chained comparison
        parts: List[Expr] = []
        left = node.left
        for op, right in zip(node.ops, node.comparators):
            parts.append(self._single_compare(op, left, right))
            left = right

        # Combine with And (nested IfNode for short-circuit)
        result = parts[-1]
        for part in reversed(parts[:-1]):
            result = IfNode(
                CmpOp("GT", part, Const(0.0)),
                result,
                Const(0.0),
            )
        return result

    def _same_name(self, a: ast.AST, b: ast.AST) -> bool:
        """Check if two AST nodes are the same Name."""
        return isinstance(a, ast.Name) and isinstance(b, ast.Name) and a.id == b.id

    def _try_numeric(self, node: ast.AST) -> float | None:
        """Extract a numeric constant, or None."""
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        return None

    def _try_name(self, node: ast.AST) -> Var | None:
        """Extract a Name as Var, or None."""
        if isinstance(node, ast.Name):
            return Var(node.id)
        return None

    def _single_compare(self, op: ast.cmpop, left: ast.AST, right: ast.AST) -> Expr:
        cmp_map = {
            ast.Lt: "LT",
            ast.LtE: "LE",
            ast.Gt: "GT",
            ast.GtE: "GE",
            ast.Eq: "EQ",
            ast.NotEq: "NE",
        }
        op_str = cmp_map.get(type(op))
        if op_str is None:
            raise FluxCompileError(f"Unsupported comparison: {type(op).__name__}")
        return CmpOp(op_str, self.translate(left), self.translate(right))

    # ── boolean logic ──

    def _boolop(self, node: ast.BoolOp) -> Expr:
        if isinstance(node.op, ast.And):
            # a and b and c → if a then (if b then c else 0) else 0
            result: Expr = self.translate(node.values[-1])
            for val in reversed(node.values[:-1]):
                result = IfNode(
                    CmpOp("GT", self.translate(val), Const(0.0)),
                    result,
                    Const(0.0),
                )
            return result
        elif isinstance(node.op, ast.Or):
            # a or b or c → if a then 1 else (if b then 1 else c)
            result = self.translate(node.values[-1])
            for val in reversed(node.values[:-1]):
                result = IfNode(
                    CmpOp("GT", self.translate(val), Const(0.0)),
                    Const(1.0),
                    result,
                )
            return result
        else:
            raise FluxCompileError(
                f"Unsupported boolean operator: {type(node.op).__name__}"
            )

    # ── conditional expression ──

    def _ifexp(self, node: ast.IfExp) -> Expr:
        return IfNode(
            CmpOp("GT", self.translate(node.test), Const(0.0)),
            self.translate(node.body),
            self.translate(node.orelse),
        )

    # ── function calls ──

    def _call(self, node: ast.Call) -> Expr:
        if not isinstance(node.func, ast.Name):
            raise FluxCompileError(f"Unsupported call: {ast.dump(node.func)}")
        fname = node.func.id
        if fname == "abs" and len(node.args) == 1:
            return UnaryOp("Abs", self.translate(node.args[0]))
        elif fname == "min" and len(node.args) == 2:
            return BinOp("Min", self.translate(node.args[0]), self.translate(node.args[1]))
        elif fname == "max" and len(node.args) == 2:
            return BinOp("Max", self.translate(node.args[0]), self.translate(node.args[1]))
        elif fname == "saturate" and len(node.args) == 3:
            return RangeCheckNode(
                self.translate(node.args[0]),
                self._extract_const(node.args[1]),
                self._extract_const(node.args[2]),
            )
        raise FluxCompileError(f"Unsupported call: {fname}")

    def _extract_const(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        raise FluxCompileError("Expected numeric constant")


# ── public convenience APIs ──


def compile_lambda(
    source: str,
    *,
    prefer_range_check: bool = True,
    with_validate: bool = True,
    var_defaults: dict[str, float] | None = None,
) -> Tuple[bytes, List[float], List[str]]:
    """Compile a Python lambda string to FLUX bytecode.

    Returns (bytecode, constant_pool, disassembly).

    Example::

        bc, pool, asm = compile_lambda("lambda x: x > 0 and x < 100")
    """
    tree = ast.parse(source, mode="eval")
    lam = tree.body
    if not isinstance(lam, ast.Lambda):
        raise FluxCompileError("Source must be a lambda expression")

    adapter = PythonASTAdapter(var_defaults)
    expr = adapter.translate(lam.body)

    compiler = FluxCompiler(prefer_range_check=prefer_range_check, var_defaults=var_defaults)
    emitter = compiler.compile_constraint(expr, with_validate=with_validate, with_halt=True)
    return emitter.to_bytes(), emitter.const_pool, emitter.disassemble()


def compile_function(
    func: Callable,
    *,
    prefer_range_check: bool = True,
    with_validate: bool = True,
    var_defaults: dict[str, float] | None = None,
) -> Tuple[bytes, List[float], List[str]]:
    """Compile a Python function to FLUX bytecode.

    Only the function body is compiled. The function must consist of
    a single return statement with a constraint expression.

    Example::

        def check(x):
            return x > 0 and x < 100
        bc, pool, asm = compile_function(check)
    """
    source = inspect.getsource(func)
    # Dedent in case the function is defined inside a class/method
    source = __import__("textwrap").dedent(source)
    tree = ast.parse(source)
    func_def = tree.body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise FluxCompileError("Expected a function definition")
    if len(func_def.body) != 1 or not isinstance(func_def.body[0], ast.Return):
        raise FluxCompileError("Function must have exactly one return statement")

    adapter = PythonASTAdapter(var_defaults)
    expr = adapter.translate(func_def.body[0].value)

    compiler = FluxCompiler(prefer_range_check=prefer_range_check, var_defaults=var_defaults)
    emitter = compiler.compile_constraint(expr, with_validate=with_validate, with_halt=True)
    return emitter.to_bytes(), emitter.const_pool, emitter.disassemble()
