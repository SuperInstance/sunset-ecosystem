"""fleet/formula_compiler.py — Spreadsheet formula compiler for fleet orchestration.

Compiles formula expressions like:
    =IF(FLEET_HEALTH() < 0.5, SPAWN("rescue"), IDLE())
into Python callables that interact with FleetConductor, BreederDaemonV2,
MeshVectorGossip, and FLUX gating.

The formula language is intentionally simple — a subset of Excel-like
functions mapped to fleet primitives. Every formula is a cell that
re-evaluates on each fleet beat, producing a side-effect or value.

Usage
-----
    from fleet.formula_compiler import FormulaCompiler, FleetFormulaEnv

    env = FleetFormulaEnv(conductor=conductor, breeder=breeder)
    compiler = FormulaCompiler(env)
    fn = compiler.compile('=IF(FLEET_HEALTH() < 0.5, SPAWN("rescue"), IDLE())')
    result = fn()  # evaluates current fleet state
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Token / AST
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    type: str
    value: str


class FormulaParser:
    """Hand-written recursive-descent parser for fleet formulas.

    Grammar:
        formula     ::= '=' expr
        expr        ::= atom | call | infix | if_expr
        atom        ::= NUMBER | STRING | NAME
        call        ::= NAME '(' [expr (',' expr)*] ')'
        infix       ::= expr OP expr
        if_expr     ::= 'IF' '(' expr ',' expr ',' expr ')'
    """

    TOKEN_SPEC = [
        ("NUMBER", r"\d+(\.\d*)?"),
        ("STRING", r'"[^"]*"'),
        ("NAME", r"[A-Za-z_][A-Za-z0-9_]*"),
        ("OP", r"[\+\-\*/<>=!]+|<=|>=|==|!="),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("COMMA", r","),
        ("SKIP", r"[ \t]+"),
    ]

    def __init__(self, source: str):
        self.source = source.strip()
        if self.source.startswith("="):
            self.source = self.source[1:]
        self.tokens: List[Token] = []
        self._pos = 0
        self._tokenize()

    def _tokenize(self) -> None:
        regex = "|".join(f"(?P<{name}>{pattern})" for name, pattern in self.TOKEN_SPEC)
        for m in re.finditer(regex, self.source):
            kind = m.lastgroup
            value = m.group()
            if kind == "SKIP":
                continue
            self.tokens.append(Token(kind, value))
        self.tokens.append(Token("EOF", ""))

    def _peek(self, offset: int = 0) -> Token:
        idx = self._pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return self.tokens[-1]

    def _consume(self, expected: Optional[str] = None) -> Token:
        tok = self._peek()
        if expected and tok.type != expected:
            raise SyntaxError(f"Expected {expected}, got {tok.type} ({tok.value})")
        self._pos += 1
        return tok

    def parse(self) -> "ExprNode":
        node = self._parse_expr()
        if self._peek().type != "EOF":
            raise SyntaxError(f"Unexpected token {self._peek().value}")
        return node

    def _parse_expr(self) -> "ExprNode":
        return self._parse_if_or_comparison()

    def _parse_if_or_comparison(self) -> "ExprNode":
        left = self._parse_add_sub()
        while self._peek().type == "OP":
            op = self._consume("OP")
            right = self._parse_add_sub()
            left = InfixNode(op.value, left, right)
        return left

    def _parse_add_sub(self) -> "ExprNode":
        left = self._parse_mul_div()
        while self._peek().type == "OP" and self._peek().value in ("+", "-"):
            op = self._consume("OP")
            right = self._parse_mul_div()
            left = InfixNode(op.value, left, right)
        return left

    def _parse_mul_div(self) -> "ExprNode":
        left = self._parse_atom()
        while self._peek().type == "OP" and self._peek().value in ("*", "/"):
            op = self._consume("OP")
            right = self._parse_atom()
            left = InfixNode(op.value, left, right)
        return left

    def _parse_atom(self) -> "ExprNode":
        tok = self._peek()
        if tok.type == "NUMBER":
            self._consume()
            return NumberNode(float(tok.value))
        if tok.type == "STRING":
            self._consume()
            return StringNode(tok.value[1:-1])  # strip quotes
        if tok.type == "NAME":
            self._consume()
            if self._peek().type == "LPAREN":
                self._consume("LPAREN")
                args: List[ExprNode] = []
                if self._peek().type != "RPAREN":
                    args.append(self._parse_expr())
                    while self._peek().type == "COMMA":
                        self._consume("COMMA")
                        args.append(self._parse_expr())
                self._consume("RPAREN")
                return CallNode(tok.value, args)
            return NameNode(tok.value)
        if tok.type == "LPAREN":
            self._consume("LPAREN")
            node = self._parse_expr()
            self._consume("RPAREN")
            return node
        raise SyntaxError(f"Unexpected token {tok.value}")


# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------


class ExprNode:
    pass


@dataclass
class NumberNode(ExprNode):
    value: float


@dataclass
class StringNode(ExprNode):
    value: str


@dataclass
class NameNode(ExprNode):
    name: str


@dataclass
class CallNode(ExprNode):
    func: str
    args: List[ExprNode]


@dataclass
class InfixNode(ExprNode):
    op: str
    left: ExprNode
    right: ExprNode


# ---------------------------------------------------------------------------
# Runtime environment
# ---------------------------------------------------------------------------


class FleetFormulaEnv:
    """Runtime bindings for fleet formulas.

    Provides the fleet-specific functions that formulas call.
    All functions are pure (no side-effects) unless documented.
    """

    def __init__(
        self,
        conductor: Any = None,
        breeder: Any = None,
        mesh: Any = None,
        flux: Any = None,
        telemetry: Any = None,
    ):
        self.conductor = conductor
        self.breeder = breeder
        self.mesh = mesh
        self.flux = flux
        self.telemetry = telemetry
        self._registry: Dict[str, Callable[..., Any]] = self._build_registry()

    def _build_registry(self) -> Dict[str, Callable[..., Any]]:
        return {
            # Fleet health
            "FLEET_HEALTH": self._fleet_health,
            "THERMAL_AVG": self._thermal_avg,
            "QUEUE_DEPTH": self._queue_depth,
            "AGENT_COUNT": self._agent_count,
            "BEAT_COUNT": self._beat_count,
            # Actions
            "SPAWN": self._spawn,
            "BREED": self._breed,
            "MESH": self._mesh,
            "FLUX_CHECK": self._flux_check,
            "ALERT": self._alert,
            "IDLE": self._idle,
            "STOP": self._stop,
            # Aggregates
            "COUNTIF": self._countif,
            "AVERAGE": self._average,
            "MAX": self._max,
            "MIN": self._min,
            # Utility
            "IF": self._if,
            "AND": self._and,
            "OR": self._or,
            "NOT": self._not,
        }

    def lookup(self, name: str) -> Callable[..., Any]:
        if name not in self._registry:
            raise NameError(f"Unknown fleet function: {name}")
        return self._registry[name]

    # --- Fleet health ------------------------------------------------------

    def _fleet_health(self) -> float:
        if self.conductor is None:
            return 1.0
        return getattr(self.conductor, "health", lambda: 1.0)()

    def _thermal_avg(self) -> float:
        if self.telemetry is None:
            return 0.0
        return getattr(self.telemetry, "thermal_avg", lambda: 0.0)()

    def _queue_depth(self) -> int:
        if self.conductor is None:
            return 0
        return getattr(self.conductor, "queue_depth", lambda: 0)()

    def _agent_count(self) -> int:
        if self.breeder is None:
            return 0
        return getattr(self.breeder, "agent_count", lambda: 0)()

    def _beat_count(self) -> int:
        if self.conductor is None:
            return 0
        return getattr(self.conductor, "beat_count", lambda: 0)()

    # --- Actions -----------------------------------------------------------

    def _spawn(self, agent_type: str, count: int = 1) -> str:
        """Return a spawn command token."""
        return f"SPAWN:{agent_type}:{count}"

    def _breed(self, strategy: str = "diversity") -> str:
        return f"BREED:{strategy}"

    def _mesh(self, *targets: str) -> str:
        return f"MESH:{','.join(targets)}"

    def _flux_check(self, constraint: str, strict: bool = False) -> bool:
        if self.flux is None:
            return True
        return getattr(self.flux, "check", lambda c, s: True)(constraint, strict)

    def _alert(self, channel: str, message: str) -> str:
        return f"ALERT:{channel}:{message}"

    def _idle(self) -> str:
        return "IDLE"

    def _stop(self) -> str:
        return "STOP"

    # --- Aggregates --------------------------------------------------------

    def _countif(self, collection: Any, predicate: Any) -> int:
        if not isinstance(collection, (list, tuple, set)):
            return 0
        return sum(1 for item in collection if item == predicate)

    def _average(self, *values: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1 and isinstance(values[0], (list, tuple)):
            values = values[0]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _max(self, *values: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1 and isinstance(values[0], (list, tuple)):
            values = values[0]
        if not values:
            return 0.0
        return max(values)

    def _min(self, *values: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1 and isinstance(values[0], (list, tuple)):
            values = values[0]
        if not values:
            return 0.0
        return min(values)

    # --- Logic -------------------------------------------------------------

    def _if(self, cond: bool, true_val: Any, false_val: Any) -> Any:
        return true_val if cond else false_val

    def _and(self, *args: bool) -> bool:
        return all(args)

    def _or(self, *args: bool) -> bool:
        return any(args)

    def _not(self, arg: bool) -> bool:
        return not arg


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------


class FormulaCompiler:
    """Compiles parsed formula AST into a Python callable."""

    def __init__(self, env: FleetFormulaEnv):
        self.env = env
        self.cell_resolver: Callable[[str], Any] | None = None

    def compile(self, source: str) -> Callable[[], Any]:
        parser = FormulaParser(source)
        ast = parser.parse()
        return lambda: self._eval_node(ast)

    def compile_batch(self, sources: List[str]) -> List[Callable[[], Any]]:
        return [self.compile(s) for s in sources]

    def _eval_node(self, node: ExprNode) -> Any:
        if isinstance(node, NumberNode):
            return node.value
        if isinstance(node, StringNode):
            return node.value
        if isinstance(node, NameNode):
            # Look up as a constant or function with no args
            try:
                return self.env.lookup(node.name)()
            except NameError:
                if self.cell_resolver is not None:
                    return self.cell_resolver(node.name)
                return node.name
            except TypeError:
                return node.name
        if isinstance(node, CallNode):
            fn = self.env.lookup(node.func)
            args = [self._eval_node(arg) for arg in node.args]
            return fn(*args)
        if isinstance(node, InfixNode):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self._eval_infix(node.op, left, right)
        raise TypeError(f"Unknown node type: {type(node)}")

    def _eval_infix(self, op: str, left: Any, right: Any) -> Any:
        # Propagate error tokens
        if isinstance(left, str) and left.startswith("#"):
            return left
        if isinstance(right, str) and right.startswith("#"):
            return right
        ops: Dict[str, Callable[[Any, Any], Any]] = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: a / b if b != 0 else float("inf"),
            "<": lambda a, b: a < b,
            ">": lambda a, b: a > b,
            "<=": lambda a, b: a <= b,
            ">=": lambda a, b: a >= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
        }
        if op not in ops:
            raise ValueError(f"Unknown operator: {op}")
        return ops[op](left, right)

    def to_python_ast(self, node: ExprNode) -> ast.AST:
        """Convert formula AST to Python AST for codegen."""
        if isinstance(node, NumberNode):
            return ast.Constant(value=node.value)
        if isinstance(node, StringNode):
            return ast.Constant(value=node.value)
        if isinstance(node, NameNode):
            return ast.Name(id=node.name, ctx=ast.Load())
        if isinstance(node, CallNode):
            return ast.Call(
                func=ast.Name(id=node.func, ctx=ast.Load()),
                args=[self.to_python_ast(arg) for arg in node.args],
                keywords=[],
            )
        if isinstance(node, InfixNode):
            op_map = {
                "+": ast.Add(),
                "-": ast.Sub(),
                "*": ast.Mult(),
                "/": ast.Div(),
                "<": ast.Lt(),
                ">": ast.Gt(),
                "<=": ast.LtE(),
                ">=": ast.GtE(),
                "==": ast.Eq(),
                "!=": ast.NotEq(),
            }
            return (
                ast.Compare(
                    left=self.to_python_ast(node.left),
                    ops=[op_map[node.op]],
                    comparators=[self.to_python_ast(node.right)],
                )
                if node.op in {"<", ">", "<=", ">=", "==", "!="}
                else ast.BinOp(
                    left=self.to_python_ast(node.left),
                    op=op_map[node.op],
                    right=self.to_python_ast(node.right),
                )
            )
        raise TypeError(f"Unknown node type: {type(node)}")

    def to_python_source(self, source: str) -> str:
        """Return Python source equivalent of the formula."""
        parser = FormulaParser(source)
        node = parser.parse()
        py_ast = self.to_python_ast(node)
        return ast.unparse(py_ast)
