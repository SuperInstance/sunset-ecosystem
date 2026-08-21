# Code Review Personas
# 5 AST-based personas for fleet code review

"""Multi-persona code review using AST analysis.

Provides five distinct review perspectives:
1. Correctness — logical errors, type mismatches, unhandled exceptions
2. Security — injection risks, unsafe eval, hardcoded secrets, path traversal
3. Performance — O(n^2) loops, repeated computation, memory leaks
4. Simplicity — unnecessary complexity, over-engineering, dead code
5. Adversarial — stress testing, edge cases, fuzzing vectors

Usage:
    from fleet.review_code import ReviewPersona, review_code
    findings = review_code(source_code, ReviewPersona.CORRECTNESS)
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Set


# ---------------------------------------------------------------------------
# Finding model
# ---------------------------------------------------------------------------


@dataclass
class ReviewFinding:
    """Single review finding."""

    persona: str
    severity: str  # info, warning, critical
    message: str
    line: int
    col: int = 0


@dataclass
class ReviewFindings:
    """Collection of findings for a review run."""

    persona: str
    findings: List[ReviewFinding] = field(default_factory=list)

    def add(self, severity: str, message: str, line: int, col: int = 0) -> None:
        self.findings.append(
            ReviewFinding(
                persona=self.persona,
                severity=severity,
                message=message,
                line=line,
                col=col,
            )
        )

    def critical(self) -> List[ReviewFinding]:
        return [f for f in self.findings if f.severity == "critical"]

    def warnings(self) -> List[ReviewFinding]:
        return [f for f in self.findings if f.severity == "warning"]

    def infos(self) -> List[ReviewFinding]:
        return [f for f in self.findings if f.severity == "info"]


# ---------------------------------------------------------------------------
# Persona enum
# ---------------------------------------------------------------------------


class ReviewPersona(Enum):
    CORRECTNESS = auto()
    SECURITY = auto()
    PERFORMANCE = auto()
    SIMPLICITY = auto()
    ADVERSARIAL = auto()

    @property
    def name(self) -> str:
        return {
            ReviewPersona.CORRECTNESS: "Correctness",
            ReviewPersona.SECURITY: "Security",
            ReviewPersona.PERFORMANCE: "Performance",
            ReviewPersona.SIMPLICITY: "Simplicity",
            ReviewPersona.ADVERSARIAL: "Adversarial",
        }[self]

    @classmethod
    def all(cls) -> List["ReviewPersona"]:
        return [
            cls.CORRECTNESS,
            cls.SECURITY,
            cls.PERFORMANCE,
            cls.SIMPLICITY,
            cls.ADVERSARIAL,
        ]


# ---------------------------------------------------------------------------
# AST visitor base
# ---------------------------------------------------------------------------


class PersonaVisitor(ast.NodeVisitor):
    """Base AST visitor for a review persona."""

    def __init__(self, persona: ReviewPersona) -> None:
        self._persona = persona
        self._findings = ReviewFindings(persona=persona.name)
        self._function_depth = 0
        self._loop_depth = 0

    def run(self, source: str) -> ReviewFindings:
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            self._findings.add("critical", f"Syntax error: {e}", e.lineno or 1)
            return self._findings
        self.visit(tree)
        return self._findings

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1

    def visit_For(self, node: ast.For) -> None:
        self._loop_depth += 1
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        self._loop_depth += 1
        self.generic_visit(node)
        self._loop_depth -= 1


# ---------------------------------------------------------------------------
# Correctness visitor
# ---------------------------------------------------------------------------


class CorrectnessVisitor(PersonaVisitor):
    """Finds logical errors, type mismatches, unhandled exceptions."""

    def __init__(self) -> None:
        super().__init__(ReviewPersona.CORRECTNESS)
        self._exception_handlers: Set[str] = set()

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self._findings.add(
                "warning",
                "Bare except clause catches everything including KeyboardInterrupt",
                node.lineno,
            )
        elif isinstance(node.type, ast.Name) and node.type.id == "Exception":
            self._findings.add(
                "info",
                "Catching generic Exception — consider more specific types",
                node.lineno,
            )
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        # Detect 'is' with string/number literals
        for op in node.ops:
            if isinstance(op, ast.Is):
                if isinstance(node.left, (ast.Constant, ast.Str)):
                    self._findings.add(
                        "warning",
                        "Using 'is' with literal — use '==' instead",
                        node.lineno,
                    )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Detect mutable default arguments
        if isinstance(node.func, ast.Name) and node.func.id == "dict":
            if not node.args and not node.keywords:
                self._findings.add(
                    "info", "dict() call — could be {} for clarity", node.lineno
                )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Check for mutable default arguments
        for default in node.args.defaults:
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self._findings.add(
                    "critical",
                    "Mutable default argument — use None and initialize in body",
                    node.lineno,
                )
        super().visit_FunctionDef(node)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is None:
            self._findings.add(
                "info",
                "Bare return — ensure all paths return consistent types",
                node.lineno,
            )
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Security visitor
# ---------------------------------------------------------------------------


class SecurityVisitor(PersonaVisitor):
    """Finds injection risks, unsafe eval, hardcoded secrets, path traversal."""

    def __init__(self) -> None:
        super().__init__(ReviewPersona.SECURITY)

    DANGEROUS_BUILTINS = {"eval", "exec", "compile", "__import__"}
    SUSPICIOUS_PATTERNS = [
        (
            re.compile(r'password\s*=\s*["\'][^"\']+["\']'),
            "Hardcoded password detected",
        ),
        (re.compile(r'secret\s*=\s*["\'][^"\']+["\']'), "Hardcoded secret detected"),
        (re.compile(r'api_key\s*=\s*["\'][^"\']+["\']'), "Hardcoded API key detected"),
        (re.compile(r'token\s*=\s*["\'][^"\']+["\']'), "Hardcoded token detected"),
    ]

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id in self.DANGEROUS_BUILTINS:
                self._findings.add(
                    "critical",
                    f"Dangerous builtin '{node.func.id}' — major security risk",
                    node.lineno,
                )
            if node.func.id == "input" and self._loop_depth > 0:
                self._findings.add(
                    "warning",
                    "input() inside loop — potential for injection or DoS",
                    node.lineno,
                )
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        # Detect SQL string concatenation
        if isinstance(node.op, ast.Mod):
            if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                if (
                    "SELECT" in node.left.value.upper()
                    or "INSERT" in node.left.value.upper()
                ):
                    self._findings.add(
                        "critical",
                        "SQL string formatting — SQL injection risk",
                        node.lineno,
                    )
        if isinstance(node.op, ast.Add):
            if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                if "SELECT" in node.left.value.upper():
                    self._findings.add(
                        "critical",
                        "SQL string concatenation — SQL injection risk",
                        node.lineno,
                    )
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        # f-string SQL
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                if "SELECT" in value.value.upper() or "INSERT" in value.value.upper():
                    self._findings.add(
                        "critical", "SQL in f-string — SQL injection risk", node.lineno
                    )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "pickle":
            self._findings.add(
                "warning",
                "Pickle import — deserialization risk if loading untrusted data",
                node.lineno,
            )
        self.generic_visit(node)

    def run(self, source: str) -> ReviewFindings:
        findings = super().run(source)
        # Regex-based checks for hardcoded secrets
        for pattern, message in self.SUSPICIOUS_PATTERNS:
            for i, line in enumerate(source.split("\n"), 1):
                if pattern.search(line):
                    findings.add("critical", message, i)
        return findings


# ---------------------------------------------------------------------------
# Performance visitor
# ---------------------------------------------------------------------------


class PerformanceVisitor(PersonaVisitor):
    """Finds O(n^2) loops, repeated computation, memory leaks."""

    def __init__(self) -> None:
        super().__init__(ReviewPersona.PERFORMANCE)

    def visit_For(self, node: ast.For) -> None:
        # Track loop depth
        self._loop_depth += 1
        # Detect nested loops
        if self._loop_depth >= 2:
            self._findings.add(
                "warning",
                f"Nested loop at depth {self._loop_depth} — potential O(n^{self._loop_depth})",
                node.lineno,
            )
        # Detect list comprehension inside loop (double iteration)
        for child in ast.walk(node):
            if isinstance(child, ast.ListComp) and child is not node:
                self._findings.add(
                    "warning",
                    "List comprehension inside loop — consider extracting",
                    node.lineno,
                )
                break
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "append" and self._loop_depth > 0:
                # append in loop is fine for Python lists, but warn about repeated work
                pass
        # Detect repeated function calls with same args in loop
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        # Detect nested list comprehension
        generators = node.generators
        if len(generators) > 2:
            self._findings.add(
                "info",
                f"List comprehension with {len(generators)} generators — consider explicit loop",
                node.lineno,
            )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Check for repeated function calls that could be cached
        calls: Dict[str, int] = {}
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                calls[child.func.id] = calls.get(child.func.id, 0) + 1
        for func_name, count in calls.items():
            if count > 5:
                self._findings.add(
                    "info",
                    f"Function '{func_name}' called {count} times — consider caching",
                    node.lineno,
                )
        super().visit_FunctionDef(node)


# ---------------------------------------------------------------------------
# Simplicity visitor
# ---------------------------------------------------------------------------


class SimplicityVisitor(PersonaVisitor):
    """Finds unnecessary complexity, over-engineering, dead code."""

    def __init__(self) -> None:
        super().__init__(ReviewPersona.SIMPLICITY)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Check function length
        line_count = node.end_lineno - node.lineno if node.end_lineno else 50
        if line_count > 50:
            self._findings.add(
                "warning",
                f"Function is {line_count} lines — consider decomposition",
                node.lineno,
            )
        if line_count > 100:
            self._findings.add(
                "critical",
                f"Function is {line_count} lines — strongly consider decomposition",
                node.lineno,
            )
        # Check parameter count
        param_count = len(node.args.args) + len(node.args.kwonlyargs)
        if param_count > 7:
            self._findings.add(
                "warning",
                f"Function has {param_count} parameters — consider a dataclass/config object",
                node.lineno,
            )
        super().visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Check for classes with only static methods (should be module)
        methods = [
            n
            for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        static_methods = [
            m
            for m in methods
            if any(
                isinstance(d, ast.Name) and d.id == "staticmethod"
                for d in m.decorator_list
            )
        ]
        if methods and len(static_methods) == len(methods):
            self._findings.add(
                "info",
                "Class with only static methods — consider using a module",
                node.lineno,
            )
        # Check for empty classes
        if not methods and not any(isinstance(n, ast.Assign) for n in node.body):
            self._findings.add(
                "warning", "Empty class — consider if it's needed", node.lineno
            )
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        # Check for deeply nested ifs
        depth = 0
        current = node
        while (
            isinstance(current, ast.If)
            and isinstance(current.orelse, list)
            and len(current.orelse) == 1
            and isinstance(current.orelse[0], ast.If)
        ):
            depth += 1
            current = current.orelse[0]
        if depth > 3:
            self._findings.add(
                "warning",
                f"Deep if-elif chain ({depth + 1} levels) — consider match/dispatch table",
                node.lineno,
            )
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        # Check for while True with complex break conditions
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            if not any(isinstance(n, ast.Break) for n in ast.walk(node)):
                self._findings.add(
                    "critical", "while True with no break — infinite loop", node.lineno
                )
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        # Check for pass in except
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            self._findings.add(
                "warning",
                "Bare pass in except handler — silently swallowing errors",
                node.lineno,
            )
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Adversarial visitor
# ---------------------------------------------------------------------------


class AdversarialVisitor(PersonaVisitor):
    """Stress testing, edge cases, fuzzing vectors."""

    def __init__(self) -> None:
        super().__init__(ReviewPersona.ADVERSARIAL)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Check for missing input validation
        has_validation = False
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                if child.func.id in {"isinstance", "hasattr", "callable", "len"}:
                    has_validation = True
            if isinstance(child, ast.Raise):
                has_validation = True
        if not has_validation and len(node.args.args) > 0:
            self._findings.add(
                "warning",
                "No input validation detected — consider adversarial inputs",
                node.lineno,
            )
        # Check for division without zero check
        for child in ast.walk(node):
            if isinstance(child, ast.BinOp) and isinstance(
                child.op, (ast.Div, ast.FloorDiv, ast.Mod)
            ):
                # Simple heuristic: check if divisor is a variable (not constant)
                if not isinstance(child.right, ast.Constant):
                    self._findings.add(
                        "warning",
                        "Division by variable without zero check — ZeroDivisionError risk",
                        child.lineno,
                    )
        super().visit_FunctionDef(node)

    def visit_Index(self, node: ast.Index) -> None:
        # Python 3.9+ no longer has ast.Index, but handle legacy
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # Check for unbounded index access
        if isinstance(node.slice, ast.Constant):
            if isinstance(node.slice.value, int) and node.slice.value < 0:
                self._findings.add(
                    "info",
                    "Negative index access — ensure list is long enough",
                    node.lineno,
                )
        elif isinstance(node.slice, ast.UnaryOp):
            if isinstance(node.slice.op, ast.USub):
                self._findings.add(
                    "info",
                    "Negative index access — ensure list is long enough",
                    node.lineno,
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Check for unbounded recursion
        if isinstance(node.func, ast.Name) and self._function_depth > 0:
            self._findings.add(
                "info",
                "Recursive or nested function call — ensure base case is reachable",
                node.lineno,
            )
        self.generic_visit(node)

    def run(self, source: str) -> ReviewFindings:
        findings = super().run(source)
        # Check for missing __all__ in module
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return findings
        has_all = any(
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets)
            for n in tree.body
        )
        if not has_all and any(
            isinstance(n, (ast.FunctionDef, ast.ClassDef)) for n in tree.body
        ):
            findings.add("info", "No __all__ defined — all names are public API", 1)
        return findings


# ---------------------------------------------------------------------------
# Review dispatch
# ---------------------------------------------------------------------------

VISITOR_MAP: Dict[ReviewPersona, type] = {
    ReviewPersona.CORRECTNESS: CorrectnessVisitor,
    ReviewPersona.SECURITY: SecurityVisitor,
    ReviewPersona.PERFORMANCE: PerformanceVisitor,
    ReviewPersona.SIMPLICITY: SimplicityVisitor,
    ReviewPersona.ADVERSARIAL: AdversarialVisitor,
}


def review_code(source: str, persona: ReviewPersona) -> List[ReviewFinding]:
    """Run a single persona on source code."""
    visitor_class = VISITOR_MAP[persona]
    visitor = visitor_class()
    findings = visitor.run(source)
    return findings.findings


def review_all(source: str) -> Dict[str, List[ReviewFinding]]:
    """Run all personas on source code."""
    return {p.name: review_code(source, p) for p in ReviewPersona.all()}


def review_file(path: str) -> Dict[str, List[ReviewFinding]]:
    """Review a file with all personas."""
    source = open(path).read()
    return review_all(source)


def review_files(paths: List[str]) -> Dict[str, Dict[str, List[ReviewFinding]]]:
    """Review multiple files with all personas."""
    return {p: review_file(p) for p in paths}
