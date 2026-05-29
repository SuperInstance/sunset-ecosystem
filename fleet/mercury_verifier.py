"""fleet/mercury_verifier.py — Mercury code generator for fleet formula verification.

Converts formula AST (from `formula_compiler.py`) into Mercury predicates.
Uses Mercury's determinism analysis to classify formulas as:
- `det` — exactly one solution (safe for breeding)
- `semidet` — zero or one solution (safe for FLUX gates)
- `multi` — multiple solutions (unpredictable)
- `nondet` — unknown number of solutions (dangerous)
- `failure` — never succeeds (invalid formula)

Usage
-----
    from fleet.mercury_verifier import MercuryVerifier, FormulaToMercury

    gen = FormulaToMercury()
    mercury_code = gen.compile('=IF(FLEET_HEALTH() < 0.5, SPAWN("rescue"), IDLE())')
    print(mercury_code)

    verifier = MercuryVerifier()
    result = verifier.analyze(mercury_code)  # requires `mmc`
    assert result.determinism == "det"

Mercury predicates are generated with explicit modes and types, enabling
static analysis of formula safety without executing them.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fleet.formula_compiler import (
    CallNode,
    ExprNode,
    FleetFormulaEnv,
    FormulaCompiler,
    FormulaParser,
    InfixNode,
    NameNode,
    NumberNode,
    StringNode,
)

logger = logging.getLogger(__name__)


# ── Mercury code generator ──────────────────────────────────────────────

class FormulaToMercury:
    """Compile formula AST into Mercury predicate code."""

    # Mapping of fleet functions to Mercury builtins or library calls
    _FLEET_TO_MERCURY: Dict[str, str] = {
        "IF": "if_then_else",
        "AND": "bool.and",
        "OR": "bool.or",
        "NOT": "bool.not",
        "AVERAGE": "list.foldl((+), List, 0) / list.length(List)",
        "MAX": "list.max",
        "MIN": "list.min",
        "COUNTIF": "list.count",
    }

    def _fleet_func_name(self, name: str) -> str:
        """Map fleet function names to Mercury predicate names."""
        if name in ("FLEET_HEALTH", "THERMAL_AVG", "QUEUE_DEPTH",
                    "AGENT_COUNT", "BEAT_COUNT"):
            return f"{name.lower()}_value()"
        return self._FLEET_TO_MERCURY.get(name, name.lower())

    def compile(self, source: str) -> str:
        """Generate Mercury module text from a formula string."""
        parser = FormulaParser(source)
        ast = parser.parse()
        body = self._expr_to_mercury(ast)

        module_name = self._sanitize_module_name(source)
        return f""":- module {module_name}.

:- interface.
:- pred evaluate(float::out) is det.

:- implementation.
:- import_module float, list, string, bool.

evaluate(Result) :-
    Result = {body}.
"""

    def compile_with_mode(self, source: str, mode: str = "det") -> str:
        """Generate Mercury with explicit determinism annotation."""
        parser = FormulaParser(source)
        ast = parser.parse()
        body = self._expr_to_mercury(ast)

        module_name = self._sanitize_module_name(source)
        return f""":- module {module_name}.

:- interface.
:- pred evaluate(float::out) is {mode}.

:- implementation.
:- import_module float, list, string, bool.

evaluate(Result) :-
    Result = {body}.
"""

    def _expr_to_mercury(self, node: ExprNode) -> str:
        if isinstance(node, NumberNode):
            return str(node.value)
        if isinstance(node, StringNode):
            return f'"{node.value}"'
        if isinstance(node, NameNode):
            mg = self._FLEET_TO_MERCURY.get(node.name)
            if mg:
                return mg
            return self._fleet_func_name(node.name)
        if isinstance(node, CallNode):
            return self._call_to_mercury(node)
        if isinstance(node, InfixNode):
            return self._infix_to_mercury(node)
        return "0.0"

    def _call_to_mercury(self, node: CallNode) -> str:
        fn = node.func
        args = [self._expr_to_mercury(arg) for arg in node.args]

        if fn == "IF" and len(args) == 3:
            return f"(if {args[0]} then {args[1]} else {args[2]})"

        if fn in ("SPAWN", "BREED", "MESH", "ALERT", "IDLE", "STOP"):
            # Actions become string tokens in Mercury (side-effect free)
            # Strip quotes from string args to avoid double-quoting
            clean_args = []
            for a in args:
                if a.startswith('"') and a.endswith('"'):
                    clean_args.append(a[1:-1])
                else:
                    clean_args.append(a)
            action_body = ":".join(clean_args)
            if action_body:
                return f'"{fn}:{action_body}"'
            return f'"{fn}"'

        if fn in ("AND", "OR"):
            op = "and" if fn == "AND" else "or"
            joined = f" {op} ".join(args)
            return f"({joined})"

        if fn == "NOT" and len(args) == 1:
            return f"(not {args[0]})"

        if fn in ("FLEET_HEALTH", "THERMAL_AVG", "QUEUE_DEPTH",
                  "AGENT_COUNT", "BEAT_COUNT") and len(args) == 0:
            return self._fleet_func_name(fn)

        if fn in ("AVERAGE", "MAX", "MIN"):
            # Convert varargs to Mercury list
            list_lit = "[" + ", ".join(args) + "]"
            if fn == "AVERAGE":
                return f"(list.foldl((+), {list_lit}, 0.0) / float(list.length({list_lit})))"
            return f"list.{fn.lower()}({list_lit})"

        if fn == "COUNTIF":
            # Naive: count occurrences in a list
            return f"0.0"  # placeholder

        if fn == "CELL" and len(args) == 1:
            return f"cell_value({args[0]})"

        if fn == "RANGE" and len(args) == 2:
            return f"range_values({args[0]}, {args[1]})"

        # Default: function call
        mg = self._FLEET_TO_MERCURY.get(fn, fn.lower())
        return f"{mg}({', '.join(args)})"

    def _infix_to_mercury(self, node: InfixNode) -> str:
        left = self._expr_to_mercury(node.left)
        right = self._expr_to_mercury(node.right)
        op_map = {
            "+": "+",
            "-": "-",
            "*": "*",
            "/": "/",
            "<": "<",
            ">": ">",
            "<=": "<=",
            ">=": ">=",
            "==": "=",
            "!=": "!=",
        }
        op = op_map.get(node.op, node.op)
        return f"({left} {op} {right})"

    @staticmethod
    def _sanitize_module_name(source: str) -> str:
        """Create a valid Mercury module name from source text."""
        # Strip = and non-alnum, take first 20 chars
        cleaned = re.sub(r"[^a-zA-Z0-9]", "", source[:40])
        if not cleaned:
            cleaned = "formula"
        return f"fleet_{cleaned.lower()[:30]}"


# ── Verifier ───────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    """Outcome of Mercury determinism analysis."""

    determinism: str  # det, semidet, multi, nondet, failure
    errors: List[str]
    warnings: List[str]
    mercury_code: str
    raw_output: str = ""
    success: bool = False


class MercuryVerifier:
    """Analyze Mercury code for determinism and safety."""

    def __init__(self, *, mmc_path: str = "mmc"):
        self.mmc_path = mmc_path

    def analyze(self, mercury_code: str) -> VerificationResult:
        """Run Mercury compiler on generated code. Returns determinism classification.

        Requires `mmc` (Mercury compiler) to be installed.
        """
        errors: List[str] = []
        warnings: List[str] = []

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".m", delete=False
        ) as f:
            f.write(mercury_code)
            f.flush()
            temp_path = f.name

        try:
            proc = subprocess.run(
                [self.mmc_path, "--make", "--grade", "hlc", temp_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            raw = proc.stdout + proc.stderr

            # Parse determinism from Mercury output
            det = self._extract_determinism(raw)
            errors = self._extract_errors(raw)
            warnings = self._extract_warnings(raw)

            return VerificationResult(
                determinism=det or "unknown",
                errors=errors,
                warnings=warnings,
                mercury_code=mercury_code,
                raw_output=raw,
                success=proc.returncode == 0 and det is not None,
            )
        except FileNotFoundError:
            return VerificationResult(
                determinism="unknown",
                errors=[f"{self.mmc_path} not found — install Mercury compiler"],
                warnings=[],
                mercury_code=mercury_code,
                raw_output="",
                success=False,
            )
        except subprocess.TimeoutExpired:
            return VerificationResult(
                determinism="unknown",
                errors=["Mercury compiler timeout"],
                warnings=[],
                mercury_code=mercury_code,
                raw_output="",
                success=False,
            )
        finally:
            import os
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def check_formula(self, source: str, expected: str = "det") -> VerificationResult:
        """One-shot: compile formula to Mercury and analyze it."""
        gen = FormulaToMercury()
        code = gen.compile(source)
        result = self.analyze(code)
        result.success = result.success and result.determinism == expected
        return result

    def is_available(self) -> bool:
        """Return True if `mmc` is installed and callable."""
        try:
            proc = subprocess.run(
                [self.mmc_path, "--version"],
                capture_output=True,
                timeout=5,
            )
            return proc.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def _extract_determinism(raw: str) -> Optional[str]:
        for line in raw.splitlines():
            if "is det" in line:
                return "det"
            if "is semidet" in line:
                return "semidet"
            if "is multi" in line:
                return "multi"
            if "is nondet" in line:
                return "nondet"
            if "is failure" in line:
                return "failure"
        return None

    @staticmethod
    def _extract_errors(raw: str) -> List[str]:
        return [line for line in raw.splitlines() if "error:" in line.lower()]

    @staticmethod
    def _extract_warnings(raw: str) -> List[str]:
        return [line for line in raw.splitlines() if "warning:" in line.lower()]


# ── Batch verifier ──────────────────────────────────────────────────────

class BatchMercuryVerifier:
    """Verify multiple formulas in one batch."""

    def __init__(self, verifier: Optional[MercuryVerifier] = None):
        self.verifier = verifier or MercuryVerifier()

    def verify_batch(
        self, formulas: List[str], expected: str = "det"
    ) -> Dict[str, VerificationResult]:
        results: Dict[str, VerificationResult] = {}
        for formula in formulas:
            results[formula] = self.verifier.check_formula(formula, expected)
        return results

    def classify_formulas(self, formulas: List[str]) -> Dict[str, str]:
        """Return {formula: determinism} classification."""
        results = self.verify_batch(formulas)
        return {f: r.determinism for f, r in results.items()}

    def filter_safe(self, formulas: List[str]) -> List[str]:
        """Return formulas classified as `det` or `semidet`."""
        results = self.verify_batch(formulas)
        return [
            f
            for f, r in results.items()
            if r.determinism in ("det", "semidet") and r.success
        ]
