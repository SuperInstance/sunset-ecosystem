"""fleet/deckboss.py — Spreadsheet-grid orchestrator for the fleet.

A grid of formula-evaluating cells that re-computes on every fleet beat.
Each cell is a formula like:
    =IF(FLEET_HEALTH() < 0.5, SPAWN("rescue"), IDLE())

The grid evaluates left-to-right, top-to-bottom (Excel order).  Cell
outputs can reference other cells by name (A1, B2) enabling complex
orchestration policies built from simple primitives.

Integration
-----------
    from fleet.deckboss import DeckbossGrid, FleetDeckbossEnv
    from fleet.formula_compiler import FormulaCompiler

    env = FleetDeckbossEnv(conductor=conductor, breeder=breeder)
    grid = DeckbossGrid(env)
    grid.set_cell("A1", '=IF(FLEET_HEALTH() < 0.5, SPAWN("rescue"), IDLE())')
    grid.set_cell("B1", '=COUNTIF(fleet.status, "idle")')

    # on each beat
    results = grid.evaluate_all()
    # results = {"A1": "SPAWN:rescue:1", "B1": 7}

The grid is deterministic: same inputs always produce same outputs.
Side effects (SPAWN, ALERT, BREED) are captured as command tokens,
not executed immediately — the caller decides when to dispatch.

Usage with FleetConductorV2
---------------------------
    conductor.register_sda_pipeline(
        name="deckboss",
        decide_fn=grid.evaluate_all,
        act_fn=grid.dispatch_commands,
    )
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from fleet.formula_compiler import FleetFormulaEnv, FormulaCompiler

logger = logging.getLogger(__name__)


# ── Cell ────────────────────────────────────────────────────────────────

@dataclass
class DeckbossCell:
    """A single cell in the orchestrator grid."""

    ref: str  # "A1", "B7", etc.
    formula: str = ""
    compiled: Optional[Callable[[], Any]] = None
    last_result: Any = None
    last_evaluated_at: float = 0.0
    eval_count: int = 0
    error_count: int = 0
    depends_on: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.formula


# ── Grid ────────────────────────────────────────────────────────────────

class DeckbossGrid:
    """Formula grid that evaluates fleet orchestration policies."""

    _CELL_REF_RE = re.compile(r"^([A-Z]+)(\d+)$")

    def __init__(
        self,
        env: FleetFormulaEnv,
        *,
        max_recursion_depth: int = 10,
        lazy_compile: bool = True,
    ):
        self.env = env
        self.compiler = FormulaCompiler(env)
        self.cells: Dict[str, DeckbossCell] = {}
        self.max_recursion_depth = max_recursion_depth
        self.lazy_compile = lazy_compile
        self._eval_depth = 0
        self._cycle_guard: set[str] = set()
        self.command_log: List[Tuple[str, str, Any, float]] = []  # (ref, formula, result, ts)
        self.compiler.cell_resolver = self._eval_cell_by_ref
        self._build_env_extensions()

    def _build_env_extensions(self) -> None:
        """Inject cell-reference lookup into the formula environment."""
        # Monkey-patch a CELL() function into env
        original_registry = self.env._registry.copy()

        def _cell(ref: str) -> Any:
            return self._eval_cell_by_ref(ref.upper())

        def _range(start: str, end: str) -> List[Any]:
            refs = self._expand_range(start.upper(), end.upper())
            return [self._eval_cell_by_ref(r) for r in refs]

        self.env._registry["CELL"] = _cell
        self.env._registry["RANGE"] = _range
        self.env._registry.update(original_registry)  # ensure no overwrite of builtins

    # ── Cell management ─────────────────────────────────────────────────

    def set_cell(self, ref: str, formula: str) -> DeckbossCell:
        """Set formula for a cell reference (e.g. 'A1')."""
        ref = ref.upper()
        cell = self.cells.get(ref)
        if cell is None:
            cell = DeckbossCell(ref=ref)
            self.cells[ref] = cell
        cell.formula = formula
        cell.compiled = None  # force recompile
        if not self.lazy_compile:
            self._compile_cell(cell)
        self._rebuild_dependencies()
        return cell

    def get_cell(self, ref: str) -> Optional[DeckbossCell]:
        return self.cells.get(ref.upper())

    def clear_cell(self, ref: str) -> None:
        self.cells.pop(ref.upper(), None)
        self._rebuild_dependencies()

    def _compile_cell(self, cell: DeckbossCell) -> None:
        if cell.compiled is not None or cell.is_empty():
            return
        try:
            cell.compiled = self.compiler.compile(cell.formula)
        except Exception as exc:
            logger.warning("Deckboss compile error in %s: %s", cell.ref, exc)
            cell.compiled = lambda: f"#ERROR: {exc}"
            cell.error_count += 1

    def _rebuild_dependencies(self) -> None:
        """Parse formulas to build dependency graph."""
        for cell in self.cells.values():
            cell.depends_on = []
            cell.dependents = []
        for cell in self.cells.values():
            if cell.is_empty():
                continue
            deps = self._extract_refs(cell.formula)
            cell.depends_on = deps
            for dep in deps:
                if dep in self.cells:
                    self.cells[dep].dependents.append(cell.ref)

    def _extract_refs(self, formula: str) -> List[str]:
        """Extract cell references like A1, B12 from formula text."""
        # naive regex: bare word that looks like a cell ref
        tokens = re.findall(r"\b([A-Z]{1,3}\d{1,6})\b", formula.upper())
        # Also catch CELL("A1") and RANGE("A1","B2")
        cell_args = re.findall(r'CELL\s*\(\s*"([A-Z]\d+)"\s*\)', formula.upper())
        range_args = re.findall(
            r'RANGE\s*\(\s*"([A-Z]\d+)"\s*,\s*"([A-Z]\d+)"\s*\)', formula.upper()
        )
        refs = set(tokens) | set(cell_args)
        for start, end in range_args:
            refs.update(self._expand_range(start, end))
        return sorted(refs)

    def _expand_range(self, start: str, end: str) -> List[str]:
        """Expand A1:B2 style range into list of cell refs."""
        s_col, s_row = self._parse_ref(start)
        e_col, e_row = self._parse_ref(end)
        if s_col is None or e_col is None:
            return []
        refs = []
        for col in range(s_col, e_col + 1):
            for row in range(s_row, e_row + 1):
                refs.append(f"{self._col_name(col)}{row}")
        return refs

    def _parse_ref(self, ref: str) -> Tuple[Optional[int], int]:
        m = self._CELL_REF_RE.match(ref.upper())
        if not m:
            return None, 0
        col_str, row_str = m.groups()
        col = 0
        for ch in col_str:
            col = col * 26 + (ord(ch) - ord("A") + 1)
        return col, int(row_str)

    def _col_name(self, col: int) -> str:
        name = ""
        while col > 0:
            col, rem = divmod(col - 1, 26)
            name = chr(ord("A") + rem) + name
        return name

    # ── Evaluation ────────────────────────────────────────────────────────

    def _eval_cell_by_ref(self, ref: str) -> Any:
        cell = self.cells.get(ref)
        if cell is None or cell.is_empty():
            return None
        return self._eval_cell(cell)

    def _eval_cell(self, cell: DeckbossCell) -> Any:
        if cell.is_empty():
            return None
        if cell.ref in self._cycle_guard:
            logger.warning("Cycle detected at %s", cell.ref)
            return "#CYCLE"
        self._compile_cell(cell)
        if cell.compiled is None:
            return "#NOCOMPILE"
        self._cycle_guard.add(cell.ref)
        self._eval_depth += 1
        try:
            if self._eval_depth > self.max_recursion_depth:
                result = "#DEPTH"
            else:
                result = cell.compiled()
            cell.last_result = result
            cell.last_evaluated_at = time.time()
            cell.eval_count += 1
        except Exception as exc:
            logger.warning("Deckboss eval error in %s: %s", cell.ref, exc)
            result = f"#ERROR: {exc}"
            cell.last_result = result
            cell.error_count += 1
        finally:
            self._eval_depth -= 1
            self._cycle_guard.discard(cell.ref)
        return result

    def evaluate_all(self) -> Dict[str, Any]:
        """Evaluate all cells in dependency order. Returns {ref: result}."""
        self.command_log.clear()
        self._cycle_guard.clear()
        self._eval_depth = 0

        # Topological sort by dependencies
        sorted_refs = self._topological_sort()
        results: Dict[str, Any] = {}
        for ref in sorted_refs:
            cell = self.cells[ref]
            result = self._eval_cell(cell)
            results[ref] = result
            if self._is_command(result):
                self.command_log.append((ref, cell.formula, result, time.time()))
        return results

    def _topological_sort(self) -> List[str]:
        """Kahn's algorithm on cell dependency graph."""
        in_degree: Dict[str, int] = {ref: 0 for ref in self.cells}
        adj: Dict[str, List[str]] = {ref: [] for ref in self.cells}
        for cell in self.cells.values():
            for dep in cell.depends_on:
                if dep in self.cells:
                    in_degree[cell.ref] += 1
                    adj[dep].append(cell.ref)
        queue = [ref for ref, deg in in_degree.items() if deg == 0]
        order: List[str] = []
        while queue:
            ref = queue.pop(0)
            order.append(ref)
            for dependent in adj[ref]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        # Append any remaining (cycle participants) in original order
        for ref in sorted(self.cells):
            if ref not in order:
                order.append(ref)
        return order

    def _is_command(self, value: Any) -> bool:
        """Return True if the value is a fleet command token."""
        if not isinstance(value, str):
            return False
        return any(
            value.startswith(prefix)
            for prefix in ("SPAWN:", "BREED:", "MESH:", "ALERT:", "STOP")
        )

    # ── Command dispatch ────────────────────────────────────────────────────

    def get_commands(self) -> List[Tuple[str, str, Any, float]]:
        """Return command log from last evaluate_all()."""
        return list(self.command_log)

    def dispatch_commands(
        self,
        *,
        spawn_fn: Optional[Callable[[str, int], None]] = None,
        breed_fn: Optional[Callable[[str], None]] = None,
        alert_fn: Optional[Callable[[str, str], None]] = None,
        mesh_fn: Optional[Callable[..., None]] = None,
    ) -> Dict[str, int]:
        """Execute captured commands. Returns dispatch counters."""
        counts: Dict[str, int] = {"spawn": 0, "breed": 0, "alert": 0, "mesh": 0, "stop": 0}
        for ref, formula, result, ts in self.command_log:
            if not isinstance(result, str):
                continue
            if result.startswith("SPAWN:") and spawn_fn:
                parts = result.split(":")
                if len(parts) >= 3:
                    count_val = float(parts[2])
                    spawn_fn(parts[1], int(count_val))
                    counts["spawn"] += 1
            elif result.startswith("BREED:") and breed_fn:
                parts = result.split(":")
                breed_fn(parts[1] if len(parts) > 1 else "diversity")
                counts["breed"] += 1
            elif result.startswith("ALERT:") and alert_fn:
                parts = result.split(":", 2)
                if len(parts) >= 3:
                    alert_fn(parts[1], parts[2])
                    counts["alert"] += 1
            elif result.startswith("MESH:") and mesh_fn:
                mesh_fn(*result.split(":")[1:])
                counts["mesh"] += 1
            elif result == "STOP":
                counts["stop"] += 1
        return counts

    # ── Persistence / serialization ─────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            ref: {
                "formula": cell.formula,
                "last_result": str(cell.last_result) if cell.last_result is not None else None,
                "eval_count": cell.eval_count,
                "error_count": cell.error_count,
            }
            for ref, cell in sorted(self.cells.items())
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        self.cells.clear()
        for ref, meta in data.items():
            self.set_cell(ref, meta.get("formula", ""))
            cell = self.cells[ref]
            cell.last_result = meta.get("last_result")
            cell.eval_count = meta.get("eval_count", 0)
            cell.error_count = meta.get("error_count", 0)

    # ── Integration helper ─────────────────────────────────────────────────

    def make_sda_pipeline(self) -> Tuple[str, Callable, Callable]:
        """Return (name, decide_fn, act_fn) for SenseDecideAct registration."""
        return (
            "deckboss_grid",
            self.evaluate_all,
            self.dispatch_commands,
        )
