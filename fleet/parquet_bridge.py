"""fleet/parquet_bridge.py — Parquet/CSV ingestion bridge for deckboss grid.

Reads Parquet and CSV files into a DeckbossGrid, evaluates fleet formulas
on the imported data, and exports results back to Parquet/CSV.

Usage
-----
    from fleet.parquet_bridge import ParquetBridge

    bridge = ParquetBridge()
    bridge.load_parquet("fleet_data.parquet", sheet_name="agents")
    bridge.set_formula("D1", "=AVERAGE(B2:B100)")
    bridge.evaluate()
    bridge.export_parquet("fleet_results.parquet")
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from fleet.deckboss import DeckbossGrid, FleetFormulaEnv

logger = logging.getLogger(__name__)

# Optional pyarrow support
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.csv as pcsv
    HAS_PYARROW = True
except ImportError:
    pa = None  # type: ignore
    pq = None  # type: ignore
    pcsv = None  # type: ignore
    HAS_PYARROW = False
    logger.warning("pyarrow not available; parquet_bridge using CSV fallback")


@dataclass
class ParquetBridge:
    """Bridge between Parquet/CSV files and DeckbossGrid."""

    grid: DeckbossGrid = field(default_factory=lambda: DeckbossGrid(FleetFormulaEnv()))
    _col_names: List[str] = field(default_factory=list)
    _sheet_name: str = "Sheet1"

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_parquet(self, path: Union[str, Path], *, sheet_name: str = "Sheet1") -> Dict[str, Any]:
        """Load a Parquet file into the deckboss grid."""
        self._sheet_name = sheet_name
        if not HAS_PYARROW:
            raise RuntimeError("pyarrow required for Parquet loading")
        path = Path(path)
        table = pq.read_table(str(path))
        return self._load_arrow_table(table)

    def load_csv(self, path: Union[str, Path], *, sheet_name: str = "Sheet1") -> Dict[str, Any]:
        """Load a CSV file into the deckboss grid."""
        self._sheet_name = sheet_name
        if HAS_PYARROW:
            try:
                table = pcsv.read_csv(str(path))
                return self._load_arrow_table(table)
            except Exception:
                # Fallback to pure Python on pyarrow parse failure (e.g. empty file)
                pass
        # Pure Python fallback
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            return {"rows_loaded": 0, "cols_loaded": 0}
        return self._load_rows(rows)

    def _load_arrow_table(self, table) -> Dict[str, Any]:
        """Load a pyarrow Table into the grid."""
        self._col_names = table.column_names
        schema = table.schema
        rows_loaded = 0
        cols_loaded = len(self._col_names)

        # Column headers in row 1
        for col_idx, name in enumerate(self._col_names, start=1):
            cell_ref = f"{self._col_letter(col_idx)}1"
            self.grid.set_cell(cell_ref, str(name))

        # Data rows — track current_row across batches to avoid overlap
        current_row = 2
        for batch in table.to_batches():
            batch_len = len(batch)
            for col_idx, col_name in enumerate(self._col_names, start=1):
                col_data = batch.column(col_name)
                for i in range(batch_len):
                    val = col_data[i].as_py()
                    cell_ref = f"{self._col_letter(col_idx)}{current_row + i}"
                    self.grid.set_cell(cell_ref, self._value_to_string(val))
            current_row += batch_len
            rows_loaded += batch_len

        return {
            "rows_loaded": rows_loaded,
            "cols_loaded": cols_loaded,
            "schema": [str(f.type) for f in schema],
        }

    def _load_rows(self, rows: List[List[str]]) -> Dict[str, Any]:
        """Load raw string rows into the grid."""
        if not rows:
            return {"rows_loaded": 0, "cols_loaded": 0}
        # First row is headers
        headers = rows[0]
        self._col_names = headers
        for col_idx, name in enumerate(headers, start=1):
            cell_ref = f"{self._col_letter(col_idx)}1"
            self.grid.set_cell(cell_ref, str(name))
        # Data rows
        for row_idx, row in enumerate(rows[1:], start=2):
            for col_idx, val in enumerate(row, start=1):
                cell_ref = f"{self._col_letter(col_idx)}{row_idx}"
                self.grid.set_cell(cell_ref, val)
        return {
            "rows_loaded": len(rows) - 1,
            "cols_loaded": len(headers),
        }

    # ------------------------------------------------------------------
    # Formula operations
    # ------------------------------------------------------------------

    def set_formula(self, cell_ref: str, formula: str) -> None:
        """Set a formula in a cell."""
        self.grid.set_cell(cell_ref, formula)

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate all formulas in the grid."""
        return self.grid.evaluate_all()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_parquet(self, path: Union[str, Path]) -> None:
        """Export the grid to a Parquet file."""
        if not HAS_PYARROW:
            raise RuntimeError("pyarrow required for Parquet export")
        path = Path(path)
        table = self._to_arrow_table()
        pq.write_table(table, str(path))

    def export_csv(self, path: Union[str, Path]) -> None:
        """Export the grid to a CSV file."""
        path = Path(path)
        if HAS_PYARROW:
            table = self._to_arrow_table()
            pcsv.write_csv(table, str(path))
        else:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for row in self._to_rows():
                    writer.writerow(row)

    def _to_arrow_table(self):
        """Convert grid to pyarrow Table."""
        rows = self._to_rows()
        if not rows:
            return pa.table({"empty": []})
        headers = rows[0]
        data_rows = rows[1:]

        arrays = []
        for col_idx in range(len(headers)):
            col_data = []
            for row in data_rows:
                if col_idx < len(row):
                    val = row[col_idx]
                    col_data.append(self._string_to_typed(val))
                else:
                    col_data.append(None)
            arrays.append(pa.array(col_data))

        return pa.table(dict(zip(headers, arrays)))

    def _to_rows(self) -> List[List[str]]:
        """Convert grid to list of rows (strings)."""
        # Find max row/col
        max_row = 0
        max_col = 0
        for cell_ref in self.grid.cells:
            col, row = self.grid._parse_ref(cell_ref)
            max_row = max(max_row, row)
            max_col = max(max_col, col)

        rows = []
        for row in range(1, max_row + 1):
            row_data = []
            for col in range(1, max_col + 1):
                cell_ref = f"{self._col_letter(col)}{row}"
                val = self.grid.cells.get(cell_ref, "")
                row_data.append(str(val))
            rows.append(row_data)
        return rows

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _col_letter(n: int) -> str:
        """Convert 1-based column index to Excel letter."""
        result = ""
        while n > 0:
            n, rem = divmod(n - 1, 26)
            result = chr(65 + rem) + result
        return result

    @staticmethod
    def _value_to_string(val: Any) -> str:
        """Convert a Python value to string for grid storage."""
        if val is None:
            return ""
        if isinstance(val, (int, float, np.integer, np.floating)):
            return str(val)
        return str(val)

    @staticmethod
    def _string_to_typed(val: str) -> Any:
        """Convert a string to a typed value."""
        if val == "" or val == "None":
            return None
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        return val

    # ------------------------------------------------------------------
    # Fleet-specific helpers
    # ------------------------------------------------------------------

    def load_fleet_snapshot(self, data: Dict[str, Any]) -> None:
        """Load a fleet status snapshot into the grid."""
        # Row 1: headers
        headers = ["metric", "value"]
        for col_idx, name in enumerate(headers, start=1):
            self.grid.set_cell(f"{self._col_letter(col_idx)}1", name)
        # Data
        row = 2
        for key, val in data.items():
            self.grid.set_cell(f"A{row}", key)
            self.grid.set_cell(f"B{row}", self._value_to_string(val))
            row += 1

    def get_fleet_summary(self) -> Dict[str, Any]:
        """Extract fleet summary from evaluated grid."""
        results = self.grid.evaluate_all()
        summary = {}
        for cell_ref, val in results.items():
            if isinstance(val, (int, float)):
                summary[cell_ref] = val
        return summary
