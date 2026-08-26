"""Tests for Parquet/CSV ingestion bridge.

Covers loading, formula evaluation, export, and fleet-specific helpers.
PyArrow-dependent tests are skipped if pyarrow is not installed.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from fleet.parquet_bridge import HAS_PYARROW, ParquetBridge
from fleet.deckboss import DeckbossGrid, FleetFormulaEnv


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_grid(self):
        bridge = ParquetBridge()
        assert bridge.grid is not None
        assert isinstance(bridge.grid, DeckbossGrid)

    def test_custom_grid(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        bridge = ParquetBridge(grid=grid)
        assert bridge.grid is grid


# ---------------------------------------------------------------------------
# CSV loading (pure Python, always works)
# ---------------------------------------------------------------------------


class TestCSVLoad:
    def test_load_csv_string(self):
        bridge = ParquetBridge()
        csv_content = "name,health,status\nagent1,0.95,active\nagent2,0.80,idle\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            path = f.name
        try:
            result = bridge.load_csv(path)
            assert result["rows_loaded"] == 2
            assert result["cols_loaded"] == 3
            assert bridge.grid.cells["A1"].formula == "name"
            assert bridge.grid.cells["A2"].formula == "agent1"
            assert bridge.grid.cells["B2"].formula == "0.95"
        finally:
            os.unlink(path)

    def test_load_csv_empty(self):
        bridge = ParquetBridge()
        csv_content = "\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            path = f.name
        try:
            result = bridge.load_csv(path)
            assert result["rows_loaded"] == 0
        finally:
            os.unlink(path)

    def test_load_csv_numeric(self):
        bridge = ParquetBridge()
        csv_content = "x,y\n1,10\n2,20\n3,30\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            path = f.name
        try:
            bridge.load_csv(path)
            assert bridge.grid.cells["A2"].formula == "1"
            assert bridge.grid.cells["B4"].formula == "30"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Parquet loading (requires pyarrow)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
class TestParquetLoad:
    def test_load_parquet_basic(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        bridge = ParquetBridge()
        table = pa.table({"a": [1, 2, 3], "b": [4.5, 5.5, 6.5]})
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        try:
            pq.write_table(table, path)
            result = bridge.load_parquet(path)
            assert result["rows_loaded"] == 3
            assert result["cols_loaded"] == 2
            assert bridge.grid.cells["A1"].formula == "a"
            assert bridge.grid.cells["A2"].formula == "1"
            assert bridge.grid.cells["B2"].formula == "4.5"
        finally:
            os.unlink(path)

    def test_load_parquet_with_strings(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        bridge = ParquetBridge()
        table = pa.table({"name": ["alice", "bob"], "score": [100, 200]})
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        try:
            pq.write_table(table, path)
            bridge.load_parquet(path)
            assert bridge.grid.cells["A2"].formula == "alice"
            assert bridge.grid.cells["B2"].formula == "100"
        finally:
            os.unlink(path)

    def test_load_parquet_missing_raises(self):
        bridge = ParquetBridge()
        with pytest.raises((RuntimeError, FileNotFoundError)):
            bridge.load_parquet("/nonexistent/file.parquet")


# ---------------------------------------------------------------------------
# Formula operations
# ---------------------------------------------------------------------------


class TestFormulaOperations:
    def test_set_formula(self):
        bridge = ParquetBridge()
        bridge.set_formula("A1", "=1 + 2")
        assert bridge.grid.cells["A1"].formula == "=1 + 2"

    def test_evaluate_csv_data(self):
        bridge = ParquetBridge()
        csv_content = "x,y\n1,10\n2,20\n3,30\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            path = f.name
        try:
            bridge.load_csv(path)
            bridge.set_formula("C2", "=A2 + B2")
            results = bridge.evaluate()
            assert results["C2"] == 11.0
        finally:
            os.unlink(path)

    def test_average_on_csv(self):
        bridge = ParquetBridge()
        csv_content = "val\n10\n20\n30\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            path = f.name
        try:
            bridge.load_csv(path)
            # Use explicit cell references since A2:A4 range syntax isn't supported
            bridge.set_formula("B2", "=AVERAGE(A2, A3, A4)")
            results = bridge.evaluate()
            assert results["B2"] == pytest.approx(20.0, abs=0.01)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestCSVExport:
    def test_export_csv_basic(self):
        bridge = ParquetBridge()
        bridge.set_formula("A1", "name")
        bridge.set_formula("B1", "value")
        bridge.set_formula("A2", "test")
        bridge.set_formula("B2", "42")
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            bridge.export_csv(path)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "name" in content
            assert "test" in content
        finally:
            os.unlink(path)


@pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
class TestParquetExport:
    def test_export_parquet_roundtrip(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        bridge = ParquetBridge()
        table = pa.table({"a": [1, 2], "b": [3.0, 4.0]})
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        try:
            pq.write_table(table, path)
            bridge.load_parquet(path)
            bridge.set_formula("C2", "=A2 + B2")
            bridge.evaluate()

            out_path = path + ".out"
            bridge.export_parquet(out_path)
            result_table = pq.read_table(out_path)
            assert result_table.num_rows == 2
            assert result_table.num_columns == 3
        finally:
            os.unlink(path)
            if os.path.exists(out_path):
                os.unlink(out_path)


# ---------------------------------------------------------------------------
# Fleet helpers
# ---------------------------------------------------------------------------


class TestFleetHelpers:
    def test_load_fleet_snapshot(self):
        bridge = ParquetBridge()
        snapshot = {
            "agent_count": 50,
            "thermal_avg": 0.75,
            "queue_depth": 12,
            "flux_gate_rate": 0.95,
        }
        bridge.load_fleet_snapshot(snapshot)
        assert bridge.grid.cells["A1"].formula == "metric"
        assert bridge.grid.cells["A2"].formula == "agent_count"
        assert bridge.grid.cells["B2"].formula == "50"
        assert bridge.grid.cells["A3"].formula == "thermal_avg"
        assert bridge.grid.cells["B3"].formula == "0.75"

    def test_get_fleet_summary(self):
        bridge = ParquetBridge()
        bridge.set_formula("A1", "=42")
        bridge.set_formula("B1", "=3.14")
        summary = bridge.get_fleet_summary()
        assert summary["A1"] == 42.0
        assert summary["B1"] == pytest.approx(3.14, abs=0.01)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_col_letter(self):
        assert ParquetBridge._col_letter(1) == "A"
        assert ParquetBridge._col_letter(26) == "Z"
        assert ParquetBridge._col_letter(27) == "AA"
        assert ParquetBridge._col_letter(52) == "AZ"
        assert ParquetBridge._col_letter(53) == "BA"

    def test_value_to_string(self):
        assert ParquetBridge._value_to_string(42) == "42"
        assert ParquetBridge._value_to_string(3.14) == "3.14"
        assert ParquetBridge._value_to_string(None) == ""
        assert ParquetBridge._value_to_string("hello") == "hello"

    def test_string_to_typed(self):
        assert ParquetBridge._string_to_typed("42") == 42
        assert ParquetBridge._string_to_typed("3.14") == 3.14
        assert ParquetBridge._string_to_typed("") is None
        assert ParquetBridge._string_to_typed("hello") == "hello"
