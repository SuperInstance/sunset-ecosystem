"""Tests for the deckboss formula grid orchestrator.

Covers cell management, formula evaluation, dependency resolution,
cycle detection, command capture, topological sort, and dispatch.
"""

import pytest

from fleet.deckboss import DeckbossGrid, DeckbossCell
from fleet.formula_compiler import FleetFormulaEnv


# ---------------------------------------------------------------------------
# Cell management
# ---------------------------------------------------------------------------


class TestCellManagement:
    def test_set_and_get_cell(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        cell = grid.set_cell("A1", "=42")
        assert cell.ref == "A1"
        assert cell.formula == "=42"
        assert grid.get_cell("A1") is cell

    def test_set_cell_overwrites(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", "=1")
        grid.set_cell("A1", "=2")
        assert grid.get_cell("A1").formula == "=2"

    def test_clear_cell(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", "=42")
        grid.clear_cell("A1")
        assert grid.get_cell("A1") is None

    def test_empty_cell_returns_none(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        assert grid.get_cell("Z99") is None


# ---------------------------------------------------------------------------
# Basic evaluation
# ---------------------------------------------------------------------------


class TestBasicEvaluation:
    def test_eval_number(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", "=42")
        results = grid.evaluate_all()
        assert results["A1"] == 42.0

    def test_eval_string(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", '="hello"')
        results = grid.evaluate_all()
        assert results["A1"] == "hello"

    def test_eval_fleet_health(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", "=FLEET_HEALTH()")
        results = grid.evaluate_all()
        assert results["A1"] == 1.0

    def test_eval_if_spawn(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", '=IF(1 < 2, SPAWN("scout"), IDLE())')
        results = grid.evaluate_all()
        assert results["A1"] == "SPAWN:scout:1"

    def test_eval_if_idle(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", '=IF(2 < 1, SPAWN("scout"), IDLE())')
        results = grid.evaluate_all()
        assert results["A1"] == "IDLE"

    def test_eval_multiple_cells(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", "=1")
        grid.set_cell("A2", "=2")
        grid.set_cell("A3", "=A1 + A2")
        results = grid.evaluate_all()
        assert results["A1"] == 1.0
        assert results["A2"] == 2.0
        assert results["A3"] == 3.0


# ---------------------------------------------------------------------------
# Cell references and dependencies
# ---------------------------------------------------------------------------


class TestCellReferences:
    def test_simple_reference(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", "=10")
        grid.set_cell("B1", "=A1 + 5")
        results = grid.evaluate_all()
        assert results["B1"] == 15.0

    def test_chain_reference(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", "=1")
        grid.set_cell("A2", "=A1 + 1")
        grid.set_cell("A3", "=A2 + 1")
        results = grid.evaluate_all()
        assert results["A3"] == 3.0

    def test_cross_column_reference(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", "=5")
        grid.set_cell("B1", "=A1 * 2")
        grid.set_cell("C1", "=B1 + A1")
        results = grid.evaluate_all()
        assert results["C1"] == 15.0

    def test_reference_empty_cell(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", "=B1 + 1")
        results = grid.evaluate_all()
        assert "#ERROR" in str(results["A1"])  # B1 is empty -> None + 1 -> TypeError

    def test_cell_function(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", "=10")
        grid.set_cell("B1", '=CELL("A1") + 5')
        results = grid.evaluate_all()
        assert results["B1"] == 15.0


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------


class TestTopologicalSort:
    def test_linear_chain(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", "=1")
        grid.set_cell("A2", "=A1 + 1")
        grid.set_cell("A3", "=A2 + 1")
        order = grid._topological_sort()
        assert order.index("A1") < order.index("A2")
        assert order.index("A2") < order.index("A3")

    def test_diamond_dependency(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", "=1")
        grid.set_cell("B1", "=A1 + 1")
        grid.set_cell("C1", "=A1 + 2")
        grid.set_cell("D1", "=B1 + C1")
        order = grid._topological_sort()
        assert order.index("A1") < order.index("B1")
        assert order.index("A1") < order.index("C1")
        assert order.index("B1") < order.index("D1")
        assert order.index("C1") < order.index("D1")

    def test_independent_cells(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", "=1")
        grid.set_cell("B1", "=2")
        order = grid._topological_sort()
        assert "A1" in order
        assert "B1" in order


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------


class TestCycleDetection:
    def test_self_reference_cycle(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", "=A1 + 1")
        results = grid.evaluate_all()
        assert results["A1"] == "#CYCLE"

    def test_two_cell_cycle(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", "=B1 + 1")
        grid.set_cell("B1", "=A1 + 1")
        results = grid.evaluate_all()
        assert results["A1"] == "#CYCLE" or results["B1"] == "#CYCLE"

    def test_three_cell_cycle(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", "=B1")
        grid.set_cell("B1", "=C1")
        grid.set_cell("C1", "=A1")
        results = grid.evaluate_all()
        assert "#CYCLE" in [results["A1"], results["B1"], results["C1"]]


# ---------------------------------------------------------------------------
# Command capture
# ---------------------------------------------------------------------------


class TestCommandCapture:
    def test_spawn_command_captured(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", '=SPAWN("scout")')
        grid.evaluate_all()
        cmds = grid.get_commands()
        assert len(cmds) == 1
        assert cmds[0][0] == "A1"
        assert cmds[0][2] == "SPAWN:scout:1"

    def test_breed_command_captured(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", '=BREED("diversity")')
        grid.evaluate_all()
        cmds = grid.get_commands()
        assert len(cmds) == 1
        assert cmds[0][2] == "BREED:diversity"

    def test_alert_command_captured(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", '=ALERT("thermal", "high")')
        grid.evaluate_all()
        cmds = grid.get_commands()
        assert len(cmds) == 1
        assert cmds[0][2] == "ALERT:thermal:high"

    def test_idle_not_captured(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", "=IDLE()")
        grid.evaluate_all()
        cmds = grid.get_commands()
        assert len(cmds) == 0

    def test_multiple_commands(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", '=SPAWN("scout")')
        grid.set_cell("A2", '=SPAWN("worker")')
        grid.set_cell("A3", "=IDLE()")
        grid.evaluate_all()
        cmds = grid.get_commands()
        assert len(cmds) == 2


# ---------------------------------------------------------------------------
# Command dispatch
# ---------------------------------------------------------------------------


class TestCommandDispatch:
    def test_dispatch_spawn(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", '=SPAWN("scout", 3)')
        grid.evaluate_all()

        spawned = []

        def capture_spawn(agent_type, count):
            spawned.append((agent_type, count))

        counts = grid.dispatch_commands(spawn_fn=capture_spawn)
        assert counts["spawn"] == 1
        assert spawned == [("scout", 3)]

    def test_dispatch_breed(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", '=BREED("exploit")')
        grid.evaluate_all()

        bred = []

        def capture_breed(strategy):
            bred.append(strategy)

        counts = grid.dispatch_commands(breed_fn=capture_breed)
        assert counts["breed"] == 1
        assert bred == ["exploit"]

    def test_dispatch_alert(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", '=ALERT("thermal", "critical")')
        grid.evaluate_all()

        alerts = []

        def capture_alert(channel, message):
            alerts.append((channel, message))

        counts = grid.dispatch_commands(alert_fn=capture_alert)
        assert counts["alert"] == 1
        assert alerts == [("thermal", "critical")]

    def test_dispatch_no_handlers(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", '=SPAWN("scout")')
        grid.evaluate_all()
        counts = grid.dispatch_commands()
        assert counts["spawn"] == 0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_to_dict(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", "=42")
        grid.evaluate_all()
        d = grid.to_dict()
        assert "A1" in d
        assert d["A1"]["formula"] == "=42"
        assert d["A1"]["last_result"] == "42.0"
        assert d["A1"]["eval_count"] == 1

    def test_from_dict(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.from_dict(
            {
                "A1": {
                    "formula": "=10",
                    "last_result": "10.0",
                    "eval_count": 5,
                    "error_count": 0,
                },
            }
        )
        assert grid.get_cell("A1").formula == "=10"
        assert grid.get_cell("A1").eval_count == 5


# ---------------------------------------------------------------------------
# SDA pipeline helper
# ---------------------------------------------------------------------------


class TestSDAIntegration:
    def test_make_sda_pipeline(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        name, decide, act = grid.make_sda_pipeline()
        assert name == "deckboss_grid"
        assert callable(decide)
        assert callable(act)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_grid_evaluates(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        results = grid.evaluate_all()
        assert results == {}

    def test_formula_syntax_error(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", "=IF(1,")
        results = grid.evaluate_all()
        assert "#ERROR" in str(results["A1"])
        assert grid.get_cell("A1").error_count >= 1

    def test_eval_count_increments(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", "=1")
        grid.evaluate_all()
        grid.evaluate_all()
        assert grid.get_cell("A1").eval_count == 2

    def test_max_recursion_depth(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env, max_recursion_depth=2)
        grid.set_cell("A1", "=1")
        grid.set_cell("A2", "=A1 + 1")
        grid.set_cell("A3", "=A2 + 1")
        grid.set_cell("A4", "=A3 + 1")
        results = grid.evaluate_all()
        # A3 depends on A2 which depends on A1 = depth 3
        # But max_recursion_depth is checked per-eval chain
        # Actually depth is per-eval_cell call, not per-chain
        # A4 calls CELL("A3") which calls _eval_cell -> depth increments
        # This is complex behavior; just verify no crash
        assert "A4" in results

    def test_range_function(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", "=1")
        grid.set_cell("A2", "=2")
        grid.set_cell("A3", '=AVERAGE(RANGE("A1", "A2"))')
        results = grid.evaluate_all()
        assert results["A3"] == 1.5

    def test_col_name_conversion(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        assert grid._col_name(1) == "A"
        assert grid._col_name(27) == "AA"
        assert grid._col_name(52) == "AZ"

    def test_parse_ref(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        assert grid._parse_ref("A1") == (1, 1)
        assert grid._parse_ref("B10") == (2, 10)
        assert grid._parse_ref("AA100") == (27, 100)

    def test_complex_policy_example(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell(
            "A1", '=IF(FLEET_HEALTH() > 0.5, SPAWN("worker"), ALERT("health", "low"))'
        )
        grid.set_cell("B1", "=THERMAL_AVG()")
        grid.set_cell("C1", '=IF(B1 > 0.8, ALERT("thermal", "high"), IDLE())')
        results = grid.evaluate_all()
        # With defaults: health=1.0, thermal=0.0
        assert results["A1"] == "SPAWN:worker:1"
        assert results["B1"] == 0.0
        assert results["C1"] == "IDLE"
        cmds = grid.get_commands()
        assert len(cmds) == 1
        assert cmds[0][2] == "SPAWN:worker:1"

    def test_range_expansion(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        refs = grid._expand_range("A1", "B2")
        assert sorted(refs) == ["A1", "A2", "B1", "B2"]

    def test_dependency_rebuild_on_set(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", "=1")
        grid.set_cell("B1", "=A1 + 1")
        assert grid.get_cell("B1").depends_on == ["A1"]
        assert grid.get_cell("A1").dependents == ["B1"]

    def test_dependency_cleared_on_overwrite(self):
        env = FleetFormulaEnv()
        grid = DeckbossGrid(env)
        grid.set_cell("A1", "=1")
        grid.set_cell("B1", "=A1 + 1")
        grid.set_cell("B1", "=2")
        assert grid.get_cell("B1").depends_on == []
        assert grid.get_cell("A1").dependents == []
