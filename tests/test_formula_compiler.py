"""Tests for the fleet formula compiler.

Covers parser, evaluator, fleet function bindings, infix operators,
batch compilation, and Python AST codegen.
"""

import pytest

from fleet.formula_compiler import (
    FormulaParser,
    FormulaCompiler,
    FleetFormulaEnv,
    NumberNode,
    StringNode,
    NameNode,
    CallNode,
    InfixNode,
)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TestParser:
    def test_number(self):
        p = FormulaParser("=42")
        node = p.parse()
        assert isinstance(node, NumberNode)
        assert node.value == 42.0

    def test_string(self):
        p = FormulaParser('="hello"')
        node = p.parse()
        assert isinstance(node, StringNode)
        assert node.value == "hello"

    def test_name(self):
        p = FormulaParser("=FLEET_HEALTH")
        node = p.parse()
        assert isinstance(node, NameNode)
        assert node.name == "FLEET_HEALTH"

    def test_call_no_args(self):
        p = FormulaParser("=FLEET_HEALTH()")
        node = p.parse()
        assert isinstance(node, CallNode)
        assert node.func == "FLEET_HEALTH"
        assert node.args == []

    def test_call_one_arg(self):
        p = FormulaParser('=SPAWN("scout")')
        node = p.parse()
        assert node.func == "SPAWN"
        assert len(node.args) == 1
        assert isinstance(node.args[0], StringNode)

    def test_call_multiple_args(self):
        p = FormulaParser('=IF(FLEET_HEALTH() < 0.5, SPAWN("rescue"), IDLE())')
        node = p.parse()
        assert isinstance(node, CallNode)
        assert node.func == "IF"
        assert len(node.args) == 3

    def test_infix_add(self):
        p = FormulaParser("=1 + 2")
        node = p.parse()
        assert isinstance(node, InfixNode)
        assert node.op == "+"
        assert node.left.value == 1.0
        assert node.right.value == 2.0

    def test_infix_comparison(self):
        p = FormulaParser("=3 < 5")
        node = p.parse()
        assert isinstance(node, InfixNode)
        assert node.op == "<"

    def test_nested_call(self):
        p = FormulaParser(
            '=IF(AND(THERMAL_AVG() < 0.8, AGENT_COUNT() > 10), SPAWN("scout"), IDLE())'
        )
        node = p.parse()
        assert isinstance(node, CallNode)
        assert node.func == "IF"

    def test_no_leading_equals(self):
        p = FormulaParser("1 + 2")
        node = p.parse()
        assert isinstance(node, InfixNode)

    def test_syntax_error(self):
        with pytest.raises(SyntaxError):
            FormulaParser("=IF(1, 2").parse()


# ---------------------------------------------------------------------------
# FleetFormulaEnv
# ---------------------------------------------------------------------------


class TestFleetFormulaEnv:
    def test_fleet_health_default(self):
        env = FleetFormulaEnv()
        assert env._fleet_health() == 1.0

    def test_thermal_avg_default(self):
        env = FleetFormulaEnv()
        assert env._thermal_avg() == 0.0

    def test_queue_depth_default(self):
        env = FleetFormulaEnv()
        assert env._queue_depth() == 0

    def test_spawn_returns_token(self):
        env = FleetFormulaEnv()
        assert env._spawn("scout") == "SPAWN:scout:1"
        assert env._spawn("rescue", 3) == "SPAWN:rescue:3"

    def test_breed_returns_token(self):
        env = FleetFormulaEnv()
        assert env._breed() == "BREED:diversity"
        assert env._breed("exploit") == "BREED:exploit"

    def test_mesh_returns_token(self):
        env = FleetFormulaEnv()
        assert env._mesh("a", "b") == "MESH:a,b"

    def test_alert_returns_token(self):
        env = FleetFormulaEnv()
        assert env._alert("thermal", "high") == "ALERT:thermal:high"

    def test_idle_stop(self):
        env = FleetFormulaEnv()
        assert env._idle() == "IDLE"
        assert env._stop() == "STOP"

    def test_countif(self):
        env = FleetFormulaEnv()
        assert env._countif(["a", "b", "a", "c"], "a") == 2

    def test_average(self):
        env = FleetFormulaEnv()
        assert env._average(1, 2, 3) == 2.0
        assert env._average() == 0.0

    def test_max_min(self):
        env = FleetFormulaEnv()
        assert env._max(1, 5, 2) == 5
        assert env._min(1, 5, 2) == 1

    def test_if(self):
        env = FleetFormulaEnv()
        assert env._if(True, "yes", "no") == "yes"
        assert env._if(False, "yes", "no") == "no"

    def test_and_or_not(self):
        env = FleetFormulaEnv()
        assert env._and(True, True) is True
        assert env._and(True, False) is False
        assert env._or(True, False) is True
        assert env._or(False, False) is False
        assert env._not(True) is False
        assert env._not(False) is True

    def test_lookup_unknown(self):
        env = FleetFormulaEnv()
        with pytest.raises(NameError):
            env.lookup("UNKNOWN_FUNC")

    def test_lookup_known(self):
        env = FleetFormulaEnv()
        assert callable(env.lookup("FLEET_HEALTH"))


# ---------------------------------------------------------------------------
# FormulaCompiler
# ---------------------------------------------------------------------------


class TestFormulaCompiler:
    def test_compile_number(self):
        env = FleetFormulaEnv()
        compiler = FormulaCompiler(env)
        fn = compiler.compile("=42")
        assert fn() == 42.0

    def test_compile_string(self):
        env = FleetFormulaEnv()
        compiler = FormulaCompiler(env)
        fn = compiler.compile('="hello"')
        assert fn() == "hello"

    def test_compile_fleet_health(self):
        env = FleetFormulaEnv()
        compiler = FormulaCompiler(env)
        fn = compiler.compile("=FLEET_HEALTH()")
        assert fn() == 1.0

    def test_compile_spawn(self):
        env = FleetFormulaEnv()
        compiler = FormulaCompiler(env)
        fn = compiler.compile('=SPAWN("scout")')
        assert fn() == "SPAWN:scout:1"

    def test_compile_if_true(self):
        env = FleetFormulaEnv()
        compiler = FormulaCompiler(env)
        fn = compiler.compile('=IF(1 < 2, SPAWN("a"), IDLE())')
        assert fn() == "SPAWN:a:1"

    def test_compile_if_false(self):
        env = FleetFormulaEnv()
        compiler = FormulaCompiler(env)
        fn = compiler.compile('=IF(2 < 1, SPAWN("a"), IDLE())')
        assert fn() == "IDLE"

    def test_compile_infix_add(self):
        env = FleetFormulaEnv()
        compiler = FormulaCompiler(env)
        fn = compiler.compile("=1 + 2 + 3")
        assert fn() == 6.0

    def test_compile_infix_mul(self):
        env = FleetFormulaEnv()
        compiler = FormulaCompiler(env)
        fn = compiler.compile("=2 * 3")
        assert fn() == 6.0

    def test_compile_infix_div(self):
        env = FleetFormulaEnv()
        compiler = FormulaCompiler(env)
        fn = compiler.compile("=6 / 2")
        assert fn() == 3.0

    def test_compile_comparison(self):
        env = FleetFormulaEnv()
        compiler = FormulaCompiler(env)
        fn = compiler.compile("=5 > 3")
        assert fn() is True
        fn = compiler.compile("=3 > 5")
        assert fn() is False

    def test_compile_complex_policy(self):
        env = FleetFormulaEnv()
        compiler = FormulaCompiler(env)
        fn = compiler.compile(
            '=IF(AND(FLEET_HEALTH() < 0.5, THERMAL_AVG() > 0.8), ALERT("critical", "thermal high"), IDLE())'
        )
        result = fn()
        assert result == "IDLE"  # defaults: health=1.0, thermal=0.0

    def test_compile_with_mock_conductor(self):
        class MockConductor:
            def health(self):
                return 0.3

            def queue_depth(self):
                return 5

        env = FleetFormulaEnv(conductor=MockConductor())
        compiler = FormulaCompiler(env)
        fn = compiler.compile("=FLEET_HEALTH()")
        assert fn() == 0.3

    def test_compile_batch(self):
        env = FleetFormulaEnv()
        compiler = FormulaCompiler(env)
        fns = compiler.compile_batch(["=1", "=2", "=3"])
        assert [f() for f in fns] == [1.0, 2.0, 3.0]

    def test_to_python_source(self):
        env = FleetFormulaEnv()
        compiler = FormulaCompiler(env)
        py = compiler.to_python_source("=1 + 2")
        assert "1.0 + 2.0" in py

    def test_to_python_source_call(self):
        env = FleetFormulaEnv()
        compiler = FormulaCompiler(env)
        py = compiler.to_python_source('=SPAWN("scout")')
        assert "SPAWN('scout')" in py


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_formula(self):
        with pytest.raises(SyntaxError):
            FormulaParser("=").parse()

    def test_whitespace_only(self):
        with pytest.raises(SyntaxError):
            FormulaParser("=   ").parse()

    def test_division_by_zero(self):
        env = FleetFormulaEnv()
        compiler = FormulaCompiler(env)
        fn = compiler.compile("=1 / 0")
        assert fn() == float("inf")

    def test_nested_if(self):
        env = FleetFormulaEnv()
        compiler = FormulaCompiler(env)
        fn = compiler.compile('=IF(1 < 2, IF(3 < 4, "a", "b"), "c")')
        assert fn() == "a"

    def test_large_expression(self):
        env = FleetFormulaEnv()
        compiler = FormulaCompiler(env)
        fn = compiler.compile("=1 + 2 * 3 + 4 / 2 - 5")
        assert fn() == 1 + 2 * 3 + 4 / 2 - 5

    def test_formula_with_conductor_health(self):
        class MockConductor:
            def health(self):
                return 0.7

        env = FleetFormulaEnv(conductor=MockConductor())
        compiler = FormulaCompiler(env)
        fn = compiler.compile('=IF(FLEET_HEALTH() > 0.5, SPAWN("worker"), STOP())')
        assert fn() == "SPAWN:worker:1"
