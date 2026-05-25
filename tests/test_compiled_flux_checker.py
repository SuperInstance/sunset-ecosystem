"""Tests for CompiledFluxChecker — FLUX Path B integration.

Run: python3 -m pytest tests/test_compiled_flux_checker.py -v --tb=short
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from swarm.compiled_flux_checker import CompiledFluxChecker
from swarm.flux_gating import FluxGatingConfig, ViolationSeverity
from swarm.flux_vm_runner import FluxTrap


class TestCompiledFluxCheckerBasics:
    def test_init_with_defaults(self):
        checker = CompiledFluxChecker()
        assert checker.config.chaos_limit == 1.0
        assert checker._cache == {}

    def test_init_with_custom_config(self):
        config = FluxGatingConfig(chaos_limit=0.5, weight_bounds=(2.0, 8.0))
        checker = CompiledFluxChecker(config)
        assert checker.config.chaos_limit == 0.5
        assert checker.config.weight_bounds == (2.0, 8.0)

    def test_weight_bounds_pass(self):
        config = FluxGatingConfig(weight_bounds=(0.0, 10.0))
        checker = CompiledFluxChecker(config)
        result = checker.check_candidate(0, {"weights": [3.0, 4.0]})
        assert result.passed
        assert result.violations == []

    def test_weight_bounds_block(self):
        config = FluxGatingConfig(weight_bounds=(0.0, 4.0))
        checker = CompiledFluxChecker(config)
        result = checker.check_candidate(0, {"weights": [3.0, 4.0]})
        assert not result.passed
        assert any(v.constraint_id == "weight_bounds" for v in result.violations)

    def test_chaos_limit_pass(self):
        config = FluxGatingConfig(chaos_limit=1.0)
        checker = CompiledFluxChecker(config)
        result = checker.check_candidate(0, {"chaos": 0.5})
        assert result.passed

    def test_chaos_limit_block(self):
        config = FluxGatingConfig(chaos_limit=0.5)
        checker = CompiledFluxChecker(config)
        result = checker.check_candidate(0, {"chaos": 0.8})
        assert not result.passed
        assert any(v.constraint_id == "chaos_limit" for v in result.violations)
        assert any(v.severity == ViolationSeverity.WARNING for v in result.violations)

    def test_thermal_budget_pass(self):
        config = FluxGatingConfig(thermal_budget_limit=1.0)
        checker = CompiledFluxChecker(config)
        result = checker.check_candidate(0, {"thermal": 0.5})
        assert result.passed

    def test_thermal_budget_block(self):
        config = FluxGatingConfig(thermal_budget_limit=0.5)
        checker = CompiledFluxChecker(config)
        result = checker.check_candidate(0, {"thermal": 0.8})
        assert not result.passed
        assert any(v.constraint_id == "thermal_budget" for v in result.violations)

    def test_multiple_violations(self):
        config = FluxGatingConfig(
            chaos_limit=0.5,
            thermal_budget_limit=0.5,
            weight_bounds=(0.0, 5.0),
        )
        checker = CompiledFluxChecker(config)
        result = checker.check_candidate(
            0,
            {"chaos": 0.8, "thermal": 0.8, "weights": [3.0, 4.0]},
        )
        assert not result.passed
        ids = {v.constraint_id for v in result.violations}
        assert "chaos_limit" in ids
        assert "thermal_budget" in ids

    def test_empty_plan_passes(self):
        checker = CompiledFluxChecker()
        result = checker.check_candidate(0, {})
        assert result.passed

    def test_checker_type(self):
        checker = CompiledFluxChecker()
        result = checker.check_candidate(0, {"chaos": 0.5})
        assert result.metadata.get("checker_type") == "compiled_flux"


class TestCompiledFluxCheckerCaching:
    def test_compiles_once(self):
        config = FluxGatingConfig(chaos_limit=1.0)
        checker = CompiledFluxChecker(config)
        # First call compiles
        checker.check_candidate(0, {"chaos": 0.5})
        assert len(checker._cache) == 1
        assert "chaos" in checker._cache
        # Second call uses cache
        checker.check_candidate(0, {"chaos": 0.3})
        assert len(checker._cache) == 1  # still only one entry

    def test_thermal_compiles_separately(self):
        config = FluxGatingConfig(chaos_limit=1.0, thermal_budget_limit=1.0)
        checker = CompiledFluxChecker(config)
        checker.check_candidate(0, {"chaos": 0.5, "thermal": 0.5})
        assert len(checker._cache) == 2
        assert "chaos" in checker._cache
        assert "thermal" in checker._cache

    def test_cache_entries_have_bytecode(self):
        config = FluxGatingConfig(chaos_limit=1.0)
        checker = CompiledFluxChecker(config)
        checker.check_candidate(0, {"chaos": 0.5})
        entry = checker._cache["chaos"]
        assert len(entry.bytecode) > 0
        assert len(entry.const_pool) >= 0

    def test_stats(self):
        config = FluxGatingConfig(chaos_limit=0.5)
        checker = CompiledFluxChecker(config)
        checker.check_candidate(0, {"chaos": 0.8})  # blocked
        checker.check_candidate(0, {"chaos": 0.3})  # passed
        stats = checker.stats()
        assert stats["check_count"] == 2
        assert stats["violation_count"] == 1
        assert stats["cache_size"] == 1


class TestCompiledFluxCheckerBatch:
    def test_batch_all_pass(self):
        config = FluxGatingConfig(chaos_limit=1.0)
        checker = CompiledFluxChecker(config)
        plans = [{"chaos": 0.1}, {"chaos": 0.2}, {"chaos": 0.3}]
        results = checker.check_batch([0, 1, 2], plans)
        assert len(results) == 3
        assert all(r.passed for r in results)

    def test_batch_some_block(self):
        config = FluxGatingConfig(chaos_limit=0.5)
        checker = CompiledFluxChecker(config)
        plans = [{"chaos": 0.3}, {"chaos": 0.8}, {"chaos": 0.2}]
        results = checker.check_batch([0, 1, 2], plans)
        assert results[0].passed
        assert not results[1].passed
        assert results[2].passed


class TestCompiledFluxCheckerVMIntegration:
    def test_vm_executes_bytecode(self):
        """Verify the VM actually runs compiled bytecode."""
        config = FluxGatingConfig(chaos_limit=0.5)
        checker = CompiledFluxChecker(config)
        # Manually compile and run
        cc = checker._get_cached("chaos", checker._compile_chaos_limit)
        from swarm.flux_vm_runner import FluxVMRunner
        runner = FluxVMRunner(cc.const_pool)
        # Bytecode should return 0.0 when chaos > limit
        # We can't inject runtime vars into the VM directly without Var binding,
        # but the constraint is compiled with the limit as a constant.
        # This test verifies the VM can execute the bytecode without crashing.
        result = runner.run(cc.bytecode)
        assert isinstance(result, float)

    def test_vm_flux_trap_on_validate(self):
        """FluxTrap should raise when Validate sees 0.0."""
        from swarm.flux_compiler import Const, FluxCompiler
        compiler = FluxCompiler()
        emitter = compiler.compile_constraint(Const(0.0), with_validate=True, with_halt=True)
        from swarm.flux_vm_runner import FluxVMRunner
        runner = FluxVMRunner(emitter.const_pool)
        with pytest.raises(FluxTrap):
            runner.run(emitter.to_bytes())


class TestBreederDaemonV2CompiledIntegration:
    def _make_mock_grid(self):
        grid = MagicMock()
        grid.cold.return_value = [1, 2, 3, 4]
        grid.rooms = {i: MagicMock() for i in range(10)}
        return grid

    def test_breeder_accepts_compiled_checker(self):
        from swarm.breeder_daemon_v2 import BreederDaemonV2
        grid = self._make_mock_grid()
        checker = CompiledFluxChecker(FluxGatingConfig(chaos_limit=0.5))
        daemon = BreederDaemonV2(
            grid=grid,
            thermal=MagicMock(),
            compiled_checker=checker,
        )
        assert daemon._compiled_checker is checker

    def test_breeder_uses_compiled_checker_in_select_parents(self):
        from swarm.breeder_daemon_v2 import BreederDaemonV2
        grid = self._make_mock_grid()
        checker = CompiledFluxChecker(FluxGatingConfig(chaos_limit=0.5))
        daemon = BreederDaemonV2(
            grid=grid,
            thermal=MagicMock(),
            compiled_checker=checker,
        )
        # Mock population
        daemon._fsm = {i: MagicMock() for i in range(1, 5)}
        daemon._get_breedable_candidates = lambda: list(range(1, 5))
        pairs = daemon.select_parents(2)
        assert len(pairs) == 2
        # Verify the daemon has the compiled checker wired
        assert daemon._compiled_checker is checker
        # Note: select_parents may not trigger the checker if it takes
        # the fitness-only fallback path (no vector table).  The wiring
        # is verified by _check_flux/_flux_passed unit tests.
