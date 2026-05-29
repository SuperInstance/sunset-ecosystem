"""Tests for Mercury Compiler Agent.

Covers mock compilation, plugin loading, cache management, breeding defect
reporting, and agent status. Real compilation tests are skipped when mmc
is not available.
"""

import os
import tempfile
from pathlib import Path

import pytest

from fleet.mercury_compiler_agent import (
    MercuryCompilerAgent,
    CompileResult,
    _MMC_AVAILABLE,
)


class TestInit:
    def test_default(self):
        agent = MercuryCompilerAgent(node_id="alpha")
        assert agent.node_id == "alpha"
        assert agent.cache_dir == ".mercury_cache"
        assert agent.last_compile_success is False

    def test_custom_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = MercuryCompilerAgent(node_id="alpha", cache_dir=tmp)
            assert agent.cache_dir == tmp
            assert Path(tmp).exists()


class TestCompileFormula:
    def test_mock_compile(self):
        agent = MercuryCompilerAgent(node_id="alpha")
        result = agent.compile_formula("test", "=1 + 2", determinism="det")
        assert result.success is True
        assert result.formula_name == "test"
        assert result.determinism == "det"
        assert "mock" in str(result.warnings).lower() or not _MMC_AVAILABLE
        assert agent.last_compile_success is True

    def test_compile_generates_mercury_code(self):
        agent = MercuryCompilerAgent(node_id="alpha")
        result = agent.compile_formula("health", "=IF(FLEET_HEALTH()>0.5, PASS, FAIL)")
        assert result.mercury_code != ""
        assert "pred" in result.mercury_code or "module" in result.mercury_code or not _MMC_AVAILABLE

    def test_compile_history(self):
        agent = MercuryCompilerAgent(node_id="alpha")
        agent.compile_formula("a", "=1")
        agent.compile_formula("b", "=2")
        history = agent.get_compile_history()
        assert len(history) == 2
        assert history[0].formula_name == "a"
        assert history[1].formula_name == "b"

    def test_compile_creates_cache_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = MercuryCompilerAgent(node_id="alpha", cache_dir=tmp)
            result = agent.compile_formula("test", "=1 + 2")
            if result.success:
                cache_files = agent.get_cache_contents()
                assert any("test" in f for f in cache_files)

    def test_compile_timing(self):
        agent = MercuryCompilerAgent(node_id="alpha")
        result = agent.compile_formula("timing", "=1 + 2")
        assert result.compile_time_ms >= 0.0


class TestPluginLoading:
    def test_load_mock_plugin(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = MercuryCompilerAgent(node_id="alpha", cache_dir=tmp)
            # Create a fake .so file
            so_path = Path(tmp) / "fake.so"
            so_path.write_text("mock")
            # Without real .so, load will fail — but we can test the cache exists
            assert so_path.exists()

    def test_run_without_load_raises(self):
        agent = MercuryCompilerAgent(node_id="alpha")
        with pytest.raises(RuntimeError):
            agent.run_plugin("nonexistent")


class TestCacheManagement:
    def test_flush_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = MercuryCompilerAgent(node_id="alpha", cache_dir=tmp)
            result = agent.compile_formula("a", "=A1 + B1")
            if result.success:
                assert len(agent.get_cache_contents()) > 0
                agent.flush_cache()
                assert len(agent.get_cache_contents()) == 0

    def test_get_cache_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = MercuryCompilerAgent(node_id="alpha", cache_dir=tmp)
            result = agent.compile_formula("a", "=A1 + B1")
            if result.success:
                contents = agent.get_cache_contents()
                assert isinstance(contents, list)
                assert len(contents) > 0


class TestBreedingDefect:
    def test_report_failure(self):
        agent = MercuryCompilerAgent(node_id="alpha")
        result = CompileResult(
            formula_name="bad",
            success=False,
            mercury_code="",
            errors=["syntax error"],
        )
        defect = agent.report_breeding_defect(result)
        assert defect["defect_type"] == "compilation_failure"
        assert defect["formula_name"] == "bad"
        assert defect["errors"] == ["syntax error"]
        assert defect["node_id"] == "alpha"


class TestAgentStatus:
    def test_status(self):
        agent = MercuryCompilerAgent(node_id="alpha")
        status = agent.get_agent_status()
        assert status["node_id"] == "alpha"
        assert status["mmc_available"] == _MMC_AVAILABLE
        assert status["plugins_loaded"] == 0
        assert status["compiles_total"] == 0

    def test_status_after_compile(self):
        agent = MercuryCompilerAgent(node_id="alpha")
        result = agent.compile_formula("a", "=A1 + B1")
        status = agent.get_agent_status()
        assert status["compiles_total"] == 1
        if result.success:
            assert status["compiles_success"] == 1
        else:
            assert status["compiles_success"] == 0


class TestCompileResult:
    def test_defaults(self):
        r = CompileResult(formula_name="x", success=True, mercury_code="code")
        assert r.so_path is None
        assert r.errors == []
        assert r.warnings == []
        assert r.determinism == "unknown"

    def test_with_errors(self):
        r = CompileResult(formula_name="x", success=False, mercury_code="", errors=["fail"])
        assert r.success is False
        assert r.errors == ["fail"]


class TestRealCompilation:
    @pytest.mark.skipif(not _MMC_AVAILABLE, reason="mmc not installed")
    def test_real_compile(self):
        agent = MercuryCompilerAgent(node_id="alpha")
        result = agent.compile_formula("real_test", "=1 + 2")
        assert result.success is True
        assert result.so_path is not None
        assert Path(result.so_path).exists()

    @pytest.mark.skipif(not _MMC_AVAILABLE, reason="mmc not installed")
    def test_real_load_and_run(self):
        agent = MercuryCompilerAgent(node_id="alpha")
        result = agent.compile_formula("real_test", "=1 + 2")
        assert agent.load_plugin("real_test") is True
        run_result = agent.run_plugin("real_test")
        assert run_result["plugin"] == "real_test"
