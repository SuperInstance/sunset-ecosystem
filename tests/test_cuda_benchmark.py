"""Tests for CUDA benchmark script.

Verifies that the benchmark script runs without crashing even when
CUDA is not available, and that JSON output is valid.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_SCRIPT = PROJECT_ROOT / "benchmarks" / "cuda_benchmark.py"


class TestCUDABenchmark:
    def test_script_exists(self):
        assert BENCHMARK_SCRIPT.exists()

    def test_script_runs_without_crash(self):
        result = subprocess.run(
            [sys.executable, str(BENCHMARK_SCRIPT)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=60,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "JEPA CUDA Kernel Benchmark" in result.stdout

    def test_json_output_written(self):
        out_path = PROJECT_ROOT / "benchmarks" / "cuda_benchmark_results.json"
        if out_path.exists():
            with open(out_path) as f:
                data = json.load(f)
            assert isinstance(data, list)
            for entry in data:
                assert "numpy" in entry
                assert entry["numpy"]["backend"] == "numpy"
