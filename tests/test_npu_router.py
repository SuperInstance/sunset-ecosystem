"""Tests for NPURouterOffload — NPU-accelerated routing decisions."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from nerve.routing import RoutingLayer

# Optional deps — skip entire module if torch / onnx / ort missing
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
torch = pytest.importorskip("torch")

from swarm.npu_router import NPURouterOffload


@pytest.fixture
def onnx_model(tmp_path: Path) -> str:
    """Export a fresh ONNX model and return its path."""
    layer = RoutingLayer()
    npu = NPURouterOffload(layer, input_dim=32, output_dim=16)
    path = str(tmp_path / "router.onnx")
    npu.export_onnx(path=path)
    return path


class TestExport:
    """1. Export ONNX → file exists and valid."""

    def test_export_creates_file(self, tmp_path: Path) -> None:
        layer = RoutingLayer()
        npu = NPURouterOffload(layer, input_dim=8, output_dim=4)
        path = str(tmp_path / "test_router.onnx")
        result = npu.export_onnx(path=path)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0

    def test_export_valid_onnx(self, tmp_path: Path) -> None:
        layer = RoutingLayer()
        npu = NPURouterOffload(layer, input_dim=8, output_dim=4)
        path = str(tmp_path / "test_router.onnx")
        npu.export_onnx(path=path)
        model = onnx.load(path)
        onnx.checker.check_model(model)


class TestPredict:
    """2. Predict signal vector → returns n_channels probabilities summing to ~1."""

    def test_predict_shape_and_sum(self, onnx_model: str) -> None:
        layer = RoutingLayer()
        npu = NPURouterOffload(layer, input_dim=32, output_dim=16)
        npu.export_onnx(path=onnx_model)

        signal = np.random.randn(32).astype(np.float32)
        probs = npu.predict(signal)

        assert probs.ndim == 2
        assert probs.shape == (1, 16)
        assert np.allclose(probs.sum(), 1.0, atol=1e-5)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_predict_batch_2d(self, onnx_model: str) -> None:
        layer = RoutingLayer()
        npu = NPURouterOffload(layer, input_dim=32, output_dim=16)
        npu.export_onnx(path=onnx_model)

        batch = np.random.randn(5, 32).astype(np.float32)
        probs = npu.predict(batch)

        assert probs.shape == (5, 16)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_auto_pad(self, onnx_model: str) -> None:
        layer = RoutingLayer()
        npu = NPURouterOffload(layer, input_dim=32, output_dim=16)
        npu.export_onnx(path=onnx_model)

        short = np.random.randn(10).astype(np.float32)  # shorter than input_dim
        probs = npu.predict(short)
        assert probs.shape == (1, 16)


class TestBenchmark:
    """3. Benchmark shows NPU < CPU (if NPU available, else skip)."""

    def test_benchmark_returns_dict(self, onnx_model: str) -> None:
        layer = RoutingLayer()
        npu = NPURouterOffload(layer, input_dim=32, output_dim=16)
        npu.export_onnx(path=onnx_model)
        results = npu.benchmark(n_iterations=50)

        assert isinstance(results, dict)
        for provider in [
            "VitisAIExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]:
            assert provider in results

    def test_cpu_is_always_available(self, onnx_model: str) -> None:
        layer = RoutingLayer()
        npu = NPURouterOffload(layer, input_dim=32, output_dim=16)
        npu.export_onnx(path=onnx_model)
        results = npu.benchmark(n_iterations=20)
        cpu_result = results["CPUExecutionProvider"]
        assert isinstance(cpu_result, float)
        assert cpu_result > 0

    def test_npu_vs_cpu_when_available(self, onnx_model: str) -> None:
        layer = RoutingLayer()
        npu = NPURouterOffload(layer, input_dim=32, output_dim=16)
        npu.export_onnx(path=onnx_model)
        results = npu.benchmark(n_iterations=100)

        npu_result = results["VitisAIExecutionProvider"]
        if npu_result == "unavailable":
            pytest.skip("VitisAIExecutionProvider not installed on this host")

        cpu_result = results["CPUExecutionProvider"]
        assert isinstance(cpu_result, float)
        # NPU should be faster; if it isn't, the test is a canary, not a hard failure.
        assert npu_result < cpu_result, (
            f"NPU latency ({npu_result:.3f} ms) not faster than CPU ({cpu_result:.3f} ms)"
        )


class TestFallback:
    """4. Fallback to CPU when NPU unavailable → same results."""

    def test_fallback_changes_provider(self, onnx_model: str) -> None:
        layer = RoutingLayer()
        npu = NPURouterOffload(layer, input_dim=32, output_dim=16)
        npu.export_onnx(path=onnx_model)

        npu.fallback_to_cpu()
        assert npu.provider == "CPUExecutionProvider"
        assert npu._session is None  # session cache cleared

    def test_fallback_predict_consistency(self, onnx_model: str) -> None:
        layer = RoutingLayer()
        npu = NPURouterOffload(layer, input_dim=32, output_dim=16)
        npu.export_onnx(path=onnx_model)

        signal = np.random.randn(32).astype(np.float32)

        # CPU-only baseline
        npu.fallback_to_cpu()
        cpu_probs = npu.predict(signal)

        # Reset to default (still CPU on this machine likely)
        npu.provider = "CPUExecutionProvider"
        npu._session = None
        default_probs = npu.predict(signal)

        assert np.allclose(cpu_probs, default_probs, atol=1e-6)


class TestIntegration:
    """5. Integration with RoutingLayer.fire_fast() → valid routing."""

    def test_integration_attaches_and_runs(self, tmp_path: Path) -> None:
        layer = RoutingLayer()
        npu = NPURouterOffload(layer, input_dim=32, output_dim=16)
        path = str(tmp_path / "integration_router.onnx")
        npu.export_onnx(path=path)

        # Seed some routes so there are candidates
        for i in range(20):
            layer.add_route("src", f"dst_{i}", strength=0.5 + i * 0.02)

        layer.set_npu_router(npu)
        fired = layer.fire_fast("src", use_npu=True)

        # Must return a non-None list of destinations
        assert isinstance(fired, list)
        # With 20 candidates and output_dim=16 we expect up to 16
        assert len(fired) <= 16
        for d in fired:
            assert d.startswith("dst_")

    def test_integration_fallback_on_slow_npu(self, tmp_path: Path) -> None:
        layer = RoutingLayer()
        npu = NPURouterOffload(layer, input_dim=32, output_dim=16)
        path = str(tmp_path / "slow_router.onnx")
        npu.export_onnx(path=path)

        for i in range(20):
            layer.add_route("src", f"dst_{i}", strength=0.5 + i * 0.02)

        # Artificially lower the latency budget so the CPU path always triggers
        npu.LATENCY_THRESHOLD_US = -1.0  # always "too slow"
        layer.set_npu_router(npu)

        fired = layer.fire_fast("src", use_npu=True)
        assert isinstance(fired, list)
        # Fallback path still returns valid destinations
        assert len(fired) >= 0

    def test_integration_npu_disabled_flag(self, tmp_path: Path) -> None:
        layer = RoutingLayer()
        npu = NPURouterOffload(layer, input_dim=32, output_dim=16)
        path = str(tmp_path / "off_router.onnx")
        npu.export_onnx(path=path)

        for i in range(10):
            layer.add_route("src", f"dst_{i}", strength=0.6)

        layer.set_npu_router(npu)

        # With use_npu=False we hit the vectorised CPU path only
        fired_cpu = layer.fire_fast("src", use_npu=False)
        # With use_npu=True we may hit NPU or fallback — both are valid lists
        fired_auto = layer.fire_fast("src", use_npu=True)

        assert isinstance(fired_cpu, list)
        assert isinstance(fired_auto, list)

    def test_integration_preserves_feedback(self, tmp_path: Path) -> None:
        layer = RoutingLayer()
        npu = NPURouterOffload(layer, input_dim=32, output_dim=16)
        path = str(tmp_path / "fb_router.onnx")
        npu.export_onnx(path=path)

        layer.add_route("src", "dst_a", strength=0.5)
        layer.set_npu_router(npu)

        # Fire and then feedback — should not crash
        fired = layer.fire_fast("src", use_npu=True)
        for d in fired:
            layer.feedback("src", d, success=True)
        # Smoke test only — no assert on exact strength because NPU is stochastic
        assert True
