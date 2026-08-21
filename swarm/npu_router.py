"""NPU Router Offload — ONNX-exported MLP for routing decisions on AMD XDNA 2."""

from __future__ import annotations

__all__ = ["NPURouterOffload"]

import math
import os
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

try:
    import onnx
except ImportError:  # pragma: no cover
    onnx = None  # type: ignore[assignment]

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None  # type: ignore[assignment]

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


class _RoutingMLP(nn.Module if nn else object):  # type: ignore[misc,valid-type]
    """Small MLP: signal vector → routing probabilities.

    Architecture:
      input_dim → Linear(input_dim*2) → ReLU
                → Linear(input_dim)    → ReLU
                → Linear(output_dim)  → softmax
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        if nn is None:
            raise ImportError("torch is required for ONNX export")
        super().__init__()
        hidden1 = input_dim * 2
        hidden2 = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)


class NPURouterOffload:
    """Offload HebbianRouter dispatch decisions to NPU via ONNX Runtime.

    The embedded MLP maps a signal vector (route-state encoding) to a
    probability distribution over output channels.  When attached to a
    :class:`nerve.routing.RoutingLayer` the layer can opt-in to the NPU
    fast-path inside :meth:`fire_fast`.
    """

    #: Latency budget (microseconds).  If a single predict() call exceeds
    #: this value we emit a warning so the caller can fallback_to_cpu().
    LATENCY_THRESHOLD_US: float = 10.0

    def __init__(
        self,
        router: Any,
        provider: str = "VitisAIExecutionProvider",
        input_dim: int = 32,
        output_dim: int = 16,
    ) -> None:
        """Export router to ONNX, run on NPU.

        Args:
            router: The owning routing layer (typically ``RoutingLayer``).
            provider: ONNX Runtime execution provider.  Use
                ``'VitisAIExecutionProvider'`` for AMD XDNA 2,
                ``'CUDAExecutionProvider'`` for GPU fallback, or
                ``'CPUExecutionProvider'`` for CPU fallback.
            input_dim: Size of the signal vector fed into the MLP.
            output_dim: Number of output channels (softmax dimension).
        """
        self.router = router
        self.provider = provider
        self.input_dim = input_dim
        self.output_dim = output_dim

        self._model_path: str | None = None
        self._session: Any = None
        self._fallback_provider = "CPUExecutionProvider"
        self._input_name: str | None = None

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_onnx(
        self,
        input_dim: int | None = None,
        output_dim: int | None = None,
        path: str = "router.onnx",
    ) -> str:
        """Export the router's routing MLP to ONNX format.

        The MLP maps signal vector → routing decision (which channel to
        activate).  Hidden layers: ``[input_dim*2, input_dim]`` with ReLU.
        Output: softmax over channels.

        Args:
            input_dim: Override default input dimension.
            output_dim: Override default output dimension.
            path: Destination file path for the ONNX model.

        Returns:
            Absolute path to the exported ONNX model.
        """
        if torch is None or onnx is None:
            raise ImportError("torch and onnx are required for export")

        idim = input_dim if input_dim is not None else self.input_dim
        odim = output_dim if output_dim is not None else self.output_dim

        model = _RoutingMLP(idim, odim)
        model.eval()

        dummy_input = torch.randn(1, idim)
        torch.onnx.export(
            model,
            dummy_input,
            path,
            input_names=["signal"],
            output_names=["probs"],
            dynamic_axes={"signal": {0: "batch"}, "probs": {0: "batch"}},
            opset_version=14,
        )

        # Verify structural correctness
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)

        self._model_path = os.path.abspath(path)
        self.input_dim = idim
        self.output_dim = odim
        return self._model_path

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _init_session(self) -> Any:
        """Lazy-init ONNX Runtime session with provider priority."""
        if self._session is not None:
            return self._session
        if ort is None:
            raise ImportError("onnxruntime is required for inference")

        path = self._model_path
        if path is None or not os.path.exists(path):
            raise RuntimeError("No ONNX model found. Call export_onnx() first.")

        # Build provider list, preferring the requested one but falling
        # back to CPU if it is not installed / not available.
        available = ort.get_available_providers()
        providers = [
            p for p in [self.provider, self._fallback_provider] if p in available
        ]
        if not providers:
            providers = ["CPUExecutionProvider"]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self._session = ort.InferenceSession(path, sess_options, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        return self._session

    def predict(self, signal_vector: np.ndarray) -> np.ndarray:
        """Run inference on NPU.

        Target: <10 μs for 128-dim input → 16-channel output.

        Args:
            signal_vector: 1-D or 2-D array.  If 1-D it is reshaped to
                ``(1, input_dim)``.  Must be float32 or convertible.

        Returns:
            2-D array of shape ``(batch, output_dim)`` with softmax
            probabilities per channel.
        """
        session = self._init_session()
        if self._input_name is None:
            raise RuntimeError("Session not initialized")

        # Normalise shape / dtype
        x = np.asarray(signal_vector, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Guard against dimension mismatch when a model was exported with
        # different geometry.
        if x.shape[1] != self.input_dim:
            # Zero-pad or truncate to the expected input_dim
            if x.shape[1] < self.input_dim:
                pad = np.zeros(
                    (x.shape[0], self.input_dim - x.shape[1]), dtype=np.float32
                )
                x = np.concatenate([x, pad], axis=1)
            else:
                x = x[:, : self.input_dim]

        # Measure wall-clock latency (same clock used in benchmark)
        t0 = time.perf_counter()
        outputs = session.run(None, {self._input_name: x})
        t1 = time.perf_counter()
        latency_us = (t1 - t0) * 1e6

        probs = outputs[0]

        if latency_us > self.LATENCY_THRESHOLD_US:
            warnings.warn(
                f"NPU latency {latency_us:.1f}μs exceeds threshold "
                f"{self.LATENCY_THRESHOLD_US}μs — consider fallback_to_cpu().",
                RuntimeWarning,
                stacklevel=2,
            )

        return probs

    # ------------------------------------------------------------------
    # Benchmark & fallback
    # ------------------------------------------------------------------

    def benchmark(self, n_iterations: int = 1000) -> dict[str, float | str]:
        """Benchmark NPU vs CPU vs GPU latency.

        Returns a mapping ``provider → median latency in milliseconds``.
        Providers that are not installed report the string
        ``"unavailable"``.
        """
        if ort is None:
            raise ImportError("onnxruntime is required")

        path = self._model_path
        if path is None or not os.path.exists(path):
            raise RuntimeError("No ONNX model found. Call export_onnx() first.")

        available = ort.get_available_providers()
        results: dict[str, float | str] = {}

        for provider in [
            "VitisAIExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]:
            if provider not in available:
                results[provider] = "unavailable"
                continue

            sess_options = ort.SessionOptions()
            sess = ort.InferenceSession(path, sess_options, providers=[provider])
            input_name = sess.get_inputs()[0].name
            dummy = np.random.randn(1, self.input_dim).astype(np.float32)

            # Warm-up
            for _ in range(max(10, n_iterations // 100)):
                sess.run(None, {input_name: dummy})

            # Timed loop
            latencies_ms = []
            for _ in range(n_iterations):
                t0 = time.perf_counter()
                sess.run(None, {input_name: dummy})
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)

            results[provider] = float(np.median(latencies_ms))

        return results

    def fallback_to_cpu(self) -> None:
        """Switch to CPU execution provider if NPU unavailable.

        This clears the cached session so the next ``predict()`` call
        re-initialises with ``CPUExecutionProvider``.
        """
        self.provider = "CPUExecutionProvider"
        self._session = None
