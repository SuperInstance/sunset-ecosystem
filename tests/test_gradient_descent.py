"""Tests for gradient_descent.py — SGD / Adam / RMSProp optimizers.

Run: python3 -m pytest tests/test_gradient_descent.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.gradient_descent import SGDOptimizer, AdamOptimizer, RMSPropOptimizer


class TestSGDOptimizer:
    def test_simple_step(self):
        opt = SGDOptimizer(lr=0.1)
        params = {"x": 1.0}
        grads = {"x": 2.0}
        new = opt.step(params, grads)
        assert abs(new["x"] - 0.8) < 1e-6

    def test_zero_grad(self):
        opt = SGDOptimizer(lr=0.1)
        params = {"x": 1.0}
        new = opt.step(params, {})
        assert new["x"] == 1.0

    def test_momentum(self):
        opt = SGDOptimizer(lr=0.1, momentum=0.9)
        params = {"x": 1.0}
        grads = {"x": 2.0}
        new = opt.step(params, grads)
        # v = -0.2
        assert abs(new["x"] - 0.8) < 1e-6
        new2 = opt.step(new, grads)
        # v = 0.9*(-0.2) - 0.1*2 = -0.18 - 0.2 = -0.38
        assert abs(new2["x"] - 0.42) < 1e-6


class TestAdamOptimizer:
    def test_simple_step(self):
        opt = AdamOptimizer(lr=0.1)
        params = {"x": 1.0}
        grads = {"x": 2.0}
        new = opt.step(params, grads)
        # t=1, m=0.2, v=0.004, m_hat=2.0, v_hat=0.004/(1-0.999)=4.0
        # delta = 0.1 * 2.0 / (2.0 + 1e-8) ≈ 0.1
        assert abs(new["x"] - 0.9) < 1e-3

    def test_state(self):
        opt = AdamOptimizer()
        params = {"x": 1.0}
        grads = {"x": 1.0}
        opt.step(params, grads)
        state = opt.state()
        assert state["t"] == 1
        assert "m" in state
        assert "v" in state

    def test_multiple_params(self):
        opt = AdamOptimizer(lr=0.1)
        params = {"a": 1.0, "b": 2.0}
        grads = {"a": 1.0, "b": -1.0}
        new = opt.step(params, grads)
        assert new["a"] < 1.0
        assert new["b"] > 2.0


class TestRMSPropOptimizer:
    def test_simple_step(self):
        opt = RMSPropOptimizer(lr=0.1)
        params = {"x": 1.0}
        grads = {"x": 2.0}
        new = opt.step(params, grads)
        # cache = 0.01*4 = 0.04
        # delta = 0.1 * 2 / sqrt(0.04) = 0.1 * 2 / 0.2 = 1.0
        assert abs(new["x"] - 0.0) < 1e-6

    def test_zero_grad(self):
        opt = RMSPropOptimizer(lr=0.1)
        params = {"x": 1.0}
        new = opt.step(params, {})
        assert new["x"] == 1.0
