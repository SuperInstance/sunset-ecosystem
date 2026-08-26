"""Mini SGD / Adam optimizer for fleet parameter tuning.

Lightweight gradient-based optimizer for tuning fleet hyperparameters,
breeding weights, and room layout scores. Pure NumPy-free implementation.

Usage:
    opt = AdamOptimizer(lr=0.01)
    params = {"lr": 0.1, "temp": 0.5}
    grads = {"lr": -0.02, "temp": 0.01}
    params = opt.step(params, grads)
"""

from __future__ import annotations

import math
from typing import Dict, Any, Callable, Optional


class Optimizer:
    """Base class for parameter optimizers."""

    def step(
        self, params: Dict[str, float], grads: Dict[str, float]
    ) -> Dict[str, float]:
        raise NotImplementedError


class SGDOptimizer(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum.

    :param lr: Learning rate.
    :param momentum: Momentum coefficient (0 = no momentum).
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self._velocity: Dict[str, float] = {}

    def step(
        self, params: Dict[str, float], grads: Dict[str, float]
    ) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for key, val in params.items():
            grad = grads.get(key, 0.0)
            v = self._velocity.get(key, 0.0)
            v = self.momentum * v - self.lr * grad
            self._velocity[key] = v
            result[key] = val + v
        return result


class AdamOptimizer(Optimizer):
    """
    Adam optimizer (Kingma & Ba, 2015).

    :param lr: Learning rate.
    :param beta1: Exponential decay for first moment.
    :param beta2: Exponential decay for second moment.
    :param eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._t = 0
        self._m: Dict[str, float] = {}
        self._v: Dict[str, float] = {}

    def step(
        self, params: Dict[str, float], grads: Dict[str, float]
    ) -> Dict[str, float]:
        self._t += 1
        result: Dict[str, float] = {}
        for key, val in params.items():
            grad = grads.get(key, 0.0)
            m = self.beta1 * self._m.get(key, 0.0) + (1 - self.beta1) * grad
            v = self.beta2 * self._v.get(key, 0.0) + (1 - self.beta2) * (grad**2)
            self._m[key] = m
            self._v[key] = v
            m_hat = m / (1 - self.beta1**self._t)
            v_hat = v / (1 - self.beta2**self._t)
            result[key] = val - self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
        return result

    def state(self) -> Dict[str, Any]:
        return {"t": self._t, "m": dict(self._m), "v": dict(self._v)}


class RMSPropOptimizer(Optimizer):
    """
    RMSProp optimizer.

    :param lr: Learning rate.
    :param alpha: Decay rate for moving average.
    :param eps: Small constant.
    """

    def __init__(self, lr: float = 0.001, alpha: float = 0.99, eps: float = 1e-8):
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self._cache: Dict[str, float] = {}

    def step(
        self, params: Dict[str, float], grads: Dict[str, float]
    ) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for key, val in params.items():
            grad = grads.get(key, 0.0)
            cache = self.alpha * self._cache.get(key, 0.0) + (1 - self.alpha) * (
                grad**2
            )
            self._cache[key] = cache
            result[key] = val - self.lr * grad / (math.sqrt(cache) + self.eps)
        return result
