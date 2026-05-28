"""simd_ops.py — SIMD-style vector operations for breeding workloads.

Uses numpy vectorization with explicit block processing to approximate
AVX-512/NEON-style parallelism. Provides:
1. Blockwise vector distance (Euclidean, cosine, Manhattan)
2. Batch mutation with SIMD-style randomness
3. Population fitness evaluation with vectorized scoring
4. Top-k selection with partial sort emulation

All operations process vectors in chunks of 16 (simulating AVX-512
registers on 64-bit floats) or 32 (simulating AVX-512 on 32-bit floats).
"""
from __future__ import annotations

__all__ = [
    "SIMDOps",
    "blockwise_euclidean",
    "blockwise_cosine",
    "batch_mutate",
    "vectorized_fitness",
    "topk_select",
]

import numpy as np

# Simulated SIMD register width (doubles)
SIMD_WIDTH = 16  # 16 × float64 = 1024-bit (AVX-512 style)


def blockwise_euclidean(a: np.ndarray, b: np.ndarray, block: int = SIMD_WIDTH) -> float:
    """Euclidean distance with block-wise reduction (SIMD-style).

    Computes Σ(a_i - b_i)² in chunks of `block` elements,
    then does final scalar reduction.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Shape mismatch")
    d = a - b
    n = len(d)
    total = 0.0
    for i in range(0, n, block):
        chunk = d[i : i + block]
        total += float(np.dot(chunk, chunk))
    return float(np.sqrt(total))


def blockwise_cosine(a: np.ndarray, b: np.ndarray, block: int = SIMD_WIDTH) -> float:
    """Cosine distance: 1 - (a·b) / (|a||b|), computed blockwise."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = len(a)
    dot = 0.0
    a_norm = 0.0
    b_norm = 0.0
    for i in range(0, n, block):
        ca = a[i : i + block]
        cb = b[i : i + block]
        dot += float(np.dot(ca, cb))
        a_norm += float(np.dot(ca, ca))
        b_norm += float(np.dot(cb, cb))
    denom = np.sqrt(a_norm * b_norm)
    if denom < 1e-15:
        return 1.0
    return 1.0 - dot / denom


def blockwise_manhattan(a: np.ndarray, b: np.ndarray, block: int = SIMD_WIDTH) -> float:
    """L1 distance computed blockwise."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = np.abs(a - b)
    n = len(d)
    total = 0.0
    for i in range(0, n, block):
        total += float(d[i : i + block].sum())
    return total


def batch_mutate(
    parents: np.ndarray,
    noise_std: float = 0.1,
    block: int = SIMD_WIDTH,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Vectorized Gaussian mutation across a batch of parents.

    parents: (batch_size, dim) array
    Returns: (batch_size, dim) mutated array

    Noise generated in SIMD-width blocks for cache efficiency.
    """
    rng = rng or np.random.default_rng()
    parents = np.asarray(parents, dtype=np.float64)
    batch_size, dim = parents.shape
    noise = np.zeros_like(parents)

    # Generate noise in blocks
    for i in range(0, dim, block):
        end = min(i + block, dim)
        noise[:, i:end] = rng.normal(0, noise_std, size=(batch_size, end - i))

    return parents + noise


def vectorized_fitness(
    population: np.ndarray,
    objective: np.ndarray,
    metric: str = "cosine",
    block: int = SIMD_WIDTH,
) -> np.ndarray:
    """Score entire population against objective vector.

    metric: 'cosine', 'euclidean', 'dot'
    Returns: (batch_size,) fitness scores (higher = better)
    """
    population = np.asarray(population, dtype=np.float64)
    objective = np.asarray(objective, dtype=np.float64)
    batch_size = population.shape[0]
    scores = np.zeros(batch_size, dtype=np.float64)

    if metric == "dot":
        for i in range(0, len(objective), block):
            end = min(i + block, len(objective))
            scores += np.dot(population[:, i:end], objective[i:end])
        return scores

    elif metric == "cosine":
        obj_norm = np.linalg.norm(objective)
        for i in range(0, len(objective), block):
            end = min(i + block, len(objective))
            scores += np.dot(population[:, i:end], objective[i:end])
        pop_norms = np.linalg.norm(population, axis=1)
        denom = pop_norms * obj_norm
        denom[denom < 1e-15] = 1e-15
        return scores / denom

    elif metric == "euclidean":
        for i in range(0, len(objective), block):
            end = min(i + block, len(objective))
            diff = population[:, i:end] - objective[i:end]
            scores += np.sum(diff * diff, axis=1)
        return -np.sqrt(scores)  # negative = higher fitness for minimization

    else:
        raise ValueError(f"Unknown metric: {metric}")


def topk_select(
    population: np.ndarray,
    fitness: np.ndarray,
    k: int,
    block: int = SIMD_WIDTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Select top-k individuals by fitness using partial sort (numpy.argpartition).

    Returns: (topk_population, topk_fitness)
    """
    population = np.asarray(population, dtype=np.float64)
    fitness = np.asarray(fitness, dtype=np.float64)
    n = len(fitness)
    k = min(k, n)
    # argpartition is O(n) vs O(n log n) for full sort
    idx = np.argpartition(fitness, -k)[-k:]
    # Sort just the top-k for deterministic order
    idx = idx[np.argsort(fitness[idx])[::-1]]
    return population[idx], fitness[idx]


# ── SIMD Ops class (convenience wrapper) ──────────────────────

class SIMDOps:
    """Namespace for SIMD-style vector operations.

    All methods operate on numpy arrays with explicit block processing
    to maximize cache efficiency and approximate vectorized hardware.
    """

    BLOCK = SIMD_WIDTH

    @classmethod
    def distance_matrix(cls, a: np.ndarray, b: np.ndarray, metric: str = "euclidean") -> np.ndarray:
        """Pairwise distance matrix between two populations.

        a: (n, dim), b: (m, dim) -> returns (n, m)
        """
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        n, dim_a = a.shape
        m, dim_b = b.shape
        if dim_a != dim_b:
            raise ValueError("Dimension mismatch")

        D = np.zeros((n, m), dtype=np.float64)

        if metric == "euclidean":
            for i in range(0, dim_a, cls.BLOCK):
                end = min(i + cls.BLOCK, dim_a)
                diff = a[:, None, i:end] - b[None, :, i:end]
                D += np.sum(diff * diff, axis=2)
            np.sqrt(D, out=D)

        elif metric == "cosine":
            a_norm = np.linalg.norm(a, axis=1, keepdims=True)
            b_norm = np.linalg.norm(b, axis=1, keepdims=True)
            dot = np.zeros((n, m), dtype=np.float64)
            for i in range(0, dim_a, cls.BLOCK):
                end = min(i + cls.BLOCK, dim_a)
                dot += np.dot(a[:, i:end], b[:, i:end].T)
            denom = a_norm @ b_norm.T
            denom[denom < 1e-15] = 1e-15
            D = 1.0 - dot / denom

        elif metric == "dot":
            for i in range(0, dim_a, cls.BLOCK):
                end = min(i + cls.BLOCK, dim_a)
                D += np.dot(a[:, i:end], b[:, i:end].T)

        else:
            raise ValueError(f"Unknown metric: {metric}")

        return D

    @classmethod
    def batch_fitness_gradient(cls, population: np.ndarray, objective: np.ndarray) -> np.ndarray:
        """Compute fitness gradient for each individual.

        For cosine similarity: gradient points toward objective.
        Returns: (batch_size, dim) gradient vectors.
        """
        population = np.asarray(population, dtype=np.float64)
        objective = np.asarray(objective, dtype=np.float64)
        # Simplified: gradient is (objective - individual) for Euclidean
        return objective[None, :] - population

    @classmethod
    def accelerate_crossover(cls, a: np.ndarray, b: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Blend crossover with SIMD-width block processing."""
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        result = np.zeros_like(a)
        for i in range(0, len(a), cls.BLOCK):
            end = min(i + cls.BLOCK, len(a))
            result[i:end] = (1 - alpha) * a[i:end] + alpha * b[i:end]
        return result
