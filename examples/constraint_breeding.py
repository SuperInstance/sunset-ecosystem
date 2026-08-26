"""examples/constraint_breeding.py — Constraint Theory breeding demo.

Demonstrates exact Pythagorean snapping for breeding population vectors.
Eliminates floating-point divergence in multi-agent breeding.

Usage:
    python examples/constraint_breeding.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from swarm.constraint_bridge import ConstraintBridge


def main():
    print("🔷 Constraint Theory Breeding Demo")
    print("=" * 40)

    bridge = ConstraintBridge(density=500)
    print(
        f"Initialized with density=500, {bridge.get_stats()['triples_cached']} triples cached"
    )

    # Simulate a breeding population with 2D direction vectors
    print("\n📊 Simulating breeding population...")
    population = [
        [0.577, 0.816],  # Approximate direction
        [0.333, 0.667],  # Another direction
        [0.123, 0.987],  # Near-vertical
        [0.999, 0.001],  # Near-horizontal
    ]

    print("\n🔍 Snapping vectors to exact Pythagorean coordinates:")
    for i, vec in enumerate(population):
        result = bridge.snap_vector(vec)
        a, b = result.exact[0], result.exact[1]
        mag_sq = a * a + b * b
        print(
            f"  Vector {i + 1}: {vec} -> [{a:.4f}, {b:.4f}] |v|² = {mag_sq:.10f} (noise: {result.noise:.4f})"
        )

    # Demonstrate exactness vs floating-point
    print("\n⚖️  Exactness comparison:")
    x = 0.6
    y = 0.8
    float_mag = x * x + y * y
    print(f"  Float: 0.6² + 0.8² = {float_mag:.16f} ❌")

    # Snap to exact triple (3, 4, 5)
    result = bridge.snap_vector([0.6, 0.8])
    exact = result.exact
    exact_mag = exact[0] * exact[0] + exact[1] * exact[1]
    print(f"  Exact: {exact[0]}² + {exact[1]}² = {exact_mag:.10f} ✅")

    # Quantization demonstration
    print("\n🎯 Quantization modes:")
    embedding = [0.5, -0.2, 0.05, -0.8, 0.33, -0.67]
    for mode in ["ternary", "turbo", "hybrid"]:
        q = bridge.quantize_embedding(embedding, mode=mode)
        print(f"  {mode:8s}: {embedding} -> {q.tolist()}")

    # Holonomy check for consensus cycle
    print("\n🔄 Holonomy verification (consistency check):")
    cycle = [
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
    ]
    is_consistent = bridge.check_holonomy(cycle)
    print(f"  Cycle consistent: {is_consistent} ✅")

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
