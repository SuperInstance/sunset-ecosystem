"""Test: Agentic Compiler — End-to-End Compilation Demo.

Demonstrates the compiler autonomously:
  1. Profiling a hot function
  2. Compiling it to Numba
  3. Validating correctness (A/B test)
  4. Measuring speedup
  5. Hot-swapping at runtime

Usage:
    python3 -m scripts.test_compiler
"""

from __future__ import annotations

import time
import numpy as np


# A hot numeric function — the kind Numba excels at
def expensive_dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Compute (a * b).sum() with manual loop — expensive in pure Python."""
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


def run_demo():
    print("=" * 60)
    print("AGENTIC COMPILER — End-to-End Demo")
    print("=" * 60)

    # Step 1: Profile the function manually
    print("\n[1] Profiling: 200 calls to expensive_dot_product()")
    a = np.random.randn(10000).astype(np.float32)
    b = np.random.randn(10000).astype(np.float32)

    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        expensive_dot_product(a, b)
        times.append((time.perf_counter() - t0) * 1000)
    avg_ms = sum(times) / len(times)
    print(f"    Average: {avg_ms:.3f}ms/call")

    # Step 2: Compile with CodeGenerator
    print("\n[2] Compiling to Numba...")
    from sunset.codegen import CodeGenerator

    gen = CodeGenerator()
    kernel = gen.compile(expensive_dot_product, test_args=(a, b))

    if not kernel.ready:
        print(f"    ❌ Compilation failed: {kernel.error}")
        return

    print(f"    ✅ Compiled in {kernel.compile_time_ms:.1f}ms ({kernel.backend})")

    # Step 3: Validate correctness
    print("\n[3] Validating (A/B test, 10 trials)...")
    validated = gen.validate(kernel, expensive_dot_product, (a, b), trials=10)
    print(
        f"    {'✅' if validated else '❌'} Correctness: {'PASS' if validated else 'FAIL'}"
    )

    if not validated:
        return

    # Step 4: Measure speedup
    print("\n[4] Measuring speedup...")
    speedup = gen.measure_speedup(kernel, expensive_dot_product, (a, b), trials=50)
    print(f"    🔥 Speedup: {speedup:.1f}×")

    # Step 5: Hot-swap demonstration
    print("\n[5] Hot-swap demonstration...")
    import types

    fake_module = types.ModuleType("test_module")
    fake_module.expensive_dot_product = expensive_dot_product

    deployed = gen.deploy(kernel, fake_module, "expensive_dot_product")
    print(f"    {'✅' if deployed else '❌'} Hot-swapped into test_module")

    # Verify the swapped function is faster
    t0 = time.perf_counter()
    for _ in range(100):
        fake_module.expensive_dot_product(a, b)
    swapped_ms = (time.perf_counter() - t0) / 100 * 1000
    print(f"    Post-swap: {swapped_ms:.3f}ms/call (was {avg_ms:.3f}ms)")

    print("\n" + "=" * 60)
    print("Demo complete. The compiler can now:")
    print("  • Profile functions at runtime")
    print("  • Compile hot paths to Numba LLVM")
    print("  • Validate correctness via A/B testing")
    print("  • Measure actual speedup (not estimates)")
    print("  • Hot-swap compiled kernels without restarting")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
