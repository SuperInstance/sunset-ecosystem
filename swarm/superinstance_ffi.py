#!/usr/bin/env python3
"""Real ctypes bindings for libsuperinstance_ffi.so — built from FM's Rust crate."""

import ctypes
import os

# ---------------------------------------------------------------------------
# Load the shared library
# ---------------------------------------------------------------------------
_LIB_DIR = os.path.join(
    os.path.dirname(__file__), "superinstance-ffi", "target", "release"
)
_LIB_PATH = os.path.join(_LIB_DIR, "libsuperinstance_ffi.so")

if not os.path.exists(_LIB_PATH):
    raise RuntimeError(
        f"libsuperinstance_ffi.so not found at {_LIB_PATH}. Run `cargo build --release` in superinstance-ffi/"
    )

lib = ctypes.CDLL(_LIB_PATH)

# ---------------------------------------------------------------------------
# 1. eisenstein_norm
# ---------------------------------------------------------------------------
lib.eisenstein_norm.argtypes = [ctypes.c_int, ctypes.c_int]
lib.eisenstein_norm.restype = ctypes.c_int


def eisenstein_norm(a: int, b: int) -> int:
    """Eisenstein integer norm N(a,b) = a² − a·b + b²."""
    return lib.eisenstein_norm(a, b)


# ---------------------------------------------------------------------------
# 2. Laman checks
# ---------------------------------------------------------------------------
lib.laman_check_subset.argtypes = [ctypes.c_uint, ctypes.c_uint]
lib.laman_check_subset.restype = ctypes.c_int
lib.laman_is_rigid.argtypes = [ctypes.c_uint, ctypes.c_uint]
lib.laman_is_rigid.restype = ctypes.c_int


def laman_check_subset(num_vertices: int, num_edges: int) -> bool:
    return bool(lib.laman_check_subset(num_vertices, num_edges))


def laman_is_rigid(num_vertices: int, num_edges: int) -> bool:
    return bool(lib.laman_is_rigid(num_vertices, num_edges))


# ---------------------------------------------------------------------------
# 3. holonomy_check
# ---------------------------------------------------------------------------
lib.holonomy_check.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_uint,
    ctypes.c_double,
]
lib.holonomy_check.restype = ctypes.c_float


def holonomy_check(states: list[float], threshold: float) -> float:
    """Cyclic drift consistency check. Returns 1.0 if consistent, 0.0 otherwise."""
    n = len(states)
    arr = (ctypes.c_double * n)(*states)
    return lib.holonomy_check(arr, n, threshold)


# ---------------------------------------------------------------------------
# 4. pythagorean48_encode
# ---------------------------------------------------------------------------
lib.pythagorean48_encode.argtypes = [ctypes.c_int, ctypes.c_int]
lib.pythagorean48_encode.restype = ctypes.c_int


def pythagorean48_encode(numerator: int, denominator: int) -> int:
    """Frequency ratio → 48-tone index."""
    return lib.pythagorean48_encode(numerator, denominator)


# ---------------------------------------------------------------------------
# 5. constraint check
# ---------------------------------------------------------------------------
lib.constraint_check.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
lib.constraint_check.restype = ctypes.c_int
lib.constraint_violation.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
lib.constraint_violation.restype = ctypes.c_double


def constraint_check(value: float, lower: float, upper: float) -> bool:
    return bool(lib.constraint_check(value, lower, upper))


def constraint_violation(value: float, lower: float, upper: float) -> float:
    return lib.constraint_violation(value, lower, upper)


# ---------------------------------------------------------------------------
# 6. spline_interpolate
# ---------------------------------------------------------------------------
lib.spline_interpolate.argtypes = [ctypes.c_double] * 5
lib.spline_interpolate.restype = ctypes.c_double


def spline_interpolate(p0: float, p1: float, m0: float, m1: float, t: float) -> float:
    """Hermite cubic spline at parameter t in [0,1]."""
    return lib.spline_interpolate(p0, p1, m0, m1, t)


# ---------------------------------------------------------------------------
# 7. deadband_filter
# ---------------------------------------------------------------------------
lib.deadband_filter.argtypes = [
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
]
lib.deadband_filter.restype = ctypes.c_double


def deadband_filter(value: float, last: float, deadband: float) -> tuple[float, float]:
    """Returns (filtered_value, updated_last)."""
    last_ptr = ctypes.c_double(last)
    result = lib.deadband_filter(value, ctypes.byref(last_ptr), deadband)
    return result, last_ptr.value


# ---------------------------------------------------------------------------
# 8. manhattan_distance
# ---------------------------------------------------------------------------
lib.manhattan_distance.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_uint,
]
lib.manhattan_distance.restype = ctypes.c_float


def manhattan_distance(a: list[float], b: list[float]) -> float:
    """L1 distance between two float arrays."""
    n = len(a)
    if len(b) != n:
        raise ValueError("Arrays must have same length")
    a_arr = (ctypes.c_float * n)(*a)
    b_arr = (ctypes.c_float * n)(*b)
    return lib.manhattan_distance(a_arr, b_arr, n)


# ---------------------------------------------------------------------------
# 9. cascade_match
# ---------------------------------------------------------------------------
lib.cascade_match.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # query
    ctypes.POINTER(ctypes.c_float),  # candidates (flat n×dim)
    ctypes.c_uint,  # n
    ctypes.c_uint,  # dim
    ctypes.POINTER(ctypes.c_float),  # thresholds
    ctypes.c_uint,  # tiers
]
lib.cascade_match.restype = ctypes.c_int


def cascade_match(
    query: list[float], candidates: list[list[float]], thresholds: list[float]
) -> int:
    """Tiered nearest-neighbor search. Returns index of first match, or -1."""
    dim = len(query)
    n = len(candidates)
    if n == 0:
        return -1
    flat = [val for cand in candidates for val in cand]
    if len(flat) != n * dim:
        raise ValueError("All candidates must have same dimension as query")
    tiers = len(thresholds)
    q_arr = (ctypes.c_float * dim)(*query)
    c_arr = (ctypes.c_float * (n * dim))(*flat)
    t_arr = (ctypes.c_float * tiers)(*thresholds)
    return lib.cascade_match(q_arr, c_arr, n, dim, t_arr, tiers)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Superinstance FFI Real Bindings — Self Test ===")
    assert eisenstein_norm(1, 0) == 1
    assert eisenstein_norm(2, 1) == 3
    print(f"  eisenstein_norm(2,1) = {eisenstein_norm(2, 1)} ✓")

    assert laman_is_rigid(3, 3)
    assert not laman_is_rigid(4, 4)
    print(f"  laman_is_rigid(3,3) = {laman_is_rigid(3, 3)} ✓")

    assert holonomy_check([0.0, 0.0, 0.0], 1e-6) == 1.0
    assert holonomy_check([0.0, 1.0, 2.0], 0.1) == 0.0
    print(f"  holonomy_check consistent = {holonomy_check([0.0, 0.0, 0.0], 1e-6)} ✓")

    assert pythagorean48_encode(2, 1) == 12
    assert pythagorean48_encode(3, 2) == 7
    print(f"  pythagorean48_encode(2,1) = {pythagorean48_encode(2, 1)} ✓")

    assert constraint_check(0.5, 0.0, 1.0)
    assert not constraint_check(1.1, 0.0, 1.0)
    print(f"  constraint_check(0.5, 0, 1) = {constraint_check(0.5, 0.0, 1.0)} ✓")

    assert abs(spline_interpolate(0.0, 1.0, 0.0, 0.0, 0.5) - 0.5) < 1e-9
    print(f"  spline at t=0.5 = {spline_interpolate(0.0, 1.0, 0.0, 0.0, 0.5)} ✓")

    val, last = deadband_filter(0.05, 0.0, 0.1)
    assert val == 0.0
    val, last = deadband_filter(0.2, last, 0.1)
    assert val == 0.2
    print(f"  deadband_filter(0.2, 0, 0.1) = {val} ✓")

    assert manhattan_distance([1.0, 2.0, 3.0], [4.0, 0.0, 3.0]) == 5.0
    print(
        f"  manhattan_distance = {manhattan_distance([1.0, 2.0, 3.0], [4.0, 0.0, 3.0])} ✓"
    )

    idx = cascade_match(
        [1.0, 1.0, 1.0], [[0.0, 0.0, 0.0], [1.1, 1.0, 1.0], [2.0, 2.0, 2.0]], [0.5, 1.5]
    )
    assert idx == 1
    print(f"  cascade_match index = {idx} ✓")

    print("\nAll 9 real FFI tests passed. libsuperinstance_ffi.so is live.")
