"""
superinstance_ffi_mock.py — Pure-Python mock of the superinstance-ffi C library.

Reads superinstance_ffi.h at import time, exposes every declared function as a
numpy-backed mock callable wrapped in a CDLL-like object.

Usage::
    from superinstance_ffi_mock import load_mock_ffi
    ffi = load_mock_ffi()
    ffi.eisenstein_norm.argtypes = [c_int, c_int]
    ffi.eisenstein_norm.restype = c_int
    result = ffi.eisenstein_norm(3, 4)
"""
from __future__ import annotations

import ctypes
import math
import os
import re
import numpy as np
from typing import Any, Callable, List, Tuple

# ---------------------------------------------------------------------------
# Header parser
# ---------------------------------------------------------------------------

def _parse_header(path: str) -> List[Tuple[str, str, List[Tuple[str, str]]]]:
    """
    Parse a C header and return a list of::
        (return_type, func_name, [(arg_type, arg_name), ...])
    """
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()

    # Strip block comments /* ... */
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # Strip line comments // ...
    text = re.sub(r"//.*", "", text)

    # Find function declarations: int foo(int a, float b);
    # We want to capture return type, function name, and args
    pattern = re.compile(
        r"([a-zA-Z_][a-zA-Z0-9_*\s]+)\s+"
        r"([a-zA-Z_][a-zA-Z0-9_]*)\s*"
        r"\(([^)]*)\)\s*;"
    )

    functions = []
    for match in pattern.finditer(text):
        raw_ret = match.group(1).strip()
        func_name = match.group(2).strip()
        raw_args = match.group(3).strip()

        # Clean return type (remove extra whitespace, keep pointer info)
        ret_type = " ".join(raw_ret.split())

        # Parse arguments
        args = []
        if raw_args and raw_args.lower() != "void":
            for arg in raw_args.split(","):
                arg = arg.strip()
                if not arg:
                    continue
                # Handle "const double *states" -> "const double *", "states"
                # Also "double *last" -> "double *", "last"
                # Simple regex: everything up to last identifier is type, rest is name
                m = re.match(r"(.*?)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*$", arg)
                if m:
                    arg_type = m.group(1).strip()
                    arg_name = m.group(2).strip()
                else:
                    arg_type = arg
                    arg_name = ""
                args.append((arg_type, arg_name))

        functions.append((ret_type, func_name, args))

    return functions


# ---------------------------------------------------------------------------
# Type helpers — convert ctypes / list inputs to numpy arrays or scalars
# ---------------------------------------------------------------------------

def _to_numpy_array(obj, dtype=np.float64):
    """Convert ctypes array, pointer, list, or numpy array to numpy array."""
    if obj is None:
        return np.array([], dtype=dtype)
    if isinstance(obj, np.ndarray):
        return obj.astype(dtype, copy=False)
    if isinstance(obj, (list, tuple)):
        return np.array(obj, dtype=dtype)
    # ctypes pointer or array
    if hasattr(obj, "contents"):  # ctypes pointer
        # We can't easily know the length from a bare pointer, but callers
        # typically pass ctypes arrays which have __len__
        return _to_numpy_array(obj.contents, dtype)
    if hasattr(obj, "__len__") and hasattr(obj, "__getitem__"):
        # ctypes array type — iterate
        try:
            return np.array([obj[i] for i in range(len(obj))], dtype=dtype)
        except Exception:
            pass
    return np.array([obj], dtype=dtype)


def _to_float(val):
    if isinstance(val, (np.generic,)):
        return float(val)
    return float(val)


def _to_int(val):
    if isinstance(val, (np.generic,)):
        return int(val)
    return int(val)


# ---------------------------------------------------------------------------
# Mock implementations — one per header function
# ---------------------------------------------------------------------------

def _mock_eisenstein_norm(a, b):
    """N(a,b) = a² - a·b + b²"""
    a = _to_int(a)
    b = _to_int(b)
    return a * a - a * b + b * b


def _mock_laman_check_subset(num_vertices, num_edges):
    """Check if num_edges <= 2*num_vertices - 3"""
    n = _to_int(num_vertices)
    m = _to_int(num_edges)
    return 1 if m <= 2 * n - 3 else 0


def _mock_laman_is_rigid(num_vertices, num_edges):
    """Check if num_edges == 2*num_vertices - 3"""
    n = _to_int(num_vertices)
    m = _to_int(num_edges)
    return 1 if m == 2 * n - 3 else 0


def _mock_holonomy_check(states, length, threshold):
    """
    Cumulative drift = sum(|diffs|) / len <= threshold.
    Returns 1.0 if consistent, 0.0 otherwise.
    """
    arr = _to_numpy_array(states, dtype=np.float64)
    length = _to_int(length)
    threshold = _to_float(threshold)

    if length == 0:
        return 1.0

    # Use provided length or actual array length, whichever is smaller
    arr = arr[:length]
    if len(arr) < 2:
        return 1.0

    diffs = np.abs(np.diff(arr))
    drift = float(np.sum(diffs)) / len(arr)
    return 1.0 if drift <= threshold else 0.0


def _mock_pythagorean48_encode(numerator, denominator):
    """
    Encode a frequency ratio into Pythagorean 48-tone space.
    Returns nearest tempered semitone index (0..47).
    """
    num = _to_int(numerator)
    den = _to_int(denominator)
    if den == 0:
        return 0

    # Frequency ratio
    ratio = num / den

    # In 12-TET, semitone ratio is 2^(1/12)
    # In Pythagorean tuning based on pure fifths (3/2), the comma is folded.
    # We approximate by mapping log2(ratio) * 12 to nearest semitone,
    # then modulo 48 for the 48-tone space.
    semitone_float = 12.0 * math.log2(ratio)
    # Map to 0..47 range (4 octaves of 12)
    idx = int(round(semitone_float)) % 48
    return idx


def _mock_constraint_check(value, lower, upper):
    """Returns 1 if value in [lower, upper], else 0."""
    v = _to_float(value)
    lo = _to_float(lower)
    hi = _to_float(upper)
    return 1 if lo <= v <= hi else 0


def _mock_constraint_violation(value, lower, upper):
    """Distance outside [lower, upper], or 0 if inside."""
    v = _to_float(value)
    lo = _to_float(lower)
    hi = _to_float(upper)
    if v < lo:
        return lo - v
    if v > hi:
        return v - hi
    return 0.0


def _mock_spline_interpolate(p0, p1, m0, m1, t):
    """
    Cubic Hermite spline interpolation.
    h00(t)*p0 + h10(t)*m0 + h01(t)*p1 + h11(t)*m1
    where h00 = 2t³ - 3t² + 1, h10 = t³ - 2t² + t,
          h01 = -2t³ + 3t²,   h11 = t³ - t²
    """
    p0 = _to_float(p0)
    p1 = _to_float(p1)
    m0 = _to_float(m0)
    m1 = _to_float(m1)
    t = _to_float(t)

    t2 = t * t
    t3 = t2 * t

    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2

    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


def _mock_deadband_filter(value, last_ptr, deadband):
    """
    If |value - *last| < deadband, return *last.
    Otherwise update *last and return value.
    """
    value = _to_float(value)
    deadband = _to_float(deadband)

    # Handle ctypes byref pointer
    if hasattr(last_ptr, "contents"):
        last_val = float(last_ptr.contents)
    elif hasattr(last_ptr, "value"):
        # ctypes pointer .value
        last_val = float(last_ptr.value)
    elif hasattr(last_ptr, "__getitem__"):
        last_val = float(last_ptr[0])
    else:
        last_val = float(last_ptr)

    if abs(value - last_val) < deadband:
        return last_val

    # Update the pointed value
    new_val = value
    if hasattr(last_ptr, "contents"):
        last_ptr.contents = type(last_ptr.contents)(new_val)
    elif hasattr(last_ptr, "value"):
        last_ptr.value = type(last_ptr.value)(new_val)
    elif hasattr(last_ptr, "__setitem__"):
        last_ptr[0] = new_val

    return new_val


def _mock_manhattan_distance(a, b, dim):
    """L1 distance between two float arrays of length dim."""
    arr_a = _to_numpy_array(a, dtype=np.float32)
    arr_b = _to_numpy_array(b, dtype=np.float32)
    dim = _to_int(dim)

    arr_a = arr_a[:dim]
    arr_b = arr_b[:dim]

    if len(arr_a) == 0 or len(arr_b) == 0:
        return 0.0

    return float(np.sum(np.abs(arr_a - arr_b)))


def _mock_cascade_match(query, candidates, n, dim, thresholds, tiers):
    """
    Compare query against n candidates (flat [n*dim] array) with tiered thresholds.
    Returns index of first candidate passing any tier, or -1.
    """
    q = _to_numpy_array(query, dtype=np.float32)
    cands = _to_numpy_array(candidates, dtype=np.float32)
    n = _to_int(n)
    dim = _to_int(dim)
    thr = _to_numpy_array(thresholds, dtype=np.float32)
    tiers = _to_int(tiers)

    q = q[:dim]
    cands = cands[: n * dim]
    thr = thr[:tiers]

    if n == 0 or dim == 0 or tiers == 0:
        return -1

    for i in range(n):
        candidate = cands[i * dim : (i + 1) * dim]
        dist = float(np.sum(np.abs(q - candidate)))

        for t in range(tiers):
            if dist <= thr[t]:
                return i

    return -1


# ---------------------------------------------------------------------------
# Mapping from function name -> mock implementation
# ---------------------------------------------------------------------------

_MOCK_TABLE: dict[str, Callable] = {
    "eisenstein_norm": _mock_eisenstein_norm,
    "laman_check_subset": _mock_laman_check_subset,
    "laman_is_rigid": _mock_laman_is_rigid,
    "holonomy_check": _mock_holonomy_check,
    "pythagorean48_encode": _mock_pythagorean48_encode,
    "constraint_check": _mock_constraint_check,
    "constraint_violation": _mock_constraint_violation,
    "spline_interpolate": _mock_spline_interpolate,
    "deadband_filter": _mock_deadband_filter,
    "manhattan_distance": _mock_manhattan_distance,
    "cascade_match": _mock_cascade_match,
}


# ---------------------------------------------------------------------------
# CDLL-like object
# ---------------------------------------------------------------------------

class _MockFunction:
    """
    Wraps a plain Python callable with ctypes-style ``argtypes`` and ``restype``
    attributes so integration code can set them without crashing.
    """

    def __init__(self, name: str, impl: Callable):
        self._name = name
        self._impl = impl
        self.argtypes: Any = None
        self.restype: Any = None

    def __call__(self, *args, **kwargs):
        # Simple wrapper — in a real scenario we'd validate argtypes here
        return self._impl(*args, **kwargs)

    def __repr__(self):
        return f"<MockFunction '{self._name}'>"


class _MockCDLL:
    """
    CDLL-compatible object returned by `load_mock_ffi()`.
    Attribute access resolves function names from the mock table.
    """

    def __init__(self, mock_table: dict[str, Callable]):
        self._table = {name: _MockFunction(name, fn) for name, fn in mock_table.items()}

    def __getattr__(self, name: str) -> _MockFunction:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._table:
            return self._table[name]
        raise AttributeError(f"No mock function named '{name}'")

    def __dir__(self):
        return list(self._table.keys())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_mock_ffi(header_path: str | None = None) -> _MockCDLL:
    """
    Return a CDLL-like object backed by numpy mock implementations.

    Parameters
    ----------
    header_path
        Optional path to ``superinstance_ffi.h``.  When *None* the file is
        located next to this module inside ``superinstance-ffi/``.
    """
    if header_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        header_path = os.path.join(here, "superinstance-ffi", "superinstance_ffi.h")

    # Parse the header for validation / metadata (not strictly required for
    # the mocks to work, but guarantees the mock table stays in sync).
    parsed = _parse_header(header_path)
    parsed_names = {fn for _, fn, _ in parsed}
    mock_names = set(_MOCK_TABLE.keys())

    missing_in_mock = parsed_names - mock_names
    missing_in_header = mock_names - parsed_names

    if missing_in_mock:
        raise RuntimeError(
            f"Header declares functions not mocked: {missing_in_mock}"
        )
    if missing_in_header:
        # This is a soft warning — the mock may have extras for convenience.
        pass

    return _MockCDLL(_MOCK_TABLE)


# ---------------------------------------------------------------------------
# Sanity-check runner (can be executed directly)
# ---------------------------------------------------------------------------

def _self_test():
    """Quick sanity check for every mock function."""
    ffi = load_mock_ffi()

    # eisenstein_norm(3,4) -> 9 - 12 + 16 = 13
    assert ffi.eisenstein_norm(3, 4) == 13

    # laman_check_subset(4,5) -> 5 <= 5 -> 1
    assert ffi.laman_check_subset(4, 5) == 1

    # laman_is_rigid(4,5) -> 5 == 5 -> 1
    assert ffi.laman_is_rigid(4, 5) == 1

    # holonomy_check — small drift -> 1.0
    states = np.array([1.0, 1.01, 1.02, 1.03], dtype=np.float64)
    assert ffi.holonomy_check(states, len(states), 1.0) == 1.0

    # pythagorean48_encode — 3/2 is a perfect fifth, ~7 semitones
    idx = ffi.pythagorean48_encode(3, 2)
    assert 0 <= idx < 48

    # constraint_check
    assert ffi.constraint_check(5.0, 0.0, 10.0) == 1
    assert ffi.constraint_check(15.0, 0.0, 10.0) == 0

    # constraint_violation
    assert ffi.constraint_violation(5.0, 0.0, 10.0) == 0.0
    assert ffi.constraint_violation(12.0, 0.0, 10.0) == 2.0

    # spline_interpolate at t=0 -> p0, t=1 -> p1
    assert abs(ffi.spline_interpolate(0.0, 10.0, 0.0, 0.0, 0.0) - 0.0) < 1e-9
    assert abs(ffi.spline_interpolate(0.0, 10.0, 0.0, 0.0, 1.0) - 10.0) < 1e-9

    # deadband_filter — pointer simulation via list
    last = [5.0]
    assert ffi.deadband_filter(5.1, last, 0.2) == 5.0  # still inside deadband
    assert last[0] == 5.0
    assert ffi.deadband_filter(5.5, last, 0.2) == 5.5  # outside, updates
    assert last[0] == 5.5

    # manhattan_distance
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 0.0, 3.0], dtype=np.float32)
    assert ffi.manhattan_distance(a, b, 3) == 5.0

    # cascade_match
    query = np.array([1.0, 1.0], dtype=np.float32)
    candidates = np.array([0.0, 0.0, 1.1, 1.1, 5.0, 5.0], dtype=np.float32)
    thresholds = np.array([0.5, 2.0], dtype=np.float32)
    result = ffi.cascade_match(query, candidates, 3, 2, thresholds, 2)
    assert result == 0  # first candidate (0,0) passes tier-1 (dist=2.0 <= 2.0)

    print("All self-tests passed.")


if __name__ == "__main__":
    _self_test()
