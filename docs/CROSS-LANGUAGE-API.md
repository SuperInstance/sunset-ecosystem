# Cross-Language API Reference

Unified reference showing the same core concepts expressed in Python, Rust, C, and CUDA where applicable.

---

## 1. Eisenstein Norm

The norm in the Eisenstein integer ring ℤ[ω] is N(a,b) = a² - a·b + b² (or equivalently a² + a·b + b² depending on ω convention). It measures distance on the hexagonal A₂ lattice.

| Language | Function | Location | Key Difference |
|----------|----------|----------|---------------|
| Python   | `norm_sq(a, b)` | `constraint_theory.eisenstein` | Returns `int`; uses `a*a + a*b + b*b` |
| Rust     | `E12::norm(self)` | `eisenstein/src/lib.rs` | Returns `u64`; uses `a*a - a*b + b*b` |
| C        | `eisenstein_norm(a, b)` | `superinstance_ffi.h` | `int` return; same formula as Rust |

### Python
```python
# constraint_theory/eisenstein.py
def norm_sq(a: int, b: int) -> int:
    """Squared A₂ norm: a² + ab + b²."""
    return a * a + a * b + b * b
```

### Rust
```rust
// eisenstein/src/lib.rs
impl E12 {
    pub fn norm(self) -> u64 {
        let a = self.a as i64;
        let b = self.b as i64;
        let n = a * a - a * b + b * b;
        n as u64
    }
}
```

### C
```c
// superinstance_ffi.h
/**
 * Norm in the Eisenstein integers: N(a,b) = a² - a·b + b².
 */
int eisenstein_norm(int a, int b);
```

---

## 2. Constraint Satisfaction

Check whether a scalar value lies within a closed interval [lower, upper].

| Language | Function | Location | Key Difference |
|----------|----------|----------|---------------|
| Python   | `ConstraintArtifact.violated` | `superinstance/plugins/constraint.py` | Dataclass with automatic violation flag |
| Rust     | `RustConstraint::check` | `constraint-theory-rust-python/src/constraint.rs` | INT8 saturation semantics |
| C        | `flux_check` | `flux-engine-c/flux_engine.h` | Batch-capable, uint8 error mask |

### Python
```python
# superinstance/plugins/constraint.py
@dataclass(frozen=True, slots=True)
class ConstraintArtifact:
    field: str
    value: float
    lower_bound: float
    upper_bound: float
    violated: bool


# Violation is computed at collection time:
violated = not (lo <= value <= hi)
```

### Rust
```rust
// constraint-theory-rust-python/src/constraint.rs
impl RustConstraint {
    pub fn check(&self, value: i32) -> RustCheckResult {
        let pass = value >= self.lo && value <= self.hi;
        RustCheckResult { passed: pass, ... }
    }
}
```

### C
```c
// flux-engine-c/flux_engine.h
typedef struct {
    float lo;
    float hi;
    const char *name;
} FluxConstraint;

uint8_t flux_check(float value, const FluxConstraint *constraints, int n);
```

---

## 3. Fleet Consensus

Zero-holonomy consensus verifies that a cycle of fleet nodes sums to identity (no drift). If Hol(γ) = I the cycle is globally consistent.

| Language | Function | Location | Key Difference |
|----------|----------|----------|---------------|
| Python   | `HolonomyBridge.verify_cycle` | `nexus/holonomy_bridge.py` | Delegates to `HolonomyConsensus` |
| Rust     | `HolonomyConsensus::check_consensus` | `holonomy-consensus/src/consensus.rs` | O(N) per cycle, O(log N) fault isolation |

### Python
```python
# nexus/holonomy_bridge.py
class HolonomyBridge:
    def verify_cycle(self, cycle: list[str]) -> CycleReport:
        """Delegate cycle verification to holonomy-consensus."""
        return self._consensus.verify_cycle(cycle)
```

### Rust
```rust
// holonomy-consensus/src/consensus.rs
impl HolonomyConsensus {
    pub fn check_consensus(&self) -> ConsensusResult {
        let cycles = self.find_all_cycles();
        let mut max_deviation = 0.0f64;
        for cycle in cycles {
            let holonomy = self.compute_cycle_holonomy(&cycle);
            let deviation = holonomy.deviation();
            if deviation > max_deviation {
                max_deviation = deviation;
            }
        }
        ConsensusResult { max_deviation }
    }
}
```

---

## 4. Signal Deadband

Suppress signal changes that are smaller than a threshold to avoid noise jitter.

| Language | Function | Location | Key Difference |
|----------|----------|----------|---------------|
| Python   | `deadband_filter` | `sunset/plato_bridge.py` (inline) | Pure Python, pointer-free |
| Rust     | `deadband_filter` | `superinstance_ffi.h` / `src/lib.rs` | Updates `*last` in-place via pointer |
| Zig      | `DeadbandFilter.next` | `deadband-zig/src/main.zig` | Struct-based with suppression counter |

### Python
```python
def deadband_filter(value: float, last: float, deadband: float) -> float:
    if abs(value - last) < deadband:
        return last
    return value
```

### Rust (C ABI)
```rust
// superinstance-ffi/src/lib.rs
#[no_mangle]
pub extern "C" fn deadband_filter(value: f64, last: *mut f64, deadband: f64) -> f64 {
    unsafe {
        if (value - *last).abs() < deadband {
            *last
        } else {
            *last = value;
            value
        }
    }
}
```

### Zig
```zig
// deadband-zig/src/main.zig
pub const DeadbandFilter = struct {
    threshold: f64,
    baseline: f64,
    last_output: f64,

    pub fn next(self: *DeadbandFilter, value: f64) f64 {
        if (@abs(value - self.baseline) < self.threshold) {
            return self.last_output;
        }
        self.last_output = value;
        self.baseline = value;
        return value;
    }
};
```

---

## 5. Spline Interpolation

Cubic spline interpolation between two control points with tangent control.

| Language | Function | Location | Key Difference |
|----------|----------|----------|---------------|
| Python   | `cubic_interpolate` | Inline (NumPy) | Vectorised over arrays |
| Rust     | `spline_interpolate` | `superinstance_ffi.h` / `src/lib.rs` | Scalar double, C ABI |

### Python
```python
import numpy as np


def cubic_interpolate(p0, p1, m0, m1, t):
    """Cubic Hermite interpolation."""
    t2 = t * t
    t3 = t2 * t
    return (
        (2 * t3 - 3 * t2 + 1) * p0
        + (t3 - 2 * t2 + t) * m0
        + (-2 * t3 + 3 * t2) * p1
        + (t3 - t2) * m1
    )
```

### Rust (C ABI)
```rust
// superinstance-ffi/src/lib.rs
#[no_mangle]
pub extern "C" fn spline_interpolate(p0: f64, p1: f64, m0: f64, m1: f64, t: f64) -> f64 {
    let t2 = t * t;
    let t3 = t2 * t;
    (2.0*t3 - 3.0*t2 + 1.0)*p0 +
    (t3 - 2.0*t2 + t)*m0 +
    (-2.0*t3 + 3.0*t2)*p1 +
    (t3 - t2)*m1
}
```

### C
```c
// superinstance_ffi.h
/**
 * Cubic spline interpolation between two points with tangent control.
 *
 * *t* in [0,1] blends from p0 to p1 using tangents m0 and m1.
 */
double spline_interpolate(double p0, double p1, double m0, double m1, double t);
```
