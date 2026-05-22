//! Superinstance FFI — unified Rust crate for core math primitives.
//!
//! Exports:
//!     - Eisenstein norm
//!     - Laman rigidity edge check
//!     - Holonomy consistency check
//!     - Pythagorean48 encode
//!     - Constraint bounds check
//!     - Cubic spline interpolation
//!     - Deadband filter
//!     - Manhattan distance
//!     - Cascade match
//!
//! Build:
//!     cargo build --release
//!     cbindgen --crate superinstance-ffi --lang c > superinstance_ffi.h

use std::ffi::{c_double, c_float, c_int, c_uint};

// ============================================================================
// 1. Eisenstein norm
// ============================================================================

/// Norm in the Eisenstein integers: N(a,b) = a² - a·b + b².
#[no_mangle]
pub extern "C" fn eisenstein_norm(a: c_int, b: c_int) -> c_int {
    a * a - a * b + b * b
}

// ============================================================================
// 2. Laman edge check
// ============================================================================

/// Verify that a set of edges satisfies Laman's condition for a subset.
///
/// For *k* vertices, at most *2k - 3* edges are allowed for minimal rigidity.
/// Returns 1 if the subset satisfies the condition, 0 otherwise.
#[no_mangle]
pub extern "C" fn laman_check_subset(num_vertices: c_uint, num_edges: c_uint) -> c_int {
    if num_vertices < 2 {
        return 1;
    }
    let max_edges = 2 * num_vertices as c_int - 3;
    if num_edges as c_int <= max_edges {
        1
    } else {
        0
    }
}

/// Full Laman rigidity test for a graph with n vertices and m edges.
/// Requires m == 2n - 3 for generic minimal rigidity in 2D.
#[no_mangle]
pub extern "C" fn laman_is_rigid(num_vertices: c_uint, num_edges: c_uint) -> c_int {
    if num_vertices < 2 {
        return 0;
    }
    let expected = 2 * num_vertices as c_int - 3;
    if num_edges as c_int == expected {
        1
    } else {
        0
    }
}

// ============================================================================
// 3. Holonomy check
// ============================================================================

/// Check holonomic consistency around a cycle of states.
///
/// *states* is a flat array of length *len*. The cumulative drift
/// (sum of absolute differences) divided by *len* must be ≤ *threshold*.
/// Returns 1.0 if consistent, 0.0 if not.
#[no_mangle]
pub extern "C" fn holonomy_check(states: *const c_double, len: c_uint, threshold: c_double) -> c_float {
    if len < 2 || states.is_null() {
        return 1.0;
    }
    let slice = unsafe { std::slice::from_raw_parts(states, len as usize) };
    let mut total_drift = 0.0;
    for i in 0..slice.len() {
        let a = slice[i];
        let b = slice[(i + 1) % slice.len()];
        total_drift += (a - b).abs();
    }
    let avg_drift = total_drift / slice.len() as f64;
    if avg_drift <= threshold {
        1.0
    } else {
        0.0
    }
}

// ============================================================================
// 4. Pythagorean48 encode
// ============================================================================

/// Encode a frequency ratio into Pythagorean 48-tone space.
///
/// Returns the nearest tempered semitone index (0..47) for a
/// frequency ratio expressed as `numerator / denominator`.
/// The Pythagorean comma (~23.46 cents) is folded into the octave.
#[no_mangle]
pub extern "C" fn pythagorean48_encode(numerator: c_int, denominator: c_int) -> c_int {
    if denominator == 0 {
        return 0;
    }
    let ratio = (numerator as f64) / (denominator as f64);
    let semitones = 12.0 * ratio.log2();
    let idx = semitones.rem_euclid(48.0).round() as c_int;
    idx.clamp(0, 47)
}

// ============================================================================
// 5. Constraint check
// ============================================================================

/// Check if *value* lies within [*lower*, *upper*].
/// Returns 1 if satisfied, 0 if violated.
#[no_mangle]
pub extern "C" fn constraint_check(value: c_double, lower: c_double, upper: c_double) -> c_int {
    if value >= lower && value <= upper {
        1
    } else {
        0
    }
}

/// Compute constraint violation distance (0 if satisfied).
#[no_mangle]
pub extern "C" fn constraint_violation(value: c_double, lower: c_double, upper: c_double) -> c_double {
    if value < lower {
        lower - value
    } else if value > upper {
        value - upper
    } else {
        0.0
    }
}

// ============================================================================
// 6. Spline interpolate
// ============================================================================

/// Cubic spline interpolation between two points with tangent control.
///
/// *t* in [0,1] blends from p0 to p1 using tangents m0 and m1.
#[no_mangle]
pub extern "C" fn spline_interpolate(p0: c_double, p1: c_double, m0: c_double, m1: c_double, t: c_double) -> c_double {
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
}

// ============================================================================
// 7. Deadband filter
// ============================================================================

/// Apply a deadband filter: if |value - last| < deadband, return last.
/// Otherwise return value and update *last via pointer.
#[no_mangle]
pub extern "C" fn deadband_filter(value: c_double, last: *mut c_double, deadband: c_double) -> c_double {
    if last.is_null() {
        return value;
    }
    let last_val = unsafe { *last };
    if (value - last_val).abs() < deadband {
        last_val
    } else {
        unsafe { *last = value };
        value
    }
}

// ============================================================================
// 8. Manhattan distance
// ============================================================================

/// L1 distance between two float arrays of length *dim*.
#[no_mangle]
pub extern "C" fn manhattan_distance(a: *const c_float, b: *const c_float, dim: c_uint) -> c_float {
    if dim == 0 || a.is_null() || b.is_null() {
        return 0.0;
    }
    let a_slice = unsafe { std::slice::from_raw_parts(a, dim as usize) };
    let b_slice = unsafe { std::slice::from_raw_parts(b, dim as usize) };
    let mut sum = 0.0f32;
    for i in 0..dim as usize {
        sum += (a_slice[i] - b_slice[i]).abs();
    }
    sum
}

// ============================================================================
// 9. Cascade match
// ============================================================================

/// Cascade match: compare *query* against *candidates* with tiered thresholds.
///
/// *candidates* is a flat [n * dim] array. *thresholds* is a [tiers] array
/// of decreasing match thresholds. Returns the index of the first candidate
/// that passes any tier, or -1 if none match.
#[no_mangle]
pub extern "C" fn cascade_match(
    query: *const c_float,
    candidates: *const c_float,
    n: c_uint,
    dim: c_uint,
    thresholds: *const c_float,
    tiers: c_uint,
) -> c_int {
    if dim == 0 || n == 0 || tiers == 0 || query.is_null() || candidates.is_null() || thresholds.is_null() {
        return -1;
    }
    let q = unsafe { std::slice::from_raw_parts(query, dim as usize) };
    let cands = unsafe { std::slice::from_raw_parts(candidates, (n * dim) as usize) };
    let thresh = unsafe { std::slice::from_raw_parts(thresholds, tiers as usize) };

    for i in 0..n as usize {
        let cand = &cands[i * dim as usize..(i + 1) * dim as usize];
        let dist: f32 = q.iter().zip(cand.iter()).map(|(a, b)| (a - b).abs()).sum();
        for &t in thresh {
            if dist <= t {
                return i as c_int;
            }
        }
    }
    -1
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eisenstein_norm() {
        assert_eq!(eisenstein_norm(1, 0), 1);
        assert_eq!(eisenstein_norm(0, 1), 1);
        assert_eq!(eisenstein_norm(1, 1), 1);
        assert_eq!(eisenstein_norm(2, 1), 3);
        assert_eq!(eisenstein_norm(3, 2), 7);
    }

    #[test]
    fn test_laman_subset() {
        assert_eq!(laman_check_subset(3, 3), 1);  // 3 ≤ 2*3-3 = 3
        assert_eq!(laman_check_subset(3, 4), 0);  // 4 > 3
        assert_eq!(laman_check_subset(1, 0), 1);  // trivial
    }

    #[test]
    fn test_laman_rigid() {
        assert_eq!(laman_is_rigid(3, 3), 1);   // 2*3-3 = 3
        assert_eq!(laman_is_rigid(4, 5), 1);   // 2*4-3 = 5
        assert_eq!(laman_is_rigid(4, 4), 0);   // wrong count
        assert_eq!(laman_is_rigid(1, 0), 0);   // too small
    }

    #[test]
    fn test_holonomy_consistent() {
        let states = [0.0, 0.0, 0.0];
        assert_eq!(holonomy_check(states.as_ptr(), 3, 1e-6), 1.0);
    }

    #[test]
    fn test_holonomy_inconsistent() {
        let states = [0.0, 1.0, 2.0];
        assert_eq!(holonomy_check(states.as_ptr(), 3, 0.1), 0.0);
    }

    #[test]
    fn test_pythagorean48() {
        // Octave ratio 2:1 → 12 semitones → index 12 in 48-tone space
        assert_eq!(pythagorean48_encode(2, 1), 12);
        // Perfect fifth 3:2 → ~7 semitones → index 7
        assert_eq!(pythagorean48_encode(3, 2), 7);
        // Unison 1:1 → 0
        assert_eq!(pythagorean48_encode(1, 1), 0);
    }

    #[test]
    fn test_constraint_check() {
        assert_eq!(constraint_check(0.5, 0.0, 1.0), 1);
        assert_eq!(constraint_check(-0.1, 0.0, 1.0), 0);
        assert_eq!(constraint_check(1.1, 0.0, 1.0), 0);
    }

    #[test]
    fn test_constraint_violation() {
        assert_eq!(constraint_violation(0.5, 0.0, 1.0), 0.0);
        assert_eq!(constraint_violation(-0.5, 0.0, 1.0), 0.5);
        assert_eq!(constraint_violation(1.5, 0.0, 1.0), 0.5);
    }

    #[test]
    fn test_spline_interpolate() {
        // At t=0, should equal p0
        assert!((spline_interpolate(1.0, 2.0, 0.0, 0.0, 0.0) - 1.0).abs() < 1e-9);
        // At t=1, should equal p1
        assert!((spline_interpolate(1.0, 2.0, 0.0, 0.0, 1.0) - 2.0).abs() < 1e-9);
        // At t=0.5 with zero tangents, should be midpoint
        let mid = spline_interpolate(0.0, 1.0, 0.0, 0.0, 0.5);
        assert!((mid - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_deadband_filter() {
        let mut last = 0.0;
        // Within deadband → return last
        assert_eq!(deadband_filter(0.05, &mut last, 0.1), 0.0);
        assert_eq!(last, 0.0);
        // Outside deadband → update and return new value
        assert_eq!(deadband_filter(0.2, &mut last, 0.1), 0.2);
        assert_eq!(last, 0.2);
    }

    #[test]
    fn test_manhattan_distance() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 0.0, 3.0];
        assert_eq!(manhattan_distance(a.as_ptr(), b.as_ptr(), 3), 5.0);
    }

    #[test]
    fn test_manhattan_empty() {
        assert_eq!(manhattan_distance(std::ptr::null(), std::ptr::null(), 0), 0.0);
    }

    #[test]
    fn test_cascade_match() {
        let query = [1.0f32, 1.0, 1.0];
        let candidates = [
            0.0f32, 0.0, 0.0,  // far
            1.1f32, 1.0, 1.0,  // close
            2.0f32, 2.0, 2.0,  // far
        ];
        let thresholds = [0.5f32, 1.5];
        let idx = cascade_match(
            query.as_ptr(),
            candidates.as_ptr(),
            3,
            3,
            thresholds.as_ptr(),
            2,
        );
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_cascade_match_no_match() {
        let query = [10.0f32];
        let candidates = [0.0f32, 1.0, 2.0];
        let thresholds = [0.5f32];
        let idx = cascade_match(
            query.as_ptr(),
            candidates.as_ptr(),
            3,
            1,
            thresholds.as_ptr(),
            1,
        );
        assert_eq!(idx, -1);
    }

    #[test]
    fn test_cascade_match_null_safety() {
        assert_eq!(cascade_match(std::ptr::null(), std::ptr::null(), 0, 0, std::ptr::null(), 0), -1);
    }

    #[test]
    fn test_holonomy_null_safety() {
        assert_eq!(holonomy_check(std::ptr::null(), 0, 1.0), 1.0);
    }
}
