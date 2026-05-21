#![allow(clippy::needless_range_loop)]
use std::thread;

// ============================================================================
// Core forward pass (unchanged)
// ============================================================================

#[inline(always)]
fn forward_room(
    x: &[f32; 64],
    w1: &[f32],
    w2: &[f32],
    w3: &[f32],
    b1: &[f32],
    b2: &[f32],
    b3: &[f32],
    ri: usize,
    out: &mut [f32],
) {
    let o1 = ri * 64 * 32;
    let o2 = ri * 32 * 16;
    let o3 = ri * 16 * 16;
    let ob1 = ri * 32;
    let ob2 = ri * 16;
    let ob3 = ri * 16;

    let mut h32 = [0.0f32; 32];
    for i in 0..32 {
        h32[i] = b1[ob1 + i];
    }
    for row in 0..64 {
        let xr = x[row];
        if xr == 0.0 {
            continue;
        }
        for col in 0..32 {
            h32[col] += xr * w1[o1 + row * 32 + col];
        }
    }
    for h in h32.iter_mut() {
        *h = h.max(0.0);
    }

    let mut h16 = [0.0f32; 16];
    for i in 0..16 {
        h16[i] = b2[ob2 + i];
    }
    for row in 0..32 {
        let hr = h32[row];
        if hr == 0.0 {
            continue;
        }
        for col in 0..16 {
            h16[col] += hr * w2[o2 + row * 16 + col];
        }
    }
    for h in h16.iter_mut() {
        *h = h.max(0.0);
    }

    for i in 0..16 {
        out[i] = b3[ob3 + i];
    }
    for row in 0..16 {
        let hr = h16[row];
        if hr == 0.0 {
            continue;
        }
        for col in 0..16 {
            out[col] += hr * w3[o3 + row * 16 + col];
        }
    }
}

// ============================================================================
// Persistent Grid — weights live in Rust, Python only sends signals
// ============================================================================

/// Opaque handle to a pre-loaded grid.
pub struct JEPAGrid {
    n: usize,
    w1: Vec<f32>,
    w2: Vec<f32>,
    w3: Vec<f32>,
    b1: Vec<f32>,
    b2: Vec<f32>,
    b3: Vec<f32>,
    nt: usize,
}

impl JEPAGrid {
    fn new(
        n: usize,
        w1: Vec<f32>,
        w2: Vec<f32>,
        w3: Vec<f32>,
        b1: Vec<f32>,
        b2: Vec<f32>,
        b3: Vec<f32>,
    ) -> Self {
        let nt = thread::available_parallelism()
            .map(|x| x.get())
            .unwrap_or(4)
            .min(12);
        Self { n, w1, w2, w3, b1, b2, b3, nt }
    }

    fn tick(&self, x: &[f32; 64], out: &mut [f32]) {
        if self.n < 100 || self.nt <= 1 {
            for ri in 0..self.n {
                forward_room(
                    x,
                    &self.w1,
                    &self.w2,
                    &self.w3,
                    &self.b1,
                    &self.b2,
                    &self.b3,
                    ri,
                    &mut out[ri * 16..(ri + 1) * 16],
                );
            }
            return;
        }

        let chunk = (self.n + self.nt - 1) / self.nt;
        let xsig = *x;
        let out_ptr = out.as_mut_ptr();
        let n = self.n;
        thread::scope(|s| {
            for t in 0..self.nt {
                let s0 = t * chunk;
                let s1 = (s0 + chunk).min(n);
                if s0 >= s1 { break; }
                let n_sub = s1 - s0;
                let w1_c = &self.w1[s0 * 64 * 32..s1 * 64 * 32];
                let w2_c = &self.w2[s0 * 32 * 16..s1 * 32 * 16];
                let w3_c = &self.w3[s0 * 16 * 16..s1 * 16 * 16];
                let b1_c = &self.b1[s0 * 32..s1 * 32];
                let b2_c = &self.b2[s0 * 16..s1 * 16];
                let b3_c = &self.b3[s0 * 16..s1 * 16];
                // Safe: sub-slices are non-overlapping
                let out_sub = unsafe {
                    std::slice::from_raw_parts_mut(out_ptr.add(s0 * 16), n_sub * 16)
                };
                s.spawn(move || {
                    for ri in 0..n_sub {
                        forward_room(
                            &xsig,
                            w1_c,
                            w2_c,
                            w3_c,
                            b1_c,
                            b2_c,
                            b3_c,
                            ri,
                            &mut out_sub[ri * 16..(ri + 1) * 16],
                        );
                    }
                });
            }
        });
    }

    /// Batched tick: process `batch` signals at once.
    /// Signals: flat [batch * 64]. Outputs: flat [batch * n * 16].
    fn tick_batch(&self, signals: &[f32], batch: usize, out: &mut [f32]) {
        for b in 0..batch {
            let x = &signals[b * 64..b * 64 + 64];
            let x_arr: &[f32; 64] = unsafe { &*(x.as_ptr() as *const [f32; 64]) };
            let out_slice = &mut out[b * self.n * 16..(b + 1) * self.n * 16];
            self.tick(x_arr, out_slice);
        }
    }
}

// ============================================================================
// FFI — C-compatible interface
// ============================================================================

use std::ffi::c_void;

/// Create a persistent grid. Returns opaque pointer.
/// Python calls once at grid init. Weights copied into Rust-owned Vec.
#[no_mangle]
pub extern "C" fn jepa_grid_create(
    n: usize,
    w1: *const f32,
    w2: *const f32,
    w3: *const f32,
    b1: *const f32,
    b2: *const f32,
    b3: *const f32,
) -> *mut c_void {
    let w1_vec = unsafe { std::slice::from_raw_parts(w1, n * 64 * 32).to_vec() };
    let w2_vec = unsafe { std::slice::from_raw_parts(w2, n * 32 * 16).to_vec() };
    let w3_vec = unsafe { std::slice::from_raw_parts(w3, n * 16 * 16).to_vec() };
    let b1_vec = unsafe { std::slice::from_raw_parts(b1, n * 32).to_vec() };
    let b2_vec = unsafe { std::slice::from_raw_parts(b2, n * 16).to_vec() };
    let b3_vec = unsafe { std::slice::from_raw_parts(b3, n * 16).to_vec() };

    let grid = Box::new(JEPAGrid::new(n, w1_vec, w2_vec, w3_vec, b1_vec, b2_vec, b3_vec));
    Box::into_raw(grid) as *mut c_void
}

/// Tick ONE signal through a persistent grid.
/// Python calls per tick with just signal + pre-allocated output buffer.
#[no_mangle]
pub extern "C" fn jepa_grid_tick(
    handle: *mut c_void,
    signal: *const f32,
    out: *mut f32,
) {
    let grid = unsafe { &*(handle as *mut JEPAGrid) };
    let x = unsafe { &*(signal as *const [f32; 64]) };
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out, grid.n * 16) };
    grid.tick(x, out_slice);
}

/// Tick BATCH signals through a persistent grid.
/// Python calls once per N ticks. Amortizes FFI overhead.
#[no_mangle]
pub extern "C" fn jepa_grid_tick_batch(
    handle: *mut c_void,
    signals: *const f32,
    batch: usize,
    out: *mut f32,
) {
    let grid = unsafe { &*(handle as *mut JEPAGrid) };
    let sig_slice = unsafe { std::slice::from_raw_parts(signals, batch * 64) };
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out, batch * grid.n * 16) };
    grid.tick_batch(sig_slice, batch, out_slice);
}

/// Destroy a persistent grid. Python calls on cleanup.
#[no_mangle]
pub extern "C" fn jepa_grid_destroy(handle: *mut c_void) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle as *mut JEPAGrid);
        }
    }
}

// ============================================================================
// Legacy one-shot API (kept for backward compatibility)
// ============================================================================

#[no_mangle]
pub extern "C" fn jepa_forward_batch(
    x_ptr: *const f32,
    w1: *const f32,
    w2: *const f32,
    w3: *const f32,
    b1: *const f32,
    b2: *const f32,
    b3: *const f32,
    n: usize,
    out_ptr: *mut f32,
) {
    let x = unsafe { &*(x_ptr as *const [f32; 64]) };
    let w1_s = unsafe { std::slice::from_raw_parts(w1, n * 64 * 32) };
    let w2_s = unsafe { std::slice::from_raw_parts(w2, n * 32 * 16) };
    let w3_s = unsafe { std::slice::from_raw_parts(w3, n * 16 * 16) };
    let b1_s = unsafe { std::slice::from_raw_parts(b1, n * 32) };
    let b2_s = unsafe { std::slice::from_raw_parts(b2, n * 16) };
    let b3_s = unsafe { std::slice::from_raw_parts(b3, n * 16) };
    let out_s = unsafe { std::slice::from_raw_parts_mut(out_ptr, n * 16) };

    let nt = thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(4)
        .min(12);
    if n < 100 || nt <= 1 {
        for ri in 0..n {
            forward_room(
                x, w1_s, w2_s, w3_s, b1_s, b2_s, b3_s, ri,
                &mut out_s[ri * 16..(ri + 1) * 16],
            );
        }
        return;
    }

    let chunk = (n + nt - 1) / nt;
    let xsig = *x;
    thread::scope(|s| {
        for t in 0..nt {
            let s0 = t * chunk;
            let s1 = (s0 + chunk).min(n);
            if s0 >= s1 {
                break;
            }
            let out_sub = unsafe {
                std::slice::from_raw_parts_mut(out_ptr.add(s0 * 16), (s1 - s0) * 16)
            };
            let w1_c = &w1_s[s0 * 64 * 32..s1 * 64 * 32];
            let w2_c = &w2_s[s0 * 32 * 16..s1 * 32 * 16];
            let w3_c = &w3_s[s0 * 16 * 16..s1 * 16 * 16];
            let b1_c = &b1_s[s0 * 32..s1 * 32];
            let b2_c = &b2_s[s0 * 16..s1 * 16];
            let b3_c = &b3_s[s0 * 16..s1 * 16];
            let n_sub = s1 - s0;
            s.spawn(move || {
                for ri in 0..n_sub {
                    forward_room(
                        &xsig, w1_c, w2_c, w3_c, b1_c, b2_c, b3_c, ri,
                        &mut out_sub[ri * 16..(ri + 1) * 16],
                    );
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    struct LCG(u64);
    impl LCG {
        fn new(s: u64) -> Self {
            Self(s)
        }
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            (self.0 >> 40) as f32 * 0.01 / 16777216.0
        }
    }

    fn make(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut r = LCG::new(42);
        let mut g = |s| {
            let mut v = vec![0.0f32; s];
            for x in v.iter_mut() {
                *x = r.next();
            }
            v
        };
        (
            g(n * 64 * 32),
            g(n * 32 * 16),
            g(n * 16 * 16),
            g(n * 32),
            g(n * 16),
            g(n * 16),
        )
    }

    #[test]
    fn test_persistent_grid() {
        let (w1, w2, w3, b1, b2, b3) = make(100);
        let handle = jepa_grid_create(
            100,
            w1.as_ptr(),
            w2.as_ptr(),
            w3.as_ptr(),
            b1.as_ptr(),
            b2.as_ptr(),
            b3.as_ptr(),
        );

        let x = [1.0f32; 64];
        let mut out = vec![0.0f32; 100 * 16];

        // Single tick
        jepa_grid_tick(handle, x.as_ptr(), out.as_mut_ptr());
        assert!(out.iter().all(|v| v.is_finite()));

        // Batched tick
        let signals: Vec<f32> = (0..10).flat_map(|_| x.iter().copied()).collect();
        let mut batch_out = vec![0.0f32; 10 * 100 * 16];
        jepa_grid_tick_batch(handle, signals.as_ptr(), 10, batch_out.as_mut_ptr());
        assert!(batch_out.iter().all(|v| v.is_finite()));

        jepa_grid_destroy(handle);
    }

    #[test]
    fn bench_persistent_vs_oneshot() {
        let n = 10000;
        let (w1, w2, w3, b1, b2, b3) = make(n);
        let x = [1.0f32; 64];
        let mut out = vec![0.0f32; n * 16];

        // One-shot (legacy API)
        let start = Instant::now();
        for _ in 0..50 {
            jepa_forward_batch(
                x.as_ptr(), w1.as_ptr(), w2.as_ptr(), w3.as_ptr(),
                b1.as_ptr(), b2.as_ptr(), b3.as_ptr(), n, out.as_mut_ptr(),
            );
        }
        let oneshot = start.elapsed() / 50;

        // Persistent
        let handle = jepa_grid_create(
            n, w1.as_ptr(), w2.as_ptr(), w3.as_ptr(),
            b1.as_ptr(), b2.as_ptr(), b3.as_ptr(),
        );
        let start = Instant::now();
        for _ in 0..50 {
            jepa_grid_tick(handle, x.as_ptr(), out.as_mut_ptr());
        }
        let persistent = start.elapsed() / 50;

        // Batched
        let signals: Vec<f32> = (0..50).flat_map(|_| x.iter().copied()).collect();
        let mut batch_out = vec![0.0f32; 50 * n * 16];
        let start = Instant::now();
        jepa_grid_tick_batch(handle, signals.as_ptr(), 50, batch_out.as_mut_ptr());
        let batch = start.elapsed() / 50;

        println!("10K rooms — oneshot: {:?}, persistent: {:?}, batch/tick: {:?}",
                 oneshot, persistent, batch);

        jepa_grid_destroy(handle);
    }
}
