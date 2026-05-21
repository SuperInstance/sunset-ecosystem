#![allow(clippy::needless_range_loop)]
use std::thread;

/// Forward ONE room: 64×32 → ReLU → 32×16 → ReLU → 16×16 → latent(16)
/// Row-outer loop for contiguous B access (cache-friendly).
#[inline(always)]
fn forward_room(x: &[f32; 64], w1: &[f32], w2: &[f32], w3: &[f32],
                b1: &[f32], b2: &[f32], b3: &[f32], ri: usize, out: &mut [f32]) {
    let o1 = ri * 64 * 32; let o2 = ri * 32 * 16; let o3 = ri * 16 * 16;
    let ob1 = ri * 32; let ob2 = ri * 16; let ob3 = ri * 16;

    let mut h32 = [0.0f32; 32];
    for i in 0..32 { h32[i] = b1[ob1 + i]; }
    for row in 0..64 {
        let xr = x[row];
        if xr == 0.0 { continue; }
        for col in 0..32 { h32[col] += xr * w1[o1 + row * 32 + col]; }
    }
    for h in h32.iter_mut() { *h = h.max(0.0); }

    let mut h16 = [0.0f32; 16];
    for i in 0..16 { h16[i] = b2[ob2 + i]; }
    for row in 0..32 {
        let hr = h32[row];
        if hr == 0.0 { continue; }
        for col in 0..16 { h16[col] += hr * w2[o2 + row * 16 + col]; }
    }
    for h in h16.iter_mut() { *h = h.max(0.0); }

    for i in 0..16 { out[i] = b3[ob3 + i]; }
    for row in 0..16 {
        let hr = h16[row];
        if hr == 0.0 { continue; }
        for col in 0..16 { out[col] += hr * w3[o3 + row * 16 + col]; }
    }
}

/// Batch forward: ALL rooms, multi-threaded.
/// Splits rooms across available cores. Ryzen AI 9 has 12.
#[no_mangle]
pub extern "C" fn jepa_forward_batch(
    x_ptr: *const f32, w1: *const f32, w2: *const f32, w3: *const f32,
    b1: *const f32, b2: *const f32, b3: *const f32, n: usize, out_ptr: *mut f32,
) {
    let x: &[f32; 64] = unsafe { &*(x_ptr as *const [f32; 64]) };
    let w1_s = unsafe { std::slice::from_raw_parts(w1, n * 64 * 32) };
    let w2_s = unsafe { std::slice::from_raw_parts(w2, n * 32 * 16) };
    let w3_s = unsafe { std::slice::from_raw_parts(w3, n * 16 * 16) };
    let b1_s = unsafe { std::slice::from_raw_parts(b1, n * 32) };
    let b2_s = unsafe { std::slice::from_raw_parts(b2, n * 16) };
    let b3_s = unsafe { std::slice::from_raw_parts(b3, n * 16) };
    let out_s = unsafe { std::slice::from_raw_parts_mut(out_ptr, n * 16) };

    let nt = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4).min(12);
    if n < 100 || nt <= 1 {
        for ri in 0..n { forward_room(x, w1_s, w2_s, w3_s, b1_s, b2_s, b3_s, ri, &mut out_s[ri*16..(ri+1)*16]); }
        return;
    }

    let chunk = (n + nt - 1) / nt;
    let xsig = *x;
    std::thread::scope(|s| {
        for t in 0..nt {
            let s0 = t * chunk;
            let s1 = (s0 + chunk).min(n);
            if s0 >= s1 { break; }
            let out_sub = unsafe { std::slice::from_raw_parts_mut(out_ptr.add(s0 * 16), (s1 - s0) * 16) };
            let w1_c = &w1_s[s0 * 64 * 32..s1 * 64 * 32];
            let w2_c = &w2_s[s0 * 32 * 16..s1 * 32 * 16];
            let w3_c = &w3_s[s0 * 16 * 16..s1 * 16 * 16];
            let b1_c = &b1_s[s0 * 32..s1 * 32];
            let b2_c = &b2_s[s0 * 16..s1 * 16];
            let b3_c = &b3_s[s0 * 16..s1 * 16];
            let n_sub = s1 - s0;
            s.spawn(move || {
                for ri in 0..n_sub { forward_room(&xsig, w1_c, w2_c, w3_c, b1_c, b2_c, b3_c, ri, &mut out_sub[ri*16..(ri+1)*16]); }
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
        fn new(s: u64) -> Self { Self(s) }
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            (self.0 >> 40) as f32 * 0.01 / 16777216.0
        }
    }

    fn make(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut r = LCG::new(42);
        let mut g = |s| { let mut v = vec![0.0f32; s]; for x in v.iter_mut() { *x = r.next(); } v };
        (g(n*64*32), g(n*32*16), g(n*16*16), g(n*32), g(n*16), g(n*16))
    }

    #[test]
    fn test_correctness() {
        let (w1,w2,w3,b1,b2,b3) = make(10);
        let x = [1.0f32; 64];
        let mut out = vec![0.0f32; 160];
        jepa_forward_batch(x.as_ptr(), w1.as_ptr(), w2.as_ptr(), w3.as_ptr(),
                          b1.as_ptr(), b2.as_ptr(), b3.as_ptr(), 10, out.as_mut_ptr());
        assert!(out[0..16].iter().zip(out[16..32].iter()).map(|(a,b)| (a-b).abs()).sum::<f32>() > 1e-8);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn bench() {
        let (w1,w2,w3,b1,b2,b3) = make(10000);
        let x = [1.0f32; 64];
        let mut out = vec![0.0f32; 10000 * 16];
        jepa_forward_batch(x.as_ptr(), w1.as_ptr(), w2.as_ptr(), w3.as_ptr(),
                          b1.as_ptr(), b2.as_ptr(), b3.as_ptr(), 10000, out.as_mut_ptr());
        let start = Instant::now();
        for _ in 0..50 {
            jepa_forward_batch(x.as_ptr(), w1.as_ptr(), w2.as_ptr(), w3.as_ptr(),
                              b1.as_ptr(), b2.as_ptr(), b3.as_ptr(), 10000, out.as_mut_ptr());
        }
        let avg = start.elapsed() / 50;
        println!("10K rooms × 50 passes: {:?} avg ({:.0}ns/room)", avg, avg.as_nanos() as f64 / 10000.0);
    }
}
