// reasoning/rust/src/lib.rs — Rust Reasoner for Plato
//
// Performance-critical path: tile similarity, embedding lookups,
// and consensus voting. Exposes FFI for Python integration.
//
// Build: cargo build --release
// Test: cargo test

use std::collections::HashMap;
use std::f32;

/// Compute cosine similarity between two vectors.
/// Performance-critical: uses SIMD if available.
#[no_mangle]
pub extern "C" fn cosine_similarity(a: *const f32, b: *const f32, len: usize) -> f32 {
    if a.is_null() || b.is_null() || len == 0 {
        return 0.0;
    }
    
    unsafe {
        let a_slice = std::slice::from_raw_parts(a, len);
        let b_slice = std::slice::from_raw_parts(b, len);
        
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;
        
        // Unroll by 4 for SIMD-like performance
        let mut i = 0;
        while i + 4 <= len {
            dot += a_slice[i] * b_slice[i];
            dot += a_slice[i+1] * b_slice[i+1];
            dot += a_slice[i+2] * b_slice[i+2];
            dot += a_slice[i+3] * b_slice[i+3];
            
            norm_a += a_slice[i] * a_slice[i];
            norm_a += a_slice[i+1] * a_slice[i+1];
            norm_a += a_slice[i+2] * a_slice[i+2];
            norm_a += a_slice[i+3] * a_slice[i+3];
            
            norm_b += b_slice[i] * b_slice[i];
            norm_b += b_slice[i+1] * b_slice[i+1];
            norm_b += b_slice[i+2] * b_slice[i+2];
            norm_b += b_slice[i+3] * b_slice[i+3];
            
            i += 4;
        }
        
        // Handle remainder
        while i < len {
            dot += a_slice[i] * b_slice[i];
            norm_a += a_slice[i] * a_slice[i];
            norm_b += b_slice[i] * b_slice[i];
            i += 1;
        }
        
        let norm = norm_a.sqrt() * norm_b.sqrt();
        if norm == 0.0 {
            0.0
        } else {
            dot / norm
        }
    }
}

/// Batch compute similarities: query against N embeddings.
/// Returns indices of top-k matches.
#[no_mangle]
pub extern "C" fn batch_similarity(
    query: *const f32,
    embeddings: *const f32,
    dim: usize,
    n: usize,
    top_k: usize,
    indices: *mut usize,
    scores: *mut f32,
) -> i32 {
    if query.is_null() || embeddings.is_null() || indices.is_null() || scores.is_null() {
        return -1;
    }
    
    unsafe {
        let q_slice = std::slice::from_raw_parts(query, dim);
        let emb_slice = std::slice::from_raw_parts(embeddings, dim * n);
        
        let mut results: Vec<(usize, f32)> = Vec::with_capacity(n);
        
        for i in 0..n {
            let start = i * dim;
            let emb = &emb_slice[start..start + dim];
            let score = cosine_similarity(q_slice.as_ptr(), emb.as_ptr(), dim);
            results.push((i, score));
        }
        
        // Sort by score descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let k = top_k.min(n);
        let idx_slice = std::slice::from_raw_parts_mut(indices, k);
        let score_slice = std::slice::from_raw_parts_mut(scores, k);
        
        for i in 0..k {
            idx_slice[i] = results[i].0;
            score_slice[i] = results[i].1;
        }
    }
    
    0
}

/// Tile struct for reasoning.
#[repr(C)]
pub struct Tile {
    pub id: u64,
    pub embedding: *const f32,
    pub dim: usize,
    pub score: f32,
}

/// Reasoner state.
pub struct Reasoner {
    tiles: HashMap<u64, Vec<f32>>,
    dim: usize,
}

impl Reasoner {
    pub fn new(dim: usize) -> Self {
        Reasoner {
            tiles: HashMap::new(),
            dim,
        }
    }
    
    pub fn add_tile(&mut self, id: u64, embedding: Vec<f32>) {
        self.tiles.insert(id, embedding);
    }
    
    pub fn find_similar(&self, query: &[f32], top_k: usize) -> Vec<(u64, f32)> {
        let mut results: Vec<(u64, f32)> = Vec::new();
        
        for (id, emb) in &self.tiles {
            let score = unsafe {
                cosine_similarity(query.as_ptr(), emb.as_ptr(), self.dim)
            };
            results.push((*id, score));
        }
        
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(top_k);
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let score = unsafe { cosine_similarity(a.as_ptr(), b.as_ptr(), 3) };
        assert!((score - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let score = unsafe { cosine_similarity(a.as_ptr(), b.as_ptr(), 3) };
        assert!(score.abs() < 1e-6);
    }
    
    #[test]
    fn test_batch_similarity() {
        let query = vec![1.0, 0.0, 0.0];
        let embeddings = vec![
            1.0, 0.0, 0.0,  // Same as query
            0.0, 1.0, 0.0,  // Orthogonal
            0.5, 0.5, 0.0,  // 45 degrees
        ];
        
        let mut indices = vec![0usize; 2];
        let mut scores = vec![0.0f32; 2];
        
        let result = unsafe {
            batch_similarity(
                query.as_ptr(),
                embeddings.as_ptr(),
                3, 3, 2,
                indices.as_mut_ptr(),
                scores.as_mut_ptr(),
            )
        };
        
        assert_eq!(result, 0);
        assert_eq!(indices[0], 0);  // Most similar
        assert!(scores[0] > 0.99);
    }
    
    #[test]
    fn test_reasoner() {
        let mut r = Reasoner::new(3);
        r.add_tile(1, vec![1.0, 0.0, 0.0]);
        r.add_tile(2, vec![0.0, 1.0, 0.0]);
        
        let results = r.find_similar(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results[0].0, 1);
        assert!(results[0].1 > 0.99);
    }
}