// reasoning/cpp/reasoner.cpp — C++ Reasoner for Plato
//
// GPU-accelerated tile reasoning using OpenMP.
// Exposes C API for FFI.
//
// Build: g++ -O3 -fopenmp -shared -fPIC -o libplato_cpp.so reasoner.cpp
// Test: g++ -O3 -fopenmp -o test_reasoner reasoner.cpp && ./test_reasoner

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>

extern "C" {

/// Compute cosine similarity between two float vectors.
float plato_cosine_similarity(const float* a, const float* b, size_t len) {
    if (!a || !b || len == 0) return 0.0f;
    
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    #pragma omp simd reduction(+:dot, norm_a, norm_b)
    for (size_t i = 0; i < len; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    float norm = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (norm == 0.0f) return 0.0f;
    return dot / norm;
}

/// Batch compute similarities.
/// Returns 0 on success, -1 on error.
int plato_batch_similarity(
    const float* query,
    const float* embeddings,
    size_t dim,
    size_t n,
    size_t top_k,
    size_t* indices,
    float* scores
) {
    if (!query || !embeddings || !indices || !scores) return -1;
    
    std::vector<std::pair<size_t, float>> results;
    results.reserve(n);
    
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        float score = plato_cosine_similarity(query, embeddings + i * dim, dim);
        #pragma omp critical
        results.emplace_back(i, score);
    }
    
    // Sort descending by score
    std::sort(results.begin(), results.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    
    size_t k = std::min(top_k, n);
    for (size_t i = 0; i < k; ++i) {
        indices[i] = results[i].first;
        scores[i] = results[i].second;
    }
    
    return 0;
}

/// Aggregate embeddings (mean pooling).
void plato_mean_pool(
    const float* embeddings,
    size_t dim,
    size_t n,
    float* output
) {
    if (!embeddings || !output || n == 0) return;
    
    std::memset(output, 0, dim * sizeof(float));
    
    for (size_t i = 0; i < n; ++i) {
        const float* emb = embeddings + i * dim;
        #pragma omp simd
        for (size_t j = 0; j < dim; ++j) {
            output[j] += emb[j];
        }
    }
    
    #pragma omp simd
    for (size_t j = 0; j < dim; ++j) {
        output[j] /= static_cast<float>(n);
    }
}

/// Normalize a vector to unit length.
void plato_normalize(float* vec, size_t len) {
    if (!vec || len == 0) return;
    
    float norm = 0.0f;
    #pragma omp simd reduction(+:norm)
    for (size_t i = 0; i < len; ++i) {
        norm += vec[i] * vec[i];
    }
    
    norm = std::sqrt(norm);
    if (norm == 0.0f) return;
    
    #pragma omp simd
    for (size_t i = 0; i < len; ++i) {
        vec[i] /= norm;
    }
}

} // extern "C"

// Simple test main
#ifdef PLATO_TEST_MAIN
#include <cstdio>
#include <cassert>

int main() {
    // Test cosine similarity
    float a[] = {1.0f, 0.0f, 0.0f};
    float b[] = {1.0f, 0.0f, 0.0f};
    float score = plato_cosine_similarity(a, b, 3);
    printf("Identical: %.6f (expected 1.0)\n", score);
    assert(std::abs(score - 1.0f) < 1e-6);
    
    // Test orthogonal
    float c[] = {0.0f, 1.0f, 0.0f};
    score = plato_cosine_similarity(a, c, 3);
    printf("Orthogonal: %.6f (expected 0.0)\n", score);
    assert(std::abs(score) < 1e-6);
    
    // Test batch
    float query[] = {1.0f, 0.0f, 0.0f};
    float embeddings[] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.5f, 0.5f, 0.0f
    };
    size_t indices[2];
    float scores[2];
    
    int result = plato_batch_similarity(query, embeddings, 3, 3, 2, indices, scores);
    assert(result == 0);
    assert(indices[0] == 0);
    assert(scores[0] > 0.99f);
    
    printf("All C++ tests passed!\n");
    return 0;
}
#endif
