/* Mock CUDA library for testing
 * Compile with: gcc -shared -fPIC -o nerve/libjepa_cuda.so tests/mock_jepa.c
 */

#include <string.h>

void jepa_cuda_tick(float* signal, float* w1, float* w2, float* w3,
                    float* b1, float* b2, float* b3, float* out, int n) {
    /* Minimal stub: zero the output */
    memset(out, 0, n * 16 * sizeof(float));
}

void jepa_cuda_tick_batch(float* signals, float* w1, float* w2, float* w3,
                          float* b1, float* b2, float* b3, float* out,
                          int n, int batch) {
    memset(out, 0, batch * n * 16 * sizeof(float));
}
