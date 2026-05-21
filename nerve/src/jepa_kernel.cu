/** JEPA Forward CUDA Kernel
 *
 *  Each room = 64→32→16→16 MLP. No training, no backprop.
 *  One thread block per room. One warp per layer.
 *  Target: 10K rooms in <2ms on RTX 4050.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define D 64   // input dim
#define H 32   // hidden dim
#define L 16   // latent dim

// ReLU activation
__device__ inline float relu(float x) { return fmaxf(x, 0.0f); }

// Layer 1: (D,) @ (D, H) -> (H,)  + bias
// Each thread handles one output neuron
__device__ void layer1(const float* __restrict__ x,
                       const float* __restrict__ w,
                       const float* __restrict__ b,
                       float* __restrict__ out,
                       int tid) {
    float acc = b[tid];
    for (int i = 0; i < D; i += 4) {
        // Unrolled 4x for ILP
        acc += x[i]   * w[(i)   * H + tid];
        acc += x[i+1] * w[(i+1) * H + tid];
        acc += x[i+2] * w[(i+2) * H + tid];
        acc += x[i+3] * w[(i+3) * H + tid];
    }
    out[tid] = relu(acc);
}

// Layer 2: (H,) @ (H, L) -> (L,) + bias
__device__ void layer2(const float* __restrict__ h1,
                       const float* __restrict__ w,
                       const float* __restrict__ b,
                       float* __restrict__ out,
                       int tid) {
    float acc = b[tid];
    for (int i = 0; i < H; i += 4) {
        acc += h1[i]   * w[(i)   * L + tid];
        acc += h1[i+1] * w[(i+1) * L + tid];
        acc += h1[i+2] * w[(i+2) * L + tid];
        acc += h1[i+3] * w[(i+3) * L + tid];
    }
    out[tid] = relu(acc);
}

// Layer 3: (L,) @ (L, L) -> (L,) + bias (near-identity w3)
__device__ void layer3(const float* __restrict__ h2,
                       const float* __restrict__ w,
                       const float* __restrict__ b,
                       float* __restrict__ out,
                       int tid) {
    float acc = b[tid];
    for (int i = 0; i < L; i += 4) {
        acc += h2[i]   * w[(i)   * L + tid];
        acc += h2[i+1] * w[(i+1) * L + tid];
        acc += h2[i+2] * w[(i+2) * L + tid];
        acc += h2[i+3] * w[(i+3) * L + tid];
    }
    out[tid] = acc;  // no ReLU on output
}

// One block = one room. Block size = 32 (one warp).
// Shared mem: H + L floats for intermediates.
__global__ void jepa_forward_kernel(
    const float* __restrict__ signal,      // (64,) input
    const float* __restrict__ w1,        // (n, D, H)
    const float* __restrict__ w2,        // (n, H, L)
    const float* __restrict__ w3,        // (n, L, L)
    const float* __restrict__ b1,        // (n, H)
    const float* __restrict__ b2,        // (n, L)
    const float* __restrict__ b3,        // (n, L)
    float* __restrict__ out,             // (n, L) output
    int n_rooms
) {
    int room = blockIdx.x;
    int tid  = threadIdx.x;

    if (room >= n_rooms) return;
    if (tid >= 32) return;  // we only need 32 threads

    // Shared memory for layer activations
    __shared__ float smem[H + L];  // 32 + 16 = 48 floats
    float* h1 = smem;       // (H,) = 32
    float* h2 = smem + H;   // (L,) = 16

    // Room offsets in weight arrays
    int off_w1 = room * D * H;
    int off_w2 = room * H * L;
    int off_w3 = room * L * L;
    int off_b1 = room * H;
    int off_b2 = room * L;
    int off_b3 = room * L;

    // Layer 1: only threads 0..31 participate (H=32)
    if (tid < H) {
        layer1(signal, w1 + off_w1, b1 + off_b1, h1, tid);
    }
    __syncthreads();

    // Layer 2: only threads 0..15 participate (L=16)
    if (tid < L) {
        layer2(h1, w2 + off_w2, b2 + off_b2, h2, tid);
    }
    __syncthreads();

    // Layer 3: only threads 0..15 participate
    if (tid < L) {
        layer3(h2, w3 + off_w3, b3 + off_b3, out + room * L, tid);
    }
}

// Batched kernel: process `batch` signals for all rooms
__global__ void jepa_forward_batch_kernel(
    const float* __restrict__ signals,   // (batch, 64)
    const float* __restrict__ w1,
    const float* __restrict__ w2,
    const float* __restrict__ w3,
    const float* __restrict__ b1,
    const float* __restrict__ b2,
    const float* __restrict__ b3,
    float* __restrict__ out,             // (batch, n, L)
    int n_rooms,
    int batch
) {
    int room = blockIdx.x;
    int b    = blockIdx.y;
    int tid  = threadIdx.x;

    if (room >= n_rooms || b >= batch) return;
    if (tid >= 32) return;

    __shared__ float smem[H + L];
    float* h1 = smem;
    float* h2 = smem + H;

    const float* signal = signals + b * D;
    float* out_ptr = out + (b * n_rooms + room) * L;

    int off_w1 = room * D * H;
    int off_w2 = room * H * L;
    int off_w3 = room * L * L;
    int off_b1 = room * H;
    int off_b2 = room * L;
    int off_b3 = room * L;

    if (tid < H) {
        layer1(signal, w1 + off_w1, b1 + off_b1, h1, tid);
    }
    __syncthreads();

    if (tid < L) {
        layer2(h1, w2 + off_w2, b2 + off_b2, h2, tid);
    }
    __syncthreads();

    if (tid < L) {
        layer3(h2, w3 + off_w3, b3 + off_b3, out_ptr, tid);
    }
}

// ============================================================================
// Host-side launcher
// ============================================================================

extern "C" {

// Single tick
void jepa_cuda_tick(
    const float* signal,
    const float* w1, const float* w2, const float* w3,
    const float* b1, const float* b2, const float* b3,
    float* out,
    int n_rooms
) {
    dim3 grid(n_rooms, 1);
    dim3 block(32, 1);
    jepa_forward_kernel<<<grid, block>>>(
        signal, w1, w2, w3, b1, b2, b3, out, n_rooms
    );
    cudaDeviceSynchronize();
}

// Batched tick
void jepa_cuda_tick_batch(
    const float* signals,
    const float* w1, const float* w2, const float* w3,
    const float* b1, const float* b2, const float* b3,
    float* out,
    int n_rooms,
    int batch
) {
    dim3 grid(n_rooms, batch);
    dim3 block(32, 1);
    jepa_forward_batch_kernel<<<grid, block>>>(
        signals, w1, w2, w3, b1, b2, b3, out, n_rooms, batch
    );
    cudaDeviceSynchronize();
}

} // extern "C"
