# NPU Offload Spec for HDC Novelty on AMD XDNA 2

> **Round 7 — Fleet Research Brief**  
> **Scope:** Evaluate whether AMD XDNA 2 NPU can accelerate Hyperdimensional Computing (HDC) binary novelty scoring, or whether the impedance mismatch is too high to justify offload.  
> **Source files:** `swarm/hdc_novelty.py`, `swarm/npu_router.py`  
> **Date:** 2026-05-24

---

## 1. What XDNA 2 Offers

AMD XDNA 2 (Strix Point / Ryzen AI 300 series) is the second-generation NPU derived from Xilinx AI Engine technology. Key specs relevant to our offload decision:

| Spec | Value | Notes |
|------|-------|-------|
| **Peak throughput** | 50 TOPS (INT8) / 25 TFLOPS (BF16) | TOPS is peak theoretical; realizable depends on memory layout and op mix |
| **Architecture** | 2D grid of AI Engine tiles | VLIW + SIMD vector cores; spatial dataflow |
| **Precision support** | INT8, INT4, Block BF16 | **No native binary (1-bit) support** |
| **ONNX Runtime path** | `VitisAIExecutionProvider` | Primary SDK: Ryzen AI Software + ONNX Runtime |
| **Quantizer** | AMD Quark (PyTorch + ONNX) | Static quantization to INT8/BF16; power-of-two scales |
| **Memory bandwidth** | ~120 GB/s LPDDR5X / 800 GB/s on-chip SRAM | Small models with tiled weights fit in SRAM; large activations spill to DRAM |
| **Programming model** | ONNX-GenAI / ONNX Runtime / IRON (bare-metal) | IRON/MLIR-AIR allows custom kernels, but this is a heavy lift |

### 1.1 Critical insight for HDC
XDNA 2 is a **matrix-engine NPU**. It excels at dense GEMM, conv2d, and MLP layers — exactly what `swarm/npu_router.py` targets with its `RoutingMLP`. It is **not** a general-purpose vector ALU with bitwise logic primitives. The AI Engine tiles do not expose POPCNT or bitwise XOR as first-class instructions in the ONNX Runtime path.

---

## 2. Which HDC Operations Could Run on NPU

From `swarm/hdc_novelty.py`, the core HDC novelty pipeline consists of:

| Operation | NumPy implementation | NPU feasibility |
|-----------|---------------------|-----------------|
| **Binarization** | `(vec > 0).astype(np.uint8)` | Can be fused as a `Sign` → `Clip` → `Cast` chain, but ONNX `Sign` is not universally quantized. Easier to keep on CPU as a pre-processing step. |
| **XOR (Hamming)** | `np.bitwise_xor(a, b)` | ONNX has `BitwiseXor` (opset 18+), but **XDNA 2 Vitis AI EP does not offload bitwise ops to NPU tiles**. They run on CPU fallback. |
| **POPCOUNT** | `np.bitwise_count(xor).sum()` or Python `int.bit_count()` | **No POPCNT operator in ONNX.** Must be emulated via shift-and-add or lookup table. |
| **Bundling (majority vote)** | Element-wise sum of binary vectors → threshold at 0 | Sum is `ReduceSum` + `Sign`; doable, but requires unpacking binary vectors to INT8, losing the 32× packing advantage. |
| **Permutation** | Circular shift / reorder | `Gather` or `Transpose`; ONNX supports it, but NPU gains are minimal for pure data-movement ops. |

### 2.1 Verdict per operation
- **XOR + POPCNT** (the hot path for novelty scoring): **Not practically off-loadable** via ONNX Runtime on XDNA 2 today. The NPU compiler will reject or CPU-fallback these ops.
- **MLP routing** (what `npu_router.py` already does): **Excellent fit**. Dense `MatMul` + `ReLU` + `Softmax` maps perfectly to INT8 GEMM arrays.

---

## 3. ONNX Model Spec for an HDC Novelty Kernel

Even though the NPU is a poor fit, here is a **reference ONNX subgraph** that computes Hamming distance without a custom POPCNT op, for completeness.

### 3.1 POPCNT emulation via 4-bit lookup
ONNX lacks `PopCount`. A portable workaround is a **lookup-table (LUT) decomposition**:

1. Split each `uint8` into two 4-bit nibbles.
2. Use a `Gather` with a precomputed 16-entry LUT `[0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4]`.
3. Sum all LUT outputs.

```
Input:     uint8[A, B]      -- two packed binary vectors, shape (n_words,)
Output:    float32 scalar   -- Hamming distance normalized to [0,1]

Subgraph:
  1. Cast A, B → uint8 (ensure dtype)
  2. BitwiseXor(A, B) → X  (opset 18)
  3. Unpack X into low/high nibbles:
       low  = X & 0x0F        -- bitwise AND via Mul + Floor + Cast hacks, or use Split if pre-shuffled
       high = X >> 4          -- no bitwise shift in ONNX either; must pre-shift weights
  4. Gather(LUT, low)  → count_low
  5. Gather(LUT, high) → count_high
  6. ReduceSum(count_low + count_high) → hamming
  7. Div(hamming, dim) → score
```

> **Reality check:** Steps 3–5 are absurdly inefficient compared to a single VPOPCNTDQ instruction. The graph is ~20+ nodes, most of which will **CPU-fallback** on Vitis AI EP because `Gather`, `And`, and `ShiftRight` are not in the NPU subgraph whitelist.

### 3.2 Alternative: float32 MLP approximation
Instead of exact Hamming distance, train a tiny MLP to approximate novelty scores from two float32 vectors:

```python
# Pseudo-PyTorch → ONNX
class HDCNoveltyApproximator(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim * 2, dim),   # concat(a,b)
            nn.ReLU(),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),              # novelty in [0,1]
        )
    def forward(self, a, b):
        x = torch.cat([a, b], dim=-1)
        return self.encoder(x)
```

- **Pros:** Runs natively on XDNA 2 INT8 GEMM arrays via Quark quantization.
- **Cons:** Approximate, not deterministic. Requires calibration data. Adds model-management overhead.
- **Fleet stance:** Deterministic Hamming distance is a design requirement for the breeding daemon’s novelty gate. An MLP approximator introduces non-determinism into agent selection — **unacceptable**.

---

## 4. Fallback Path When NPU Unavailable

The existing code in `swarm/hdc_novelty.py` already implements a clean fallback hierarchy:

```
HAS_AVX512 ──yes──► AVX-512 VPOPCNTDQ fast path (np.bitwise_count on uint64)
    │
    no ────► NumPy bitwise_count (if NumPy ≥ 2.0, still vectorised)
                │
                no ────► Pure-Python per-word popcount (int.bit_count)
                            │
                            last resort ────► np.unpackbits fallback
```

**Recommendation:** Preserve this hierarchy exactly. The NPU path, if ever added, should sit **alongside** AVX-512, not replace it:

```python
if use_npu and HAS_XDNA2:
    score = npu_hdc_approximator(a, b)   # optional, approximate
else:
    score = avx512_hdc_score(a, b)       # exact, deterministic
```

---

## 5. Expected Latency: CPU AVX-512 vs NPU

### 5.1 Workload definition
- **Dataset:** 1,000 vectors × 512-bit hypervectors
- **Task:** Full pairwise novelty matrix (1,000 × 1,000 = 1M comparisons)
- **Packed representation:** 512 bits = 8 × `uint64` words per vector

### 5.2 CPU AVX-512 estimate
From the benchmark in `hdc_novelty.py`:

| Metric | Value |
|--------|-------|
| Per-comparison cost | 8 XORs + 8 POPCNTs + 1 sum |
| AVX-512 throughput | VPOPCNTDQ: 8× uint64 per zmm register per instruction |
| Effective vector width | 512 bits |
| Estimated 1M comparisons | **~2–4 ms** on a single Zen 5 core at 4 GHz |
| vs cosine (float32) | ~100× faster (as documented) |

The CPU path is **memory-bandwidth-bound**, not compute-bound. 1M comparisons × 8 words × 8 bytes = 64 MB of XOR traffic — easily saturating L3 but still completing in milliseconds.

### 5.3 NPU estimate (theoretical best case)
Even if we could somehow map the workload:

| Metric | Value |
|--------|-------|
| NPU peak | 50 TOPS INT8 |
| Useful ops per comparison | 512 XORs + 512 POPCNTs = ~1K ops (if emulated as int8 adds) |
| Theoretical throughput | 50e12 / 1e3 = **50M comparisons/sec** |
| 1M comparisons (theoretical) | **~20 ms** |
| Overhead | Model load, PCIe/DMA setup, INT8 unpack/pack, CPU fallback for unsupported ops |
| **Realistic expectation** | **>100 ms with massive code complexity** |

### 5.4 Latency conclusion
For this workload, **AVX-512 on CPU beats XDNA 2 NPU** even in theory, and the implementation gap is enormous. The NPU’s 50 TOPS are for dense INT8 MAC arrays, not bitwise logic on packed binary data.

---

## 6. Blockers: Why This Is Hard

| Blocker | Severity | Details |
|---------|----------|---------|
| **ONNX has no POPCNT** | 🔴 Critical | No operator exists. Emulation via LUT/shift-add is 20–50× slower than native VPOPCNTDQ. |
| **int8-only vs binary** | 🔴 Critical | XDNA 2 computes in INT8/BF16. Binary vectors packed into uint64 must be unpacked to int8, losing the 32× density advantage. Memory bandwidth explodes. |
| **Bitwise ops not in NPU whitelist** | 🔴 Critical | `BitwiseXor` (opset 18) is not supported by Vitis AI EP. It will CPU-fallback, defeating the purpose. |
| **No shift/and/or ops on NPU** | 🟠 High | Even if we wanted to emulate POPCNT with bit-manipulation, the NPU lacks these primitives in the ONNX path. |
| **Determinism requirement** | 🟠 High | The breeding daemon’s novelty gate requires exact, reproducible Hamming distance. MLP approximation introduces variance. |
| **ROI vs complexity** | 🟡 Medium | Engineering a custom MLIR-AIR kernel via IRON is possible but requires Xilinx expertise and months of work for marginal (if any) speedup over AVX-512. |

---

## Executive Summary

> **NPU offload for HDC novelty scoring is NOT viable on AMD XDNA 2 today.**

The impedance mismatch is too high:

1. **Architecture mismatch:** XDNA 2 is a matrix-multiply engine. HDC novelty is a bitwise-logic workload.
2. **Precision mismatch:** Binary hypervectors (1-bit) vs NPU native INT8. Unpacking destroys the memory-density advantage.
3. **ONNX operator gap:** No POPCNT, no bitwise shift, no native XOR offload on Vitis AI EP.
4. **Latency reality:** AVX-512 VPOPCNTDQ completes the 1M×512-bit matrix in ~2–4 ms. The NPU cannot beat this even theoretically, and practically would require 100+ ms with enormous code complexity.

### What SHOULD run on the NPU
The existing `swarm/npu_router.py` `RoutingMLP` (signal → channel softmax) is a **perfect fit** for XDNA 2. Keep that path. It is a dense MLP with ReLU + MatMul + Softmax — exactly what the INT8 GEMM arrays were designed for.

### Recommendation to Forgemaster / Fleet
- **Close the NPU-HDC path.** Do not invest engineering time in a custom MLIR-AIR kernel or LUT-emulated ONNX graph.
- **Double down on AVX-512.** The `hdc_novelty.py` VPOPCNTDQ path is already world-class for this workload.
- **Future watch:** If AMD adds `uint1` or bitwise logic support in XDNA 3, re-evaluate. Until then, the CPU is the right silicon for HDC novelty.

---

*Researched by CCC, Cocapn Fleet.*  
*Sources: AMD XDNA 2 whitepapers, ONNX Runtime op support registry, Ryzen AI Software 1.7.1 release notes, `swarm/hdc_novelty.py` AVX-512 benchmark data.*
