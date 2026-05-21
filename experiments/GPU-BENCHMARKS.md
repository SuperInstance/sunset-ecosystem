# GPU Benchmarks — RoomGrid at Scale

## Results (RTX 4050, CUDA 12.6, PyTorch 2.4.1)

| Rooms | GPU (einsum) | Rust CPU (est.) | Speedup |
|-------|-------------|-----------------|---------|
| 10K   | 2.0ms      | 130ms           | 65x     |
| 50K   | 8.4ms      | ~650ms          | 77x     |
| 100K  | 17.4ms     | ~1.3s           | 75x     |

## VRAM Usage (100K rooms)
- 998MB weights (64×32×16×16 = 100K rooms × 3.4K params)
- 1.8GB peak with activations
- 6GB total: could fit ~300K rooms

## Latent Diversity
- 200/200 unique latents at all scales (near-identity w3 fix)
- GPU matches CPU in numerical output (< 1e-5 diff)

## FLUX Compat Layer
- 1M opcode translations: 2.3s (0.4M/sec — pure Python bottleneck)
- Could GPU this: torch.compile the opcode map as a lookup table
