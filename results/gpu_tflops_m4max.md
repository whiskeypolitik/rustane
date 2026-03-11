# GPU TFLOPS Benchmark — M4 Max 128GB

**Date**: 2026-03-10
**Hardware**: Apple M4 Max, 128GB unified memory, 40 GPU cores
**Software**: MLX 0.31.0, macOS
**Method**: 100 iterations after 10 warmup, wall clock via `time.perf_counter()`

## Results

| Config | Median ms | p5 ms | p95 ms | TFLOPS | p5 | p95 |
|--------|-----------|-------|--------|--------|-----|-----|
| 4K x 4K fp16 | 9.325 | 9.291 | 9.467 | **14.74** | 14.52 | 14.79 |
| 4K x 4K fp32 | 10.519 | 10.445 | 10.614 | 13.07 | 12.95 | 13.16 |
| SEQ×FFN×DIM fp16 (FFN fwd) | 0.306 | 0.297 | 0.322 | 7.90 | 7.50 | 8.13 |
| SEQ×DIM×FFN fp16 (FFN proj) | 0.328 | 0.319 | 0.336 | 7.37 | 7.19 | 7.58 |
| 0.6B FFN fwd fp16 | 0.365 | 0.354 | 0.377 | 8.82 | 8.54 | 9.10 |
| 8K x 8K fp16 (peak) | 77.111 | 72.668 | 82.679 | 14.26 | 13.30 | 15.13 |

## Key Findings

- **Peak measured: 14.74 TFLOPS** (fp16, 4K×4K matmul)
- ~~Apple claimed: ~36.86 TFLOPS (M4 Max fp16 peak)~~
- **CORRECTION (2026-03-11):** Apple's 36.86T assumes fp16 runs at 2x fp32. **M4 GPU does NOT have fp16 double rate.** Real fp32 peak is ~18.4T. fp16 and fp32 have identical ALU throughput on M1-M4.
- **Actual utilization: ~80%** of real 18.4T peak — this is good!
- **fp32 nearly matches fp16**: 13.07 vs 14.74 — only 12% faster. This proves fp16≠2x on M4 (the 12% gain is from reduced memory traffic, not faster ALUs).
- **Training-relevant sizes**: 7-9 TFLOPS at gpt_karpathy dimensions — smaller matrices don't saturate the GPU
- **8K peak similar to 4K**: No significant scaling benefit going larger, confirms 4K already saturates compute

## Implications for Rustane

1. **GPU is already well-utilized at ~80% MFU.** Custom Metal shaders could gain 10-15% more.
2. **25 TFLOPS is physically impossible on M4 Max** (would need 135% of real peak). Revise targets.
3. ANE with fused mega-kernels (~17-19T) could match or exceed GPU at training-relevant matmul sizes (7-9T)
4. Dual-accelerator strategy confirmed: ANE trains while GPU is free for inference
5. **Note:** M5 (Apple10 GPU) genuinely doubles fp16 throughput — our Metal shaders will benefit automatically
