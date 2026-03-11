# Results Index

> Master log. One-line summaries pointing to detailed test logs.

| Date | Phase | Test | Result | Log |
|------|-------|------|--------|-----|
| 2026-03-11 | — | Performance analysis & optimization | GPU at 80% real MFU (not 40%). ANE needs fused kernels (32+ ops) for 19T. | [performance_analysis.md](performance_analysis.md) |
| 2026-03-10 | 0.5.1 | GPU TFLOPS (MLX matmul) | Peak 14.74 TFLOPS fp16, **80% of real 18.4T peak** (Apple's 36.86T claim is misleading) | [gpu_tflops_m4max.md](gpu_tflops_m4max.md) |
| 2026-03-10 | 0 | ANE crate source analysis | IOSurface layout matches, weight blobs work, 1 gap (write helpers) | [phase0_analysis.md](phase0_analysis.md) |
| 2026-03-10 | 0 | Phase 0 verification (5 tests) | ALL PASS — multi-blob compile, ANE eval (1+1)*2=4.0, IOSurface roundtrip, RAII guards, interleaved layout | tests/phase0_verify.rs |
| 2026-03-10 | 0.5.2 | ANE TFLOPS (conv1x1) | Peak 7.3 TFLOPS at 768→3072 w=512. Single kernels = 38% of Orion's 19 TFLOPS claim. | [ane_tflops_m4max.md](ane_tflops_m4max.md) |
| 2026-03-10 | 0.5.5 | f32↔f16 conversion | half crate 3-4x faster than ane crate. 13.55 GB/s peak. <1ms for 2.4M elements. | [f16_convert_m4max.md](f16_convert_m4max.md) |
| 2026-03-10 | 1 | DynMatmul training probe | slice+transpose+matmul COMPILES + RUNS on ANE. 1000 iters ✓. Dynamic weight update ✓. | [phase1_training_probe.md](phase1_training_probe.md) |
| 2026-03-11 | 0.5.3 | IOSurface staging bandwidth | Flat memcpy: 90 GB/s. Interleaved write: 10 GB/s (9x slower). Staging = 2.87x compute for single kernels. | [iosurface_staging_m4max.md](iosurface_staging_m4max.md) |
| 2026-03-11 | 0.5.4 | Dual load (ANE + GPU) | <5% degradation. ANE 1.5%, GPU 2.6%. Concurrent use is viable. | [dual_load_m4max.md](dual_load_m4max.md) |
