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
| 2026-03-11 | 2 | CPU training ops (vDSP) | 26/26 tests pass. vDSP, RMSNorm, cross-entropy, Adam, SiLU, embedding. | crates/engine/src/cpu/ |
| 2026-03-11 | 3.1 | Forward kernel generation | sdpaFwd, woFwd, ffnFused all compile + run on ANE. 3000 evals, 0 crashes, 3.8s. | crates/engine/tests/phase3_kernels.rs |
| 2026-03-11 | 3.2 | Backward kernel generation | sdpaBwd1, sdpaBwd2 compile + run on ANE. All 10/10 kernels operational. | crates/engine/src/kernels/sdpa_bwd.rs |
| 2026-03-11 | 3.3-3.5 | Single-layer training | Forward+backward+Adam update on ANE. Loss: 927K→675K in 10 steps (27% decrease). **Phase 3 exit criteria met.** | crates/engine/tests/phase3_training.rs |
| 2026-03-11 | 4.1 | Full model scaffolding | 6-layer fwd+bwd, overfit test: loss 9.04→6.55 in 10 steps (28% decrease). Random init loss = 9.06 ≈ ln(8192) ✓. | crates/engine/tests/phase4_training.rs |
| 2026-03-11 | 4.2 | Complete backward pass | Fixed softcap bwd, final RMSNorm bwd, embedding bwd. Fixed dembed accumulation bug (beta=0→1). | crates/engine/src/full_model.rs |
| 2026-03-11 | 4.3 | Data + training binary | mmap'd uint16, val_bpb via token_bytes. Smoke test on 631M real tokens: loss decreasing, val_bpb 3.13→3.11. | crates/engine/src/bin/train.rs |
| 2026-03-11 | 4.4 | Obj-C → Rust port audit | 5 bugs fixed (init scales, grad clipping). 14 items verified. RMSNorm bwd: Rust correct, Obj-C has bug. | crates/engine/tests/phase4_pretraining_checks.rs |
| 2026-03-11 | — | Rust vs Obj-C optimization | 7 optimization opportunities, ~15-25% speedup. IOSurface reuse is #1. Honest: ANE dominates 70-80%. | [rust_vs_objc_optimization.md](rust_vs_objc_optimization.md) |
| 2026-03-11 | 4.5 | Optimization round 1 | IOSurface reuse + pre-alloc scratch + vectorized Adam + LTO. 525→397ms/step (**24% faster**). 47/47 tests pass. | crates/engine/tests/bench_step_time.rs |
| 2026-03-11 | — | Deep Rust vs Obj-C comparison | Rust 3.7x slower per microbatch (326ms vs 89ms). Root cause: synchronous ANE dispatch. Path to parity via async pipeline. | [rust_vs_objc_deep_comparison.md](rust_vs_objc_deep_comparison.md) |
| 2026-03-18 | — | Parameter sweep (600M→5B) | 25 configs across 5 scales. Wide+shallow wins at 600M-1.5B (+11-17%). Pattern reverses: 3B crossover, 5B deep+narrow wins (+16%). ANE trains at 5B on 128GB M4 Max. | [2026-03-18_1830_param-sweep.md](2026-03-18_1830_param-sweep.md) |
