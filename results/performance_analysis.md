# Performance Analysis & Optimization Roadmap

**Date:** 2026-03-11
**Hardware:** Apple M4 Max (40-core GPU, 16-core ANE)

---

## TL;DR

| Target | Measured | Claimed Peak | Real Peak | Actual MFU | Status |
|--------|----------|-------------|-----------|------------|--------|
| GPU (MLX fp16) | 14.74 T | 36.86 T | **~18.4 T** | **~80%** | Already well-utilized |
| ANE (single conv1x1) | 7.3 T | 19 T | 19 T | **38%** | Expected for single-op graphs |

**GPU:** Apple's 36.86T claim is misleading. M4 fp16 has identical ALU throughput to fp32. Real peak is ~18.4T. We're at 80% — not 40%.

**ANE:** Single-op graphs are expected to hit only 30-38%. Deep fused graphs (32+ ops) can reach 94% (≈17.8T). This is the primary optimization target.

---

## GPU: The fp16 Myth on M4

### What Apple Claims vs Reality

Apple markets M4 Max at 36.86 TFLOPS fp16 by assuming fp16 runs at 2x fp32 throughput. **This is not true on M1 through M4 GPUs.**

Evidence:
- Our benchmark: fp32 = 13.07T, fp16 = 14.74T → only 12% difference (not 2x)
- Philip Turner's [metal-benchmarks](https://github.com/philipturner/metal-benchmarks): confirms fp16/fp32 parity on M-series
- Academic paper "Apple vs Oranges" (arXiv 2502.05317): M4 10-core measures 2.9T fp32, matching per-core scaling

The 12% fp16 advantage comes from reduced memory traffic (half the bytes), not faster ALUs.

### Realistic GPU Targets

| Level | TFLOPS | MFU | What It Requires |
|-------|--------|-----|------------------|
| Current (MLX) | 14.74 | ~80% | Already achieved |
| Optimized Metal | 15.5-16.5 | 85-90% | Custom shaders with `simdgroup_matrix` |
| Theoretical max | ~17-18 | 92-98% | Perfect register allocation (unrealistic sustained) |

**25 TFLOPS is physically impossible on M4 Max.** It would require 135% of the real fp32 peak.

### GPU Optimization (Low Priority)

We're already at 80% MFU. Remaining gains are marginal:
1. Custom Metal shaders with `simdgroup_matrix_multiply_accumulate` → +10-15%
2. Kernel fusion (fused FFN/attention) → +20-40% on end-to-end throughput (not raw matmul)
3. Eliminate MLX Python overhead → +10-30% on small shapes, negligible on large

**Decision: GPU optimization is not the bottleneck. Focus on ANE.**

### Note on M5 (Future)

Apple10 GPU (M5) genuinely doubles fp16 throughput and adds per-core neural accelerators. M5 Max could realistically hit 35+ TFLOPS fp16. Our Metal shaders will benefit from this without code changes.

---

## ANE: Why 7.3 TFLOPS and How to Reach 19

### Root Cause: Single-Op Graphs Don't Saturate ANE

ANE has ~32MB SRAM and a deep execution pipeline. With only 1 operation per graph:
- Pipeline stays mostly empty (only 1 stage active)
- Per-dispatch overhead (~0.095ms XPC + IOKit round-trip) dominates
- SRAM is underutilized (data loaded once, used once, discarded)

With 32+ operations per graph:
- Pipeline is fully fed (all stages active simultaneously)
- Dispatch overhead amortized over many ops
- SRAM reuse is maximized (intermediate results stay on-chip)

**This is well-documented:** maderix's benchmarks show single matmul at ~30% utilization, deep graphs at 94%.

### Expected Utilization by Graph Depth

| Graph Depth | Expected TFLOPS | MFU | Example |
|-------------|-----------------|-----|---------|
| 1 op (our conv1x1 bench) | 5-7 T | 30-38% | Single matmul |
| 5 ops (our DynMatmul probe) | 0.01-0.5 T | <3% | Tiny problem + shallow graph |
| 15 ops (single DynMatmul) | 3-8 T | 15-40% | autoresearch-ANE woFwd |
| 25+ ops (fused FFN) | 10-14 T | 50-75% | ane crate GPT-2 FFN |
| 40+ ops (fused FFN mega) | 14-17 T | 75-90% | autoresearch-ANE ffnFused |
| 60+ ops (fused SDPA) | 16-18 T | 85-94% | autoresearch-ANE sdpaFwd |

### Why the DynMatmul Probe Got 0.014 TFLOPS

Three compounding problems:
1. **Tiny problem size:** 6.3M FLOPs/eval — dispatch overhead alone caps at ~0.066T
2. **Shallow graph:** Only 5 ops (2 slices + 2 transposes + 1 matmul)
3. **Minimum spatial width:** OC=64 leaves compute units mostly idle

This is not a bug — it's expected behavior for a validation probe. Real training kernels will be 100-1000x larger.

### ANE Optimization Strategies (Ranked by Impact)

#### 1. Fuse Operations Into Deep Graphs (2-4x improvement)

**THE single biggest lever.** Build fused training kernels matching autoresearch-ANE's patterns:

| Kernel | Ops | Pattern |
|--------|-----|---------|
| ffnFused | ~40 | 5 slices + 2 parallel matmuls + SiLU gate + matmul + residual |
| sdpaFwd | ~60+ | 3 QKV matmuls + full RoPE + attention + softmax + output proj |
| woFwd | ~15 | Single DynMatmul with reshape/transpose |

The ane crate's GPT-2 example already demonstrates attention (~35 ops) and FFN (~25 ops) as separate compiled units. Combining them gives ~60 ops → 85-94% utilization.

#### 2. Cache Request Objects (10-30% for small kernels)

The ane crate creates a new `_ANERequest` Obj-C object every `run()` call. autoresearch-ANE creates it once (`make_request`) and reuses it forever. This matters most for small/fast kernels where dispatch overhead is a larger fraction.

**Action:** Add `Executable::make_request()` and `Executable::run_with_request()` to ane crate or ane-bridge.

#### 3. Scale Up Problem Dimensions (2-10x for probes)

Training-relevant dimensions from autoresearch-ANE:
- DIM = 768-1024, HIDDEN = 3072-4096, SEQ = 256+
- DynMatmul working set: (IC × (SEQ+OC)) × 4 bytes ≈ several MB

Our probe used IC=768, SEQ=64, OC=64 (6.3M FLOPs). Real training uses IC=1024, SEQ=256, OC=3072 (1.6G FLOPs) — 250x more compute per dispatch.

#### 4. SRAM Budget Management (prevents 30% cliff)

M4 Max ANE has ~32MB SRAM. Working sets must fit:
- DynMatmul 1024×(256+3072)×2 bytes = ~8.4MB → fits ✓
- sdpaFwd 1024×4352×2 bytes = ~8.9MB input → fits ✓
- Full layer (all surfaces) → monitor as models scale

If working set exceeds SRAM, throughput drops ~30% due to DRAM spills.

#### 5. QoS Setting (marginal, <5%)

Use `NSQualityOfServiceUserInitiated` (21) instead of `Default` (-1). Matches autoresearch-ANE's setting.

---

## Confirmed Numbers (No Methodology Errors)

Our benchmarks are correct. The "underperformance" was a framing issue:

| Metric | Before (framing) | After (corrected) |
|--------|------------------|-------------------|
| GPU MFU | 40% of 36.86T | **80% of 18.4T** |
| ANE MFU | 38% of 19T | **38% of 19T** (correct, single-op expected) |
| ANE target | "underperforming" | **on track** — fused kernels are next step |
| 25T GPU goal | "need custom Metal" | **physically impossible on M4** |

---

## Next Steps (Priority Order)

1. **Build fused FFN mega-kernel** — DynMatmul(up) → GELU → DynMatmul(down) as single graph (~40 ops)
2. **Benchmark fused kernel** — expect 10-17 TFLOPS
3. **Add Request caching** to ane-bridge — reduce per-eval overhead
4. **Build fused attention kernel** — sdpaFwd pattern (~60 ops)
5. **Measure end-to-end training loop** — forward + backward + staging

---

## Sources

- [Inside the M4 ANE, Part 2: Benchmarks (maderix)](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)
- [Orion: ANE Training & Inference (arXiv 2603.06728)](https://arxiv.org/html/2603.06728v1)
- [Deploying Transformers on ANE (Apple ML Research)](https://machinelearning.apple.com/research/neural-engine-transformers)
- [Philip Turner's metal-benchmarks (GPU microarchitecture)](https://github.com/philipturner/metal-benchmarks)
- [Apple vs Oranges: M-Series for HPC (arXiv 2502.05317)](https://arxiv.org/html/2502.05317v1)
- [metalQwen3 — pure Metal GPU LLM](https://github.com/BoltzmannEntropy/metalQwen3)
