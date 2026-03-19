# Pre-Merge Benchmark Results: ane-utilization

**Date**: 2026-03-18
**Branch**: `ane-utilization`
**Hardware**: M4 Max, 128GB UMA
**Software**: Rust 1.94.0, macOS 15.x

## Phase 0: Unit Tests

All 32 test suites pass, 0 failures.

## Phase 1: Scale Correctness

| Config | Kernels | Forward | Train (10 steps) | ms/step | Status |
|--------|---------|---------|-------------------|---------|--------|
| **110M** (1024d/8L) | ✅ 0.5s | loss=9.017 | 9.017→8.989 (Δ-0.028) | **213ms** | PASS |
| **600M** (1536d/20L) | ✅ 0.5s | loss=9.011 | 9.011→8.986 (Δ-0.025) | **970ms** | PASS |
| **800M** (1792d/24L) | ✅ 0.5s | loss=9.018 | 9.018→8.996 (Δ-0.022) | **1454ms** | PASS |
| **1B** (2048d/28L) | ✅ 0.6s | loss=9.018 | 9.018→8.996 (Δ-0.022) | **2194ms** | PASS |

All configs: kernels compile, forward works, loss decreases, no NaN/Inf.

## Phase 2: Step Time Breakdown

| Config | total | fwd | bwd | norm | upd |
|--------|-------|-----|-----|------|-----|
| **110M** (1024d/8L) | 213ms | — | — | — | — |
| **600M** (1536d/20L) | 970ms | 261ms | 637ms | 14ms | 59ms |
| **800M** (1792d/24L) | 1453ms | 398ms | 937ms | 26ms | 93ms |
| **1B** (2048d/28L) | 2204ms | 588ms | 1441ms | 38ms | 137ms |

### Backward dominates at all scales

| Config | bwd % | fwd % | upd % | norm % |
|--------|-------|-------|-------|--------|
| 600M | 66% | 27% | 6% | 1% |
| 800M | 64% | 27% | 6% | 2% |
| 1B | 65% | 27% | 6% | 2% |

Backward is consistently ~65% of step time across all scales.

## Phase 3: Per-Layer Forward Profile

### 600M (1536d, 20 layers)

| Operation | Time | % |
|-----------|------|---|
| RMSNorm1 (CPU) | 0.10ms | 1% |
| Stage sdpaFwd | 2.52ms | 15% |
| ANE sdpaFwd | 4.27ms | 25% |
| Read sdpaFwd | 1.03ms | 6% |
| Stage woFwd | 0.38ms | 2% |
| ANE woFwd | 0.77ms | 5% |
| Read woFwd | 0.30ms | 2% |
| Residual + RMSNorm2 | 0.44ms | 3% |
| Stage ffnFused | 1.60ms | 9% |
| ANE ffnFused | 3.57ms | 21% |
| Read ffnFused | 2.07ms | 12% |
| **TOTAL** | **17.05ms** | |

| Category | ms | % |
|----------|-----|---|
| ANE compute | 8.62 | 51% |
| IOSurf staging | 4.50 | 26% |
| IOSurf readback | 3.40 | 20% |
| CPU (rmsnorm) | 0.54 | 3% |

### 800M (1792d, 24 layers)

| Category | ms | % |
|----------|-----|---|
| ANE compute | 10.54 | 52% |
| IOSurf staging | 5.60 | 28% |
| IOSurf readback | 3.22 | 16% |
| CPU (rmsnorm) | 0.90 | 4% |
| **TOTAL** | **20.26** | |

### 1B (2048d, 28 layers)

| Category | ms | % |
|----------|-----|---|
| ANE compute | 12.76 | 50% |
| IOSurf staging | 7.59 | 30% |
| IOSurf readback | 3.65 | 14% |
| CPU (rmsnorm) | 1.32 | 5% |
| **TOTAL** | **25.32** | |

### Scaling trend (forward only)

| Metric | 600M→800M | 800M→1B |
|--------|-----------|---------|
| ANE compute | +22% | +21% |
| IOSurf staging | +24% | +36% |
| IOSurf readback | -5% | +13% |
| Total per-layer | +19% | +25% |
| Total fwd (layers × per-layer) | +53% (×1.2 layers × 1.19/layer) | +48% |

Staging grows faster than ANE compute at larger dims — it becomes a bigger fraction.

## Key Observations

1. **Backward = 65% at all scales**. This is the GPU backward target.
2. **Forward staging = 26-30%** of per-layer forward. IOSurface-native weights would cut this.
3. **ANE compute = 50-52%** of per-layer forward. Stable ratio — ANE scales linearly with work.
4. **1B works fine** — no IOSurface allocation failures, no ANE compiler rejections, all 28 layers train.
5. **Adam scales super-linearly** — 59ms at 600M, 93ms at 800M, 137ms at 1B (param count scales ~2×, Adam ~2.3×). Cache pressure at larger sizes.

## Baseline Numbers for GPU Backward Branch

| Config | Current ms/step | bwd ms | GPU target bwd ms | Target total |
|--------|----------------|--------|-------------------|-------------|
| 600M | 970 | 637 | ~250 | ~580 (40% faster) |
| 800M | 1453 | 937 | ~370 | ~890 (39% faster) |
| 1B | 2204 | 1441 | ~570 | ~1330 (40% faster) |

Note: 600M was 710ms/step at seq=256. These benchmarks run at seq=512 (the `target_600m` config). Reconcile with training config before quoting numbers.
