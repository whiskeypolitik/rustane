# IOSurface Staging Bandwidth — M4 Max

**Date:** 2026-03-11
**Hardware:** Apple M4 Max
**Mode:** Release build, 1000 iterations

## Lock/Unlock Overhead

| Config | µs/cycle |
|--------|----------|
| 768×128 (393K B) | 0.70 |
| 768×576 (1.8M B) | 0.62 |
| 768×3584 (11M B) | 0.58 |
| 1024×4096 (16.8M B) | 0.51 |

**Negligible.** ~0.5-0.7µs per lock/unlock cycle regardless of buffer size.

## Full Write (flat memcpy via copy_from_f32)

| Config | µs | MB | GB/s |
|--------|----|----|------|
| 768×128 (small DynMatmul) | 7.0 | 0.39 | 56.0 |
| 768×576 (Wo) | 24.0 | 1.77 | 73.7 |
| 768×3584 (FFN) | 122.7 | 11.01 | 89.8 |
| 1024×4096 (Qwen3 FFN) | 193.5 | 16.78 | 86.7 |

**~90 GB/s peak** for flat memcpy. 16% of M4 Max's 546 GB/s memory bandwidth.

## Channel-Interleaved Weight Staging (DynMatmul pattern)

| Config (IC×OC) | µs | Weight MB | GB/s |
|----------------|----|-----------|----- |
| 768×64 (probe) | 20.2 | 0.20 | 9.75 |
| 768×512 (Wo) | 162.0 | 1.57 | 9.71 |
| 768×3072 (FFN up) | 939.0 | 9.44 | 10.05 |
| 1024×3072 (Qwen3 FFN) | 1251.1 | 12.58 | 10.06 |

**~10 GB/s** for interleaved writes — **9x slower than flat memcpy.**

## Analysis

The interleaved write pattern (per-channel strided access) is the bottleneck:
- Flat memcpy: 90 GB/s (good cache behavior, sequential access)
- Interleaved: 10 GB/s (strided access, poor cache utilization)

For the FFN-sized case (768×3072):
- Weight staging: **939 µs**
- ANE compute (conv1x1): **~330 µs**
- Staging is **2.87x the compute time** for single-kernel operations

## Optimization Opportunities

1. **Use `copy_from_slice` per channel stripe** instead of element-by-element write → approach flat memcpy speed (~90 GB/s)
2. **NEON vectorized interleaved write** (autoresearch-ANE does this)
3. **For fused mega-kernels** (40+ ops, ~1-5ms compute), staging overhead becomes a smaller fraction
4. **Double-buffer**: stage weights to next IOSurface while current kernel runs

## Implication for Training

At FFN scale with fused mega-kernels (~2-5ms compute), 939µs staging = 19-47% overhead.
With optimized flat write (~123µs), staging drops to 2.5-6% overhead.
**Optimizing the write pattern is important for training throughput.**
