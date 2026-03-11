# Dual Load (ANE + GPU) Bandwidth Test — M4 Max

**Date:** 2026-03-11
**Hardware:** Apple M4 Max, 128GB unified memory
**Method:** ANE conv1x1 benchmark + MLX 4K×4K fp16 matmul running simultaneously

## Results

| Accelerator | Standalone | Dual Load | Degradation |
|-------------|-----------|-----------|-------------|
| ANE (768→3072 conv1x1) | 6.431 TFLOPS | 6.337 TFLOPS | **1.5%** |
| GPU (4K×4K fp16 matmul) | 14.728 TFLOPS | 14.347 TFLOPS | **2.6%** |

## Analysis

**<5% degradation under dual load.** ANE and GPU run concurrently with negligible bandwidth contention.

This confirms:
1. M4 Max has sufficient memory bandwidth (546 GB/s) to serve both accelerators simultaneously
2. The dual-accelerator strategy is sound: ANE training + GPU inference can coexist
3. No need for time-slicing or scheduling between ANE and GPU workloads

## Implication for Rustane

The hybrid architecture works as designed:
- **Training:** ANE runs forward pass → GPU could handle backward/gradient computation simultaneously
- **Inference:** ANE prefill + Metal GPU decode can overlap
- **Dual use:** Train on ANE while serving inference on GPU, with <3% throughput loss
