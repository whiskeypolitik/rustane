# rustane

Rust training + inference engine for Apple Neural Engine (ANE) + Metal GPU.

Training pipeline validated from 48M to 5B parameters. Forward pass confirmed to 30B. All on M4 Max 128GB using reverse-engineered private ANE APIs.

## What This Does

Rustane is the first training-capable, memory-safe Rust engine for the Apple Neural Engine. It uses reverse-engineered private APIs (`_ANEClient`, `_ANEInMemoryModel`) to compile and evaluate MIL kernels directly on ANE hardware — no CoreML, no black-box scheduler.

The engine trains transformer models at 3-5W power draw, leaving the GPU completely free. Trained weights export via SafeTensors for inference anywhere.

## Scale Results (M4 Max 128GB)

### Training Pipeline Validation

25 architecture configs tested across 5 scales. Each validates: compile + forward + backward + Adam + loss decrease.

| Scale | Params | ms/step | tok/s | RAM | Status |
|-------|--------|---------|-------|-----|--------|
| 600M | 579M | 865 | 592 | ~12GB | Pass |
| 1B | 1.3B | 2,012 | 254 | ~25GB | Pass |
| 1.5B | 1.9B | 2,775 | 184 | ~30GB | Pass |
| 3B | 3.2B | 4,639 | 110 | ~55GB | Pass |
| **5B** | **5.0B** | **7,940** | **64** | **~85GB** | **Pass** |

### Forward-Only (no backward/optimizer)

| Scale | Forward Time | RAM |
|-------|-------------|-----|
| 7B | 3.1s | 31GB |
| 10B | 4.7s | 46GB |
| 15B | 27s | 70GB |
| 20B | 41s | 93GB |
| **30B** | **75s** | **130GB** |

No ANE compilation ceiling found. The limit is RAM, not the chip.

### Key Findings

- **Architecture crossover at 3B**: wide+shallow wins below (fewer ANE dispatches), deep+narrow wins above (smaller matmuls more efficient)
- **Efficiency cliff at dim=5120**: forward time jumps 4.7x per layer. Keep dim at or below 4096 for ANE.
- **Practical training ceiling**: ~5B on 128GB. An M3/M4 Ultra with 512GB could reach ~20B.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                      engine crate                    │
│  Training loop | Forward/Backward | Adam optimizer   │
│  ANE kernels (10) | CPU ops (vDSP) | Metal Adam     │
└───┬─────────────────────────────────────────┬───────┘
    │                                         │
┌───▼──────────────┐           ┌──────────────▼───────┐
│   ane-bridge     │           │   metal-decode       │
│   ANE private    │           │   Metal compute      │
│   API FFI        │           │   shaders (planned)  │
│   IOSurface I/O  │           │   Single-token       │
│   MIL generation │           │   decode path        │
└──────────────────┘           └──────────────────────┘
```

### Crates

- **ane-bridge** — Safe Rust FFI to ANE private APIs via objc2/dlopen. MIL kernel generation, IOSurface weight packing, dynamic weight pipeline.
- **metal-decode** — Metal shaders for single-token decode (planned).
- **engine** — Training orchestrator: ANE forward (10 fused kernels), CPU backward (Accelerate sgemm), Metal/CPU Adam optimizer.

## Quick Start

```bash
# Requires Rust 1.94.0+ and macOS 15+ on Apple Silicon
cargo build
cargo test -p engine --release

# Run training validation at 600M
cargo test -p engine --test bench_param_sweep --release -- --ignored --nocapture sweep_600m_a

# Run the full parameter sweep (600M to 5B, ~60 min)
cargo test -p engine --test bench_param_sweep --release -- --ignored --nocapture sweep_full

# Forward-only scale ladder (5B to 30B, ~8 min)
cargo test -p engine --test bench_fwd_only_scale --release -- --ignored --nocapture fwd_scale_ladder

# Train on real data (needs climbmix-400B tokenized data)
cargo run -p engine --release --bin train -- \
  --model custom:1536,4096,20,512 --data /path/to/train.bin \
  --lr 3e-4 --accum 1 --warmup 3% \
  --embed-lr 1.0 --beta2 0.99 \
  --loss-scale 1 --grad-clip 1 \
  --steps 72000
```

## Hardware Requirements

Any Apple Silicon Mac with 18GB+ RAM. The ANE is the same 16-core design across M1-M4. Only RAM differs.

Tested on M4 Max 128GB. Other configs are estimates based on RAM scaling.

| Hardware | Memory | Training Ceiling | Forward Ceiling |
|----------|--------|-----------------|-----------------|
| M1/M2/M3 Pro 18GB | 18 GB | ~300M | ~3B |
| M1/M2/M3 Pro 36GB | 36 GB | ~1B | ~7B |
| M1/M2/M3/M4 Max 64GB | 64 GB | ~3B | ~15B |
| M3/M4 Max 96GB | 96 GB | ~5B | ~20B |
| **M3/M4 Max 128GB** | **128 GB** | **~5B** (tested) | **~30B** (tested) |
| M3 Ultra 192GB | 192 GB | ~10B | ~40B+ |
| M3 Ultra 512GB | 512 GB | ~20B | ~100B+ |

## ANE Gotchas

- IOSurface spatial width must be multiple of 16 (silent data corruption otherwise)
- ANE compiler fails on rsqrt/sqrt after reduce ops — use pow(-0.5)
- Per-ANE-dispatch overhead: ~0.095ms (XPC + IOKit round-trip)
- IOSurface stores fp32, ANE casts to fp16 internally
- dim must be divisible by 128 (heads = dim/128, hd=128)
- hidden must be divisible by 16

## Sister Projects

- [autoresearch-ANE](https://github.com/ncdrone/autoresearch-ANE) — ANE training in native Obj-C (private APIs). The research foundation.
- [autoresearch-mlx](https://github.com/ncdrone/autoresearch-mlx) — MLX training in Python. Architecture exploration (241 experiments).

## Credits

See [CREDITS.md](CREDITS.md) for the full list. Key acknowledgments:

- **maderix** — ANE private API reverse engineering, the foundational work everything builds on
- **ane crate** (computer-graphics-tools) — Rust FFI to ANE, our ane-bridge base
- **thebasedcapital/ane-infer** — First Rust + ANE + Metal prototype
- **karpathy/llm.c** — Training architecture reference, climbmix-400B dataset

## License

MIT
