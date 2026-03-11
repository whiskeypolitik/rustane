# rustane

Rust-native training + inference engine for Apple Neural Engine and Metal GPU.

Train on ANE (private APIs) → export weights → run hybrid ANE prefill + Metal decode → port to Jetson via candle-rs + CUDA.

## What This Does

Rustane is the first training-capable, memory-safe Rust stack for the Apple Neural Engine. It uses reverse-engineered private APIs (`_ANEClient`, `_ANEInMemoryModel`) to compile and evaluate MIL kernels directly on ANE hardware — no CoreML, no black-box scheduler.

The engine trains transformer models (up to ~1.5B parameters on 128GB M4 Max) at 3-5W power draw, leaving the GPU completely free for other work. Trained weights export via SafeTensors for inference anywhere.

## Status

Research complete (10 documents, 16 parallel research agents). Pre-coding validation in progress.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                      cli crate                       │
│              (train / infer / export)                 │
└─────────────┬───────────────────────┬───────────────┘
              │                       │
┌─────────────▼───────────────────────▼───────────────┐
│                    engine crate                       │
│  Training loop │ Inference │ Optimizer │ CPU ops      │
│  Checkpoint I/O │ Config │ KV cache │ f32↔f16        │
└───┬─────────────────────────────────────────┬───────┘
    │                                         │
┌───▼──────────────┐           ┌──────────────▼───────┐
│   ane-bridge     │           │   metal-decode       │
│   (Apple only)   │           │   (Apple only)       │
│                  │           │                      │
│ Pure Rust objc2  │           │ Metal compute shaders│
│ dlopen + FFI     │           │ q8_gemv, rmsnorm,    │
│ IOSurface I/O    │           │ rope, sdpa_causal    │
│ MIL generation   │           │ Single cmd buffer    │
│ Dynamic weights  │           │ Zero allocations     │
└──────────────────┘           └──────────────────────┘

Future backend:
┌──────────────────┐
│ candle + CUDA    │
│ (Jetson / edge)  │
└──────────────────┘
```

### Crates

- **ane-bridge** — Safe Rust FFI to ANE private APIs via dlopen. MIL kernel generation, IOSurface weight packing, dynamic weight pipeline (compile once, update via memcpy).
- **metal-decode** — Custom Metal shaders for single-token decode. One command buffer per token, zero allocations. Kernels: q8_gemv, q4_gemv, rmsnorm, rope, sdpa_causal.
- **engine** — Hybrid orchestrator. Training: ANE forward/backward + CPU optimizer + async dW overlap. Inference: ANE prefill (fused FFN mega-kernels) + Metal GPU decode.

## Hardware

Optimized for 128GB Apple Silicon. The install script (coming soon) will detect your hardware and tell you what's possible.

| Hardware | Memory | Max Model | Notes |
|----------|--------|-----------|-------|
| M4 Max 128GB | 128 GB | **~1.5B** | Sweet spot: DIM=1536, HIDDEN=5120, NL=48 |
| M4 Max 64GB | 64 GB | ~0.8B | Comfortable for 0.6B Qwen3-style |
| M4 Pro 48GB | 48 GB | ~0.6B | Full training feasible |
| M4 Pro 24GB | 24 GB | ~0.3B | Tight but works |
| M-series 16GB | 16 GB | ~0.1B | Research/prototype only |

The bottleneck is ANE compiler limits (16,384 spatial dim, ~32K channels, ~32MB SRAM), not memory. See `dev/research/10-remaining-gaps-and-decisions.md` for the full scaling analysis.

Future: NVIDIA Jetson deployment via candle-rs + CUDA backend.

## Setup

```bash
# Requires Rust 1.94.0+ and macOS 15+
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable

cargo build
cargo test
```

## Training

```bash
# Phase 1: ~48.8M param GPT (gpt_karpathy config)
cargo run -p engine -- train --config gpt_karpathy

# Phase 2: ~600M param Qwen3-style
cargo run -p engine -- train --config qwen3_06b

# Phase 3: ~1.5B param (128GB M4 Max sweet spot)
cargo run -p engine -- train --config qwen3_15b
```

*(Commands not yet functional — implementation in progress.)*

## Key Numbers

| Metric | ANE Training | MLX (GPU) Baseline |
|--------|-------------|-------------------|
| Peak TFLOPS | ~19 FP16 | ~36.86 FP16 |
| Sustained TFLOPS | 15-19 (mega-kernels) | ~10.6 (~29% MFU) |
| Power | 3-5W | 30-60W |
| GPU availability | **Free** | Occupied |
| Hardware used | ANE (16 cores) | GPU (40 cores) |

ANE training leaves the GPU free for inference serving, display rendering, or simultaneous MLX experiments.

## Sister Projects

- [autoresearch-ANE](https://github.com/ncdrone/autoresearch-ANE) — ANE training in native Obj-C (private APIs). The research foundation.
- [autoresearch-mlx](https://github.com/ncdrone/autoresearch-mlx) — MLX training in Python. Architecture exploration (241 experiments, val_bpb 1.664→1.266).

## Credits

See [CREDITS.md](CREDITS.md) for the projects and people this builds on.

## License

MIT
