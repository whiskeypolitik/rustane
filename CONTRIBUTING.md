# Contributing to Rustane

Thanks for your interest in ANE training research. This project is actively developed and we welcome contributions.

## Getting Started

```bash
git clone https://github.com/ncdrone/rustane.git
cd rustane
cargo build
cargo test -p engine --release
```

Requires:
- Rust 1.94.0+
- macOS 15+
- Apple Silicon (M1 or later)
- 18GB+ RAM (64GB+ recommended for billion-scale)

## Running Tests

```bash
# Unit tests (no ANE hardware needed for most)
cargo test -p engine --release

# Training validation at 600M (needs ANE, ~17s)
cargo test -p engine --test bench_param_sweep --release -- --ignored --nocapture sweep_600m_a

# Full parameter sweep 600M-5B (~60 min, needs 85GB+ RAM)
cargo test -p engine --test bench_param_sweep --release -- --ignored --nocapture sweep_full

# Forward-only scale ladder 5B-30B (~8 min, needs 128GB)
cargo test -p engine --test bench_fwd_only_scale --release -- --ignored --nocapture fwd_scale_ladder
```

## Submitting a PR

1. Fork the repo
2. Create a branch from `master`
3. Make your changes
4. Run `cargo test -p engine --release` (all unit tests must pass)
5. Run at least one `--ignored` benchmark test relevant to your change
6. Open a PR against `master`

### PR checklist

- [ ] `cargo check` passes
- [ ] `cargo test -p engine --release` passes (28 unit tests + integration tests)
- [ ] Relevant benchmark test passes (if touching engine code)
- [ ] No new warnings from `cargo check`
- [ ] Commit messages describe what changed and why

## What We Need Help With

### High impact
- **GPU backward pass** — Metal compute shaders for backward matmuls (currently CPU-bound, 63-67% of step time)
- **Long training runs** — 10K+ step validation on real data at 600M-1.5B
- **M3 Ultra testing** — see below

### Medium impact
- IOSurface-native weight storage (zero-copy staging)
- Workspace correctness tests (forward_ws vs forward equivalence)
- Mixed precision experiments

### Research
- GQA (grouped query attention) support
- Sequence length scaling beyond 512
- ANE kernel fusion strategies at different dims

## M3 Ultra Contributors

If you have access to an M3 Ultra (192GB or 512GB), we have a dedicated branch for you:

**Branch: `m3-ultra`**

The M3 Ultra has 2 ANE dies (2x the ANE cores of a Max). We want to know:

1. **Does training work at 10B+?** The 192GB model should fit ~10B training state comfortably. The 512GB model could reach 20B.
2. **Does Apple's ANE scheduler split work across both dies?** If so, we'd expect ~2x ANE throughput vs Max. If not, one die sits idle.
3. **What happens at dim=5120+?** We found an efficiency cliff at dim=5120 on M4 Max. Does the Ultra behave the same?

To test:
```bash
git checkout m3-ultra

# Training validation at 10B (should fit on 192GB)
cargo test -p engine --test bench_param_sweep --release -- --ignored --nocapture sweep_10b

# Forward pass — push to 50B+ on 512GB
cargo test -p engine --test bench_fwd_only_scale --release -- --ignored --nocapture fwd_find_ceiling
```

Report your results as an issue with the `m3-ultra` label. Include:
- Chip model (M3 Ultra)
- RAM (192GB or 512GB)
- macOS version
- Output from the test run

Even a single test result from an Ultra is valuable. Nobody has published ANE data from this hardware.

## Project Structure

```
crates/
  ane-bridge/     — Rust FFI to ANE private APIs
  metal-decode/   — Metal shaders for decode (planned)
  engine/         — Training orchestrator
    src/
      bin/train.rs    — Training binary with CLI flags
      full_model.rs   — Forward/backward/train_step
      layer.rs        — Per-layer ANE kernels
      model.rs        — Model configs (48M to 5B+)
      cpu/            — Accelerate FFI, Adam, RMSNorm, etc.
    tests/
      bench_*.rs      — Benchmarks (--ignored, run manually)
      auto_*.rs       — A/B optimization tests
      phase*.rs       — Phase validation tests
results/              — Benchmark results and sweep data
```

## ANE Gotchas

If you're touching ANE kernel code:

- IOSurface spatial width must be multiple of 16 (silent data corruption otherwise)
- dim must be divisible by 128, hidden must be divisible by 16
- ANE compiler fails on rsqrt/sqrt after reduce ops — use pow(-0.5)
- Per-dispatch overhead is ~0.095ms (XPC + IOKit round-trip)
- IOSurface stores fp32, ANE casts to fp16 internally
- `sgemm_at` uses beta=1.0 — always zero the output buffer first

## Code Style

- No unnecessary abstractions — three similar lines beats a premature helper
- Keep commits focused — one change per commit
- Commit messages: what changed and why, not just what
- No docs/comments on code you didn't change

## Leaderboard

After running any benchmark, submit your results to the community leaderboard:

```bash
make sweep-600m    # or any benchmark
make submit        # prompts for name + X handle, posts to bench.rustane.org
```

Requires `jq` for JSON processing (`brew install jq` if you don't have it).

View the leaderboard: [bench.rustane.org](https://bench.rustane.org)

## License

By contributing, you agree that your contributions will be licensed under MIT.
