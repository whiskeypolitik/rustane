# Experiment: Split Compile/Allocate Phases in parallel_bench.rs

## Hypothesis

> [!IMPORTANT]
> This is the **leading hypothesis**, not a confirmed root cause. Success criteria are empirical.

The ane debrief ([debrief-round-2.md](file:///Users/andrewgordon/other-projects/ane/debrief-round-2.md)) proved compile-payload IOSurface allocation works fine in isolation — 12KB MIL, 86MB weights, same-process and cross-process, all pass. The original failure is a **small compile-time MIL surface** (~11,865 bytes) inside `ane::client::compile_network`, not a rustane post-compile buffer allocation.

The hypothesis: the failure occurs because `parallel_bench.rs` **interleaves compilation and TensorData allocation**, so later kernel compilations attempt their MIL payload IOSurface while earlier kernels' runtime TensorData IOSurfaces are already live. This overlap may exhaust a shared kernel resource that the ane-isolated tests never stress.

Current flow in `compile_mode_runner`:
1. `compile_sdpa_runner` → compile sdpa exe → allocate **8 TensorData surfaces** → return
2. `compile_wo_runner` → compile wo exe (8 surfaces alive) → allocate **2 more** → return
3. `compile_baseline_ffn_runner` → compile 2 FFN exes (10 surfaces alive) → allocate **4 more** → return

This is the cleanest mismatch vs ane's tests: ane proved compiles are fine alone, but rustane compiles later kernels while earlier kernels' runtime surfaces are already allocated.

> [!NOTE]
> If the issue is purely "small MIL surface allocation is flaky on cold compile" independent of concurrent surface pressure, this refactor will not help. The experiment will tell us.

## Observed Failure Data

- **Failing allocation**: `ane::client::compile_network` → `nsdata_on_surface("mil", ...)` → `IOSurface::initWithProperties` returns nil
- **Requested size**: 11,865 bytes (11.6 KB)
- **Error**: `IOSurfaceAlloc { context: "mil", byte_count: 11865, width: 11865, bpr: 11865 }`
- **Kernel**: `sdpaFwd`
- **Behavior**: nondeterministic — same code path sometimes passes, sometimes fails

## Status of Prior Work

- [x] Forward-only kernel compile path — already landed in [layer.rs L756–812](file:///Users/andrewgordon/RustRover-Projects/rustane/crates/engine/src/layer.rs#L756)
- [x] `catch_unwind` retry wrapper — landed in [parallel_bench.rs](file:///Users/andrewgordon/RustRover-Projects/rustane/crates/engine/src/parallel_bench.rs#L112)
- [x] ane-side `Result`-based `IOSurfaceAlloc` error — landed in ane `compile-fixed` branch
- [ ] **Split compile/allocate phases in `parallel_bench.rs`** — this experiment
- [ ] Fix `tensor_data_new_logged` byte count (fp32 → fp16)

## Proposed Changes

### [MODIFY] [parallel_bench.rs](file:///Users/andrewgordon/RustRover-Projects/rustane/crates/engine/src/parallel_bench.rs)

Split each `compile_*_runner` function into two phases:

| Function | Phase 1 (compile only) | Phase 2 (allocate + assemble) |
|---|---|---|
| `compile_sdpa_runner` (L561) | `compile_sdpa_exe(cfg) -> Result<Executable>` | `allocate_sdpa_buffers(exe, cfg) -> SdpaRunner` |
| `compile_wo_runner` (L577) | `compile_wo_exe(cfg) -> Result<Executable>` | `allocate_wo_buffers(exe, cfg) -> WoRunner` |
| `compile_baseline_ffn_runner` (L628) | `compile_baseline_ffn_exes(cfg) -> Result<(Exe, Exe, bool)>` | `allocate_baseline_ffn_buffers(...) -> BaselineFfnRunner` |
| `compile_sharded_attention_runner` (L606) | Split per worker similarly | |
| `compile_sharded_ffn_runner` (L683) | Split per worker similarly | |

Restructure `compile_mode_runner` (L733):
```
Phase 1: compile all executables (no TensorData alive)
Phase 2: allocate all TensorData buffers, assemble runners
```

### [MODIFY] [layer.rs](file:///Users/andrewgordon/RustRover-Projects/rustane/crates/engine/src/layer.rs)

Fix `tensor_data_new_logged` (L35): change `elements * size_of::<f32>()` to `elements * 2` (fp16 = 2 bytes per element).

## Verification Plan

### Primary Gates (branch's current reduced battery)

```bash
# Gate 1: forward ladder
make forward-ladder

# Gate 2: sharded forward ladder
make forward-ladder ATTN_SHARDS=10 FFN_SHARDS=10

# Gate 3: 30B smoke (~30s)
make forward-ceiling
```

### IOSurface Repro Path (optional, for testing this specific hypothesis)

```bash
# Run fwd_7b multiple times (5-10x) to check reduced failure rate
cargo test -p engine --test bench_fwd_only_scale --release -- --ignored --nocapture fwd_7b
```

### Existing Test Suite

```bash
cargo test -p engine --release
```

> [!IMPORTANT]
> Due to the nondeterministic nature of this bug, a single passing run is not conclusive. Multiple runs of the repro path are needed to assess whether the refactor reduces failure frequency.
