# Split Compile/Allocate Phases in parallel_bench.rs

## Problem

The ane debrief ([debrief-round-2.md](file:///Users/andrewgordon/other-projects/ane/debrief-round-2.md)) confirmed that compile-payload IOSurface allocation works fine in isolation. The failure requires **compile-payload surfaces and runtime TensorData surfaces to be alive simultaneously** — which is exactly what `parallel_bench.rs` does.

Current flow in `compile_mode_runner`:
1. `compile_sdpa_runner` → compile sdpa_fwd exe → allocate **8 TensorData surfaces** → return
2. `compile_wo_runner` → compile wo_fwd exe (while 8 SDPA surfaces alive) → allocate **2 more surfaces** → return
3. `compile_baseline_ffn_runner` → compile 2 FFN exes (while 10 surfaces alive) → allocate **4 more surfaces** → return

By step 3, the FFN compilation's MIL payload IOSurface competes with ~10 already-live TensorData IOSurfaces for shared kernel resources.

Target flow (following `CompiledKernels::compile` pattern in [layer.rs](file:///Users/andrewgordon/RustRover-Projects/rustane/crates/engine/src/layer.rs#L688)):
1. Compile **all** executables (sdpa, wo, w13, w2) — compile payloads live only during each `.compile()` call
2. Allocate **all** TensorData surfaces after all compilation finishes

## Proposed Changes

### Engine

---

#### [MODIFY] [parallel_bench.rs](file:///Users/andrewgordon/RustRover-Projects/rustane/crates/engine/src/parallel_bench.rs)

Split each `compile_*_runner` function into two phases:

**`compile_sdpa_runner`** (L561) → split into:
- `compile_sdpa_exe(cfg) -> Result<Executable, AneError>` — compile only
- `allocate_sdpa_buffers(exe, cfg) -> SdpaRunner` — allocate TensorData, assemble struct

**`compile_wo_runner`** (L577) → split into:
- `compile_wo_exe(cfg) -> Result<Executable, AneError>` — compile only
- `allocate_wo_buffers(exe, cfg) -> WoRunner` — allocate TensorData, assemble struct

**`compile_baseline_ffn_runner`** (L628) → split into:
- `compile_baseline_ffn_exes(cfg) -> Result<(Executable, Executable, bool), AneError>` — compile w13 + w2
- `allocate_baseline_ffn_buffers(w13_exe, w2_exe, use_dual_w13, cfg) -> BaselineFfnRunner` — allocate

**`compile_sharded_attention_runner`** (L606) → split similarly per worker

**`compile_sharded_ffn_runner`** (L683) → split similarly per worker

**`compile_mode_runner`** (L733) → restructure to:
```
Phase 1: compile all executables
Phase 2: allocate all TensorData buffers, assemble runners
```

---

#### [MODIFY] [layer.rs](file:///Users/andrewgordon/RustRover-Projects/rustane/crates/engine/src/layer.rs)

Fix `tensor_data_new_logged` byte logging: currently computes `elements * size_of::<f32>()` (4 bytes) but `TensorData` surfaces are fp16-sized (2 bytes). Change to `elements * 2`.

## Verification Plan

### Manual Verification

The primary test is `make forward-7b` — the command that has been failing nondeterministically:

```bash
make forward-7b
```

- **Before this change**: fails nondeterministically with `IOSurfaceAlloc` error
- **After this change**: should pass reliably since compile payloads no longer overlap with runtime surfaces

Also run:
```bash
# Forward ladder to confirm no regression
make forward-ladder

# Sharded modes if available
ATTN_SHARDS=2 make forward-7b
FFN_SHARDS=4 make forward-7b
```

### Existing Tests

```bash
# Full engine test suite
cargo test -p engine --release

# Forward-only scale tests
cargo test -p engine --test bench_fwd_only_scale --release -- --ignored --nocapture fwd_7b
```

> [!IMPORTANT]
> Due to the nondeterministic nature of this bug, a single passing run is not conclusive. The user should run `make forward-7b` multiple times (5–10x) to confirm reduced failure rate.
