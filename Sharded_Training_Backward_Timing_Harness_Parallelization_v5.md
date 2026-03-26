# Sharded Training Backward Timing Harness + Parallelization

Sharding regressions from `sweep-600m` show backward time ballooning from 710ms (baseline) to 1174ms (ATTN_SHARDS=2), 1978ms (FFN_SHARDS=4), and 2258ms (both). This plan instruments the sharded backward paths to isolate the bottleneck, then parallelizes the serial shard loops.

## Phased approach

| Phase | Goal | Deliverable | Touches training path? |
| --- | --- | --- | --- |
| 1 | Timing harness | New benchmark test | No |
| 2 | FFN backward parallelization | Production code change | Yes |
| 3 | Re-measure | Harness re-run + data | No |
| 4 | Attention backward parallelization | Production code change | Yes |

> [!IMPORTANT]
> Phase 2 should not start until Phase 1 data confirms the serial loop is the dominant contributor. Phase 4 should not start until Phase 3 data confirms FFN backward regression collapsed.

---

## Phase 1 — Sharded backward timing harness

### Timing structs

#### [MODIFY] [layer.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs)

```rust
#[derive(Debug, Clone, Default)]
pub struct DispatchTiming {
    pub kernel_name: &'static str,
    pub staging_ms: f32,
    pub ane_wall_ms: f32,
    pub ane_hw_ns: Option<u64>,     // None in WallOnly mode
    pub readback_ms: f32,
}

#[derive(Debug, Clone, Default)]
pub struct ShardTiming {
    pub shard_idx: usize,
    pub dispatches: Vec<DispatchTiming>,
    pub cpu_grad_accum_ms: f32,
    pub cpu_silu_ms: f32,
    pub shard_total_ms: f32,
}

#[derive(Debug, Clone, Default)]
pub struct ShardedBackwardTimings {
    // FFN backward
    pub ffn_scale_dy_ms: f32,
    pub ffn_shards: Vec<ShardTiming>,
    pub ffn_grad_scatter_ms: f32,
    pub ffn_dx_merge_ms: f32,
    pub ffn_wall_ms: f32,
    pub ffn_sum_ane_wall_ms: f32,
    pub ffn_sum_cpu_ms: f32,
    pub ffn_sum_ane_hw_ms: Option<f32>,     // None in WallOnly mode
    // Attention backward
    pub attn_transpose_ms: f32,
    pub attn_shards: Vec<ShardTiming>,
    pub attn_grad_scatter_ms: f32,
    pub attn_dx_merge_ms: f32,
    pub attn_wall_ms: f32,
    pub attn_sum_ane_wall_ms: f32,
    pub attn_sum_cpu_ms: f32,
    pub attn_sum_ane_hw_ms: Option<f32>,    // None in WallOnly mode
    // Derived (populated only in WallAndHw mode)
    pub host_overhead_ms: Option<f32>,      // sum_ane_wall - sum_ane_hw
}
```

> [!NOTE]
> HW-derived fields (`*_sum_ane_hw_ms`, `host_overhead_ms`) are `Option<f32>` — `None` in `WallOnly` mode, `Some` in `WallAndHw` mode. The harness prints "n/a" for these columns in `WallOnly` results.

Post-loop reduction is timed separately via `ffn_grad_scatter_ms` / `ffn_dx_merge_ms` (and attention equivalents). After Phase 2, these may become the new bottleneck; without separate buckets, Phase 3 re-measurement would misattribute reduction cost.

### Dual-mode timing API

```rust
#[derive(Debug, Clone, Copy)]
pub enum TimingMode {
    /// Instant around run_cached_direct. Production-equivalent latency.
    WallOnly,
    /// Instant around run_cached_with_stats. Higher overhead, provides ane_hw_ns.
    WallAndHw,
}

pub fn backward_into_sharded_timed(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &LayerWeights,
    cache: &ForwardCache,
    dy: &[f32],
    grads: &mut LayerGrads,
    ws: &mut BackwardWorkspace,
    dx_out: &mut [f32],
    sharded_attn: Option<&mut ShardedAttentionBackwardRuntime>,
    sharded_ffn: Option<&mut ShardedFfnBackwardRuntime>,
    mode: TimingMode,
) -> ShardedBackwardTimings { ... }
```

### Benchmark test

#### [NEW] [bench_sharded_backward_timing.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/tests/bench_sharded_backward_timing.rs)

`#[ignore]`d. 600M config. Construct runtimes explicitly (no env vars). 2 warmup + 5 timed, calling wrapper twice per step (`WallOnly` + `WallAndHw`). JSON to `results/sharded_backward_timing/`.

```bash
cargo test -p engine --test bench_sharded_backward_timing --release -- --ignored --nocapture
```

---

## Phase 2 — Parallelize FFN backward shard loop

#### [MODIFY] [layer.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs)

**`ShardedFfnBackwardWorker` additions (L724):**

```rust
dw1_local: Vec<f32>,    // [dim * shard_hidden]
dw2_local: Vec<f32>,    // [dim * shard_hidden]
dw3_local: Vec<f32>,    // [dim * shard_hidden]
```

> [!CAUTION]
> `accumulate_dw` uses `beta=1.0` (additive). Worker-local buffers must be zeroed before each backward call.

**Serial reference preservation:**

When the serial loop at L1655 is replaced with `thread::scope`, preserve the original serial implementation under `#[cfg(test)]`:

```rust
/// Serial reference implementation — test-only, stripped from release builds.
/// Used by correctness tests to produce bit-exact reference output for
/// comparison against the parallel implementation.
#[cfg(test)]
fn run_sharded_ffn_backward_into_serial(
    cfg: &ModelConfig, weights: &LayerWeights, cache: &ForwardCache,
    dy: &[f32], grads: &mut LayerGrads, ws: &mut BackwardWorkspace,
    runtime: &mut ShardedFfnBackwardRuntime,
) { /* current serial loop body, unchanged */ }

/// Production parallel implementation.
fn run_sharded_ffn_backward_into(
    cfg: &ModelConfig, weights: &LayerWeights, cache: &ForwardCache,
    dy: &[f32], grads: &mut LayerGrads, ws: &mut BackwardWorkspace,
    runtime: &mut ShardedFfnBackwardRuntime,
) { /* new thread::scope parallel loop */ }
```

> [!NOTE]
> The `#[cfg(test)]` serial reference is compiled only during `cargo test`. No dead code in release builds. The test calls both and compares bit-exactly, directly isolating any regression introduced by the parallelization.

**Expose the serial reference for tests:**

```rust
/// Test-only entry point: runs the serial sharded FFN backward.
#[cfg(test)]
pub fn backward_into_sharded_serial(
    cfg: &ModelConfig, kernels: &CompiledKernels, weights: &LayerWeights,
    cache: &ForwardCache, dy: &[f32], grads: &mut LayerGrads,
    ws: &mut BackwardWorkspace, dx_out: &mut [f32],
    sharded_attn: Option<&mut ShardedAttentionBackwardRuntime>,
    sharded_ffn: Option<&mut ShardedFfnBackwardRuntime>,
) { /* wraps run_sharded_ffn_backward_into_serial */ }
```

**Parallel loop:** `thread::scope` + barrier, worker-local `dw*_local`, post-loop `scatter_dw_columns` + `dx_ffn vadd` merge. See v3/v4 for pseudocode — unchanged.

### Phase 2 verification

#### [NEW] [test_sharded_ffn_backward_correctness.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/tests/test_sharded_ffn_backward_correctness.rs)

Non-`#[ignore]`d. Construct runtimes explicitly (no env vars).

**Test 1 — Bit-exact: parallel vs serial reference.**
Calls `backward_into_with_training_ffn` (parallel, production) and `backward_into_sharded_serial` (`#[cfg(test)]`, serial). Compares all 9 gradient fields + `dx` with `to_bits()` exact match. This is a schedule-only change (same per-shard ops, same merge order), so bit-exact is expected.

**Test 2 — Toleranced: sharded vs non-sharded baseline.**
Calls `backward_into_with_training_ffn(sharded_ffn=Some(...))` vs `backward_into(no sharding)`. All 9 gradient fields. Tolerances: max abs ≤ 2e-2, cosine ≥ 0.999.

**Test 3 — Bit-exact idempotency.**
Two runs of the parallel path with fresh `LayerGrads` produce `to_bits()` identical output. Catches races and non-deterministic merge.

> [!IMPORTANT]
> **Tier 3 accumulation test is removed.** The post-loop `scatter_dw_columns` ([L1499](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L1499)) uses `copy_from_slice` — it **overwrites**, not adds. This means calling `backward_into_with_training_ffn` twice into the same `grads` overwrites the sharded columns rather than doubling them. This is the same behavior as the existing serial sharded path, so it's not a bug — but it means a multi-call accumulation test would fail for the wrong reason. If multi-microbatch accumulation is needed for sharded training in the future, `scatter_dw_columns` must be changed to additive, but that is out of scope for this plan.

Sweep gate:
```bash
make sweep-600m FFN_SHARDS=4
make sweep-600m
```

---

## Phase 3 — Re-measure with timing harness

Re-run Phase 1 harness. Key metrics:

| Metric | Phase 1 (serial) | Phase 3 (parallel) | Interpretation |
| --- | --- | --- | --- |
| `ffn_wall_ms` | ≈ sum(shards) | should drop | main success metric |
| `max(shard_total)` | = last shard | ≈ wall - reduction | useful overlap |
| `ffn_grad_scatter_ms` | small | may grow | new bottleneck? |
| `ffn_dx_merge_ms` | small | unchanged | sequential reduction |
| `sum(ane_wall) / ffn_wall` | ≈ 1.0 | > 1.0 | overlap ratio |

---

## Phase 4 — Parallelize attention backward shard loop

Same pattern as Phase 2. Key differences:

**Worker-local gradient buffers:**

```rust
dwq_local: Vec<f32>,   // [dim * q_dim_shard], column-sharded
dwk_local: Vec<f32>,   // [dim * kv_dim_shard], column-sharded
dwv_local: Vec<f32>,   // [dim * kv_dim_shard], column-sharded
dwo_local: Vec<f32>,   // [q_dim_shard * dim], ROW-sharded
```

> [!CAUTION]
> **`dwo` is row-sharded** (`[q_dim × dim]` partitioned by `q_col_start`). Needs `scatter_dw_rows` (memcpy of contiguous row blocks), not `scatter_dw_columns`.

> [!IMPORTANT]
> Current sharded attention backward (L1792–2010) **never computes `dwo`**. Phase 4 must add per-worker `dwo_local` computation: `accumulate_dw(attn_out_shard, q_dim_shard, dx2_scaled, dim, seq, dwo_local)`.

Serial reference preserved as `#[cfg(test)] run_sharded_attention_backward_into_serial`. All buffers zeroed each call. Weight transposes (L1785–1788) before thread spawn.

### Phase 4 verification

Same three-test structure: bit-exact parallel-vs-serial, toleranced sharded-vs-baseline, bit-exact idempotency. Sweep gate:
```bash
make sweep-600m ATTN_SHARDS=2
make sweep-600m ATTN_SHARDS=2 FFN_SHARDS=4
```

---

## Data collection checkpoints

| Config | Phase 0 (current) | Phase 1 | Phase 2 | Phase 4 |
| --- | --- | --- | --- | --- |
| Baseline | 1046 ms/step, bwd=710 | — | — | — |
| ATTN_SHARDS=2 | 1541 ms/step, bwd=1174 | — | — | — |
| FFN_SHARDS=4 | 2993 ms/step, bwd=1978 | — | — | — |
| Both | 3297 ms/step, bwd=2258 | — | — | — |
