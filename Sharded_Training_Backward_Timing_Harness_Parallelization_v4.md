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

Per-dispatch timing captures each individual ANE round-trip within a shard. FFN backward has 2 dispatches per shard (W2, W13t); attention backward has 5 (wot, bwd1, bwd2, q_bwd, kv_bwd):

```rust
/// Timing for a single ANE dispatch round-trip within a shard.
#[derive(Debug, Clone, Default)]
pub struct DispatchTiming {
    pub kernel_name: &'static str,
    pub staging_ms: f32,
    pub ane_wall_ms: f32,           // wall clock around dispatch (always populated)
    pub ane_hw_ns: Option<u64>,     // from run_cached_with_stats (None in direct mode)
    pub readback_ms: f32,
}

/// All dispatches + CPU work for one shard in one backward call.
#[derive(Debug, Clone, Default)]
pub struct ShardTiming {
    pub shard_idx: usize,
    pub dispatches: Vec<DispatchTiming>,
    pub cpu_grad_accum_ms: f32,
    pub cpu_silu_ms: f32,           // FFN only
    pub shard_total_ms: f32,        // outer Instant for entire shard
}

/// Aggregate timing for one sharded backward call.
#[derive(Debug, Clone, Default)]
pub struct ShardedBackwardTimings {
    // FFN backward
    pub ffn_scale_dy_ms: f32,
    pub ffn_shards: Vec<ShardTiming>,
    pub ffn_grad_scatter_ms: f32,    // post-loop scatter_dw_columns for dw1/dw2/dw3
    pub ffn_dx_merge_ms: f32,        // post-loop dx_ffn vadd reduction
    pub ffn_wall_ms: f32,            // outer wall (end-to-end including reduction)
    pub ffn_sum_ane_hw_ms: f32,
    pub ffn_sum_ane_wall_ms: f32,
    pub ffn_sum_cpu_ms: f32,
    // Attention backward
    pub attn_transpose_ms: f32,
    pub attn_shards: Vec<ShardTiming>,
    pub attn_grad_scatter_ms: f32,   // post-loop scatter for dwq/dwk/dwv/dwo
    pub attn_dx_merge_ms: f32,       // post-loop dx_merged vadd reduction
    pub attn_wall_ms: f32,
    pub attn_sum_ane_hw_ms: f32,
    pub attn_sum_ane_wall_ms: f32,
    pub attn_sum_cpu_ms: f32,
    // Derived
    pub host_overhead_ms: f32,       // sum_ane_wall - sum_ane_hw
}
```

> [!NOTE]
> **Post-loop reduction is timed separately** via `ffn_grad_scatter_ms` / `ffn_dx_merge_ms` (and attention equivalents). After Phase 2 introduces parallel dispatch, these buckets may become the new bottleneck. Without them, Phase 3 re-measurement would misattribute reduction cost to the shard loop and overstate the amount of useful overlap.
>
> `ffn_wall_ms` is the outer `Instant` around the **entire** FFN backward including reduction. In Phase 1 (serial): `ffn_wall_ms ≈ sum(shard_totals) + grad_scatter + dx_merge`. After Phase 2 (parallel): `ffn_wall_ms ≈ max(shard_totals) + grad_scatter + dx_merge`. The ratio quantifies effective overlap.

### Dual-mode timing API

#### [MODIFY] [layer.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs)

`run_cached_with_stats` has measurable overhead vs `run_cached_direct` (see [bench_hw_execution_time.rs L47–66](file:///Users/USER/RustRover-Projects/rustane/crates/engine/tests/bench_hw_execution_time.rs#L47-L66)). The public wrapper takes an explicit mode:

```rust
/// Controls whether the timing harness uses run_cached_direct or run_cached_with_stats.
#[derive(Debug, Clone, Copy)]
pub enum TimingMode {
    /// Wall clock only (Instant around run_cached_direct). Production-equivalent latency.
    WallOnly,
    /// Wall clock + ANE hardware ns (run_cached_with_stats). Higher overhead.
    WallAndHw,
}

/// Run one sharded backward step with per-dispatch timing.
/// Benchmark-only entry point — not used by the training loop.
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

In `WallOnly` mode, `DispatchTiming.ane_hw_ns` is `None` and `ane_wall_ms` reflects production-equivalent latency. In `WallAndHw` mode, `ane_hw_ns` is `Some(ns)` but `ane_wall_ms` includes the stats collection overhead. The harness calls both modes and reports side-by-side.

### Benchmark test

#### [NEW] [bench_sharded_backward_timing.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/tests/bench_sharded_backward_timing.rs)

An `#[ignore]`d test:

1. 600M config (dim=1536, hidden=4096, heads=20, seq=512)
2. Construct runtimes explicitly — `ShardedFfnBackwardRuntime::compile()`, `ShardedAttentionBackwardRuntime::compile()` — for FFN_SHARDS ∈ {2,4}, ATTN_SHARDS ∈ {2}
3. 2 warmup + 5 timed steps, calling `backward_into_sharded_timed` **twice per step**: once `WallOnly`, once `WallAndHw`
4. Print per-dispatch + per-shard detail, with separate columns for direct-wall and stats-wall
5. Write JSON to `results/sharded_backward_timing/`

```bash
cargo test -p engine --test bench_sharded_backward_timing --release -- --ignored --nocapture
```

---

## Phase 2 — Parallelize FFN backward shard loop

> [!WARNING]
> The current serial loop at L1655 writes into shared `grads.dw1/dw2/dw3` via `accumulate_dw` + `scatter_dw_columns` inside the per-shard iteration. Parallelizing requires worker-local gradient temporaries and a post-loop reduction step.

#### [MODIFY] [layer.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs)

**Changes to `ShardedFfnBackwardWorker` (L724):**

```rust
pub struct ShardedFfnBackwardWorker {
    // ... existing fields ...
    dw_tmp: Vec<f32>,       // existing: [dim * shard_hidden]
    dw1_local: Vec<f32>,    // NEW: [dim * shard_hidden]
    dw2_local: Vec<f32>,    // NEW: [dim * shard_hidden]
    dw3_local: Vec<f32>,    // NEW: [dim * shard_hidden]
}
```

> [!CAUTION]
> `accumulate_dw` (L4751) uses `beta=1.0` (additive). Worker-local buffers must be zeroed before each backward call:
> ```rust
> worker.dw1_local.fill(0.0);
> worker.dw2_local.fill(0.0);
> worker.dw3_local.fill(0.0);
> ```

**Changes to `run_sharded_ffn_backward_into` (L1637):**

Replace the serial `for` loop (L1655–1768) with `thread::scope` + barrier. Each worker writes only to its own `dw*_local` buffers. Post-loop reduction scatters into `grads.*` and merges `dx_ffn`. See Phase 2 of v3 plan for full pseudocode — unchanged.

### Phase 2 verification

#### [NEW] [test_sharded_ffn_backward_correctness.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/tests/test_sharded_ffn_backward_correctness.rs)

> [!IMPORTANT]
> **Sharding is NOT a schedule-only change.** Unlike the `auto_move_dw2` and `auto_fuse_qbwd` optimizations (which rearrange identical FP operations), sharding changes ANE tile sizes — each shard does `shard_hidden`-wide matmuls instead of `hidden`-wide. The ANE compiler may tile/quantize differently, producing non-identical FP results. **Bit-exact comparison between sharded and non-sharded paths is not achievable.**
>
> However, the Phase 2 rewrite (serial → parallel) within the sharded path IS a schedule-only change. The problem is that after the rewrite, the serial implementation no longer exists as a callable path — both routes through `backward_into_with_training_ffn` (L2014) will execute the parallel code.

**Correctness strategy (two-tier):**

**Tier 1 — Toleranced: sharded vs non-sharded baseline.** Compares `backward_into_with_training_ffn(sharded_ffn=Some(...))` against `backward_into(no sharding)`. Uses tolerances because sharding genuinely changes the computation:
- Max abs diff ≤ 2e-2
- Mean abs diff ≤ 5e-3
- Cosine similarity ≥ 0.999

Compares **all** gradient fields: `dx`, `dw1/dw2/dw3`, `dwo/dwq/dwk/dwv`, `dgamma1/dgamma2` (FFN backward feeds attention via RMSNorm2 at [L2097](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L2097)).

**Tier 2 — Bit-exact idempotency.** Two consecutive runs of the sharded path with fresh `LayerGrads` produce **bit-identical** output (`to_bits()` comparison). This catches:
- Write races from incorrect parallelization
- Stale IOSurface buffer data
- Non-deterministic merge order

**Tier 3 — Accumulation.** Two backward calls into the same `grads` produce `2× single_call` within `1e-4` relative tolerance.

All three tiers are non-`#[ignore]`d. Construct runtimes explicitly via `ShardedFfnBackwardRuntime::compile(cfg, 4)` — no env vars.

Then sweep gate:
```bash
make sweep-600m FFN_SHARDS=4
make sweep-600m   # baseline unaffected
```

---

## Phase 3 — Re-measure with timing harness

Re-run Phase 1 harness after Phase 2 lands. Key metrics to compare:

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

#### [MODIFY] [layer.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs)

**`ShardedAttentionBackwardWorker` additions:**

```rust
dwq_local: Vec<f32>,   // [dim * q_dim_shard], column-sharded
dwk_local: Vec<f32>,   // [dim * kv_dim_shard], column-sharded
dwv_local: Vec<f32>,   // [dim * kv_dim_shard], column-sharded
dwo_local: Vec<f32>,   // [q_dim_shard * dim], ROW-sharded (not column!)
```

> [!CAUTION]
> **`dwo` is `[q_dim × dim]`, row-sharded by `q_col_start`.** The baseline computes it at [L2163](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L2163) as `accumulate_dw(attn_out, q_dim, dx2_scaled, dim, seq, dwo)`. Each shard sees `attn_out_shard [q_dim_shard × seq]`, producing `dwo_local [q_dim_shard × dim]` — contiguous row slices. Needs `scatter_dw_rows` (simple memcpy), not `scatter_dw_columns`.

> [!IMPORTANT]
> The current sharded attention backward (L1792–2010) **never computes `dwo`**. Phase 4 must add per-worker `dwo_local` computation.

All worker-local gradient buffers must be zeroed each call. Weight transposes (L1785–1788) must complete before thread spawn.

### Phase 4 verification

Same two-tier strategy as Phase 2: toleranced sharded-vs-baseline + bit-exact idempotency. All gradient fields compared. Sweep gate:
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
