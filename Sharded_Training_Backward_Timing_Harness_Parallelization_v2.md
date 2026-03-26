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

#### [MODIFY] [layer.rs](file:///Users/andrewgordon/RustRover-Projects/rustane/crates/engine/src/layer.rs)

Add new public structs alongside the existing `ForwardTimings` (L3304) and `BackwardTimings` (L3640).

Per-dispatch timing captures each individual ANE round-trip within a shard. FFN backward has 2 dispatches per shard (W2, W13t); attention backward has 5 (wot, bwd1, bwd2, q_bwd, kv_bwd):

```rust
/// Timing for a single ANE dispatch round-trip within a shard.
#[derive(Debug, Clone, Default)]
pub struct DispatchTiming {
    pub kernel_name: &'static str,  // "w2", "w13t", "wot", "bwd1", "bwd2", "q_bwd", "kv_bwd"
    pub staging_ms: f32,            // CPU: copy_from_f32 + stage_spatial + stage_weight_columns
    pub ane_wall_ms: f32,           // wall clock around run_cached_with_stats
    pub ane_hw_ns: u64,             // from run_cached_with_stats
    pub readback_ms: f32,           // CPU: as_f32_slice + copy_from_slice
}

/// All dispatches + CPU work for one shard in one backward call.
#[derive(Debug, Clone, Default)]
pub struct ShardTiming {
    pub shard_idx: usize,
    pub dispatches: Vec<DispatchTiming>,
    pub cpu_grad_accum_ms: f32,     // accumulate_dw + scatter_dw_columns
    pub cpu_silu_ms: f32,           // SiLU derivative (FFN only)
    pub shard_total_ms: f32,        // wall clock for entire shard (outer Instant)
}
```

Aggregate view for the whole backward call. Records both sum-of-parts (for bucket analysis) and outer wall clock (for actual end-to-end time, which diverges from sum once shards overlap in Phase 2/4):

```rust
/// Aggregate timing for one sharded backward call.
#[derive(Debug, Clone, Default)]
pub struct ShardedBackwardTimings {
    // FFN backward
    pub ffn_scale_dy_ms: f32,
    pub ffn_shards: Vec<ShardTiming>,
    pub ffn_merge_ms: f32,
    pub ffn_wall_ms: f32,               // outer wall clock (end-to-end)
    pub ffn_sum_ane_hw_ms: f32,          // sum of all ane_hw across shards
    pub ffn_sum_ane_wall_ms: f32,        // sum of all ane_wall across shards
    pub ffn_sum_cpu_ms: f32,             // sum of all CPU buckets
    // Attention backward
    pub attn_transpose_ms: f32,
    pub attn_shards: Vec<ShardTiming>,
    pub attn_merge_ms: f32,
    pub attn_wall_ms: f32,
    pub attn_sum_ane_hw_ms: f32,
    pub attn_sum_ane_wall_ms: f32,
    pub attn_sum_cpu_ms: f32,
    // Derived (once shards are parallel, sum > wall reveals overlap)
    pub host_overhead_ms: f32,           // sum_ane_wall - sum_ane_hw
}
```

> [!NOTE]
> Recording both `ffn_wall_ms` (outer `Instant` around the entire shard loop) and the per-dispatch sums lets us distinguish "serial dispatch overhead" from "actual ANE contention" after Phase 2 introduces overlap. In Phase 1 measurements, `ffn_wall_ms ≈ sum(shard totals)` since everything is serial. After parallelization, `ffn_wall_ms < sum(shard totals)` and the ratio quantifies effective overlap.

Add `_with_stats` variants of the sharded backward functions:

- `run_sharded_ffn_backward_into_with_stats` — wraps each `run_cached_direct` call at L1679–1682 (W2) and L1755–1758 (W13t) with `Instant` timing and swaps to `run_cached_with_stats`, populating a `DispatchTiming` per dispatch and a `ShardTiming` per shard
- `run_sharded_attention_backward_into_with_stats` — same pattern for the 5 ANE dispatches per shard: wot (L1815), bwd1 (L1860), bwd2 (L1907), q_bwd (L1944), kv_bwd (L1968)

These variants mirror the existing dispatch code but add `Instant::now()` guards around each stage/dispatch/readback/grad-accum section. They do **not** change the computation.

> [!IMPORTANT]
> `run_cached_with_stats` has measurable overhead vs `run_cached_direct` (see [bench_hw_execution_time.rs L47–66](file:///Users/andrewgordon/RustRover-Projects/rustane/crates/engine/tests/bench_hw_execution_time.rs#L47-L66), which treats them as distinct measurements). The harness should run two passes: one with `run_cached_direct` + `Instant` for production-equivalent wall timing, and one with `run_cached_with_stats` for HW ns. Report both.

### Benchmark test

#### [NEW] [bench_sharded_backward_timing.rs](file:///Users/andrewgordon/RustRover-Projects/rustane/crates/engine/tests/bench_sharded_backward_timing.rs)

An `#[ignore]`d test following the pattern of [bench_hw_execution_time.rs](file:///Users/andrewgordon/RustRover-Projects/rustane/crates/engine/tests/bench_hw_execution_time.rs):

1. Use the 600M config matching `sweep-600m` (dim=1536, hidden=4096, heads=20, seq=512 — from [Makefile L101](file:///Users/andrewgordon/RustRover-Projects/rustane/Makefile#L101))
2. Compile baseline kernels + sharded runtimes for FFN_SHARDS ∈ {2,4} and ATTN_SHARDS ∈ {2}
   - `FFN_SHARDS=1` is invalid per [full_model.rs L121–123](file:///Users/andrewgordon/RustRover-Projects/rustane/crates/engine/src/full_model.rs#L121-L123); omit to run baseline
3. For each config, run 2 warmup + 5 timed steps calling both direct-timing and stats-timing variants
4. Print per-dispatch detail:

```
=== FFN backward, FFN_SHARDS=4, 600M ===
  shard 0:
    w2:    stage=X.Xms  ane_wall=X.Xms  ane_hw=XXXXns  readback=X.Xms
    silu:  X.Xms
    w13t:  stage=X.Xms  ane_wall=X.Xms  ane_hw=XXXXns  readback=X.Xms
    grad_accum: X.Xms
    shard_total: X.Xms
  shard 1:
    ...
  merge: X.Xms
  ffn_wall (end-to-end): X.Xms
  ---
  sum(ane_hw): X.Xms | sum(ane_wall): X.Xms | host_overhead: X.Xms | sum(cpu): X.Xms
```

5. Write JSON results to `results/sharded_backward_timing/`
6. Print summary table: baseline vs sharded, percentage breakdown by bucket

```bash
cargo test -p engine --test bench_sharded_backward_timing --release -- --ignored --nocapture
```

---

## Phase 2 — Parallelize FFN backward shard loop

> [!WARNING]
> The current serial loop at L1655 writes into shared `grads.dw1/dw2/dw3` via `accumulate_dw` + `scatter_dw_columns` inside the per-shard iteration. Parallelizing requires worker-local gradient temporaries and a post-loop reduction step, otherwise there will be write races and nondeterministic gradients.

#### [MODIFY] [layer.rs](file:///Users/andrewgordon/RustRover-Projects/rustane/crates/engine/src/layer.rs)

**Changes to `ShardedFfnBackwardWorker` (L724):**

The existing `dw_tmp` field (L737) is `[dim × shard_hidden]` and reused across multiple `accumulate_dw` calls within the loop. Add three persistent worker-local gradient buffers:

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
> `accumulate_dw` (L4751) calls `vdsp::sgemm_at` which uses `beta=1.0` — it **adds** into its output buffer. These worker-local buffers must be zeroed at the top of every backward call before `accumulate_dw` writes to them:
> ```rust
> worker.dw1_local.fill(0.0);
> worker.dw2_local.fill(0.0);
> worker.dw3_local.fill(0.0);
> ```
> Without this, gradients leak across training steps and layers.

**Changes to `run_sharded_ffn_backward_into` (L1637):**

Replace the serial `for` loop (L1655–1768) with `thread::scope`:

```rust
thread::scope(|scope| {
    let barrier = Arc::new(Barrier::new(shard_count + 1));
    let mut handles = Vec::with_capacity(shard_count);

    for (worker, layout) in workers.iter_mut().zip(layouts.iter()) {
        let b = Arc::clone(&barrier);
        handles.push(scope.spawn(move || {
            // Zero worker-local gradient buffers
            worker.dw1_local.fill(0.0);
            worker.dw2_local.fill(0.0);
            worker.dw3_local.fill(0.0);

            b.wait();
            // 1. Stage W2 shard weights + dffn activations → worker.w2_in
            // 2. run_cached_direct W2
            // 3. SiLU derivative (worker-local h1/h3/gate/dsilu_raw/dh1/dh3)
            // 4. accumulate_dw into worker.dw1_local/dw2_local/dw3_local (NOT into grads.*)
            // 5. Stage W13t shard → worker.w13t_in
            // 6. run_cached_direct W13t
        }));
    }

    barrier.wait();
    for handle in handles {
        handle.join().expect("FFN backward shard panicked");
    }
});

// Post-loop reduction: scatter worker-local grads into full gradient tensors
for (worker, layout) in workers.iter().zip(layouts.iter()) {
    scatter_dw_columns(&mut grads.dw1, dim, hidden, &worker.dw1_local, shard_hidden, layout.col_start);
    scatter_dw_columns(&mut grads.dw2, dim, hidden, &worker.dw2_local, shard_hidden, layout.col_start);
    scatter_dw_columns(&mut grads.dw3, dim, hidden, &worker.dw3_local, shard_hidden, layout.col_start);
}

// dx_ffn merge: sum partial dx from each shard's W13t output
runtime.dx_ffn.fill(0.0);
for worker in &workers {
    let locked = worker.w13t_out.as_f32_slice();
    vdsp::vadd(&runtime.dx_ffn, &locked[..dim * seq], &mut runtime.merge_tmp);
    runtime.dx_ffn.copy_from_slice(&runtime.merge_tmp);
}
```

**Thread-safety of shared reads:**
- `ws.dffn` — written once at L1652, read-only during shard loop ✓
- `cache.h1/h3/gate/x2norm` — forward cache, immutable ✓
- `weights.w1/w2/w3` — read-only ✓

### Phase 2 verification

> [!IMPORTANT]
> Sweep-level loss convergence alone is too coarse to catch race conditions or gradient mis-scattering. Before using `sweep-600m` as a secondary gate, add an exact correctness smoke test.

#### [NEW] [test_sharded_backward_correctness.rs](file:///Users/andrewgordon/RustRover-Projects/rustane/crates/engine/tests/test_sharded_backward_correctness.rs)

Following the existing pattern in [auto_parallel_grad_norm.rs](file:///Users/andrewgordon/RustRover-Projects/rustane/crates/engine/tests/auto_parallel_grad_norm.rs) and [bench_attn_ffn_parallel_full_model.rs L1394](file:///Users/andrewgordon/RustRover-Projects/rustane/crates/engine/tests/bench_attn_ffn_parallel_full_model.rs#L1394):

1. Run one forward+backward step using the baseline path (`backward_into`)
2. Run the same step using the sharded path (`backward_into_with_training_ffn`, `FFN_SHARDS=4`)
3. Compare `dx`, `dw1`, `dw2`, `dw3`, `dgamma2` element-wise:
   - Max abs diff ≤ 2e-2
   - Mean abs diff ≤ 5e-3
   - Cosine similarity ≥ 0.999
4. Run twice and compare outputs to verify determinism
5. Non-`#[ignore]`d — runs as part of `cargo test -p engine`

Then use `sweep-600m` as the secondary gate:
```bash
make sweep-600m FFN_SHARDS=4
make sweep-600m   # baseline unaffected
```

---

## Phase 3 — Re-measure with timing harness

Re-run the Phase 1 harness after Phase 2 lands:

```bash
cargo test -p engine --test bench_sharded_backward_timing --release -- --ignored --nocapture
```

**Expected outcome:** With shards running in parallel, `ffn_wall_ms` should drop well below `sum(shard_total_ms)`, and `ffn_wall_ms` should approach `max(shard_total_ms) + merge_ms`. If `ffn_wall_ms` doesn't collapse, the data will point to staging, dispatch overhead, or ANE contention as the next target.

---

## Phase 4 — Parallelize attention backward shard loop

Same `thread::scope` + barrier + worker-local-grad + post-loop-reduction pattern as Phase 2, applied to `run_sharded_attention_backward_into` (L1771).

#### [MODIFY] [layer.rs](file:///Users/andrewgordon/RustRover-Projects/rustane/crates/engine/src/layer.rs)

**Changes to `ShardedAttentionBackwardWorker` (L748):**

Each worker already has its own `dw_tmp` (L775), `da`, `dv`, `dq`, `dk`, `dx_attn`, `dx_kv` — correctly scoped per-worker. Add worker-local gradient buffers:

```rust
pub struct ShardedAttentionBackwardWorker {
    // ... existing fields ...
    dwq_local: Vec<f32>,   // NEW: [dim * q_dim_shard], column-sharded
    dwk_local: Vec<f32>,   // NEW: [dim * kv_dim_shard], column-sharded
    dwv_local: Vec<f32>,   // NEW: [dim * kv_dim_shard], column-sharded
    dwo_local: Vec<f32>,   // NEW: [q_dim_shard * dim], ROW-sharded
}
```

> [!CAUTION]
> **`dwo` is row-sharded, not column-sharded.** The baseline `dwo` computation at [L2163](file:///Users/andrewgordon/RustRover-Projects/rustane/crates/engine/src/layer.rs#L2163) is `accumulate_dw(attn_out, q_dim, dx2_scaled, dim, seq, &mut grads.dwo)`, producing `[q_dim × dim]`. When attention is sharded by heads, each worker sees only `attn_out_shard` of shape `[q_dim_shard × seq]`, so `dwo_local` is `[q_dim_shard × dim]` — a contiguous **row slice** of the full `[q_dim × dim]` gradient.
>
> This means the post-loop reduction needs a **row-oriented scatter** (simple `memcpy` of contiguous row blocks), **not** the column-oriented `scatter_dw_columns` used for `dwq/dwk/dwv`. Add a `scatter_dw_rows` helper:
> ```rust
> fn scatter_dw_rows(
>     dst: &mut [f32], total_rows: usize, cols: usize,
>     src: &[f32], src_rows: usize, row_start: usize,
> ) {
>     let offset = row_start * cols;
>     dst[offset..offset + src_rows * cols].copy_from_slice(&src[..src_rows * cols]);
> }
> ```

> [!IMPORTANT]
> The current sharded attention backward path (L1792–2010) **never computes `dwo`** — it only handles `dwq`, `dwk`, `dwv`. The baseline non-sharded path computes `dwo` at [L2163](file:///Users/andrewgordon/RustRover-Projects/rustane/crates/engine/src/layer.rs#L2163) via `accumulate_dw(attn_out, q_dim, dx2_scaled, dim, seq, dwo)`. Phase 4 must add this computation inside each worker:
> ```rust
> worker.dwo_local.fill(0.0);
> // attn_out_shard = cache.attn_out[q_col_start*seq .. (q_col_start+q_dim_shard)*seq]
> accumulate_dw(attn_out_shard, q_dim_shard, &ws.dx2_scaled, dim, seq, &mut worker.dwo_local);
> ```
> Then scatter with `scatter_dw_rows` using `q_col_start` as the row offset.

**All worker-local gradient buffers must be zeroed** at the top of each backward call (same `accumulate_dw` additivity issue as Phase 2).

**Changes to `run_sharded_attention_backward_into` (L1771):**
- Weight transposes at L1785–1788 (`wot`, `wqt`, `wkt`, `wvt`) must happen **before** spawning threads — they write into shared `ws.*t` buffers that threads read
- Replace serial `for worker in &mut runtime.workers` (L1792) with `thread::scope` + barrier
- Post-loop: scatter worker-local `dwq/dwk/dwv` via `scatter_dw_columns`, scatter `dwo` via `scatter_dw_rows`, merge `dx_merged` from worker `dx_attn + dx_kv`

### Phase 4 verification

Same pattern as Phase 2: exact `dx/dw*/dgamma*` comparison test, then sweep gate:

```bash
make sweep-600m ATTN_SHARDS=2
make sweep-600m ATTN_SHARDS=2 FFN_SHARDS=4
```

---

## Data collection checkpoints

After each phase, record the `sweep-600m` numbers to track progress:

| Config | Phase 0 (current) | Phase 1 | Phase 2 | Phase 4 |
| --- | --- | --- | --- | --- |
| Baseline | 1046 ms/step, bwd=710 | — | — | — |
| ATTN_SHARDS=2 | 1541 ms/step, bwd=1174 | — | — | — |
| FFN_SHARDS=4 | 2993 ms/step, bwd=1978 | — | — | — |
| Both | 3297 ms/step, bwd=2258 | — | — | — |
