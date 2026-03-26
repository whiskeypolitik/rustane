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

Add two new public structs alongside the existing `ForwardTimings` (L3304) and `BackwardTimings` (L3640):

```rust
/// Per-shard timing for one ANE dispatch round-trip.
#[derive(Debug, Clone, Default)]
pub struct ShardDispatchTiming {
    pub shard_idx: usize,
    pub staging_ms: f32,       // CPU: copy_from_f32 + stage_spatial + stage_weight_columns
    pub ane_wall_ms: f32,      // wall clock around run_cached{_with_stats}
    pub ane_hw_ns: u64,        // from run_cached_with_stats
    pub readback_ms: f32,      // CPU: as_f32_slice + copy_from_slice
    pub grad_accum_ms: f32,    // CPU: accumulate_dw + scatter_dw_columns (FFN only)
}

/// Aggregate timing for one sharded backward call.
#[derive(Debug, Clone, Default)]
pub struct ShardedBackwardTimings {
    // FFN backward buckets
    pub ffn_scale_dy_ms: f32,
    pub ffn_shard_dispatches: Vec<ShardDispatchTiming>,  // one per shard, 2 dispatches each (W2 + W13t)
    pub ffn_silu_ms: f32,
    pub ffn_merge_ms: f32,
    pub ffn_total_ms: f32,
    // Attention backward buckets
    pub attn_transpose_ms: f32,
    pub attn_shard_dispatches: Vec<ShardDispatchTiming>,  // one per shard, 5 dispatches each
    pub attn_rope_bwd_ms: f32,
    pub attn_merge_ms: f32,
    pub attn_total_ms: f32,
    // Derived
    pub total_ane_hw_ms: f32,
    pub total_ane_wall_ms: f32,
    pub total_cpu_ms: f32,
    pub host_overhead_ms: f32,  // ane_wall - ane_hw
}
```

Add `_with_stats` variants of the sharded backward functions:

- `run_sharded_ffn_backward_into_with_stats` — wraps each `run_cached_direct` call at L1679–1682 and L1755–1758 with `Instant` timing and swaps to `run_cached_with_stats`, collecting per-shard `ShardDispatchTiming`
- `run_sharded_attention_backward_into_with_stats` — same pattern for the 5 ANE dispatches per shard at L1815–1818, L1860–1863, L1907–1910, L1944 (q_bwd), and kv_bwd

These variants mirror the existing dispatch code but add `Instant::now()` guards around each stage/dispatch/readback/grad-accum section. They do **not** change the computation — same inputs, same outputs.

### Benchmark test

#### [NEW] [bench_sharded_backward_timing.rs](file:///Users/andrewgordon/RustRover-Projects/rustane/crates/engine/tests/bench_sharded_backward_timing.rs)

An `#[ignore]`d test following the pattern of [bench_hw_execution_time.rs](file:///Users/andrewgordon/RustRover-Projects/rustane/crates/engine/tests/bench_hw_execution_time.rs):

1. Use the 600M config matching `sweep-600m` (`custom:1536,4096,20,512` → dim=1536, hidden=4096, heads=20, seq=512, inferred from [Makefile L101](file:///Users/andrewgordon/RustRover-Projects/rustane/Makefile#L101))
2. Compile baseline kernels + sharded runtimes for FFN_SHARDS ∈ {1,2,4} and ATTN_SHARDS ∈ {1,2}
3. For each config, run 2 warmup + 5 timed steps calling the `_with_stats` variants
4. Collect and print:

```
=== FFN backward, FFN_SHARDS=4, 600M ===
  shard 0: stage=X.Xms  ane_wall=X.Xms  ane_hw=XXXXns  readback=X.Xms  grad_accum=X.Xms
  shard 1: stage=X.Xms  ane_wall=X.Xms  ane_hw=XXXXns  readback=X.Xms  grad_accum=X.Xms
  ...
  silu_deriv: X.Xms
  merge: X.Xms
  total: X.Xms
  ---
  total_ane_hw: X.Xms  total_ane_wall: X.Xms  host_overhead: X.Xms  total_cpu: X.Xms
```

5. Write JSON results to `results/sharded_backward_timing/` for comparison across runs
6. Print a summary table comparing baseline vs sharded, with percentage breakdown by bucket

**Run command:**
```bash
cargo test -p engine --test bench_sharded_backward_timing --release -- --ignored --nocapture
```

---

## Phase 2 — Parallelize FFN backward shard loop

> [!WARNING]
> The current serial loop at L1655 writes into shared `grads.dw1/dw2/dw3` via `accumulate_dw` + `scatter_dw_columns` inside the per-shard iteration. Parallelizing requires worker-local gradient temporaries and a post-loop reduction step, otherwise there will be write races and nondeterministic gradients.

#### [MODIFY] [layer.rs](file:///Users/andrewgordon/RustRover-Projects/rustane/crates/engine/src/layer.rs)

**Changes to `ShardedFfnBackwardWorker` (L724):**
- The existing `dw_tmp: Vec<f32>` field (L737) already serves as a worker-local scratch buffer of size `dim × shard_hidden`
- Add two additional worker-local gradient buffers:

```rust
pub struct ShardedFfnBackwardWorker {
    // ... existing fields ...
    dw_tmp: Vec<f32>,       // existing: [dim * shard_hidden], reused for each accumulate_dw call
    dw1_local: Vec<f32>,    // NEW: [dim * shard_hidden], worker-local dW1 shard
    dw2_local: Vec<f32>,    // NEW: [dim * shard_hidden], worker-local dW2 shard  
    dw3_local: Vec<f32>,    // NEW: [dim * shard_hidden], worker-local dW3 shard
}
```

**Changes to `run_sharded_ffn_backward_into` (L1637):**

Replace the serial `for` loop (L1655–1768) with `thread::scope`:

```rust
thread::scope(|scope| {
    let barrier = Arc::new(Barrier::new(shard_count + 1));
    let mut handles = Vec::with_capacity(shard_count);
    
    for (worker, layout) in workers.iter_mut().zip(layouts.iter()) {
        let b = Arc::clone(&barrier);
        handles.push(scope.spawn(move || {
            b.wait();
            // 1. Stage W2 shard weights + dffn activations → worker.w2_in
            // 2. run_cached_direct W2
            // 3. SiLU derivative (worker-local)
            // 4. accumulate_dw into worker.dw1_local/dw2_local/dw3_local (NO scatter yet)
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

**Key constraint for `dffn` access:** `ws.dffn` is read-only during the shard loop (written once at L1652). It can be shared across threads via `&ws.dffn`. Similarly, `cache.h1/h3/gate/x2norm` are read-only — these are safe to share.

**Key constraint for weight access:** `weights.w1/w2/w3` are `&[f32]` — read-only, safe to share across threads.

---

## Phase 3 — Re-measure with timing harness

Re-run the Phase 1 harness after Phase 2 lands:

```bash
cargo test -p engine --test bench_sharded_backward_timing --release -- --ignored --nocapture
```

And re-run the sweep to confirm:
```bash
make sweep-600m FFN_SHARDS=4
```

**Expected outcome:** FFN backward regression should collapse significantly if serial dispatch was the dominant issue. If it doesn't, the data will point to staging or dispatch overhead as the next target.

---

## Phase 4 — Parallelize attention backward shard loop

Same pattern as Phase 2, applied to `run_sharded_attention_backward_into` (L1771):

#### [MODIFY] [layer.rs](file:///Users/andrewgordon/RustRover-Projects/rustane/crates/engine/src/layer.rs)

**Changes to `ShardedAttentionBackwardWorker` (L748):**
- Add worker-local gradient buffers for `dwq_local`, `dwk_local`, `dwv_local`, `dwo_local`
- Each worker already has its own `dw_tmp` (L775), `da`, `dv`, `dq`, `dk`, `dx_attn`, `dx_kv` — these are correctly scoped per-worker

**Changes to `run_sharded_attention_backward_into` (L1771):**
- Replace serial `for worker in &mut runtime.workers` (L1792) with `thread::scope` + barrier
- Weight transposes at L1785–1788 (`wot`, `wqt`, `wkt`, `wvt`) must happen **before** spawning threads — they write into shared `ws.*t` buffers that threads read
- Post-loop: scatter worker-local `dwq/dwk/dwv/dwo` into `grads`, merge `dx_merged` from worker `dx_attn + dx_kv`

> [!NOTE]
> Attention backward has 5 serial ANE dispatches per shard (wot, bwd1, bwd2, q_bwd, kv_bwd) vs FFN's 2, so the potential improvement from parallelization is higher per shard, but the implementation complexity is also higher due to the RoPE backward step and score-channel intermediates.

---

## Verification plan

### Automated tests

**Phase 1 — Harness produces valid output:**
```bash
cargo test -p engine --test bench_sharded_backward_timing --release -- --ignored --nocapture
```
Gate: test runs without panics, all `ane_hw_ns > 0`, `ane_wall_ms > ane_hw_ms`, JSON written to `results/`.

**Phase 2 — FFN backward parallelization correctness:**
```bash
make sweep-600m FFN_SHARDS=4
```
Gate: loss matches baseline within existing tolerance (the sweep already validates fwd+bwd+update loss convergence). The sweep logs `ms/step`, `fwd`, `bwd`, `upd` — compare `bwd` against the Phase 1 baseline number.

Also run the full sweep to catch regressions:
```bash
make sweep-600m
```
Gate: baseline (no sharding) timing should be unaffected.

**Phase 4 — Attention backward parallelization correctness:**
```bash
make sweep-600m ATTN_SHARDS=2
make sweep-600m ATTN_SHARDS=2 FFN_SHARDS=4
```
Gate: same loss tolerance, `bwd` timing compared.

### Data collection checkpoints

After each phase, record the `sweep-600m` numbers in this table to track progress:

| Config | Phase 0 (current) | Phase 1 | Phase 2 | Phase 4 |
| --- | --- | --- | --- | --- |
| Baseline | 1046 ms/step, bwd=710 | — | — | — |
| ATTN_SHARDS=2 | 1541 ms/step, bwd=1174 | — | — | — |
| FFN_SHARDS=4 | 2993 ms/step, bwd=1978 | — | — | — |
| Both | 3297 ms/step, bwd=2258 | — | — | — |
