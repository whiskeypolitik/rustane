# Sharded FFN Forward Optimization

## Problem

600M sharded training forward with `FFN_SHARDS=4` regresses to 911ms vs 259ms baseline (3.5×). Per-layer timing harness shows:
- Per-layer wall: ~57ms, of which FFN section is ~51ms
- FFN ANE wall: only ~7.6ms (15% of FFN wall)
- **`w2` staging: 7.5–8.3ms per shard** — the dominant cost

Root cause has two parts:
1. The shard loop at [L1559](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L1559) is serial
2. `stage_transposed_weight_columns` ([L1393](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L1393)) does element-by-element gather with stride-`hidden` reads from `weights.w2` (24MB at 600M) — cache-hostile, ~8ms per call

## Phased approach

| Phase | Goal | Expected fwd improvement |
| --- | --- | --- |
| A | Prepack sharded W2 weights | Eliminate ~24ms of w2 staging (4 shards × ~6ms saved) |
| B | Parallelize forward shard loop | Wall → max(shard) + merge instead of sum(shards) |

> [!IMPORTANT]
> **I recommend the opposite ordering from the engineer's plan.** The engineer suggests parallelizing first, then optimizing staging. I recommend prepacking first because:
> 1. Parallelizing 4 shards that each do 8ms of cache-hostile `stage_transposed_weight_columns` will create **L2/L3 contention** — 4 threads simultaneously scattering across the same 24MB `weights.w2` matrix. Parallelization ROI will be limited by shared cache pressure.
> 2. Prepacking replaces the per-call transpose+scatter with a one-time (~8ms) prepack per generation change + a per-call memcpy (~1ms). This cuts per-shard w2 staging from ~8ms to ~1ms, making each shard ~7ms cheaper.
> 3. After prepacking, each shard's total work is small enough (~4ms: 1ms w13 stage + 1ms ANE + 1ms silu + 1ms w2 stage) that parallelization yields clean overlap with minimal cache contention.
>
> Estimated impact: Phase A alone should cut per-layer FFN from ~51ms to ~23ms (20 layers: 911ms → ~460ms fwd). Phase B then cuts ~23ms to ~8ms (fwd → ~160ms).

---

## Phase A — Prepack sharded W2 weights

### Analysis

The non-sharded forward path already has this optimization at [L4152–4163](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L4152): it transposes `weights.w2` into `cache.w2t_scratch` once per generation, then stages from contiguous rows via `stage_spatial`. The sharded path at [L1611](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L1611) skips this and calls `stage_transposed_weight_columns` every forward call — the entire cost is redundant.

The fix: each worker pre-packs its shard's W2 weight region into a contiguous CPU buffer. The prepack runs once when `w2_generation` changes. Subsequent forward calls stage from the prepacked buffer via `stage_spatial` (memcpy-speed).

### Changes

#### [MODIFY] [layer.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs)

**`ShardedFfnForwardWorker` (L703):**

```rust
pub struct ShardedFfnForwardWorker {
    // ... existing fields ...
    w2_packed: Vec<f32>,          // NEW: [dim * shard_hidden], pre-transposed W2 shard
    w2_packed_generation: u64,    // NEW: tracks w2_generation
}
```

Size: `dim × shard_hidden` = 1536 × 1024 = 6.3MB per worker (25MB total for 4 shards). Already dwarfed by the IOSurface allocations.

**`ShardedFfnForwardRuntime::compile` (initialization):**

Set `w2_packed_generation: 0` and allocate `w2_packed: vec![0.0; dim * shard_hidden]`.

**`run_sharded_ffn_forward_into` (L1542), inside the shard loop:**

Replace L1611–1620:
```rust
// BEFORE (8ms per shard):
stage_transposed_weight_columns(
    buf, &weights.w2, dim, hidden, layout.col_start, shard_hidden, w2_sp, seq,
);

// AFTER (~1ms per shard):
if worker.w2_packed_generation != weights.w2_generation {
    // One-time prepack: transpose W2 shard columns into contiguous rows
    for c in 0..shard_hidden {
        for r in 0..dim {
            worker.w2_packed[c * dim + r] = weights.w2[r * hidden + layout.col_start + c];
        }
    }
    worker.w2_packed_generation = weights.w2_generation;
}
stage_spatial(buf, shard_hidden, w2_sp, &worker.w2_packed, dim, seq);
```

The prepack (~8ms) runs once per training step (when optimizer updates `weights.w2` and bumps `w2_generation`). All 20 layers share the same `w2_generation` per layer, so each layer's first forward call pays the prepack, subsequent calls use cache. Across a training step, the prepack cost is amortized: 8ms × 4 shards = 32ms once, vs 8ms × 4 shards × 20 layers = 640ms currently.

> [!NOTE]
> The prepack loop can be further optimized with `vdsp::mtrans` for the sub-matrix, but even the naive element-wise version runs only once per generation change — perfectingit is diminishing returns.

### Phase A verification

**Serial reference:** The serial loop body is already preserved as `#[doc(hidden)]` from the backward plan. Apply the same pattern here.

**Test (non-`#[ignore]`d):** Bit-exact comparison of `x_next`, `cache.h1/h3/gate` between prepacked path and non-sharded baseline (toleranced, since sharding changes tile sizes).

**Timing:** Re-run forward timing harness:
```bash
cargo test -p engine --test bench_sharded_forward_timing --release -- --ignored --nocapture
```

Expected: `w2_staging_ms` drops from ~8ms to ~1ms per shard. Per-layer FFN wall drops from ~51ms to ~23ms.

---

## Phase B — Parallelize forward shard loop

After Phase A, each shard's work is roughly uniform and ~4ms:
- w13 staging: ~2ms
- w13 ANE: ~1ms
- silu_gate: ~0.7ms
- w2 staging: ~1ms (prepacked)
- w2 ANE: ~1ms

Worth parallelizing now with minimal cache contention.

### Changes

#### [MODIFY] [layer.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs)

**Preserve serial reference (same pattern as backward):**

```rust
#[doc(hidden)]
pub fn run_sharded_ffn_forward_into_serial(...) { /* current serial body */ }
```

**New parallel implementation at `run_sharded_ffn_forward_into`:**

Following the existing parallel bench pattern at [bench_ffn_latency_parallel_full_model.rs L743](file:///Users/USER/RustRover-Projects/rustane/crates/engine/tests/bench_ffn_latency_parallel_full_model.rs#L743):

```rust
fn run_sharded_ffn_forward_into(
    cfg: &ModelConfig, weights: &LayerWeights,
    cache: &mut ForwardCache, x_next: &mut [f32],
    runtime: &mut ShardedFfnForwardRuntime,
) {
    // Pre-loop: prepack W2 shards if generation changed (Phase A)
    for (worker, layout) in runtime.workers.iter_mut().zip(runtime.layouts.iter()) {
        if worker.w2_packed_generation != weights.w2_generation {
            // prepack...
            worker.w2_packed_generation = weights.w2_generation;
        }
    }

    let x2norm_ref: &[f32] = &cache.x2norm;
    thread::scope(|scope| {
        let barrier = Arc::new(Barrier::new(runtime.shard_count + 1));
        let mut handles = Vec::with_capacity(runtime.shard_count);
        for (worker, layout) in runtime.workers.iter_mut().zip(runtime.layouts.iter()) {
            let b = Arc::clone(&barrier);
            handles.push(scope.spawn(move || {
                b.wait();
                // 1. stage W13 (x2norm + weight columns)
                // 2. run_cached_direct W13
                // 3. read h1, h3
                // 4. silu_gate
                // 5. stage W2 (gate + prepacked weights via stage_spatial)
                // 6. run_cached_direct W2
            }));
        }
        barrier.wait();
        for h in handles { h.join().expect("FFN forward shard panicked"); }
    });

    // Post-join: deterministic merge (same order as serial)
    runtime.ffn_out.fill(0.0);
    for worker in &runtime.workers {
        let locked = worker.w2_out.as_f32_slice();
        vdsp::vadd(&runtime.ffn_out, &locked[..runtime.ffn_out.len()], &mut runtime.merge_tmp);
        runtime.ffn_out.copy_from_slice(&runtime.merge_tmp);
    }
    // Store h1/h3/gate into cache (same order as serial)
    for (worker, layout) in runtime.workers.iter().zip(runtime.layouts.iter()) {
        store_shard_channels(&mut cache.h1, hidden, seq, layout.col_start, &worker.h1, shard_hidden);
        store_shard_channels(&mut cache.h3, hidden, seq, layout.col_start, &worker.h3, shard_hidden);
        store_shard_channels(&mut cache.gate, hidden, seq, layout.col_start, &worker.gate, shard_hidden);
    }
    let alpha = 1.0 / (2.0 * cfg.nlayers as f32).sqrt();
    vdsp::vsma(&runtime.ffn_out, alpha, &cache.x2, x_next);
}
```

**Thread safety:**
- `x2norm_ref` — immutable, shared across threads ✓
- `weights.w1/w3` — immutable ✓
- `worker.w2_packed` — worker-local, read-only after Phase A prepack ✓
- `worker.{w13,w2}_{in,out}` — worker-local TensorData ✓
- `worker.{h1,h3,gate}` — worker-local ✓
- `cache.h1/h3/gate` — written only in post-join merge ✓

### Phase B verification

**Test 1 (bit-exact):** parallel vs `#[doc(hidden)]` serial reference. All outputs: `x_next`, `cache.h1/h3/gate`.

**Test 2 (toleranced):** parallel-sharded vs non-sharded `forward_into`. Same tolerance as backward tests.

**Test 3 (idempotency):** bit-exact between consecutive runs.

**Timing:**
```bash
cargo test -p engine --test bench_sharded_forward_timing --release -- --ignored --nocapture
make sweep-600m FFN_SHARDS=4
```

---

## Expected end state

| Config | Before | After Phase A (est.) | After Phase B (est.) |
| --- | --- | --- | --- |
| FFN_SHARDS=4 fwd | 911ms | ~460ms | ~180ms |
| FFN_SHARDS=4 bwd | 1023ms | 1023ms | 1023ms |
| FFN_SHARDS=4 total | 2017ms | ~1550ms | ~1280ms |
| Baseline fwd | 259ms | 259ms | 259ms |

> [!NOTE]
> The sharded forward will likely not match baseline (259ms) even after both phases because each shard still pays its own ANE dispatch overhead and the attention section runs unsharded. The goal is to get close enough that FFN sharding is viable for larger models where it's required.
