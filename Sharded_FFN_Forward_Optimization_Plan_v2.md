# Sharded FFN Forward Optimization

## Problem

`FFN_SHARDS=4` forward regresses to 911ms vs 259ms baseline (3.5×). Per-layer timing:
- FFN wall: ~51ms, ANE wall: ~7.6ms (15%)
- **`w2` staging: 7.5–8.3ms per shard** — dominant cost
- Root cause: `stage_transposed_weight_columns` ([L1393](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L1393)) does element-by-element gather with stride-`hidden` reads across 24MB `weights.w2`

## Phases

| Phase | Goal | Expected fwd improvement |
| --- | --- | --- |
| A | Use per-layer cached W2 transpose for sharded staging | ~51ms → ~23ms per-layer FFN |
| B | Parallelize forward shard loop | ~23ms → ~8ms per-layer FFN |

> [!IMPORTANT]
> Phase A first. After Phase A, each shard's w2 staging drops from ~8ms to ~1ms, making Phase B's parallelization clean with minimal cache contention.

---

## Phase A — Use existing `cache.w2t_scratch` for sharded W2 staging

### Key insight

The non-sharded forward already caches a full transpose of `weights.w2` into `cache.w2t_scratch` ([L4152–4163](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L4152)), keyed by `w2t_generation`. `ForwardCache` is **per-layer** (one per `ws.caches[l]` in the training loop), so the generation tracking is correct — each layer's cache tracks its own layer's weights.

The sharded forward path at [L1611](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L1611) ignores this and calls `stage_transposed_weight_columns` from raw `weights.w2` every call. The fix: transpose once into `cache.w2t_scratch`, then stage each shard from **contiguous rows** of the transposed matrix.

In the transposed layout `[hidden × dim]`, shard columns `col_start..col_start+shard_hidden` of the original `[dim × hidden]` become contiguous rows — a single `stage_spatial` call at memcpy speed.

### Changes

#### [MODIFY] [layer.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs)

In `run_sharded_ffn_forward_into` (L1542), add before the shard loop:

```rust
// Reuse per-layer cached W2 transpose (same pattern as non-sharded path at L4152)
if cache.w2t_generation != weights.w2_generation {
    vdsp::mtrans(&weights.w2, hidden, &mut cache.w2t_scratch, dim, dim, hidden);
    cache.w2t_generation = weights.w2_generation;
}
```

Replace L1611–1620 inside the shard loop:

```rust
// BEFORE (~8ms per shard — element-by-element gather):
stage_transposed_weight_columns(buf, &weights.w2, dim, hidden, ...);

// AFTER (~1ms per shard — contiguous memcpy):
let w2t_shard_start = layout.col_start * dim;
let w2t_shard = &cache.w2t_scratch[w2t_shard_start..w2t_shard_start + shard_hidden * dim];
stage_spatial(buf, shard_hidden, w2_sp, w2t_shard, dim, seq);
```

**Cost model per layer:**
- Transpose: one `vdsp::mtrans` (~1ms, vectorized, runs once per layer per step)
- 4× `stage_spatial`: ~1ms each (contiguous memcpy)
- Total: ~5ms vs current ~32ms

> [!NOTE]
> `cache.w2t_generation` initializes to `u64::MAX` ([L122](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L122)), guaranteeing the first call always transposes. `LayerWeights.w2_generation` starts at 0 ([L1307](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L1307)). No init-skip bug.

### Phase A verification

**Test (non-`#[ignore]`d):** Toleranced comparison of `x_next`, `cache.h1/h3/gate` between sharded and non-sharded forward. Explicit runtime construction, no env vars.

**Timing:**
```bash
cargo test -p engine --test bench_sharded_forward_timing --release -- --ignored --nocapture
make sweep-600m FFN_SHARDS=4
```

---

## Phase B — Parallelize forward shard loop

After Phase A, each shard is ~4ms of cache-local work. Parallelize with `thread::scope` + barrier, following the existing bench pattern at [bench_ffn_latency_parallel_full_model.rs L743](file:///Users/USER/RustRover-Projects/rustane/crates/engine/tests/bench_ffn_latency_parallel_full_model.rs#L743).

### Changes

#### [MODIFY] [layer.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs)

Preserve serial reference as `#[doc(hidden)] pub`.

```rust
fn run_sharded_ffn_forward_into(...) {
    // Pre-loop: cache W2 transpose (Phase A)
    if cache.w2t_generation != weights.w2_generation { ... }

    let x2norm_ref: &[f32] = &cache.x2norm;
    let w2t_ref: &[f32] = &cache.w2t_scratch;  // per-layer, safe to share
    thread::scope(|scope| {
        let barrier = Arc::new(Barrier::new(shard_count + 1));
        for (worker, layout) in workers.iter_mut().zip(layouts.iter()) {
            let b = Arc::clone(&barrier);
            scope.spawn(move || {
                b.wait();
                // 1. stage W13 (x2norm + weight columns from weights.w1/w3)
                // 2. run_cached_direct W13
                // 3. read h1, h3
                // 4. silu_gate
                // 5. stage W2 (gate + contiguous rows from w2t_ref)
                // 6. run_cached_direct W2
            });
        }
        barrier.wait();
    });
    // Post-join: merge ffn_out, store h1/h3/gate into cache (deterministic order)
}
```

**Thread safety:**
- `x2norm_ref`, `w2t_ref`, `weights.w1/w3` — immutable shared reads ✓
- `worker.{w13,w2}_{in,out,h1,h3,gate}` — worker-local ✓
- `cache.h1/h3/gate`, `runtime.ffn_out` — written only in post-join merge ✓

### Phase B verification

**Test 1 (bit-exact):** parallel vs `#[doc(hidden)]` serial reference.
**Test 2 (toleranced):** sharded vs non-sharded baseline.
**Test 3 (idempotency):** two consecutive runs produce bit-identical output.

```bash
cargo test -p engine --test bench_sharded_forward_timing --release -- --ignored --nocapture
make sweep-600m FFN_SHARDS=4
```

---

## Expected end state

| Config | Before | After A (est.) | After A+B (est.) |
| --- | --- | --- | --- |
| FFN_SHARDS=4 fwd | 911ms | ~460ms | ~180ms |
| FFN_SHARDS=4 total | 2017ms | ~1550ms | ~1280ms |
| Baseline fwd | 259ms | 259ms | 259ms |
