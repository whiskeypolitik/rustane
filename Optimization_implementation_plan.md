# Sharding Optimization Plan — Phased Workstreams

**Targets:** 30B training, 122B forward-only

---

## P0-A: FFN backward transpose reuse

#### [MODIFY] [layer.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs)

`BackwardWorkspace` already has `w1t`/`w3t` fields ([L1266–1267](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L1266)) — transient shared scratch, reused across layers, zero new memory.

In `run_sharded_ffn_backward_into` ([L2827](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L2827)), before `thread::scope`:

```rust
vdsp::mtrans(&weights.w1, hidden, &mut ws.w1t, dim, dim, hidden);
vdsp::mtrans(&weights.w3, hidden, &mut ws.w3t, dim, dim, hidden);
```

Pass `ws.w1t`/`ws.w3t` as shared read-only refs into workers. In `run_sharded_ffn_backward_worker` ([L2704](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L2704)), replace:

```diff
-stage_transposed_weight_columns(buf, &weights.w1, dim, hidden, col_start, shard_hidden, w13t_sp, 2*seq);
-stage_transposed_weight_columns(buf, &weights.w3, dim, hidden, col_start, shard_hidden, w13t_sp, 2*seq+dim);
+let w1t_start = layout.col_start * dim;
+stage_spatial(buf, shard_hidden, w13t_sp, &w1t[w1t_start..w1t_start + shard_hidden*dim], dim, 2*seq);
+let w3t_start = layout.col_start * dim;
+stage_spatial(buf, shard_hidden, w13t_sp, &w3t[w3t_start..w3t_start + shard_hidden*dim], dim, 2*seq+dim);
```

Same pattern as attention backward's `ws.wot` at [L3377](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L3377).

**Verification:**
- Bit-exact parallel vs `#[doc(hidden)]` serial reference
- `make sweep-600m` for baseline, `FFN_SHARDS=4`, `ATTN_SHARDS=2`, combined

**Expected:** FFN_SHARDS=4 backward ~1014ms → ~500ms.

---

## P0-B: Extract reusable forward-mode runtime from `parallel_bench.rs`

#### [NEW] Production forward-mode runtime structs in [layer.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs)

Extract from [parallel_bench.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/parallel_bench.rs):
- `ShardedAttentionForwardRuntime` — does not exist in production today, only in bench code at [L1380](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/parallel_bench.rs#L1380)
- Forward-mode shard layout resolution (baseline / attn-only / FFN-only / attn+FFN)
- Layer-dispatch composition (which runner to call per component)

**Do not carry over:**
- Layer-by-layer `x_buf.clone()` ([parallel_bench.rs L1457](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/parallel_bench.rs#L1457))
- Benchmark config, retry policy, reporting ([L144](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/parallel_bench.rs#L144))

After extraction, `parallel_bench.rs` should call the new production runtime.

---

## P1-A: Wire extracted forward runtime into forward-only

#### [MODIFY] [full_model.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/full_model.rs)

Add `forward_only_ws_with_options` alongside existing `forward_only_ws` ([L286](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/full_model.rs#L286)). Support all four forward modes: baseline, attn-only, FFN-only, combined.

**Verification:**
- Layer-level correctness tests (toleranced sharded vs baseline)
- `#[ignore]`d large-scale smoke tests at 30B and 122B configs

---

## P1-B: Wire forward runtime into training

#### [MODIFY] [full_model.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/full_model.rs)

1. Stop forcing `forward_attn_request: None` at [L133](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/full_model.rs#L133)
2. Extend `ModelForwardWorkspace` ([L223](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/full_model.rs#L223)) to hold combined forward runtime (not only `training_ffn_sharded`)
3. Make `forward_ws_with_options` ([L495](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/full_model.rs#L495)) handle all four forward modes

**Verification:**
- Layer-level bit-exact + toleranced tests
- `make sweep-600m` for all four `ATTN_SHARDS` × `FFN_SHARDS` combinations

---

## P1-C: Research-gated ANE overlap spike

> [!WARNING]
> Do not commit runtime complexity before proving viability.

**Go/no-go criterion:** Demonstrate measurable ANE/CPU overlap in a microbench using an async dispatch pattern. If no such pattern exists in the current ANE stack, defer indefinitely.

**If viable:** In FFN backward workers, overlap W2 ANE dispatch with W13t CPU staging (W2 inputs are available pre-loop; no data dependency). ~2–4ms/shard saving.
