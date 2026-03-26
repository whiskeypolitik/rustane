# Sharding Optimization Plan v3

**Targets:** 30B training, 122B forward-only

---

## P0-A: FFN backward transpose reuse

Transpose `weights.w1/w3` into existing `ws.w1t/ws.w3t` ([L1266–1267](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L1266)) once before `thread::scope` in `run_sharded_ffn_backward_into` ([L2827](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L2827)). Workers stage shard rows via `stage_spatial` instead of `stage_transposed_weight_columns` ([L2704](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L2704)). Zero new memory. Same pattern as `ws.wot` at [L3377](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L3377).

**Verify:** bit-exact parallel vs serial reference, `make sweep-600m` all four configs.

---

## P0-B: Shared mode resolution + forward runtime extraction

### P0-B.1: Shared `ResolvedForwardMode`

> [!IMPORTANT]
> Today, mode resolution is split: bench resolves 4 modes at [parallel_bench.rs L252](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/parallel_bench.rs#L252), training env parsing forces `forward_attn_request: None` at [full_model.rs L133](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/full_model.rs#L133). If P0-B lands without unifying these, bench, forward-only, and training will drift on what `ATTN_SHARDS`/`FFN_SHARDS` mean.

Add to [layer.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs) (or a new `sharding.rs` module):

```rust
pub enum ResolvedForwardMode {
    Baseline,
    AttentionOnly { attn_shards: usize },
    FfnOnly { ffn_shards: usize },
    AttentionFfn { attn_shards: usize, ffn_shards: usize },
}
```

Single validation point for shard divisibility. `TrainingParallelOptions::from_env_for_cfg` and `parallel_bench` both resolve through this.

### P0-B.2: `CombinedForwardRuntime`

Extract reusable parts from `ModeRunner` in [parallel_bench.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/parallel_bench.rs) into production:

```rust
pub struct CombinedForwardRuntime {
    pub mode: ResolvedForwardMode,
    pub attn: AttentionForwardRunner,  // enum: Baseline | Sharded
    pub ffn: FfnForwardRunner,         // enum: Baseline | Sharded
    pub scratch: LayerScratch,
}
```

`parallel_bench.rs` becomes a thin caller that compiles a `CombinedForwardRuntime` and runs it.

> [!WARNING]
> Do NOT carry over from bench code: `x_buf.clone()` per layer ([L1457](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/parallel_bench.rs#L1457)), retry policy, benchmark config, reporting ([L144](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/parallel_bench.rs#L144)).

### P0-B.3: Training attention-forward cache contract

> [!IMPORTANT]
> Bench attention forward at [L1031](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/parallel_bench.rs#L1031) fills `scratch.{xnorm, rms_inv1, o_out, x2, x2norm, rms_inv2}` but does NOT populate `ForwardCache.{q_rope, k_rope, v, attn_out}` — training backward depends on all of these ([L80–92](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L80)).

The production sharded attention forward runner must write back into `ForwardCache`:
- `q_rope`: read from each worker's `sdpa.q_rope_out`, scatter by `q_col_start`
- `k_rope`: read from each worker's `sdpa.k_rope_out`, scatter by `kv_col_start`
- `v`: read from each worker's `sdpa.v_out`, scatter by `kv_col_start`
- `attn_out`: read from each worker's `sdpa.attn_out`, scatter by `q_col_start`

This is post-join work (same pattern as `store_shard_channels` in FFN forward). The bench path skips it because it has no backward.

---

## P1-A: Wire into forward-only

Add `CombinedForwardRuntime` ownership to both `ModelForwardWorkspace` and lean forward workspace:

```rust
pub struct ModelForwardWorkspace {
    // existing fields...
    pub combined_forward: Option<CombinedForwardRuntime>,  // replaces training_ffn_sharded
}
```

Add `forward_only_ws_with_options` alongside [forward_only_ws L286](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/full_model.rs#L286). Runtime compiled once and stored persistently — no recompile per call.

**Verify:** toleranced sharded vs baseline. `#[ignore]`d smoke at 30B and 122B configs.

---

## P1-B: Wire into training forward

1. Stop forcing `forward_attn_request: None` at [L133](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/full_model.rs#L133)
2. `forward_ws_with_options` ([L495](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/full_model.rs#L495)) delegates to `CombinedForwardRuntime` for all four modes, writing into per-layer `ForwardCache` (via cache contract from P0-B.3)
3. Training loop calls combined runtime with cache population; backward reads from cache as before

**Verify:** bit-exact + toleranced layer tests, `make sweep-600m` all four `ATTN_SHARDS` × `FFN_SHARDS` combinations.

---

## P1-C: Research-gated ANE overlap spike

> [!WARNING]
> Not a committed build item. Go/no-go: prove async ANE dispatch yields measurable overlap in a microbench before committing runtime complexity.

If viable: overlap W2 ANE dispatch with W13t CPU staging in FFN backward workers. ~2–4ms/shard.

---

## Env var policy

Six independent controls:

| Env var | Direction | Default |
| --- | --- | --- |
| `ATTN_FWD_SHARDS` | forward attention | None (baseline) |
| `ATTN_BWD_SHARDS` | backward attention | None (baseline) |
| `FFN_FWD_SHARDS` | forward FFN | None (baseline) |
| `FFN_BWD_SHARDS` | backward FFN | None (baseline) |
| `ATTN_SHARDS` | shorthand: sets both `ATTN_FWD_SHARDS` and `ATTN_BWD_SHARDS` | — |
| `FFN_SHARDS` | shorthand: sets both `FFN_FWD_SHARDS` and `FFN_BWD_SHARDS` | — |

If both a shorthand and an explicit per-direction var are set for the same axis, **error immediately** with a clear message (e.g., `"FFN_SHARDS=4 conflicts with FFN_FWD_SHARDS=2; use per-direction vars or the shorthand, not both"`). No silent override — prevents misconfigured experiments from wasting compute. Use the shorthand for uniform policy, per-direction vars for asymmetric configurations.

`ResolvedForwardMode` and `ResolvedBackwardMode` are resolved independently from these vars. `TrainingParallelOptions` stores both. Validation (head/hidden divisibility) applies per-direction since forward and backward shard counts may differ.
