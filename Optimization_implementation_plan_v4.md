# Sharding Optimization Plan v4

**Targets:** 30B training, 122B forward-only

---

## P0-A: FFN backward transpose reuse

Transpose `weights.w1/w3` into existing `ws.w1t/ws.w3t` ([L1266–1267](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L1266)) once before `thread::scope` in `run_sharded_ffn_backward_into` ([L2827](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L2827)). Workers stage shard rows via `stage_spatial` instead of `stage_transposed_weight_columns` ([L2704](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L2704)). Zero new memory. Same pattern as `ws.wot` at [L3377](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L3377).

**Verify:** bit-exact parallel vs serial reference, `make sweep-600m` all four configs.

---

## P0-B: Shared mode resolution, runtime extraction, production layer API

### P0-B.1: Shared mode resolution for both directions

Today: monolithic parsing at [full_model.rs L99](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/full_model.rs#L99), hardcoded `forward_attn_request: None` at [L133](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/full_model.rs#L133). Bench has its own mode enum at [parallel_bench.rs L252](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/parallel_bench.rs#L252).

Add to production (new `sharding.rs` module or top of `layer.rs`):

```rust
pub enum ResolvedForwardMode {
    Baseline,
    AttentionOnly { attn_shards: usize },
    FfnOnly { ffn_shards: usize },
    AttentionFfn { attn_shards: usize, ffn_shards: usize },
}

pub enum ResolvedBackwardMode {
    Baseline,
    AttentionOnly { attn_shards: usize },
    FfnOnly { ffn_shards: usize },
    AttentionFfn { attn_shards: usize, ffn_shards: usize },
}
```

Single `resolve_modes_from_env(cfg) -> Result<(ResolvedForwardMode, ResolvedBackwardMode), String>` that parses all six env vars, applies conflict-error policy, validates divisibility per-direction. `TrainingParallelOptions` stores both resolved modes. `parallel_bench` calls the same resolver.

### P0-B.2: `CombinedForwardRuntime`

Extract from [parallel_bench.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/parallel_bench.rs):

```rust
pub struct CombinedForwardRuntime {
    pub mode: ResolvedForwardMode,
    pub attn: AttentionForwardRunner,  // enum: Baseline | Sharded
    pub ffn: FfnForwardRunner,         // enum: Baseline | Sharded
    scratch: LayerScratch,             // INTERNAL temporary only
}
```

> [!IMPORTANT]
> `LayerScratch` is an **internal temporary** — not the source of truth for training state. The production API (P0-B.4) writes final results into `ForwardCache`. Bench code may use scratch-only since it has no backward.

Do NOT carry over from bench: `x_buf.clone()` ([L1457](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/parallel_bench.rs#L1457)), retry policy, reporting ([L144](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/parallel_bench.rs#L144)).

### P0-B.3: ForwardCache write contract

The production forward path writes **directly into `ForwardCache`** — this is the single source of truth for training backward. `LayerScratch` is used only as intermediate workspace within the combined runtime.

Sharded attention forward must populate (post-join scatter):

| ForwardCache field | Source | Scatter key |
| --- | --- | --- |
| `q_rope` | worker `sdpa.q_rope_out` | `q_col_start` |
| `k_rope` | worker `sdpa.k_rope_out` | `kv_col_start` |
| `v` | worker `sdpa.v_out` | `kv_col_start` |
| `attn_out` | worker `sdpa.attn_out` | `q_col_start` |
| `xnorm`, `rms_inv1` | computed pre-dispatch | direct write |
| `o_out` | merged wo outputs | direct write |
| `x2`, `x2norm`, `rms_inv2` | computed post-merge | direct write |

Sharded FFN forward already fills `h1`, `h3`, `gate` via `store_shard_channels` post-join.

### P0-B.4: Production layer-level forward API

> [!IMPORTANT]
> Today's only production training-forward entrypoint is [forward_into_with_training_ffn L4704](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L4704) — it keeps attention baseline and only optionally shards FFN. Without a new layer API, `full_model.rs` would duplicate bench orchestration or reach into runtime internals.

Add to [layer.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs):

```rust
pub fn forward_into_combined(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,       // for baseline attention fallback
    weights: &LayerWeights,
    x: &[f32],
    cache: &mut ForwardCache,        // THE write target
    x_next: &mut [f32],
    runtime: &mut CombinedForwardRuntime,
) {
    // 1. RMSNorm1 → cache.xnorm, cache.rms_inv1
    // 2. Attention (baseline or sharded) → cache.{q_rope,k_rope,v,attn_out,o_out}
    // 3. Residual → cache.x2
    // 4. RMSNorm2 → cache.x2norm, cache.rms_inv2
    // 5. FFN (baseline or sharded) → cache.{h1,h3,gate}, x_next
}
```

This is the single production layer-forward entrypoint for all four modes. `forward_into_with_training_ffn` becomes a thin wrapper calling this with `AttentionOnly::Baseline`. `full_model.rs` calls `forward_into_combined` unconditionally.

---

## P1-A: Wire into forward-only

Add `CombinedForwardRuntime` ownership to workspaces:

```rust
pub struct ModelForwardWorkspace {
    // existing fields...
    pub combined_forward: Option<CombinedForwardRuntime>,  // replaces training_ffn_sharded
}
```

Add `forward_only_ws_with_options`. Runtime compiled once, stored persistently.

**Verify:** toleranced sharded vs baseline. `#[ignore]`d smoke at 30B and 122B.

---

## P1-B: Wire into training forward

1. Stop forcing `forward_attn_request: None`
2. `forward_ws_with_options` delegates to `forward_into_combined` for all four modes, writing into per-layer `ForwardCache`
3. Backward reads from cache as before — no changes needed

**Verify:** bit-exact + toleranced layer tests, `make sweep-600m` all four combinations.

---

## P1-C: Research-gated ANE overlap spike

Go/no-go: prove async ANE dispatch yields measurable overlap in a microbench before committing runtime complexity. If viable: overlap W2 ANE dispatch with W13t CPU staging in FFN backward workers.

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

`ResolvedForwardMode` and `ResolvedBackwardMode` are resolved independently via `resolve_modes_from_env`. Validation (head/hidden divisibility) applies per-direction since forward and backward shard counts may differ.
