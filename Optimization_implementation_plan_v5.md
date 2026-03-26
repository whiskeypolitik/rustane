# Sharding Optimization Plan v5

**Targets:** 30B training, 122B forward-only

---

## P0-A: FFN backward transpose reuse

Transpose `weights.w1/w3` into existing `ws.w1t/ws.w3t` ([L1266–1267](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L1266)) once before `thread::scope` in `run_sharded_ffn_backward_into` ([L2827](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L2827)). Workers stage shard rows via `stage_spatial` instead of `stage_transposed_weight_columns` ([L2704](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L2704)). Zero new memory.

**Verify:** bit-exact parallel vs serial reference, `make sweep-600m` all four configs.

---

## P0-B: Shared resolution, runtime extraction, production layer API

### P0-B.1: Shared resolver with caller-selected policy

```rust
pub enum ShardPolicy {
    FailFast,                    // training, sweeps — reject invalid shard counts
    AutoAdjustNearest { log: bool }, // ladder, ceiling — round to nearest valid, print adjustment
}

pub enum ResolvedForwardMode { Baseline, AttentionOnly{..}, FfnOnly{..}, AttentionFfn{..} }
pub enum ResolvedBackwardMode { Baseline, AttentionOnly{..}, FfnOnly{..}, AttentionFfn{..} }

pub fn resolve_modes(
    cfg: &ModelConfig,
    requested: ShardRequest,  // parsed from env vars or explicit args
    policy: ShardPolicy,
) -> Result<(ResolvedForwardMode, ResolvedBackwardMode), String>;
```

- `make sweep-600m` / training: call with `FailFast`
- `make forward-ladder` / `forward-ceiling`: call with `AutoAdjustNearest { log: true }`
- When `AutoAdjustNearest` changes a request, print clearly: `"requested FFN_SHARDS=8, applied FFN_FWD_SHARDS=6"`
- `parallel_bench.rs` keeps existing `ShardPolicy` semantics but delegates to this shared core
- Env parsing is a thin wrapper over `resolve_modes`; not the resolver itself

### P0-B.2: `CombinedForwardRuntime`

Extract from [parallel_bench.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/parallel_bench.rs). `LayerScratch` is **internal temporary only**, not the training state source of truth.

```rust
pub struct CombinedForwardRuntime {
    pub mode: ResolvedForwardMode,
    pub attn: AttentionForwardRunner,
    pub ffn: FfnForwardRunner,
    scratch: LayerScratch,
}
```

### P0-B.3: ForwardCache write contract

`ForwardCache` is the single write target for training. `forward_into_combined` must populate **all** fields backward depends on:

| Field | Source | Notes |
| --- | --- | --- |
| **`x`** | layer input | **Copy before any computation** ([L4769](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L4769)) — backward RMSNorm1 needs it ([L4216](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L4216)) |
| `xnorm`, `rms_inv1` | RMSNorm1 output | Direct write |
| `q_rope` | worker `sdpa.q_rope_out` | Scatter by `q_col_start` |
| `k_rope` | worker `sdpa.k_rope_out` | Scatter by `kv_col_start` |
| `v` | worker `sdpa.v_out` | Scatter by `kv_col_start` |
| `attn_out` | worker `sdpa.attn_out` | Scatter by `q_col_start` |
| `o_out` | merged wo outputs | Direct write |
| `x2`, `x2norm`, `rms_inv2` | post-attention | Direct write |
| `h1`, `h3`, `gate` | FFN shard outputs | `store_shard_channels` post-join |

### P0-B.4: Production layer API — sharded modes only

> [!IMPORTANT]
> `forward_into_combined` is used **only when shard env vars are present**. Baseline mode continues using the existing pipelined path (`forward_into_pipelined` + deferred `read_ffn_cache`) to preserve its latency advantage at [full_model.rs L314-377](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/full_model.rs#L314).

```rust
pub fn forward_into_combined(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,       // baseline attention fallback
    weights: &LayerWeights,
    x: &[f32],
    cache: &mut ForwardCache,        // THE write target
    x_next: &mut [f32],
    runtime: &mut CombinedForwardRuntime,
) {
    cache.x.copy_from_slice(x);      // preserve for backward
    // 1. RMSNorm1 → cache.xnorm, cache.rms_inv1
    // 2. Attention (baseline or sharded) → cache.{q_rope,k_rope,v,attn_out,o_out}
    // 3. Residual → cache.x2
    // 4. RMSNorm2 → cache.x2norm, cache.rms_inv2
    // 5. FFN (baseline or sharded) → cache.{h1,h3,gate}, x_next
}
```

Dispatch in `full_model.rs`:
- **No shard vars set:** existing pipelined baseline — no change
- **Any shard var set:** `forward_into_combined` with the resolved mode

---

## P1-A: Wire into forward-only

Add `CombinedForwardRuntime` ownership to workspaces:

```rust
pub struct ModelForwardWorkspace {
    pub combined_forward: Option<CombinedForwardRuntime>,  // None = baseline pipelined
}
```

Add `forward_only_ws_with_options`. Runtime compiled once, stored persistently.

**Verify:** toleranced sharded vs baseline. `#[ignore]`d smoke at 30B and 122B.

---

## P1-B: Wire into training forward + backward migration

1. Stop forcing `forward_attn_request: None` at [L133](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/full_model.rs#L133)
2. `forward_ws_with_options` dispatches: baseline → pipelined, sharded → `forward_into_combined` writing into per-layer `ForwardCache`
3. **Backward wiring migration:** `backward_ws_with_options` ([L572](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/full_model.rs#L572)) currently branches on `options.backward_ffn_request: Option<usize>` and `options.backward_attn_request: Option<usize>`. Migrate to branch on `ResolvedBackwardMode` from `TrainingParallelOptions`. Update runtime compile/reuse checks to match resolved mode variants, not raw `Option<usize>` fields.

**Verify:** bit-exact + toleranced layer tests, `make sweep-600m` all four combinations.

---

## P1-C: Research-gated ANE overlap spike

Go/no-go: prove async ANE dispatch yields measurable overlap in a microbench.

---

## Env var policy

| Env var | Direction | Default |
| --- | --- | --- |
| `ATTN_FWD_SHARDS` | forward attention | None (baseline) |
| `ATTN_BWD_SHARDS` | backward attention | None (baseline) |
| `FFN_FWD_SHARDS` | forward FFN | None (baseline) |
| `FFN_BWD_SHARDS` | backward FFN | None (baseline) |
| `ATTN_SHARDS` | shorthand: both attn directions | — |
| `FFN_SHARDS` | shorthand: both FFN directions | — |

If both a shorthand and a per-direction var are set for the same axis, **error immediately**: `"FFN_SHARDS=4 conflicts with FFN_FWD_SHARDS=2; use per-direction vars or the shorthand, not both"`.

No shard vars → baseline pipelined path. Any shard var → combined runtime with resolved mode.
