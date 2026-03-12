//! Single transformer layer: compile kernels, run forward/backward on ANE + CPU.
//!
//! Forward: RMSNorm1(CPU) → sdpaFwd(ANE) → woFwd(ANE) → residual+RMSNorm2(CPU) → ffnFused(ANE)
//! Backward: Scale dy → ffnBwdW2t(ANE) → SiLU'(CPU) → ffnBwdW13t(ANE) → RMSNorm2 bwd(CPU)
//!           → wotBwd(ANE) → sdpaBwd1(ANE) → sdpaBwd2(ANE) → RoPE bwd(CPU)
//!           → qBwd(ANE) → kvBwd(ANE) → RMSNorm1 bwd(CPU)

use ane_bridge::ane::{Executable, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use crate::cpu::{rmsnorm, vdsp};
use crate::kernels::{dyn_matmul, sdpa_fwd, sdpa_bwd, ffn_fused};
use crate::model::ModelConfig;

/// Per-layer weights (f32, CPU-side).
pub struct LayerWeights {
    pub wq: Vec<f32>,      // [DIM * Q_DIM]
    pub wk: Vec<f32>,      // [DIM * KV_DIM]
    pub wv: Vec<f32>,      // [DIM * KV_DIM]
    pub wo: Vec<f32>,      // [Q_DIM * DIM]
    pub w1: Vec<f32>,      // [DIM * HIDDEN]
    pub w3: Vec<f32>,      // [DIM * HIDDEN]
    pub w2: Vec<f32>,      // [DIM * HIDDEN]
    pub gamma1: Vec<f32>,  // [DIM]
    pub gamma2: Vec<f32>,  // [DIM]
}

/// Weight gradients (same layout as weights).
pub struct LayerGrads {
    pub dwq: Vec<f32>,
    pub dwk: Vec<f32>,
    pub dwv: Vec<f32>,
    pub dwo: Vec<f32>,
    pub dw1: Vec<f32>,
    pub dw3: Vec<f32>,
    pub dw2: Vec<f32>,
    pub dgamma1: Vec<f32>,
    pub dgamma2: Vec<f32>,
}

/// Cached activations from forward pass, needed for backward.
pub struct ForwardCache {
    pub x: Vec<f32>,           // layer input [DIM * SEQ]
    pub xnorm: Vec<f32>,       // after RMSNorm1 [DIM * SEQ]
    pub rms_inv1: Vec<f32>,    // per-position rms_inv [SEQ]
    pub q_rope: Vec<f32>,      // [Q_DIM * SEQ]
    pub k_rope: Vec<f32>,      // [KV_DIM * SEQ]
    pub v: Vec<f32>,           // [KV_DIM * SEQ]
    pub attn_out: Vec<f32>,    // [Q_DIM * SEQ]
    pub o_out: Vec<f32>,       // woFwd output [DIM * SEQ]
    pub x2: Vec<f32>,          // post-attn residual [DIM * SEQ]
    pub x2norm: Vec<f32>,      // after RMSNorm2 [DIM * SEQ]
    pub rms_inv2: Vec<f32>,    // per-position rms_inv [SEQ]
    pub h1: Vec<f32>,          // gate projection [HIDDEN * SEQ]
    pub h3: Vec<f32>,          // up projection [HIDDEN * SEQ]
    pub gate: Vec<f32>,        // silu(h1) * h3 [HIDDEN * SEQ]
}

/// Pre-allocated CPU-side staging scratch buffer.
/// A single Vec<f32> reused for every kernel call, sized to the largest
/// staging buffer needed. Eliminates ~60 Vec allocations per training step.
pub struct LayerScratch {
    pub stage: Vec<f32>,
}

impl LayerScratch {
    /// Allocate scratch buffer large enough for any kernel's staging data.
    pub fn allocate(cfg: &ModelConfig) -> Self {
        let dim = cfg.dim;
        let seq = cfg.seq;
        let q_dim = cfg.q_dim;
        let kv_dim = cfg.kv_dim;
        let hidden = cfg.hidden;

        // Compute max staging buffer size across all 10 kernels
        let sdpa_sp = sdpa_fwd::input_spatial_width(cfg);
        let wo_sp = dyn_matmul::spatial_width(seq, dim);
        let ffn_sp = ffn_fused::input_spatial_width(cfg);
        let w2t_sp = dyn_matmul::spatial_width(seq, hidden);
        let w13t_sp = dyn_matmul::dual_spatial_width(seq, dim);
        let wot_sp = dyn_matmul::spatial_width(seq, q_dim);
        let bwd1_in_ch = sdpa_bwd::bwd1_input_channels(cfg);
        let bwd2_in_ch = sdpa_bwd::bwd2_input_channels(cfg);
        let q_bwd_sp = dyn_matmul::spatial_width(seq, dim);
        let kv_bwd_sp = dyn_matmul::dual_spatial_width(seq, dim);

        let sizes = [
            dim * sdpa_sp,
            q_dim * wo_sp,
            dim * ffn_sp,
            dim * w2t_sp,
            hidden * w13t_sp,
            dim * wot_sp,
            bwd1_in_ch * seq,
            bwd2_in_ch * seq,
            q_dim * q_bwd_sp,
            kv_dim * kv_bwd_sp,
        ];
        let max_size = sizes.iter().copied().max().unwrap();

        Self {
            stage: vec![0.0f32; max_size],
        }
    }
}

/// Pre-allocated IOSurface buffers for all 10 kernels (input + output each).
/// Eliminates ~100 IOSurface alloc/dealloc cycles per training step.
/// All writes use `TensorData::copy_from_f32(&self, ..)` which takes `&self`,
/// so no interior mutability wrapper is needed.
pub struct KernelBuffers {
    // Forward: sdpa_fwd, wo_fwd, ffn_fused
    sdpa_fwd_in: TensorData,
    sdpa_fwd_out: TensorData,
    wo_fwd_in: TensorData,
    wo_fwd_out: TensorData,
    ffn_fused_in: TensorData,
    ffn_fused_out: TensorData,
    // Backward: ffn_bwd_w2t, ffn_bwd_w13t, wot_bwd, sdpa_bwd1, sdpa_bwd2, q_bwd, kv_bwd
    ffn_bwd_w2t_in: TensorData,
    ffn_bwd_w2t_out: TensorData,
    ffn_bwd_w13t_in: TensorData,
    ffn_bwd_w13t_out: TensorData,
    wot_bwd_in: TensorData,
    wot_bwd_out: TensorData,
    sdpa_bwd1_in: TensorData,
    sdpa_bwd1_out: TensorData,
    sdpa_bwd2_in: TensorData,
    sdpa_bwd2_out: TensorData,
    q_bwd_in: TensorData,
    q_bwd_out: TensorData,
    kv_bwd_in: TensorData,
    kv_bwd_out: TensorData,
}

impl KernelBuffers {
    /// Pre-allocate all IOSurface buffers for the given model config.
    fn allocate(cfg: &ModelConfig) -> Self {
        let dim = cfg.dim;
        let seq = cfg.seq;
        let q_dim = cfg.q_dim;
        let kv_dim = cfg.kv_dim;
        let hidden = cfg.hidden;

        // Forward: sdpa_fwd
        let sdpa_sp = sdpa_fwd::input_spatial_width(cfg);
        let sdpa_out_ch = sdpa_fwd::output_channels(cfg);
        let sdpa_fwd_in = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: sdpa_sp });
        let sdpa_fwd_out = TensorData::new(Shape { batch: 1, channels: sdpa_out_ch, height: 1, width: seq });

        // Forward: wo_fwd
        let wo_sp = dyn_matmul::spatial_width(seq, dim);
        let wo_fwd_in = TensorData::new(Shape { batch: 1, channels: q_dim, height: 1, width: wo_sp });
        let wo_fwd_out = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq });

        // Forward: ffn_fused
        let ffn_sp = ffn_fused::input_spatial_width(cfg);
        let ffn_out_ch = ffn_fused::output_channels(cfg);
        let ffn_fused_in = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: ffn_sp });
        let ffn_fused_out = TensorData::new(Shape { batch: 1, channels: ffn_out_ch, height: 1, width: seq });

        // Backward: ffn_bwd_w2t
        let w2t_sp = dyn_matmul::spatial_width(seq, hidden);
        let ffn_bwd_w2t_in = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: w2t_sp });
        let ffn_bwd_w2t_out = TensorData::new(Shape { batch: 1, channels: hidden, height: 1, width: seq });

        // Backward: ffn_bwd_w13t
        let w13t_sp = dyn_matmul::dual_spatial_width(seq, dim);
        let ffn_bwd_w13t_in = TensorData::new(Shape { batch: 1, channels: hidden, height: 1, width: w13t_sp });
        let ffn_bwd_w13t_out = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq });

        // Backward: wot_bwd
        let wot_sp = dyn_matmul::spatial_width(seq, q_dim);
        let wot_bwd_in = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: wot_sp });
        let wot_bwd_out = TensorData::new(Shape { batch: 1, channels: q_dim, height: 1, width: seq });

        // Backward: sdpa_bwd1
        let bwd1_in_ch = sdpa_bwd::bwd1_input_channels(cfg);
        let bwd1_out_ch = sdpa_bwd::bwd1_output_channels(cfg);
        let sdpa_bwd1_in = TensorData::new(Shape { batch: 1, channels: bwd1_in_ch, height: 1, width: seq });
        let sdpa_bwd1_out = TensorData::new(Shape { batch: 1, channels: bwd1_out_ch, height: 1, width: seq });

        // Backward: sdpa_bwd2
        let bwd2_in_ch = sdpa_bwd::bwd2_input_channels(cfg);
        let bwd2_out_ch = sdpa_bwd::bwd2_output_channels(cfg);
        let sdpa_bwd2_in = TensorData::new(Shape { batch: 1, channels: bwd2_in_ch, height: 1, width: seq });
        let sdpa_bwd2_out = TensorData::new(Shape { batch: 1, channels: bwd2_out_ch, height: 1, width: seq });

        // Backward: q_bwd
        let q_bwd_sp = dyn_matmul::spatial_width(seq, dim);
        let q_bwd_in = TensorData::new(Shape { batch: 1, channels: q_dim, height: 1, width: q_bwd_sp });
        let q_bwd_out = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq });

        // Backward: kv_bwd
        let kv_bwd_sp = dyn_matmul::dual_spatial_width(seq, dim);
        let kv_bwd_in = TensorData::new(Shape { batch: 1, channels: kv_dim, height: 1, width: kv_bwd_sp });
        let kv_bwd_out = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq });

        Self {
            sdpa_fwd_in, sdpa_fwd_out,
            wo_fwd_in, wo_fwd_out,
            ffn_fused_in, ffn_fused_out,
            ffn_bwd_w2t_in, ffn_bwd_w2t_out,
            ffn_bwd_w13t_in, ffn_bwd_w13t_out,
            wot_bwd_in, wot_bwd_out,
            sdpa_bwd1_in, sdpa_bwd1_out,
            sdpa_bwd2_in, sdpa_bwd2_out,
            q_bwd_in, q_bwd_out,
            kv_bwd_in, kv_bwd_out,
        }
    }
}

/// Compiled kernels for one layer (shared across layers since same dims).
pub struct CompiledKernels {
    pub sdpa_fwd: Executable,
    pub wo_fwd: Executable,
    pub ffn_fused: Executable,
    pub ffn_bwd_w2t: Executable,
    pub ffn_bwd_w13t: Executable,
    pub wot_bwd: Executable,
    pub sdpa_bwd1: Executable,
    pub sdpa_bwd2: Executable,
    pub q_bwd: Executable,
    pub kv_bwd: Executable,
    /// Pre-allocated IOSurface buffers for all kernels (avoids alloc/dealloc per call).
    bufs: KernelBuffers,
}

impl CompiledKernels {
    /// Compile all 10 kernels for the given model config.
    pub fn compile(cfg: &ModelConfig) -> Self {
        let qos = NSQualityOfService::UserInteractive;

        // Forward kernels
        let sdpa_fwd = sdpa_fwd::build(cfg).compile(qos).expect("sdpaFwd compile");
        let wo_fwd = dyn_matmul::build(cfg.q_dim, cfg.dim, cfg.seq)
            .compile(qos).expect("woFwd compile");
        let ffn_fused = ffn_fused::build(cfg).compile(qos).expect("ffnFused compile");

        // Backward kernels
        let ffn_bwd_w2t = dyn_matmul::build(cfg.dim, cfg.hidden, cfg.seq)
            .compile(qos).expect("ffnBwdW2t compile");
        let ffn_bwd_w13t = dyn_matmul::build_dual(cfg.hidden, cfg.dim, cfg.seq)
            .compile(qos).expect("ffnBwdW13t compile");
        let wot_bwd = dyn_matmul::build(cfg.dim, cfg.q_dim, cfg.seq)
            .compile(qos).expect("wotBwd compile");
        let sdpa_bwd1 = sdpa_bwd::build_bwd1(cfg).compile(qos).expect("sdpaBwd1 compile");
        let sdpa_bwd2 = sdpa_bwd::build_bwd2(cfg).compile(qos).expect("sdpaBwd2 compile");
        let q_bwd = dyn_matmul::build(cfg.q_dim, cfg.dim, cfg.seq)
            .compile(qos).expect("qBwd compile");
        let kv_bwd = dyn_matmul::build_dual(cfg.kv_dim, cfg.dim, cfg.seq)
            .compile(qos).expect("kvBwd compile");

        // Pre-allocate IOSurface buffers for all kernels
        let bufs = KernelBuffers::allocate(cfg);

        Self {
            sdpa_fwd, wo_fwd, ffn_fused,
            ffn_bwd_w2t, ffn_bwd_w13t, wot_bwd,
            sdpa_bwd1, sdpa_bwd2, q_bwd, kv_bwd,
            bufs,
        }
    }
}

impl LayerWeights {
    /// Initialize to match Obj-C reference (train.m):
    /// Wq/Wk/Wv: 1/√DIM, Wo/W2: zero-init (DeepNet), W1/W3: 1/√HIDDEN.
    pub fn random(cfg: &ModelConfig) -> Self {
        let scale_qkv = 1.0 / (cfg.dim as f32).sqrt();
        let scale_ffn = 1.0 / (cfg.hidden as f32).sqrt();
        Self {
            wq: random_vec(cfg.dim * cfg.q_dim, scale_qkv),
            wk: random_vec(cfg.dim * cfg.kv_dim, scale_qkv),
            wv: random_vec(cfg.dim * cfg.kv_dim, scale_qkv),
            wo: vec![0.0; cfg.q_dim * cfg.dim],     // zero-init (DeepNet)
            w1: random_vec(cfg.dim * cfg.hidden, scale_ffn),
            w3: random_vec(cfg.dim * cfg.hidden, scale_ffn),
            w2: vec![0.0; cfg.dim * cfg.hidden],     // zero-init (DeepNet)
            gamma1: vec![1.0; cfg.dim],
            gamma2: vec![1.0; cfg.dim],
        }
    }
}

impl LayerGrads {
    pub fn zeros(cfg: &ModelConfig) -> Self {
        Self {
            dwq: vec![0.0; cfg.dim * cfg.q_dim],
            dwk: vec![0.0; cfg.dim * cfg.kv_dim],
            dwv: vec![0.0; cfg.dim * cfg.kv_dim],
            dwo: vec![0.0; cfg.q_dim * cfg.dim],
            dw1: vec![0.0; cfg.dim * cfg.hidden],
            dw3: vec![0.0; cfg.dim * cfg.hidden],
            dw2: vec![0.0; cfg.dim * cfg.hidden],
            dgamma1: vec![0.0; cfg.dim],
            dgamma2: vec![0.0; cfg.dim],
        }
    }

    pub fn zero_out(&mut self) {
        self.dwq.fill(0.0);
        self.dwk.fill(0.0);
        self.dwv.fill(0.0);
        self.dwo.fill(0.0);
        self.dw1.fill(0.0);
        self.dw3.fill(0.0);
        self.dw2.fill(0.0);
        self.dgamma1.fill(0.0);
        self.dgamma2.fill(0.0);
    }
}

/// Simple LCG pseudo-random for reproducible init (no external dep).
fn random_vec(n: usize, scale: f32) -> Vec<f32> {
    let mut v = vec![0.0f32; n];
    let mut seed: u64 = 42 + n as u64;
    for x in v.iter_mut() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r = ((seed >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0;
        *x = r * scale;
    }
    v
}

// ── Helper: pack f32 data into ANE spatial layout ──

/// Stage activations into IOSurface spatial dimension.
/// `dst` is [channels * sp_width], `src` is [channels * src_width].
/// Writes src at spatial offset `sp_offset`.
/// Uses copy_from_slice for vectorized memcpy on inner dimension.
fn stage_spatial(dst: &mut [f32], channels: usize, sp_width: usize, src: &[f32], src_width: usize, sp_offset: usize) {
    for c in 0..channels {
        let d = c * sp_width + sp_offset;
        let s = c * src_width;
        dst[d..d + src_width].copy_from_slice(&src[s..s + src_width]);
    }
}

/// Read a slice of channels from ANE output buffer into a pre-allocated destination.
/// No-alloc version of the former `read_channels`.
/// Uses copy_from_slice for vectorized memcpy on inner dimension.
fn read_channels_into(src: &[f32], _total_ch: usize, seq: usize, ch_start: usize, ch_count: usize, dst: &mut [f32]) {
    for c in 0..ch_count {
        let d = c * seq;
        let s = (ch_start + c) * seq;
        dst[d..d + seq].copy_from_slice(&src[s..s + seq]);
    }
}

// ── Forward pass ──

/// Run forward pass for one transformer layer.
/// Returns (x_next, cache) where x_next is [DIM * SEQ].
pub fn forward(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &LayerWeights,
    x: &[f32],
) -> (Vec<f32>, ForwardCache) {
    let dim = cfg.dim;
    let seq = cfg.seq;
    let q_dim = cfg.q_dim;
    let kv_dim = cfg.kv_dim;
    let hidden = cfg.hidden;
    let alpha = 1.0 / (2.0 * cfg.nlayers as f32).sqrt();

    // 1. RMSNorm1 (CPU): per-position normalization
    let mut xnorm = vec![0.0f32; dim * seq];
    let mut rms_inv1 = vec![0.0f32; seq];
    let mut x_pos = vec![0.0f32; dim];
    let mut out_pos = vec![0.0f32; dim];
    for s in 0..seq {
        for c in 0..dim { x_pos[c] = x[c * seq + s]; }
        rms_inv1[s] = rmsnorm::forward(&x_pos, &weights.gamma1, &mut out_pos);
        for c in 0..dim { xnorm[c * seq + s] = out_pos[c]; }
    }

    // 2. Stage sdpaFwd directly into IOSurface (skip scratch buffer)
    let sdpa_sp = sdpa_fwd::input_spatial_width(cfg);
    let sdpa_out_ch = sdpa_fwd::output_channels(cfg);
    {
        let mut locked = kernels.bufs.sdpa_fwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, dim, sdpa_sp, &xnorm, seq, 0);
        stage_spatial(buf, dim, sdpa_sp, &weights.wq, q_dim, seq);
        stage_spatial(buf, dim, sdpa_sp, &weights.wk, kv_dim, seq + q_dim);
        stage_spatial(buf, dim, sdpa_sp, &weights.wv, kv_dim, seq + q_dim + kv_dim);
    }

    // 3. Run sdpaFwd (ANE)
    kernels.sdpa_fwd.run(&[&kernels.bufs.sdpa_fwd_in], &[&kernels.bufs.sdpa_fwd_out]).expect("ANE eval failed");

    // Extract: attn_out[Q_DIM,SEQ], Q_rope[Q_DIM,SEQ], K_rope[KV_DIM,SEQ], V[KV_DIM,SEQ]
    let mut attn_out = vec![0.0f32; q_dim * seq];
    let mut q_rope = vec![0.0f32; q_dim * seq];
    let mut k_rope = vec![0.0f32; kv_dim * seq];
    let mut v = vec![0.0f32; kv_dim * seq];
    {
        let locked = kernels.bufs.sdpa_fwd_out.as_f32_slice();
        read_channels_into(&locked, sdpa_out_ch, seq, 0, q_dim, &mut attn_out);
        read_channels_into(&locked, sdpa_out_ch, seq, q_dim, q_dim, &mut q_rope);
        read_channels_into(&locked, sdpa_out_ch, seq, 2 * q_dim, kv_dim, &mut k_rope);
        read_channels_into(&locked, sdpa_out_ch, seq, 2 * q_dim + kv_dim, kv_dim, &mut v);
    }

    // 4. Stage woFwd directly into IOSurface
    let wo_sp = dyn_matmul::spatial_width(seq, dim);
    {
        let mut locked = kernels.bufs.wo_fwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, q_dim, wo_sp, &attn_out, seq, 0);
        stage_spatial(buf, q_dim, wo_sp, &weights.wo, dim, seq);
    }

    // 5. Run woFwd (ANE)
    kernels.wo_fwd.run(&[&kernels.bufs.wo_fwd_in], &[&kernels.bufs.wo_fwd_out]).expect("ANE eval failed");

    // Read o_out directly from output IOSurface
    let mut o_out = vec![0.0f32; dim * seq];
    {
        let locked = kernels.bufs.wo_fwd_out.as_f32_slice();
        o_out.copy_from_slice(&locked[..dim * seq]);
    }

    // 6. Residual + RMSNorm2 (CPU)
    // x2 = x + alpha * o_out  (vDSP: vsma = o_out * alpha + x)
    let mut x2 = vec![0.0f32; dim * seq];
    vdsp::vsma(&o_out, alpha, x, &mut x2);
    let mut x2norm = vec![0.0f32; dim * seq];
    let mut rms_inv2 = vec![0.0f32; seq];
    // Reuse scratch buffers from RMSNorm1 above
    for s in 0..seq {
        for c in 0..dim { x_pos[c] = x2[c * seq + s]; }
        rms_inv2[s] = rmsnorm::forward(&x_pos, &weights.gamma2, &mut out_pos);
        for c in 0..dim { x2norm[c * seq + s] = out_pos[c]; }
    }

    // 7. Stage ffnFused directly into IOSurface
    let ffn_sp = ffn_fused::input_spatial_width(cfg);
    let ffn_out_ch = ffn_fused::output_channels(cfg);
    {
        let mut locked = kernels.bufs.ffn_fused_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, dim, ffn_sp, &x2norm, seq, 0);
        stage_spatial(buf, dim, ffn_sp, &x2, seq, seq);
        stage_spatial(buf, dim, ffn_sp, &weights.w1, hidden, 2 * seq);
        stage_spatial(buf, dim, ffn_sp, &weights.w3, hidden, 2 * seq + hidden);
        stage_spatial(buf, dim, ffn_sp, &weights.w2, hidden, 2 * seq + 2 * hidden);
    }

    // 8. Run ffnFused (ANE)
    kernels.ffn_fused.run(&[&kernels.bufs.ffn_fused_in], &[&kernels.bufs.ffn_fused_out]).expect("ANE eval failed");

    // Extract: x_next[DIM,SEQ], h1[HIDDEN,SEQ], h3[HIDDEN,SEQ], gate[HIDDEN,SEQ]
    let mut x_next = vec![0.0f32; dim * seq];
    let mut h1 = vec![0.0f32; hidden * seq];
    let mut h3 = vec![0.0f32; hidden * seq];
    let mut gate = vec![0.0f32; hidden * seq];
    {
        let locked = kernels.bufs.ffn_fused_out.as_f32_slice();
        read_channels_into(&locked, ffn_out_ch, seq, 0, dim, &mut x_next);
        read_channels_into(&locked, ffn_out_ch, seq, dim, hidden, &mut h1);
        read_channels_into(&locked, ffn_out_ch, seq, dim + hidden, hidden, &mut h3);
        read_channels_into(&locked, ffn_out_ch, seq, dim + 2 * hidden, hidden, &mut gate);
    }

    let cache = ForwardCache {
        x: x.to_vec(), xnorm, rms_inv1, q_rope, k_rope, v, attn_out, o_out,
        x2, x2norm, rms_inv2, h1, h3, gate,
    };

    (x_next, cache)
}

// ── Backward pass ──

/// Run backward pass for one transformer layer.
/// `dy` is gradient of loss w.r.t. layer output [DIM * SEQ].
/// Returns `dx` (gradient w.r.t. layer input) and fills `grads`.
pub fn backward(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &LayerWeights,
    cache: &ForwardCache,
    dy: &[f32],
    grads: &mut LayerGrads,
) -> Vec<f32> {
    let dim = cfg.dim;
    let seq = cfg.seq;
    let q_dim = cfg.q_dim;
    let kv_dim = cfg.kv_dim;
    let hidden = cfg.hidden;
    let heads = cfg.heads;
    let hd = cfg.hd;
    let alpha = 1.0 / (2.0 * cfg.nlayers as f32).sqrt();

    // ── 1. Scale dy for FFN residual (vDSP vectorized) ──
    let mut dffn = vec![0.0f32; dim * seq];
    vdsp::vsmul(dy, alpha, &mut dffn);

    // ── 2. ffnBwdW2t(ANE): dffn @ W2 → dsilu_raw [HIDDEN, SEQ] ──
    let w2t_sp = dyn_matmul::spatial_width(seq, hidden);
    {
        let mut locked = kernels.bufs.ffn_bwd_w2t_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, dim, w2t_sp, &dffn, seq, 0);
        stage_spatial(buf, dim, w2t_sp, &weights.w2, hidden, seq);
    }
    kernels.ffn_bwd_w2t.run(&[&kernels.bufs.ffn_bwd_w2t_in], &[&kernels.bufs.ffn_bwd_w2t_out]).expect("ANE eval failed");

    // Read dsilu_raw directly from output IOSurface
    let mut dsilu_raw = vec![0.0f32; hidden * seq];
    {
        let locked = kernels.bufs.ffn_bwd_w2t_out.as_f32_slice();
        dsilu_raw.copy_from_slice(&locked[..hidden * seq]);
    }

    // ── 3. SiLU derivative (CPU): dh1, dh3 from dsilu_raw, h1, h3 ──
    // dsilu_raw = dL/d(gate_out) where gate_out = silu(h1) * h3
    // dh1 = dsilu_raw * h3 * silu'(h1)
    // dh3 = dsilu_raw * silu(h1)
    let mut dh1 = vec![0.0f32; hidden * seq];
    let mut dh3 = vec![0.0f32; hidden * seq];
    for i in 0..hidden * seq {
        let sig = 1.0 / (1.0 + (-cache.h1[i]).exp());
        let silu_val = cache.h1[i] * sig;
        let silu_deriv = sig * (1.0 + cache.h1[i] * (1.0 - sig));
        dh3[i] = dsilu_raw[i] * silu_val;
        dh1[i] = dsilu_raw[i] * cache.h3[i] * silu_deriv;
    }

    // ── 4. Stage ffnBwdW13t directly into IOSurface, then overlap dW with ANE ──
    let w13t_sp = dyn_matmul::dual_spatial_width(seq, dim);
    {
        let mut locked = kernels.bufs.ffn_bwd_w13t_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, hidden, w13t_sp, &dh1, seq, 0);
        stage_spatial(buf, hidden, w13t_sp, &dh3, seq, seq);
        stage_spatial_transposed(buf, hidden, w13t_sp, &weights.w1, dim, hidden, 2 * seq);
        stage_spatial_transposed(buf, hidden, w13t_sp, &weights.w3, dim, hidden, 2 * seq + dim);
    }

    // ── 5. ASYNC: ANE ffnBwdW13t || CPU dW2+dW1+dW3 accumulation ──
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels.ffn_bwd_w13t.run(
                &[&kernels.bufs.ffn_bwd_w13t_in],
                &[&kernels.bufs.ffn_bwd_w13t_out],
            ).expect("ANE eval failed");
        });
        // CPU: dW accumulation while ANE runs (no data race — different memory)
        accumulate_dw(&dffn, dim, &cache.gate, hidden, seq, &mut grads.dw2);
        accumulate_dw(&cache.x2norm, dim, &dh1, hidden, seq, &mut grads.dw1);
        accumulate_dw(&cache.x2norm, dim, &dh3, hidden, seq, &mut grads.dw3);
        ane_handle.join().expect("ANE thread panicked");
    });

    // Read dx_ffn directly from output IOSurface
    let mut dx_ffn = vec![0.0f32; dim * seq];
    {
        let locked = kernels.bufs.ffn_bwd_w13t_out.as_f32_slice();
        dx_ffn.copy_from_slice(&locked[..dim * seq]);
    }

    // ── 6. RMSNorm2 backward (CPU) ──
    let mut dx2 = vec![0.0f32; dim * seq];
    let mut dy_pos = vec![0.0f32; dim];
    let mut x2_pos = vec![0.0f32; dim];
    let mut dx_pos = vec![0.0f32; dim];
    for s in 0..seq {
        for c in 0..dim {
            dy_pos[c] = dx_ffn[c * seq + s];
            x2_pos[c] = cache.x2[c * seq + s];
        }
        rmsnorm::backward(&dy_pos, &x2_pos, &weights.gamma2, cache.rms_inv2[s], &mut dx_pos, &mut grads.dgamma2);
        for c in 0..dim { dx2[c * seq + s] = dx_pos[c]; }
    }
    // Add residual gradient: dx2 += dy (FFN residual branch)
    for i in 0..dim * seq {
        dx2[i] += dy[i];
    }

    // ── 7. Scale dx2 for attention residual (vDSP vectorized) ──
    let mut dx2_scaled = vec![0.0f32; dim * seq];
    vdsp::vsmul(&dx2, alpha, &mut dx2_scaled);

    // ── 8. wotBwd(ANE): dx2_scaled @ Wo → da [Q_DIM, SEQ] ──
    let wot_sp = dyn_matmul::spatial_width(seq, q_dim);
    {
        let mut locked = kernels.bufs.wot_bwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, dim, wot_sp, &dx2_scaled, seq, 0);
        stage_spatial_transposed(buf, dim, wot_sp, &weights.wo, q_dim, dim, seq);
    }
    kernels.wot_bwd.run(&[&kernels.bufs.wot_bwd_in], &[&kernels.bufs.wot_bwd_out]).expect("ANE eval failed");

    // Read da directly from output IOSurface
    let mut da = vec![0.0f32; q_dim * seq];
    {
        let locked = kernels.bufs.wot_bwd_out.as_f32_slice();
        da.copy_from_slice(&locked[..q_dim * seq]);
    }

    // ── 9. Stage sdpaBwd1 directly into IOSurface, then overlap dWo with ANE ──
    let bwd1_in_ch = sdpa_bwd::bwd1_input_channels(cfg);
    let bwd1_out_ch = sdpa_bwd::bwd1_output_channels(cfg);
    {
        let mut locked = kernels.bufs.sdpa_bwd1_in.as_f32_slice_mut();
        let buf = &mut *locked;
        pack_channels(buf, bwd1_in_ch, seq, &cache.q_rope, q_dim, 0);
        pack_channels(buf, bwd1_in_ch, seq, &cache.k_rope, q_dim, q_dim);
        pack_channels(buf, bwd1_in_ch, seq, &cache.v, q_dim, 2 * q_dim);
        pack_channels(buf, bwd1_in_ch, seq, &da, q_dim, 3 * q_dim);
    }

    // ASYNC: ANE sdpaBwd1 || CPU dWo accumulation
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels.sdpa_bwd1.run(
                &[&kernels.bufs.sdpa_bwd1_in],
                &[&kernels.bufs.sdpa_bwd1_out],
            ).expect("ANE eval failed");
        });
        accumulate_dw(&cache.attn_out, q_dim, &dx2_scaled, dim, seq, &mut grads.dwo);
        ane_handle.join().expect("ANE thread panicked");
    });

    let score_ch = heads * seq;
    let mut dv_full = vec![0.0f32; q_dim * seq];
    let mut probs_flat = vec![0.0f32; score_ch * seq];
    let mut dp_flat = vec![0.0f32; score_ch * seq];
    {
        let locked = kernels.bufs.sdpa_bwd1_out.as_f32_slice();
        read_channels_into(&locked, bwd1_out_ch, seq, 0, q_dim, &mut dv_full);
        read_channels_into(&locked, bwd1_out_ch, seq, q_dim, score_ch, &mut probs_flat);
        read_channels_into(&locked, bwd1_out_ch, seq, q_dim + score_ch, score_ch, &mut dp_flat);
    }

    // ── 10. sdpaBwd2(ANE): probs, dp, Q_rope, K_rope → dQ, dK ──
    let bwd2_in_ch = sdpa_bwd::bwd2_input_channels(cfg);
    let bwd2_out_ch = sdpa_bwd::bwd2_output_channels(cfg);
    {
        let mut locked = kernels.bufs.sdpa_bwd2_in.as_f32_slice_mut();
        let buf = &mut *locked;
        pack_channels(buf, bwd2_in_ch, seq, &probs_flat, score_ch, 0);
        pack_channels(buf, bwd2_in_ch, seq, &dp_flat, score_ch, score_ch);
        pack_channels(buf, bwd2_in_ch, seq, &cache.q_rope, q_dim, 2 * score_ch);
        pack_channels(buf, bwd2_in_ch, seq, &cache.k_rope, q_dim, 2 * score_ch + q_dim);
    }
    kernels.sdpa_bwd2.run(&[&kernels.bufs.sdpa_bwd2_in], &[&kernels.bufs.sdpa_bwd2_out]).expect("ANE eval failed");

    let mut dq = vec![0.0f32; q_dim * seq];
    let mut dk = vec![0.0f32; q_dim * seq];
    {
        let locked = kernels.bufs.sdpa_bwd2_out.as_f32_slice();
        read_channels_into(&locked, bwd2_out_ch, seq, 0, q_dim, &mut dq);
        read_channels_into(&locked, bwd2_out_ch, seq, q_dim, q_dim, &mut dk);
    }
    // For MHA, dV_full = dV (no GQA reduce needed)
    let dv = dv_full;

    // ── 11. RoPE backward in-place (CPU) ──
    rope_backward_inplace(&mut dq, heads, hd, seq);
    rope_backward_inplace(&mut dk, heads, hd, seq);

    // ── 12. Stage qBwd directly into IOSurface, then overlap dW with ANE ──
    let q_bwd_sp = dyn_matmul::spatial_width(seq, dim);
    {
        let mut locked = kernels.bufs.q_bwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, q_dim, q_bwd_sp, &dq, seq, 0);
        stage_spatial_transposed(buf, q_dim, q_bwd_sp, &weights.wq, dim, q_dim, seq);
    }

    // ASYNC: ANE qBwd || CPU dWq+dWk+dWv accumulation
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels.q_bwd.run(
                &[&kernels.bufs.q_bwd_in],
                &[&kernels.bufs.q_bwd_out],
            ).expect("ANE eval failed");
        });
        accumulate_dw(&cache.xnorm, dim, &dq, q_dim, seq, &mut grads.dwq);
        accumulate_dw(&cache.xnorm, dim, &dk, kv_dim, seq, &mut grads.dwk);
        accumulate_dw(&cache.xnorm, dim, &dv, kv_dim, seq, &mut grads.dwv);
        ane_handle.join().expect("ANE thread panicked");
    });

    // Read dx_attn directly from output IOSurface
    let mut dx_attn = vec![0.0f32; dim * seq];
    {
        let locked = kernels.bufs.q_bwd_out.as_f32_slice();
        dx_attn.copy_from_slice(&locked[..dim * seq]);
    }

    // ── 14. kvBwd(ANE): dk@Wk + dv@Wv → dx_kv [DIM, SEQ] ──
    let kv_bwd_sp = dyn_matmul::dual_spatial_width(seq, dim);
    {
        let mut locked = kernels.bufs.kv_bwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, kv_dim, kv_bwd_sp, &dk, seq, 0);
        stage_spatial(buf, kv_dim, kv_bwd_sp, &dv, seq, seq);
        stage_spatial_transposed(buf, kv_dim, kv_bwd_sp, &weights.wk, dim, kv_dim, seq + seq);
        stage_spatial_transposed(buf, kv_dim, kv_bwd_sp, &weights.wv, dim, kv_dim, 2 * seq + dim);
    }
    kernels.kv_bwd.run(&[&kernels.bufs.kv_bwd_in], &[&kernels.bufs.kv_bwd_out]).expect("ANE eval failed");

    // Read dx_kv directly from output IOSurface
    let mut dx_kv = vec![0.0f32; dim * seq];
    {
        let locked = kernels.bufs.kv_bwd_out.as_f32_slice();
        dx_kv.copy_from_slice(&locked[..dim * seq]);
    }

    // ── 15. Merge: dx_attn + dx_kv (vDSP vectorized) ──
    let mut dx_merged = vec![0.0f32; dim * seq];
    vdsp::vadd(&dx_attn, &dx_kv, &mut dx_merged);

    // ── 16. RMSNorm1 backward (CPU) ──
    let mut dx_rms1 = vec![0.0f32; dim * seq];
    // Reuse scratch buffers from RMSNorm2 backward above
    for s in 0..seq {
        for c in 0..dim {
            dy_pos[c] = dx_merged[c * seq + s];
            x2_pos[c] = cache.x[c * seq + s];
        }
        rmsnorm::backward(&dy_pos, &x2_pos, &weights.gamma1, cache.rms_inv1[s], &mut dx_pos, &mut grads.dgamma1);
        for c in 0..dim { dx_rms1[c * seq + s] = dx_pos[c]; }
    }

    // ── 17. Final: dx = dx_rms1 + dx2 (residual from attention branch, vDSP vectorized) ──
    let mut dx = vec![0.0f32; dim * seq];
    vdsp::vadd(&dx_rms1, &dx2, &mut dx);

    dx
}

// ── CPU helpers ──

/// Accumulate weight gradient via BLAS: dW[a_ch, b_ch] += A[a_ch, seq] @ B[b_ch, seq]^T
/// `a` is [a_ch * seq] row-major, `b` is [b_ch * seq] row-major, `dw` is [a_ch * b_ch].
fn accumulate_dw(a: &[f32], a_ch: usize, b: &[f32], b_ch: usize, seq: usize, dw: &mut [f32]) {
    vdsp::sgemm_at(a, a_ch, seq, b, b_ch, dw);
}

/// Stage a transposed weight matrix into the IOSurface spatial dimension.
/// Source is [src_rows, src_cols], destination needs [src_cols, src_rows] at sp_offset.
fn stage_spatial_transposed(
    dst: &mut [f32], _channels: usize, sp_width: usize,
    src: &[f32], src_rows: usize, src_cols: usize, sp_offset: usize,
) {
    // src layout: src[row * src_cols + col]
    // dst layout: dst[col * sp_width + sp_offset + row]
    // where channels = src_cols, transposed width = src_rows
    for col in 0..src_cols {
        for row in 0..src_rows {
            dst[col * sp_width + sp_offset + row] = src[row * src_cols + col];
        }
    }
}

/// Pack activation data into the channel dimension of an IOSurface.
/// Uses copy_from_slice for vectorized memcpy on inner dimension.
fn pack_channels(dst: &mut [f32], _total_ch: usize, seq: usize, src: &[f32], src_ch: usize, ch_offset: usize) {
    for c in 0..src_ch {
        let d = (ch_offset + c) * seq;
        let s = c * seq;
        dst[d..d + seq].copy_from_slice(&src[s..s + seq]);
    }
}

/// RoPE backward: inverse rotation applied in-place.
/// For each head h, position p, pair index i:
///   [dq0, dq1] = [[cos, sin], [-sin, cos]] @ [dx0, dx1]
fn rope_backward_inplace(dx: &mut [f32], heads: usize, hd: usize, seq: usize) {
    for h in 0..heads {
        for p in 0..seq {
            for i in 0..hd / 2 {
                let theta = p as f32 / 10000.0f32.powf(2.0 * i as f32 / hd as f32);
                let c = theta.cos();
                let s = theta.sin();
                let idx0 = (h * hd + 2 * i) * seq + p;
                let idx1 = (h * hd + 2 * i + 1) * seq + p;
                let d0 = dx[idx0];
                let d1 = dx[idx1];
                // Inverse rotation: [cos, sin; -sin, cos]
                dx[idx0] = c * d0 + s * d1;
                dx[idx1] = -s * d0 + c * d1;
            }
        }
    }
}
