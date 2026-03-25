//! Single transformer layer: compile kernels, run forward/backward on ANE + CPU.
//!
//! Forward: RMSNorm1(CPU) → sdpaFwd(ANE) → woFwd(ANE) → residual+RMSNorm2(CPU) → ffnFused(ANE)
//! Backward: Scale dy → ffnBwdW2t(ANE) → SiLU'(CPU) → ffnBwdW13t(ANE) → RMSNorm2 bwd(CPU)
//!           → wotBwd(ANE) → sdpaBwd1(ANE) → sdpaBwd2(ANE) → RoPE bwd(CPU)
//!           → qBwd(ANE) → kvBwd(ANE) → RMSNorm1 bwd(CPU)

use crate::cpu::{rmsnorm, silu, vdsp};
use crate::kernels::{dyn_matmul, ffn_fused, sdpa_bwd, sdpa_fwd};
use crate::model::ModelConfig;
use ane_bridge::ane::{Executable, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::sync::OnceLock;
use std::time::Instant;

fn shape_elements(shape: &Shape) -> usize {
    shape.batch * shape.channels * shape.height * shape.width
}

fn iosurface_alloc_logging_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("RUSTANE_LOG_IOSURFACE_ALLOC")
            .ok()
            .map(|value| {
                matches!(
                    value.as_str(),
                    "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
                )
            })
            .unwrap_or(false)
    })
}

pub(crate) fn tensor_data_new_logged(label: &'static str, shape: Shape) -> TensorData {
    if iosurface_alloc_logging_enabled() {
        let elements = shape_elements(&shape);
        let bytes = elements * std::mem::size_of::<f32>();
        eprintln!(
            "IOSurface alloc {label}: batch={} channels={} height={} width={} elems={} bytes={}",
            shape.batch, shape.channels, shape.height, shape.width, elements, bytes
        );
    }
    TensorData::new(shape)
}

/// Per-layer weights (f32, CPU-side).
pub struct LayerWeights {
    pub wq: Vec<f32>,     // [DIM * Q_DIM]
    pub wk: Vec<f32>,     // [DIM * KV_DIM]
    pub wv: Vec<f32>,     // [DIM * KV_DIM]
    pub wo: Vec<f32>,     // [Q_DIM * DIM]
    pub w1: Vec<f32>,     // [DIM * HIDDEN]
    pub w3: Vec<f32>,     // [DIM * HIDDEN]
    pub w2: Vec<f32>,     // [DIM * HIDDEN]
    pub gamma1: Vec<f32>, // [DIM]
    pub gamma2: Vec<f32>, // [DIM]
    pub w2_generation: u64,
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
    pub w2t_scratch: Vec<f32>, // [HIDDEN * DIM] for W2 transpose (decomposed FFN only)
    pub w2t_generation: u64,   // generation of weights.w2 currently packed into w2t_scratch
}

impl ForwardCache {
    /// Pre-allocate all cache buffers for the given model config.
    /// Buffers are fully overwritten by forward_into — no zeroing needed at reuse.
    pub fn new(cfg: &ModelConfig) -> Self {
        let dim = cfg.dim;
        let seq = cfg.seq;
        let q_dim = cfg.q_dim;
        let kv_dim = cfg.kv_dim;
        let hidden = cfg.hidden;
        Self {
            x: vec![0.0; dim * seq],
            xnorm: vec![0.0; dim * seq],
            rms_inv1: vec![0.0; seq],
            q_rope: vec![0.0; q_dim * seq],
            k_rope: vec![0.0; kv_dim * seq],
            v: vec![0.0; kv_dim * seq],
            attn_out: vec![0.0; q_dim * seq],
            o_out: vec![0.0; dim * seq],
            x2: vec![0.0; dim * seq],
            x2norm: vec![0.0; dim * seq],
            rms_inv2: vec![0.0; seq],
            h1: vec![0.0; hidden * seq],
            h3: vec![0.0; hidden * seq],
            gate: vec![0.0; hidden * seq],
            w2t_scratch: vec![0.0; hidden * dim],
            w2t_generation: u64::MAX,
        }
    }
}

/// Pre-allocated IOSurface buffers for all 10 kernels (input + output each).
/// Eliminates ~100 IOSurface alloc/dealloc cycles per training step.
/// All writes use `TensorData::copy_from_f32(&self, ..)` which takes `&self`,
/// so no interior mutability wrapper is needed.
pub struct KernelBuffers {
    // Forward: sdpa_fwd, wo_fwd, ffn_fused
    sdpa_fwd_xnorm_in: TensorData,
    sdpa_fwd_wq_in: TensorData,
    sdpa_fwd_wk_in: TensorData,
    sdpa_fwd_wv_in: TensorData,
    sdpa_fwd_attn_out: TensorData,
    sdpa_fwd_q_rope_out: TensorData,
    sdpa_fwd_k_rope_out: TensorData,
    sdpa_fwd_v_out: TensorData,
    wo_fwd_in: TensorData,
    wo_fwd_out: TensorData,
    ffn_fused_in: Option<TensorData>,
    ffn_fused_out: Option<TensorData>,
    // Decomposed FFN forward (only populated when needs_decomposition)
    ffn_w13_in: Option<TensorData>,  // [1, dim, 1, seq+hidden]
    ffn_w13_out: Option<TensorData>, // [1, hidden, 1, seq]
    ffn_w2_in: Option<TensorData>,   // [1, hidden, 1, seq+dim]
    ffn_w2_out: Option<TensorData>,  // [1, dim, 1, seq]
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
        let td = |label: &'static str, shape: Shape| tensor_data_new_logged(label, shape);
        let dim = cfg.dim;
        let seq = cfg.seq;
        let q_dim = cfg.q_dim;
        let kv_dim = cfg.kv_dim;
        let hidden = cfg.hidden;

        // Forward: sdpa_fwd
        let sdpa_fwd_xnorm_in = td("sdpa_fwd_xnorm_in", sdpa_fwd::xnorm_shape(cfg));
        let sdpa_fwd_wq_in = td("sdpa_fwd_wq_in", sdpa_fwd::wq_shape(cfg));
        let sdpa_fwd_wk_in = td("sdpa_fwd_wk_in", sdpa_fwd::wk_shape(cfg));
        let sdpa_fwd_wv_in = td("sdpa_fwd_wv_in", sdpa_fwd::wv_shape(cfg));
        let sdpa_fwd_attn_out = td("sdpa_fwd_attn_out", sdpa_fwd::attn_out_shape(cfg));
        let sdpa_fwd_q_rope_out = td("sdpa_fwd_q_rope_out", sdpa_fwd::q_rope_shape(cfg));
        let sdpa_fwd_k_rope_out = td("sdpa_fwd_k_rope_out", sdpa_fwd::k_rope_shape(cfg));
        let sdpa_fwd_v_out = td("sdpa_fwd_v_out", sdpa_fwd::v_shape(cfg));

        // Forward: wo_fwd
        let wo_sp = dyn_matmul::spatial_width(seq, dim);
        let wo_fwd_in = td(
            "wo_fwd_in",
            Shape {
                batch: 1,
                channels: q_dim,
                height: 1,
                width: wo_sp,
            },
        );
        let wo_fwd_out = td(
            "wo_fwd_out",
            Shape {
                batch: 1,
                channels: dim,
                height: 1,
                width: seq,
            },
        );

        // Forward: ffn_fused (or decomposed)
        let decomposed = ffn_fused::needs_decomposition(cfg);
        let use_dual_w13 = decomposed && ffn_fused::can_use_dual_w13(cfg);
        let (ffn_fused_in, ffn_fused_out, ffn_w13_in, ffn_w13_out, ffn_w2_in, ffn_w2_out) =
            if decomposed {
                let (w13_sp, w13_out_ch) = if use_dual_w13 {
                    (
                        dyn_matmul::dual_separate_spatial_width(seq, hidden),
                        2 * hidden,
                    )
                } else {
                    (dyn_matmul::spatial_width(seq, hidden), hidden)
                };
                let w2_sp = dyn_matmul::spatial_width(seq, dim);
                (
                    None,
                    None,
                    Some(td(
                        "ffn_w13_in",
                        Shape {
                            batch: 1,
                            channels: dim,
                            height: 1,
                            width: w13_sp,
                        },
                    )),
                    Some(td(
                        "ffn_w13_out",
                        Shape {
                            batch: 1,
                            channels: w13_out_ch,
                            height: 1,
                            width: seq,
                        },
                    )),
                    Some(td(
                        "ffn_w2_in",
                        Shape {
                            batch: 1,
                            channels: hidden,
                            height: 1,
                            width: w2_sp,
                        },
                    )),
                    Some(td(
                        "ffn_w2_out",
                        Shape {
                            batch: 1,
                            channels: dim,
                            height: 1,
                            width: seq,
                        },
                    )),
                )
            } else {
                let ffn_sp = ffn_fused::input_spatial_width(cfg);
                let ffn_out_ch = ffn_fused::output_channels(cfg);
                (
                    Some(td(
                        "ffn_fused_in",
                        Shape {
                            batch: 1,
                            channels: dim,
                            height: 1,
                            width: ffn_sp,
                        },
                    )),
                    Some(td(
                        "ffn_fused_out",
                        Shape {
                            batch: 1,
                            channels: ffn_out_ch,
                            height: 1,
                            width: seq,
                        },
                    )),
                    None,
                    None,
                    None,
                    None,
                )
            };

        // Backward: ffn_bwd_w2t
        let w2t_sp = dyn_matmul::spatial_width(seq, hidden);
        let ffn_bwd_w2t_in = td(
            "ffn_bwd_w2t_in",
            Shape {
                batch: 1,
                channels: dim,
                height: 1,
                width: w2t_sp,
            },
        );
        let ffn_bwd_w2t_out = td(
            "ffn_bwd_w2t_out",
            Shape {
                batch: 1,
                channels: hidden,
                height: 1,
                width: seq,
            },
        );

        // Backward: ffn_bwd_w13t
        let w13t_sp = dyn_matmul::dual_spatial_width(seq, dim);
        let ffn_bwd_w13t_in = td(
            "ffn_bwd_w13t_in",
            Shape {
                batch: 1,
                channels: hidden,
                height: 1,
                width: w13t_sp,
            },
        );
        let ffn_bwd_w13t_out = td(
            "ffn_bwd_w13t_out",
            Shape {
                batch: 1,
                channels: dim,
                height: 1,
                width: seq,
            },
        );

        // Backward: wot_bwd
        let wot_sp = dyn_matmul::spatial_width(seq, q_dim);
        let wot_bwd_in = td(
            "wot_bwd_in",
            Shape {
                batch: 1,
                channels: dim,
                height: 1,
                width: wot_sp,
            },
        );
        let wot_bwd_out = td(
            "wot_bwd_out",
            Shape {
                batch: 1,
                channels: q_dim,
                height: 1,
                width: seq,
            },
        );

        // Backward: sdpa_bwd1
        let bwd1_in_ch = sdpa_bwd::bwd1_input_channels(cfg);
        let bwd1_out_ch = sdpa_bwd::bwd1_output_channels(cfg);
        let sdpa_bwd1_in = td(
            "sdpa_bwd1_in",
            Shape {
                batch: 1,
                channels: bwd1_in_ch,
                height: 1,
                width: seq,
            },
        );
        let sdpa_bwd1_out = td(
            "sdpa_bwd1_out",
            Shape {
                batch: 1,
                channels: bwd1_out_ch,
                height: 1,
                width: seq,
            },
        );

        // Backward: sdpa_bwd2
        let bwd2_in_ch = sdpa_bwd::bwd2_input_channels(cfg);
        let bwd2_out_ch = sdpa_bwd::bwd2_output_channels(cfg);
        let sdpa_bwd2_in = td(
            "sdpa_bwd2_in",
            Shape {
                batch: 1,
                channels: bwd2_in_ch,
                height: 1,
                width: seq,
            },
        );
        let sdpa_bwd2_out = td(
            "sdpa_bwd2_out",
            Shape {
                batch: 1,
                channels: bwd2_out_ch,
                height: 1,
                width: seq,
            },
        );

        // Backward: q_bwd
        let q_bwd_sp = dyn_matmul::spatial_width(seq, dim);
        let q_bwd_in = td(
            "q_bwd_in",
            Shape {
                batch: 1,
                channels: q_dim,
                height: 1,
                width: q_bwd_sp,
            },
        );
        let q_bwd_out = td(
            "q_bwd_out",
            Shape {
                batch: 1,
                channels: dim,
                height: 1,
                width: seq,
            },
        );

        // Backward: kv_bwd
        let kv_bwd_sp = dyn_matmul::dual_spatial_width(seq, dim);
        let kv_bwd_in = td(
            "kv_bwd_in",
            Shape {
                batch: 1,
                channels: kv_dim,
                height: 1,
                width: kv_bwd_sp,
            },
        );
        let kv_bwd_out = td(
            "kv_bwd_out",
            Shape {
                batch: 1,
                channels: dim,
                height: 1,
                width: seq,
            },
        );

        Self {
            sdpa_fwd_xnorm_in,
            sdpa_fwd_wq_in,
            sdpa_fwd_wk_in,
            sdpa_fwd_wv_in,
            sdpa_fwd_attn_out,
            sdpa_fwd_q_rope_out,
            sdpa_fwd_k_rope_out,
            sdpa_fwd_v_out,
            wo_fwd_in,
            wo_fwd_out,
            ffn_fused_in,
            ffn_fused_out,
            ffn_w13_in,
            ffn_w13_out,
            ffn_w2_in,
            ffn_w2_out,
            ffn_bwd_w2t_in,
            ffn_bwd_w2t_out,
            ffn_bwd_w13t_in,
            ffn_bwd_w13t_out,
            wot_bwd_in,
            wot_bwd_out,
            sdpa_bwd1_in,
            sdpa_bwd1_out,
            sdpa_bwd2_in,
            sdpa_bwd2_out,
            q_bwd_in,
            q_bwd_out,
            kv_bwd_in,
            kv_bwd_out,
        }
    }

    /// Pre-allocate only the forward IOSurface buffers.
    /// Backward buffers are reduced to tiny placeholders because forward-only
    /// benchmarks never touch them.
    fn allocate_forward_only(cfg: &ModelConfig) -> Self {
        let td = |label: &'static str, shape: Shape| tensor_data_new_logged(label, shape);
        let dim = cfg.dim;
        let seq = cfg.seq;
        let q_dim = cfg.q_dim;
        let hidden = cfg.hidden;

        let sdpa_fwd_xnorm_in = td("sdpa_fwd_xnorm_in", sdpa_fwd::xnorm_shape(cfg));
        let sdpa_fwd_wq_in = td("sdpa_fwd_wq_in", sdpa_fwd::wq_shape(cfg));
        let sdpa_fwd_wk_in = td("sdpa_fwd_wk_in", sdpa_fwd::wk_shape(cfg));
        let sdpa_fwd_wv_in = td("sdpa_fwd_wv_in", sdpa_fwd::wv_shape(cfg));
        let sdpa_fwd_attn_out = td("sdpa_fwd_attn_out", sdpa_fwd::attn_out_shape(cfg));
        let sdpa_fwd_q_rope_out = td("sdpa_fwd_q_rope_out", sdpa_fwd::q_rope_shape(cfg));
        let sdpa_fwd_k_rope_out = td("sdpa_fwd_k_rope_out", sdpa_fwd::k_rope_shape(cfg));
        let sdpa_fwd_v_out = td("sdpa_fwd_v_out", sdpa_fwd::v_shape(cfg));

        let wo_sp = dyn_matmul::spatial_width(seq, dim);
        let wo_fwd_in = td(
            "wo_fwd_in",
            Shape {
                batch: 1,
                channels: q_dim,
                height: 1,
                width: wo_sp,
            },
        );
        let wo_fwd_out = td(
            "wo_fwd_out",
            Shape {
                batch: 1,
                channels: dim,
                height: 1,
                width: seq,
            },
        );

        let decomposed = ffn_fused::needs_decomposition(cfg);
        let use_dual_w13 = decomposed && ffn_fused::can_use_dual_w13(cfg);
        let (ffn_fused_in, ffn_fused_out, ffn_w13_in, ffn_w13_out, ffn_w2_in, ffn_w2_out) =
            if decomposed {
                let (w13_sp, w13_out_ch) = if use_dual_w13 {
                    (
                        dyn_matmul::dual_separate_spatial_width(seq, hidden),
                        2 * hidden,
                    )
                } else {
                    (dyn_matmul::spatial_width(seq, hidden), hidden)
                };
                let w2_sp = dyn_matmul::spatial_width(seq, dim);
                (
                    None,
                    None,
                    Some(td(
                        "ffn_w13_in",
                        Shape {
                            batch: 1,
                            channels: dim,
                            height: 1,
                            width: w13_sp,
                        },
                    )),
                    Some(td(
                        "ffn_w13_out",
                        Shape {
                            batch: 1,
                            channels: w13_out_ch,
                            height: 1,
                            width: seq,
                        },
                    )),
                    Some(td(
                        "ffn_w2_in",
                        Shape {
                            batch: 1,
                            channels: hidden,
                            height: 1,
                            width: w2_sp,
                        },
                    )),
                    Some(td(
                        "ffn_w2_out",
                        Shape {
                            batch: 1,
                            channels: dim,
                            height: 1,
                            width: seq,
                        },
                    )),
                )
            } else {
                let ffn_sp = ffn_fused::input_spatial_width(cfg);
                let ffn_out_ch = ffn_fused::output_channels(cfg);
                (
                    Some(td(
                        "ffn_fused_in",
                        Shape {
                            batch: 1,
                            channels: dim,
                            height: 1,
                            width: ffn_sp,
                        },
                    )),
                    Some(td(
                        "ffn_fused_out",
                        Shape {
                            batch: 1,
                            channels: ffn_out_ch,
                            height: 1,
                            width: seq,
                        },
                    )),
                    None,
                    None,
                    None,
                    None,
                )
            };

        let tiny = |label: &'static str| {
            td(
                label,
                Shape {
                    batch: 1,
                    channels: 1,
                    height: 1,
                    width: 1,
                },
            )
        };

        Self {
            sdpa_fwd_xnorm_in,
            sdpa_fwd_wq_in,
            sdpa_fwd_wk_in,
            sdpa_fwd_wv_in,
            sdpa_fwd_attn_out,
            sdpa_fwd_q_rope_out,
            sdpa_fwd_k_rope_out,
            sdpa_fwd_v_out,
            wo_fwd_in,
            wo_fwd_out,
            ffn_fused_in,
            ffn_fused_out,
            ffn_w13_in,
            ffn_w13_out,
            ffn_w2_in,
            ffn_w2_out,
            ffn_bwd_w2t_in: tiny("ffn_bwd_w2t_in_tiny"),
            ffn_bwd_w2t_out: tiny("ffn_bwd_w2t_out_tiny"),
            ffn_bwd_w13t_in: tiny("ffn_bwd_w13t_in_tiny"),
            ffn_bwd_w13t_out: tiny("ffn_bwd_w13t_out_tiny"),
            wot_bwd_in: tiny("wot_bwd_in_tiny"),
            wot_bwd_out: tiny("wot_bwd_out_tiny"),
            sdpa_bwd1_in: tiny("sdpa_bwd1_in_tiny"),
            sdpa_bwd1_out: tiny("sdpa_bwd1_out_tiny"),
            sdpa_bwd2_in: tiny("sdpa_bwd2_in_tiny"),
            sdpa_bwd2_out: tiny("sdpa_bwd2_out_tiny"),
            q_bwd_in: tiny("q_bwd_in_tiny"),
            q_bwd_out: tiny("q_bwd_out_tiny"),
            kv_bwd_in: tiny("kv_bwd_in_tiny"),
            kv_bwd_out: tiny("kv_bwd_out_tiny"),
        }
    }
}

/// Pre-computed RoPE cos/sin tables (deterministic, depends only on hd and seq).
/// Eliminates 12× per-step recomputation of powf+cos+sin over 16K elements.
pub struct RopeTable {
    pub cos: Vec<f32>, // [pairs * seq] where pairs = hd/2
    pub sin: Vec<f32>, // [pairs * seq]
}

impl RopeTable {
    fn compute(hd: usize, seq: usize) -> Self {
        let pairs = hd / 2;
        let mut cos = vec![0.0f32; pairs * seq];
        let mut sin = vec![0.0f32; pairs * seq];
        for i in 0..pairs {
            let freq = 1.0 / 10000.0f32.powf(2.0 * i as f32 / hd as f32);
            for p in 0..seq {
                let theta = p as f32 * freq;
                cos[i * seq + p] = theta.cos();
                sin[i * seq + p] = theta.sin();
            }
        }
        Self { cos, sin }
    }
}

/// Compiled kernels for one layer (shared across layers since same dims).
pub struct CompiledKernels {
    pub sdpa_fwd: Executable,
    pub wo_fwd: Executable,
    pub ffn_fused: Option<Executable>, // Some when fused, None when decomposed
    // Decomposed FFN forward (only populated when needs_decomposition)
    ffn_w13_fwd: Option<Executable>, // dual_separate or single dyn_matmul
    ffn_w2_fwd: Option<Executable>,  // dyn_matmul(hidden, dim, seq)
    pub use_decomposed_ffn: bool,
    pub use_dual_w13: bool, // true when W1+W3 fused into single dispatch
    pub ffn_bwd_w2t: Executable,
    pub ffn_bwd_w13t: Executable,
    pub wot_bwd: Executable,
    pub sdpa_bwd1: Executable,
    pub sdpa_bwd2: Executable,
    pub q_bwd: Executable,
    pub kv_bwd: Executable,
    /// Pre-allocated IOSurface buffers for all kernels (avoids alloc/dealloc per call).
    bufs: KernelBuffers,
    /// Pre-computed RoPE tables (avoids 12× per-step recomputation).
    pub rope: RopeTable,
}

impl CompiledKernels {
    /// Compile all 10 kernels for the given model config.
    pub fn compile(cfg: &ModelConfig) -> Self {
        let qos = NSQualityOfService::UserInteractive;

        // Forward kernels
        let sdpa_fwd = sdpa_fwd::build(cfg).compile(qos).expect("sdpaFwd compile");
        let wo_fwd = dyn_matmul::build(cfg.q_dim, cfg.dim, cfg.seq)
            .compile(qos)
            .expect("woFwd compile");

        let decomposed = ffn_fused::needs_decomposition(cfg);
        let ffn_fused_exe = if decomposed {
            None
        } else {
            Some(
                ffn_fused::build(cfg)
                    .compile(qos)
                    .expect("ffnFused compile"),
            )
        };
        let use_dual_w13 = decomposed && ffn_fused::can_use_dual_w13(cfg);
        let ffn_w13_fwd = if decomposed {
            if use_dual_w13 {
                Some(
                    dyn_matmul::build_dual_separate(cfg.dim, cfg.hidden, cfg.seq)
                        .compile(qos)
                        .expect("ffnW13DualFwd compile"),
                )
            } else {
                Some(
                    dyn_matmul::build(cfg.dim, cfg.hidden, cfg.seq)
                        .compile(qos)
                        .expect("ffnW13Fwd compile"),
                )
            }
        } else {
            None
        };
        let ffn_w2_fwd = if decomposed {
            Some(
                dyn_matmul::build(cfg.hidden, cfg.dim, cfg.seq)
                    .compile(qos)
                    .expect("ffnW2Fwd compile"),
            )
        } else {
            None
        };

        // Backward kernels
        let ffn_bwd_w2t = dyn_matmul::build(cfg.dim, cfg.hidden, cfg.seq)
            .compile(qos)
            .expect("ffnBwdW2t compile");
        let ffn_bwd_w13t = dyn_matmul::build_dual(cfg.hidden, cfg.dim, cfg.seq)
            .compile(qos)
            .expect("ffnBwdW13t compile");
        let wot_bwd = dyn_matmul::build(cfg.dim, cfg.q_dim, cfg.seq)
            .compile(qos)
            .expect("wotBwd compile");
        let sdpa_bwd1 = sdpa_bwd::build_bwd1(cfg)
            .compile(qos)
            .expect("sdpaBwd1 compile");
        let sdpa_bwd2 = sdpa_bwd::build_bwd2(cfg)
            .compile(qos)
            .expect("sdpaBwd2 compile");
        let q_bwd = dyn_matmul::build(cfg.q_dim, cfg.dim, cfg.seq)
            .compile(qos)
            .expect("qBwd compile");
        let kv_bwd = dyn_matmul::build_dual(cfg.kv_dim, cfg.dim, cfg.seq)
            .compile(qos)
            .expect("kvBwd compile");

        // Pre-allocate IOSurface buffers for all kernels
        let bufs = KernelBuffers::allocate(cfg);

        // Pre-compute RoPE tables (deterministic, reused 12× per step)
        let rope = RopeTable::compute(cfg.hd, cfg.seq);

        Self {
            sdpa_fwd,
            wo_fwd,
            ffn_fused: ffn_fused_exe,
            ffn_w13_fwd,
            ffn_w2_fwd,
            use_decomposed_ffn: decomposed,
            use_dual_w13,
            ffn_bwd_w2t,
            ffn_bwd_w13t,
            wot_bwd,
            sdpa_bwd1,
            sdpa_bwd2,
            q_bwd,
            kv_bwd,
            bufs,
            rope,
        }
    }

    /// Compile kernels for forward-only workloads.
    /// Reuses the real forward executables but shrinks backward IOSurface state.
    pub fn compile_forward_only(cfg: &ModelConfig) -> Self {
        Self::compile(cfg)
    }
}

/// Pre-allocated scratch buffers for backward pass.
/// Eliminates ~32 vec allocations per layer × 6 layers = 192 malloc+memset+free cycles.
/// All buffers are fully overwritten before use — no zeroing needed.
pub struct BackwardWorkspace {
    // Weight transposes [hidden*dim] or [dim*q_dim]
    pub w1t: Vec<f32>,
    pub w3t: Vec<f32>,
    pub wot: Vec<f32>,
    pub wqt: Vec<f32>,
    pub wkt: Vec<f32>,
    pub wvt: Vec<f32>,
    // Activation buffers [dim*seq] or [q_dim*seq]
    pub dffn: Vec<f32>,
    pub dx_ffn: Vec<f32>,
    pub dx2: Vec<f32>,
    pub dx2_tmp: Vec<f32>,
    pub dx2_scaled: Vec<f32>,
    pub da: Vec<f32>,
    pub dv_full: Vec<f32>,
    pub dq: Vec<f32>,
    pub dk: Vec<f32>,
    pub dx_attn: Vec<f32>,
    pub dx_kv: Vec<f32>,
    pub dx_merged: Vec<f32>,
    pub dx_rms1: Vec<f32>,
    // Hidden-sized buffers [hidden*seq]
    pub dsilu_raw: Vec<f32>,
    pub dh1: Vec<f32>,
    pub dh3: Vec<f32>,
    pub neg_h1: Vec<f32>,
    pub exp_neg: Vec<f32>,
    // Score buffers [heads*seq*seq]
    pub probs_flat: Vec<f32>,
    pub dp_flat: Vec<f32>,
    // Channel-first RMSNorm scratch [seq]
    pub rms_dot_buf: Vec<f32>,
}

impl BackwardWorkspace {
    pub fn new(cfg: &ModelConfig) -> Self {
        let dim = cfg.dim;
        let seq = cfg.seq;
        let q_dim = cfg.q_dim;
        let kv_dim = cfg.kv_dim;
        let hidden = cfg.hidden;
        let heads = cfg.heads;
        Self {
            w1t: vec![0.0; hidden * dim],
            w3t: vec![0.0; hidden * dim],
            wot: vec![0.0; dim * q_dim],
            wqt: vec![0.0; q_dim * dim],
            wkt: vec![0.0; kv_dim * dim],
            wvt: vec![0.0; kv_dim * dim],
            dffn: vec![0.0; dim * seq],
            dx_ffn: vec![0.0; dim * seq],
            dx2: vec![0.0; dim * seq],
            dx2_tmp: vec![0.0; dim * seq],
            dx2_scaled: vec![0.0; dim * seq],
            da: vec![0.0; q_dim * seq],
            dv_full: vec![0.0; q_dim * seq],
            dq: vec![0.0; q_dim * seq],
            dk: vec![0.0; q_dim * seq],
            dx_attn: vec![0.0; dim * seq],
            dx_kv: vec![0.0; dim * seq],
            dx_merged: vec![0.0; dim * seq],
            dx_rms1: vec![0.0; dim * seq],
            dsilu_raw: vec![0.0; hidden * seq],
            dh1: vec![0.0; hidden * seq],
            dh3: vec![0.0; hidden * seq],
            neg_h1: vec![0.0; hidden * seq],
            exp_neg: vec![0.0; hidden * seq],
            probs_flat: vec![0.0; heads * seq * seq],
            dp_flat: vec![0.0; heads * seq * seq],
            rms_dot_buf: vec![0.0; seq],
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
            wo: vec![0.0; cfg.q_dim * cfg.dim], // zero-init (DeepNet)
            w1: random_vec(cfg.dim * cfg.hidden, scale_ffn),
            w3: random_vec(cfg.dim * cfg.hidden, scale_ffn),
            w2: vec![0.0; cfg.dim * cfg.hidden], // zero-init (DeepNet)
            gamma1: vec![1.0; cfg.dim],
            gamma2: vec![1.0; cfg.dim],
            w2_generation: 0,
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
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
fn stage_spatial(
    dst: &mut [f32],
    channels: usize,
    sp_width: usize,
    src: &[f32],
    src_width: usize,
    sp_offset: usize,
) {
    for c in 0..channels {
        let d = c * sp_width + sp_offset;
        let s = c * src_width;
        dst[d..d + src_width].copy_from_slice(&src[s..s + src_width]);
    }
}

/// Read a slice of channels from ANE output buffer into a pre-allocated destination.
/// No-alloc version of the former `read_channels`.
/// Uses copy_from_slice for vectorized memcpy on inner dimension.
/// Read contiguous channels from an IOSurface output buffer (stride = seq, no spatial padding).
/// Single memcpy instead of per-channel loop.
fn read_channels_into(
    src: &[f32],
    _total_ch: usize,
    seq: usize,
    ch_start: usize,
    ch_count: usize,
    dst: &mut [f32],
) {
    let start = ch_start * seq;
    dst.copy_from_slice(&src[start..start + ch_count * seq]);
}

fn stage_sdpa_fwd_inputs(bufs: &KernelBuffers, xnorm: &[f32], wq: &[f32], wk: &[f32], wv: &[f32]) {
    {
        let mut locked = bufs.sdpa_fwd_xnorm_in.as_f32_slice_mut();
        locked.copy_from_slice(xnorm);
    }
    {
        let mut locked = bufs.sdpa_fwd_wq_in.as_f32_slice_mut();
        locked.copy_from_slice(wq);
    }
    {
        let mut locked = bufs.sdpa_fwd_wk_in.as_f32_slice_mut();
        locked.copy_from_slice(wk);
    }
    {
        let mut locked = bufs.sdpa_fwd_wv_in.as_f32_slice_mut();
        locked.copy_from_slice(wv);
    }
}

fn run_sdpa_fwd(kernels: &CompiledKernels) {
    kernels
        .sdpa_fwd
        .run_cached_direct(
            &[
                &kernels.bufs.sdpa_fwd_xnorm_in,
                &kernels.bufs.sdpa_fwd_wq_in,
                &kernels.bufs.sdpa_fwd_wk_in,
                &kernels.bufs.sdpa_fwd_wv_in,
            ],
            &[
                &kernels.bufs.sdpa_fwd_attn_out,
                &kernels.bufs.sdpa_fwd_q_rope_out,
                &kernels.bufs.sdpa_fwd_k_rope_out,
                &kernels.bufs.sdpa_fwd_v_out,
            ],
        )
        .expect("ANE eval failed");
}

fn read_sdpa_fwd_outputs(
    bufs: &KernelBuffers,
    attn_out: &mut [f32],
    q_rope: &mut [f32],
    k_rope: &mut [f32],
    v: &mut [f32],
) {
    {
        let locked = bufs.sdpa_fwd_attn_out.as_f32_slice();
        attn_out.copy_from_slice(&locked[..attn_out.len()]);
    }
    {
        let locked = bufs.sdpa_fwd_q_rope_out.as_f32_slice();
        q_rope.copy_from_slice(&locked[..q_rope.len()]);
    }
    {
        let locked = bufs.sdpa_fwd_k_rope_out.as_f32_slice();
        k_rope.copy_from_slice(&locked[..k_rope.len()]);
    }
    {
        let locked = bufs.sdpa_fwd_v_out.as_f32_slice();
        v.copy_from_slice(&locked[..v.len()]);
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

    // 1. RMSNorm1 (CPU): channel-first, no transpose needed
    let mut xnorm = vec![0.0f32; dim * seq];
    let mut rms_inv1 = vec![0.0f32; seq];
    rmsnorm::forward_channel_first(x, &weights.gamma1, &mut xnorm, &mut rms_inv1, dim, seq);

    // 2. Stage sdpaFwd inputs directly into dedicated IOSurfaces
    stage_sdpa_fwd_inputs(&kernels.bufs, &xnorm, &weights.wq, &weights.wk, &weights.wv);

    // 3. Run sdpaFwd (ANE)
    run_sdpa_fwd(kernels);

    // Extract: attn_out[Q_DIM,SEQ], Q_rope[Q_DIM,SEQ], K_rope[KV_DIM,SEQ], V[KV_DIM,SEQ]
    let mut attn_out = vec![0.0f32; q_dim * seq];
    let mut q_rope = vec![0.0f32; q_dim * seq];
    let mut k_rope = vec![0.0f32; kv_dim * seq];
    let mut v = vec![0.0f32; kv_dim * seq];
    read_sdpa_fwd_outputs(
        &kernels.bufs,
        &mut attn_out,
        &mut q_rope,
        &mut k_rope,
        &mut v,
    );

    // 4. Stage woFwd directly into IOSurface
    let wo_sp = dyn_matmul::spatial_width(seq, dim);
    {
        let mut locked = kernels.bufs.wo_fwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, q_dim, wo_sp, &attn_out, seq, 0);
        stage_spatial(buf, q_dim, wo_sp, &weights.wo, dim, seq);
    }

    // 5. Run woFwd (ANE)
    kernels
        .wo_fwd
        .run_cached_direct(&[&kernels.bufs.wo_fwd_in], &[&kernels.bufs.wo_fwd_out])
        .expect("ANE eval failed");

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
    rmsnorm::forward_channel_first(&x2, &weights.gamma2, &mut x2norm, &mut rms_inv2, dim, seq);

    // 7-8. FFN: fused or decomposed
    let mut x_next = vec![0.0f32; dim * seq];
    let mut h1 = vec![0.0f32; hidden * seq];
    let mut h3 = vec![0.0f32; hidden * seq];
    let mut gate = vec![0.0f32; hidden * seq];
    let mut w2t_scratch = vec![0.0f32; hidden * dim];

    if kernels.use_decomposed_ffn {
        // Decomposed path: dual W1+W3 dispatch + CPU SiLU + W2 dispatch + CPU residual
        let w13_sp = if kernels.use_dual_w13 {
            dyn_matmul::dual_separate_spatial_width(seq, hidden)
        } else {
            dyn_matmul::spatial_width(seq, hidden)
        };
        let w2_sp = dyn_matmul::spatial_width(seq, dim);
        let w13_exe = kernels.ffn_w13_fwd.as_ref().unwrap();
        let w2_exe = kernels.ffn_w2_fwd.as_ref().unwrap();
        let w13_in = kernels.bufs.ffn_w13_in.as_ref().unwrap();
        let w13_out = kernels.bufs.ffn_w13_out.as_ref().unwrap();
        let w2_in = kernels.bufs.ffn_w2_in.as_ref().unwrap();
        let w2_out = kernels.bufs.ffn_w2_out.as_ref().unwrap();

        // W1+W3 projection: dual dispatch if fits, else separate
        if kernels.use_dual_w13 {
            {
                let mut locked = w13_in.as_f32_slice_mut();
                let buf = &mut *locked;
                stage_spatial(buf, dim, w13_sp, &x2norm, seq, 0);
                stage_spatial(buf, dim, w13_sp, &weights.w1, hidden, seq);
                stage_spatial(buf, dim, w13_sp, &weights.w3, hidden, seq + hidden);
            }
            w13_exe
                .run_cached_direct(&[w13_in], &[w13_out])
                .expect("ANE w13 dual failed");
            {
                let locked = w13_out.as_f32_slice();
                read_channels_into(&locked, 2 * hidden, seq, 0, hidden, &mut h1);
                read_channels_into(&locked, 2 * hidden, seq, hidden, hidden, &mut h3);
            }
        } else {
            // Separate W1, W3 dispatches
            {
                let mut locked = w13_in.as_f32_slice_mut();
                let buf = &mut *locked;
                stage_spatial(buf, dim, w13_sp, &x2norm, seq, 0);
                stage_spatial(buf, dim, w13_sp, &weights.w1, hidden, seq);
            }
            w13_exe
                .run_cached_direct(&[w13_in], &[w13_out])
                .expect("ANE w13 W1 failed");
            {
                let locked = w13_out.as_f32_slice();
                h1.copy_from_slice(&locked[..hidden * seq]);
            }
            {
                let mut locked = w13_in.as_f32_slice_mut();
                let buf = &mut *locked;
                stage_spatial(buf, dim, w13_sp, &weights.w3, hidden, seq);
            }
            w13_exe
                .run_cached_direct(&[w13_in], &[w13_out])
                .expect("ANE w13 W3 failed");
            {
                let locked = w13_out.as_f32_slice();
                h3.copy_from_slice(&locked[..hidden * seq]);
            }
        }

        // CPU SiLU gate: gate = silu(h1) * h3
        silu::silu_gate(&h1, &h3, &mut gate);

        // Transpose W2 from [DIM, HIDDEN] → [HIDDEN, DIM]
        vdsp::mtrans(&weights.w2, hidden, &mut w2t_scratch, dim, dim, hidden);

        // W2 fwd: ffn_out = gate @ W2^T
        {
            let mut locked = w2_in.as_f32_slice_mut();
            let buf = &mut *locked;
            stage_spatial(buf, hidden, w2_sp, &gate, seq, 0);
            stage_spatial(buf, hidden, w2_sp, &w2t_scratch, dim, seq);
        }
        w2_exe
            .run_cached_direct(&[w2_in], &[w2_out])
            .expect("ANE ffnW2Fwd failed");
        // Read ffn_out and compute residual: x_next = x2 + alpha * ffn_out
        {
            let locked = w2_out.as_f32_slice();
            // Use x_next as temporary for ffn_out, then compute residual in-place
            x_next.copy_from_slice(&locked[..dim * seq]);
        }
        vdsp::vsma(&x_next.clone(), alpha, &x2, &mut x_next);
    } else {
        // Fused path (existing)
        let ffn_sp = ffn_fused::input_spatial_width(cfg);
        let ffn_out_ch = ffn_fused::output_channels(cfg);
        let ffn_in = kernels.bufs.ffn_fused_in.as_ref().unwrap();
        let ffn_out = kernels.bufs.ffn_fused_out.as_ref().unwrap();
        {
            let mut locked = ffn_in.as_f32_slice_mut();
            let buf = &mut *locked;
            stage_spatial(buf, dim, ffn_sp, &x2norm, seq, 0);
            stage_spatial(buf, dim, ffn_sp, &x2, seq, seq);
            stage_spatial(buf, dim, ffn_sp, &weights.w1, hidden, 2 * seq);
            stage_spatial(buf, dim, ffn_sp, &weights.w3, hidden, 2 * seq + hidden);
            stage_spatial(buf, dim, ffn_sp, &weights.w2, hidden, 2 * seq + 2 * hidden);
        }
        kernels
            .ffn_fused
            .as_ref()
            .unwrap()
            .run_cached_direct(&[ffn_in], &[ffn_out])
            .expect("ANE eval failed");
        {
            let locked = ffn_out.as_f32_slice();
            read_channels_into(&locked, ffn_out_ch, seq, 0, dim, &mut x_next);
            read_channels_into(&locked, ffn_out_ch, seq, dim, hidden, &mut h1);
            read_channels_into(&locked, ffn_out_ch, seq, dim + hidden, hidden, &mut h3);
            read_channels_into(
                &locked,
                ffn_out_ch,
                seq,
                dim + 2 * hidden,
                hidden,
                &mut gate,
            );
        }
    }

    let cache = ForwardCache {
        x: x.to_vec(),
        xnorm,
        rms_inv1,
        q_rope,
        k_rope,
        v,
        attn_out,
        o_out,
        x2,
        x2norm,
        rms_inv2,
        h1,
        h3,
        gate,
        w2t_scratch,
        w2t_generation: weights.w2_generation,
    };

    (x_next, cache)
}

/// Forward pass writing into pre-allocated cache (zero allocations).
/// `x_next` is written with the layer output [DIM * SEQ].
pub fn forward_into(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &LayerWeights,
    x: &[f32],
    cache: &mut ForwardCache,
    x_next: &mut [f32],
) {
    let dim = cfg.dim;
    let seq = cfg.seq;
    let q_dim = cfg.q_dim;
    let kv_dim = cfg.kv_dim;
    let hidden = cfg.hidden;
    let alpha = 1.0 / (2.0 * cfg.nlayers as f32).sqrt();

    // Save layer input
    cache.x.copy_from_slice(x);

    // 1. RMSNorm1 (CPU)
    rmsnorm::forward_channel_first(
        x,
        &weights.gamma1,
        &mut cache.xnorm,
        &mut cache.rms_inv1,
        dim,
        seq,
    );

    // 2. Stage sdpaFwd inputs
    stage_sdpa_fwd_inputs(
        &kernels.bufs,
        &cache.xnorm,
        &weights.wq,
        &weights.wk,
        &weights.wv,
    );

    // 3. Run sdpaFwd (ANE) || pre-stage woFwd weights + ffn weights
    let wo_sp = dyn_matmul::spatial_width(seq, dim);
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            run_sdpa_fwd(kernels);
        });
        // Stage woFwd weights
        {
            let mut locked = kernels.bufs.wo_fwd_in.as_f32_slice_mut();
            let buf = &mut *locked;
            stage_spatial(buf, q_dim, wo_sp, &weights.wo, dim, seq);
        }
        // Pre-stage FFN weights (fused or decomposed W1 + optional W3)
        if kernels.use_decomposed_ffn {
            let w13_sp = if kernels.use_dual_w13 {
                dyn_matmul::dual_separate_spatial_width(seq, hidden)
            } else {
                dyn_matmul::spatial_width(seq, hidden)
            };
            let w13_in = kernels.bufs.ffn_w13_in.as_ref().unwrap();
            let mut locked = w13_in.as_f32_slice_mut();
            let buf = &mut *locked;
            stage_spatial(buf, dim, w13_sp, &weights.w1, hidden, seq);
            if kernels.use_dual_w13 {
                stage_spatial(buf, dim, w13_sp, &weights.w3, hidden, seq + hidden);
            }
        } else {
            let ffn_sp = ffn_fused::input_spatial_width(cfg);
            let ffn_in = kernels.bufs.ffn_fused_in.as_ref().unwrap();
            let mut locked = ffn_in.as_f32_slice_mut();
            let buf = &mut *locked;
            let w_off = 2 * seq;
            for c in 0..dim {
                let row = c * ffn_sp;
                buf[row + w_off..row + w_off + hidden]
                    .copy_from_slice(&weights.w1[c * hidden..c * hidden + hidden]);
                buf[row + w_off + hidden..row + w_off + 2 * hidden]
                    .copy_from_slice(&weights.w3[c * hidden..c * hidden + hidden]);
                buf[row + w_off + 2 * hidden..row + w_off + 3 * hidden]
                    .copy_from_slice(&weights.w2[c * hidden..c * hidden + hidden]);
            }
        }
        ane_handle.join().expect("ANE thread panicked");
    });

    // Extract sdpaFwd output
    read_sdpa_fwd_outputs(
        &kernels.bufs,
        &mut cache.attn_out,
        &mut cache.q_rope,
        &mut cache.k_rope,
        &mut cache.v,
    );

    // 4. Stage woFwd activations only (weights already staged during sdpaFwd)
    {
        let mut locked = kernels.bufs.wo_fwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, q_dim, wo_sp, &cache.attn_out, seq, 0);
    }

    // 5. Run woFwd (ANE)
    kernels
        .wo_fwd
        .run_cached_direct(&[&kernels.bufs.wo_fwd_in], &[&kernels.bufs.wo_fwd_out])
        .expect("ANE eval failed");

    // Read o_out
    {
        let locked = kernels.bufs.wo_fwd_out.as_f32_slice();
        cache.o_out.copy_from_slice(&locked[..dim * seq]);
    }

    // 6. Residual + RMSNorm2
    vdsp::vsma(&cache.o_out, alpha, x, &mut cache.x2);
    rmsnorm::forward_channel_first(
        &cache.x2,
        &weights.gamma2,
        &mut cache.x2norm,
        &mut cache.rms_inv2,
        dim,
        seq,
    );

    // 7-8. FFN: fused or decomposed
    if kernels.use_decomposed_ffn {
        let w13_sp = if kernels.use_dual_w13 {
            dyn_matmul::dual_separate_spatial_width(seq, hidden)
        } else {
            dyn_matmul::spatial_width(seq, hidden)
        };
        let w2_sp = dyn_matmul::spatial_width(seq, dim);
        let w13_exe = kernels.ffn_w13_fwd.as_ref().unwrap();
        let w2_exe = kernels.ffn_w2_fwd.as_ref().unwrap();
        let w13_in = kernels.bufs.ffn_w13_in.as_ref().unwrap();
        let w13_out = kernels.bufs.ffn_w13_out.as_ref().unwrap();
        let w2_in = kernels.bufs.ffn_w2_in.as_ref().unwrap();
        let w2_out = kernels.bufs.ffn_w2_out.as_ref().unwrap();

        // W1+W3 projection (weights pre-staged in step 3)
        {
            let mut locked = w13_in.as_f32_slice_mut();
            let buf = &mut *locked;
            stage_spatial(buf, dim, w13_sp, &cache.x2norm, seq, 0);
        }
        if kernels.use_dual_w13 {
            w13_exe
                .run_cached_direct(&[w13_in], &[w13_out])
                .expect("ANE w13 dual failed");
            let locked = w13_out.as_f32_slice();
            read_channels_into(&locked, 2 * hidden, seq, 0, hidden, &mut cache.h1);
            read_channels_into(&locked, 2 * hidden, seq, hidden, hidden, &mut cache.h3);
        } else {
            w13_exe
                .run_cached_direct(&[w13_in], &[w13_out])
                .expect("ANE w13 W1 failed");
            {
                let locked = w13_out.as_f32_slice();
                cache.h1.copy_from_slice(&locked[..hidden * seq]);
            }
            {
                let mut locked = w13_in.as_f32_slice_mut();
                stage_spatial(&mut locked, dim, w13_sp, &weights.w3, hidden, seq);
            }
            w13_exe
                .run_cached_direct(&[w13_in], &[w13_out])
                .expect("ANE w13 W3 failed");
            {
                let locked = w13_out.as_f32_slice();
                cache.h3.copy_from_slice(&locked[..hidden * seq]);
            }
        }

        // CPU SiLU gate
        silu::silu_gate(&cache.h1, &cache.h3, &mut cache.gate);

        // Reuse the cached W2^T until training updates W2.
        if cache.w2t_generation != weights.w2_generation {
            vdsp::mtrans(
                &weights.w2,
                hidden,
                &mut cache.w2t_scratch,
                dim,
                dim,
                hidden,
            );
            cache.w2t_generation = weights.w2_generation;
        }
        {
            let mut locked = w2_in.as_f32_slice_mut();
            let buf = &mut *locked;
            stage_spatial(buf, hidden, w2_sp, &cache.gate, seq, 0);
            stage_spatial(buf, hidden, w2_sp, &cache.w2t_scratch, dim, seq);
        }
        w2_exe
            .run_cached_direct(&[w2_in], &[w2_out])
            .expect("ANE ffnW2Fwd failed");

        // Read ffn_out into x_next, then residual
        {
            let locked = w2_out.as_f32_slice();
            x_next.copy_from_slice(&locked[..dim * seq]);
        }
        // x_next = alpha * ffn_out + x2  (vsma: a*scalar + b → out)
        let ffn_tmp: Vec<f32> = x_next.to_vec();
        vdsp::vsma(&ffn_tmp, alpha, &cache.x2, x_next);
    } else {
        let ffn_sp = ffn_fused::input_spatial_width(cfg);
        let ffn_out_ch = ffn_fused::output_channels(cfg);
        let ffn_in = kernels.bufs.ffn_fused_in.as_ref().unwrap();
        let ffn_out = kernels.bufs.ffn_fused_out.as_ref().unwrap();

        // Stage activations only (weights already staged in step 3)
        {
            let mut locked = ffn_in.as_f32_slice_mut();
            let buf = &mut *locked;
            for c in 0..dim {
                let row = c * ffn_sp;
                buf[row..row + seq].copy_from_slice(&cache.x2norm[c * seq..c * seq + seq]);
                buf[row + seq..row + 2 * seq].copy_from_slice(&cache.x2[c * seq..c * seq + seq]);
            }
        }
        kernels
            .ffn_fused
            .as_ref()
            .unwrap()
            .run_cached_direct(&[ffn_in], &[ffn_out])
            .expect("ANE eval failed");
        {
            let locked = ffn_out.as_f32_slice();
            read_channels_into(&locked, ffn_out_ch, seq, 0, dim, x_next);
            read_channels_into(&locked, ffn_out_ch, seq, dim, hidden, &mut cache.h1);
            read_channels_into(
                &locked,
                ffn_out_ch,
                seq,
                dim + hidden,
                hidden,
                &mut cache.h3,
            );
            read_channels_into(
                &locked,
                ffn_out_ch,
                seq,
                dim + 2 * hidden,
                hidden,
                &mut cache.gate,
            );
        }
    }
}

/// Pipelined forward: defers own h1/h3/gate readback, optionally reads previous
/// layer's deferred h1/h3/gate during sdpaFwd ANE overlap (step 3).
///
/// The ffnFused output IOSurface retains the previous layer's data until this
/// layer's ffnFused runs (~5ms later), giving ample time to read during step 3.
/// sdpaFwd ANE takes ~3.2ms, CPU staging takes ~1.5ms → ~1.7ms spare for readback.
/// The 12MB readback takes ~0.8ms, fitting within the spare window.
pub fn forward_into_pipelined(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &LayerWeights,
    x: &[f32],
    cache: &mut ForwardCache,
    x_next: &mut [f32],
    prev_cache: Option<&mut ForwardCache>,
) {
    let dim = cfg.dim;
    let seq = cfg.seq;
    let q_dim = cfg.q_dim;
    let kv_dim = cfg.kv_dim;
    let hidden = cfg.hidden;
    let alpha = 1.0 / (2.0 * cfg.nlayers as f32).sqrt();

    cache.x.copy_from_slice(x);

    // 1. RMSNorm1 (CPU)
    rmsnorm::forward_channel_first(
        x,
        &weights.gamma1,
        &mut cache.xnorm,
        &mut cache.rms_inv1,
        dim,
        seq,
    );

    // 2. Stage sdpaFwd inputs
    stage_sdpa_fwd_inputs(
        &kernels.bufs,
        &cache.xnorm,
        &weights.wq,
        &weights.wk,
        &weights.wv,
    );

    // 3. Run sdpaFwd (ANE) || pre-stage weights + deferred prev-layer cache readback
    let wo_sp = dyn_matmul::spatial_width(seq, dim);
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            run_sdpa_fwd(kernels);
        });
        // Stage woFwd weights
        {
            let mut locked = kernels.bufs.wo_fwd_in.as_f32_slice_mut();
            let buf = &mut *locked;
            stage_spatial(buf, q_dim, wo_sp, &weights.wo, dim, seq);
        }
        // Pre-stage FFN weights + deferred prev-layer readback
        if kernels.use_decomposed_ffn {
            // Pre-stage W1 (+ W3 if dual) weights
            let w13_sp = if kernels.use_dual_w13 {
                dyn_matmul::dual_separate_spatial_width(seq, hidden)
            } else {
                dyn_matmul::spatial_width(seq, hidden)
            };
            let w13_in = kernels.bufs.ffn_w13_in.as_ref().unwrap();
            {
                let mut locked = w13_in.as_f32_slice_mut();
                let buf = &mut *locked;
                stage_spatial(buf, dim, w13_sp, &weights.w1, hidden, seq);
                if kernels.use_dual_w13 {
                    stage_spatial(buf, dim, w13_sp, &weights.w3, hidden, seq + hidden);
                }
            }
            // When decomposed, prev_cache h1/h3/gate were already written during prev layer's
            // decomposed FFN sequence — no deferred readback needed.
        } else {
            let ffn_sp = ffn_fused::input_spatial_width(cfg);
            let ffn_out_ch = ffn_fused::output_channels(cfg);
            let ffn_in = kernels.bufs.ffn_fused_in.as_ref().unwrap();
            {
                let mut locked = ffn_in.as_f32_slice_mut();
                let buf = &mut *locked;
                let w_off = 2 * seq;
                for c in 0..dim {
                    let row = c * ffn_sp;
                    buf[row + w_off..row + w_off + hidden]
                        .copy_from_slice(&weights.w1[c * hidden..c * hidden + hidden]);
                    buf[row + w_off + hidden..row + w_off + 2 * hidden]
                        .copy_from_slice(&weights.w3[c * hidden..c * hidden + hidden]);
                    buf[row + w_off + 2 * hidden..row + w_off + 3 * hidden]
                        .copy_from_slice(&weights.w2[c * hidden..c * hidden + hidden]);
                }
            }
            // Deferred readback: read PREVIOUS layer's h1/h3/gate from ffn_fused_out.
            if let Some(prev) = prev_cache {
                let ffn_out = kernels.bufs.ffn_fused_out.as_ref().unwrap();
                let locked = ffn_out.as_f32_slice();
                read_channels_into(&locked, ffn_out_ch, seq, dim, hidden, &mut prev.h1);
                read_channels_into(&locked, ffn_out_ch, seq, dim + hidden, hidden, &mut prev.h3);
                read_channels_into(
                    &locked,
                    ffn_out_ch,
                    seq,
                    dim + 2 * hidden,
                    hidden,
                    &mut prev.gate,
                );
            }
        }
        ane_handle.join().expect("ANE thread panicked");
    });

    // Extract sdpaFwd output
    read_sdpa_fwd_outputs(
        &kernels.bufs,
        &mut cache.attn_out,
        &mut cache.q_rope,
        &mut cache.k_rope,
        &mut cache.v,
    );

    // 4. Stage woFwd activations only
    {
        let mut locked = kernels.bufs.wo_fwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, q_dim, wo_sp, &cache.attn_out, seq, 0);
    }

    // 5. Run woFwd (ANE)
    kernels
        .wo_fwd
        .run_cached_direct(&[&kernels.bufs.wo_fwd_in], &[&kernels.bufs.wo_fwd_out])
        .expect("ANE eval failed");

    {
        let locked = kernels.bufs.wo_fwd_out.as_f32_slice();
        cache.o_out.copy_from_slice(&locked[..dim * seq]);
    }

    // 6. Residual + RMSNorm2
    vdsp::vsma(&cache.o_out, alpha, x, &mut cache.x2);
    rmsnorm::forward_channel_first(
        &cache.x2,
        &weights.gamma2,
        &mut cache.x2norm,
        &mut cache.rms_inv2,
        dim,
        seq,
    );

    // 7-8. FFN: fused or decomposed
    if kernels.use_decomposed_ffn {
        let w13_sp = if kernels.use_dual_w13 {
            dyn_matmul::dual_separate_spatial_width(seq, hidden)
        } else {
            dyn_matmul::spatial_width(seq, hidden)
        };
        let w2_sp = dyn_matmul::spatial_width(seq, dim);
        let w13_exe = kernels.ffn_w13_fwd.as_ref().unwrap();
        let w2_exe = kernels.ffn_w2_fwd.as_ref().unwrap();
        let w13_in = kernels.bufs.ffn_w13_in.as_ref().unwrap();
        let w13_out = kernels.bufs.ffn_w13_out.as_ref().unwrap();
        let w2_in = kernels.bufs.ffn_w2_in.as_ref().unwrap();
        let w2_out = kernels.bufs.ffn_w2_out.as_ref().unwrap();

        // W1+W3 projection (weights pre-staged in step 3)
        {
            let mut locked = w13_in.as_f32_slice_mut();
            let buf = &mut *locked;
            stage_spatial(buf, dim, w13_sp, &cache.x2norm, seq, 0);
        }
        if kernels.use_dual_w13 {
            w13_exe
                .run_cached_direct(&[w13_in], &[w13_out])
                .expect("ANE w13 dual failed");
            let locked = w13_out.as_f32_slice();
            read_channels_into(&locked, 2 * hidden, seq, 0, hidden, &mut cache.h1);
            read_channels_into(&locked, 2 * hidden, seq, hidden, hidden, &mut cache.h3);
        } else {
            w13_exe
                .run_cached_direct(&[w13_in], &[w13_out])
                .expect("ANE w13 W1 failed");
            {
                let locked = w13_out.as_f32_slice();
                cache.h1.copy_from_slice(&locked[..hidden * seq]);
            }
            {
                let mut locked = w13_in.as_f32_slice_mut();
                stage_spatial(&mut locked, dim, w13_sp, &weights.w3, hidden, seq);
            }
            w13_exe
                .run_cached_direct(&[w13_in], &[w13_out])
                .expect("ANE w13 W3 failed");
            {
                let locked = w13_out.as_f32_slice();
                cache.h3.copy_from_slice(&locked[..hidden * seq]);
            }
        }

        // CPU SiLU gate
        silu::silu_gate(&cache.h1, &cache.h3, &mut cache.gate);

        // Reuse the cached W2^T until training updates W2.
        if cache.w2t_generation != weights.w2_generation {
            vdsp::mtrans(
                &weights.w2,
                hidden,
                &mut cache.w2t_scratch,
                dim,
                dim,
                hidden,
            );
            cache.w2t_generation = weights.w2_generation;
        }
        {
            let mut locked = w2_in.as_f32_slice_mut();
            let buf = &mut *locked;
            stage_spatial(buf, hidden, w2_sp, &cache.gate, seq, 0);
            stage_spatial(buf, hidden, w2_sp, &cache.w2t_scratch, dim, seq);
        }
        w2_exe
            .run_cached_direct(&[w2_in], &[w2_out])
            .expect("ANE ffnW2Fwd failed");

        // Read ffn_out into x_next, then residual
        {
            let locked = w2_out.as_f32_slice();
            x_next.copy_from_slice(&locked[..dim * seq]);
        }
        let ffn_tmp: Vec<f32> = x_next.to_vec();
        vdsp::vsma(&ffn_tmp, alpha, &cache.x2, x_next);
        // h1/h3/gate already populated — no deferred readback needed
    } else {
        let ffn_sp = ffn_fused::input_spatial_width(cfg);
        let ffn_out_ch = ffn_fused::output_channels(cfg);
        let ffn_in = kernels.bufs.ffn_fused_in.as_ref().unwrap();
        let ffn_out = kernels.bufs.ffn_fused_out.as_ref().unwrap();

        // Stage activations only (weights already staged in step 3)
        {
            let mut locked = ffn_in.as_f32_slice_mut();
            let buf = &mut *locked;
            for c in 0..dim {
                let row = c * ffn_sp;
                buf[row..row + seq].copy_from_slice(&cache.x2norm[c * seq..c * seq + seq]);
                buf[row + seq..row + 2 * seq].copy_from_slice(&cache.x2[c * seq..c * seq + seq]);
            }
        }
        kernels
            .ffn_fused
            .as_ref()
            .unwrap()
            .run_cached_direct(&[ffn_in], &[ffn_out])
            .expect("ANE eval failed");

        // Extract x_next ONLY — h1/h3/gate deferred to next layer's step 3
        {
            let locked = ffn_out.as_f32_slice();
            read_channels_into(&locked, ffn_out_ch, seq, 0, dim, x_next);
        }
    }
}

/// Read deferred h1/h3/gate from ffnFused IOSurface into cache.
/// Used for the last layer (no next layer to overlap with).
/// No-op when decomposed (cache already populated during forward).
pub fn read_ffn_cache(cfg: &ModelConfig, kernels: &CompiledKernels, cache: &mut ForwardCache) {
    if kernels.use_decomposed_ffn {
        return; // h1/h3/gate already written during decomposed FFN sequence
    }
    let dim = cfg.dim;
    let seq = cfg.seq;
    let hidden = cfg.hidden;
    let ffn_out_ch = ffn_fused::output_channels(cfg);

    let ffn_out = kernels.bufs.ffn_fused_out.as_ref().unwrap();
    let locked = ffn_out.as_f32_slice();
    read_channels_into(&locked, ffn_out_ch, seq, dim, hidden, &mut cache.h1);
    read_channels_into(
        &locked,
        ffn_out_ch,
        seq,
        dim + hidden,
        hidden,
        &mut cache.h3,
    );
    read_channels_into(
        &locked,
        ffn_out_ch,
        seq,
        dim + 2 * hidden,
        hidden,
        &mut cache.gate,
    );
}

/// Timing breakdown for forward pass.
#[derive(Debug, Clone)]
pub struct ForwardTimings {
    pub rmsnorm1_ms: f32,
    pub stage_sdpa_ms: f32,
    pub ane_sdpa_ms: f32,
    pub read_sdpa_ms: f32,
    pub stage_wo_ms: f32,
    pub ane_wo_ms: f32,
    pub read_wo_ms: f32,
    pub residual_rmsnorm2_ms: f32,
    pub stage_ffn_ms: f32,
    pub ane_ffn_ms: f32,
    pub read_ffn_ms: f32,
    pub total_ms: f32,
}

impl ForwardTimings {
    pub fn print(&self) {
        println!("  {:<30} {:>6.2}ms", "RMSNorm1 (CPU)", self.rmsnorm1_ms);
        println!(
            "  {:<30} {:>6.2}ms",
            "stage sdpaFwd IOSurf", self.stage_sdpa_ms
        );
        println!("  {:<30} {:>6.2}ms", "ANE sdpaFwd", self.ane_sdpa_ms);
        println!(
            "  {:<30} {:>6.2}ms",
            "read sdpaFwd output", self.read_sdpa_ms
        );
        println!("  {:<30} {:>6.2}ms", "stage woFwd IOSurf", self.stage_wo_ms);
        println!("  {:<30} {:>6.2}ms", "ANE woFwd", self.ane_wo_ms);
        println!("  {:<30} {:>6.2}ms", "read woFwd output", self.read_wo_ms);
        println!(
            "  {:<30} {:>6.2}ms",
            "residual + RMSNorm2 (CPU)", self.residual_rmsnorm2_ms
        );
        println!(
            "  {:<30} {:>6.2}ms",
            "stage ffnFused IOSurf", self.stage_ffn_ms
        );
        println!("  {:<30} {:>6.2}ms", "ANE ffnFused", self.ane_ffn_ms);
        println!(
            "  {:<30} {:>6.2}ms",
            "read ffnFused output", self.read_ffn_ms
        );
        println!("  {:<30} {:>6.2}ms", "TOTAL", self.total_ms);
    }
}

/// Forward pass with per-operation timing (same output as `forward`).
pub fn forward_timed(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &LayerWeights,
    x: &[f32],
) -> (Vec<f32>, ForwardCache, ForwardTimings) {
    let t_total = Instant::now();
    let dim = cfg.dim;
    let seq = cfg.seq;
    let q_dim = cfg.q_dim;
    let kv_dim = cfg.kv_dim;
    let hidden = cfg.hidden;
    let alpha = 1.0 / (2.0 * cfg.nlayers as f32).sqrt();

    // 1. RMSNorm1 (channel-first, no transpose)
    let t = Instant::now();
    let mut xnorm = vec![0.0f32; dim * seq];
    let mut rms_inv1 = vec![0.0f32; seq];
    rmsnorm::forward_channel_first(x, &weights.gamma1, &mut xnorm, &mut rms_inv1, dim, seq);
    let rmsnorm1_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 2. Stage sdpaFwd
    let t = Instant::now();
    stage_sdpa_fwd_inputs(&kernels.bufs, &xnorm, &weights.wq, &weights.wk, &weights.wv);
    let stage_sdpa_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 3. ANE sdpaFwd || pre-stage woFwd weights + FFN weights
    let t = Instant::now();
    let wo_sp = dyn_matmul::spatial_width(seq, dim);
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            run_sdpa_fwd(kernels);
        });
        // Stage woFwd weights
        {
            let mut locked = kernels.bufs.wo_fwd_in.as_f32_slice_mut();
            let buf = &mut *locked;
            stage_spatial(buf, q_dim, wo_sp, &weights.wo, dim, seq);
        }
        // Pre-stage FFN weights
        if kernels.use_decomposed_ffn {
            let w13_sp = if kernels.use_dual_w13 {
                dyn_matmul::dual_separate_spatial_width(seq, hidden)
            } else {
                dyn_matmul::spatial_width(seq, hidden)
            };
            let w13_in = kernels.bufs.ffn_w13_in.as_ref().unwrap();
            let mut locked = w13_in.as_f32_slice_mut();
            let buf = &mut *locked;
            stage_spatial(buf, dim, w13_sp, &weights.w1, hidden, seq);
            if kernels.use_dual_w13 {
                stage_spatial(buf, dim, w13_sp, &weights.w3, hidden, seq + hidden);
            }
        } else {
            let ffn_sp = ffn_fused::input_spatial_width(cfg);
            let ffn_in = kernels.bufs.ffn_fused_in.as_ref().unwrap();
            let mut locked = ffn_in.as_f32_slice_mut();
            let buf = &mut *locked;
            let w_off = 2 * seq;
            for c in 0..dim {
                let row = c * ffn_sp;
                buf[row + w_off..row + w_off + hidden]
                    .copy_from_slice(&weights.w1[c * hidden..c * hidden + hidden]);
                buf[row + w_off + hidden..row + w_off + 2 * hidden]
                    .copy_from_slice(&weights.w3[c * hidden..c * hidden + hidden]);
                buf[row + w_off + 2 * hidden..row + w_off + 3 * hidden]
                    .copy_from_slice(&weights.w2[c * hidden..c * hidden + hidden]);
            }
        }
        ane_handle.join().expect("ANE thread panicked");
    });
    let ane_sdpa_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 4. Read output
    let t = Instant::now();
    let mut attn_out = vec![0.0f32; q_dim * seq];
    let mut q_rope = vec![0.0f32; q_dim * seq];
    let mut k_rope = vec![0.0f32; kv_dim * seq];
    let mut v = vec![0.0f32; kv_dim * seq];
    read_sdpa_fwd_outputs(
        &kernels.bufs,
        &mut attn_out,
        &mut q_rope,
        &mut k_rope,
        &mut v,
    );
    let read_sdpa_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 5. Stage woFwd activations only (weights already staged during sdpaFwd)
    let t = Instant::now();
    {
        let mut locked = kernels.bufs.wo_fwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, q_dim, wo_sp, &attn_out, seq, 0);
    }
    let stage_wo_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 6. ANE woFwd — ffnFused weights already staged in step 3
    let t = Instant::now();
    kernels
        .wo_fwd
        .run_cached_direct(&[&kernels.bufs.wo_fwd_in], &[&kernels.bufs.wo_fwd_out])
        .expect("ANE eval failed");
    let ane_wo_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 7. Read woFwd output
    let t = Instant::now();
    let mut o_out = vec![0.0f32; dim * seq];
    {
        let locked = kernels.bufs.wo_fwd_out.as_f32_slice();
        o_out.copy_from_slice(&locked[..dim * seq]);
    }
    let read_wo_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 8. Residual + RMSNorm2 (bulk transpose)
    let t = Instant::now();
    let mut x2 = vec![0.0f32; dim * seq];
    vdsp::vsma(&o_out, alpha, x, &mut x2);
    let mut x2norm = vec![0.0f32; dim * seq];
    let mut rms_inv2 = vec![0.0f32; seq];
    rmsnorm::forward_channel_first(&x2, &weights.gamma2, &mut x2norm, &mut rms_inv2, dim, seq);
    let residual_rmsnorm2_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 9-11. FFN: fused or decomposed (stage + dispatch + read)
    let mut x_next = vec![0.0f32; dim * seq];
    let mut h1 = vec![0.0f32; hidden * seq];
    let mut h3 = vec![0.0f32; hidden * seq];
    let mut gate = vec![0.0f32; hidden * seq];
    let mut w2t_scratch = vec![0.0f32; hidden * dim];

    let (stage_ffn_ms, ane_ffn_ms, read_ffn_ms);
    if kernels.use_decomposed_ffn {
        let w13_sp = if kernels.use_dual_w13 {
            dyn_matmul::dual_separate_spatial_width(seq, hidden)
        } else {
            dyn_matmul::spatial_width(seq, hidden)
        };
        let w2_sp = dyn_matmul::spatial_width(seq, dim);
        let w13_exe = kernels.ffn_w13_fwd.as_ref().unwrap();
        let w2_exe = kernels.ffn_w2_fwd.as_ref().unwrap();
        let w13_in = kernels.bufs.ffn_w13_in.as_ref().unwrap();
        let w13_out = kernels.bufs.ffn_w13_out.as_ref().unwrap();
        let w2_in = kernels.bufs.ffn_w2_in.as_ref().unwrap();
        let w2_out = kernels.bufs.ffn_w2_out.as_ref().unwrap();

        // Stage: x2norm (W1+W3 weights already pre-staged)
        let t = Instant::now();
        {
            let mut locked = w13_in.as_f32_slice_mut();
            let buf = &mut *locked;
            stage_spatial(buf, dim, w13_sp, &x2norm, seq, 0);
        }
        stage_ffn_ms = t.elapsed().as_secs_f32() * 1000.0;

        // ANE: W1+W3 dispatch (dual or separate) + CPU SiLU + W2 dispatch
        let t = Instant::now();
        if kernels.use_dual_w13 {
            w13_exe
                .run_cached_direct(&[w13_in], &[w13_out])
                .expect("ANE w13 dual failed");
            let locked = w13_out.as_f32_slice();
            read_channels_into(&locked, 2 * hidden, seq, 0, hidden, &mut h1);
            read_channels_into(&locked, 2 * hidden, seq, hidden, hidden, &mut h3);
        } else {
            w13_exe
                .run_cached_direct(&[w13_in], &[w13_out])
                .expect("ANE w13 W1 failed");
            {
                let locked = w13_out.as_f32_slice();
                h1.copy_from_slice(&locked[..hidden * seq]);
            }
            {
                let mut locked = w13_in.as_f32_slice_mut();
                stage_spatial(&mut locked, dim, w13_sp, &weights.w3, hidden, seq);
            }
            w13_exe
                .run_cached_direct(&[w13_in], &[w13_out])
                .expect("ANE w13 W3 failed");
            {
                let locked = w13_out.as_f32_slice();
                h3.copy_from_slice(&locked[..hidden * seq]);
            }
        }
        silu::silu_gate(&h1, &h3, &mut gate);
        vdsp::mtrans(&weights.w2, hidden, &mut w2t_scratch, dim, dim, hidden);
        {
            let mut locked = w2_in.as_f32_slice_mut();
            let buf = &mut *locked;
            stage_spatial(buf, hidden, w2_sp, &gate, seq, 0);
            stage_spatial(buf, hidden, w2_sp, &w2t_scratch, dim, seq);
        }
        w2_exe
            .run_cached_direct(&[w2_in], &[w2_out])
            .expect("ANE ffnW2Fwd failed");
        ane_ffn_ms = t.elapsed().as_secs_f32() * 1000.0;

        // Read ffn_out + residual
        let t = Instant::now();
        {
            let locked = w2_out.as_f32_slice();
            x_next.copy_from_slice(&locked[..dim * seq]);
        }
        let ffn_tmp: Vec<f32> = x_next.clone();
        vdsp::vsma(&ffn_tmp, alpha, &x2, &mut x_next);
        read_ffn_ms = t.elapsed().as_secs_f32() * 1000.0;
    } else {
        let ffn_sp = ffn_fused::input_spatial_width(cfg);
        let ffn_out_ch = ffn_fused::output_channels(cfg);
        let ffn_in = kernels.bufs.ffn_fused_in.as_ref().unwrap();
        let ffn_out = kernels.bufs.ffn_fused_out.as_ref().unwrap();

        let t = Instant::now();
        {
            let mut locked = ffn_in.as_f32_slice_mut();
            let buf = &mut *locked;
            stage_spatial(buf, dim, ffn_sp, &x2norm, seq, 0);
            stage_spatial(buf, dim, ffn_sp, &x2, seq, seq);
        }
        stage_ffn_ms = t.elapsed().as_secs_f32() * 1000.0;

        let t = Instant::now();
        kernels
            .ffn_fused
            .as_ref()
            .unwrap()
            .run_cached_direct(&[ffn_in], &[ffn_out])
            .expect("ANE eval failed");
        ane_ffn_ms = t.elapsed().as_secs_f32() * 1000.0;

        let t = Instant::now();
        {
            let locked = ffn_out.as_f32_slice();
            read_channels_into(&locked, ffn_out_ch, seq, 0, dim, &mut x_next);
            read_channels_into(&locked, ffn_out_ch, seq, dim, hidden, &mut h1);
            read_channels_into(&locked, ffn_out_ch, seq, dim + hidden, hidden, &mut h3);
            read_channels_into(
                &locked,
                ffn_out_ch,
                seq,
                dim + 2 * hidden,
                hidden,
                &mut gate,
            );
        }
        read_ffn_ms = t.elapsed().as_secs_f32() * 1000.0;
    }

    let total_ms = t_total.elapsed().as_secs_f32() * 1000.0;

    let cache = ForwardCache {
        x: x.to_vec(),
        xnorm,
        rms_inv1,
        q_rope,
        k_rope,
        v,
        attn_out,
        o_out,
        x2,
        x2norm,
        rms_inv2,
        h1,
        h3,
        gate,
        w2t_scratch,
        w2t_generation: weights.w2_generation,
    };

    let timings = ForwardTimings {
        rmsnorm1_ms,
        stage_sdpa_ms,
        ane_sdpa_ms,
        read_sdpa_ms,
        stage_wo_ms,
        ane_wo_ms,
        read_wo_ms,
        residual_rmsnorm2_ms,
        stage_ffn_ms,
        ane_ffn_ms,
        read_ffn_ms,
        total_ms,
    };

    (x_next, cache, timings)
}

/// Timing breakdown for backward pass.
#[derive(Debug, Clone)]
pub struct BackwardTimings {
    pub scale_dy_ms: f32,
    pub stage_run_ffn_bwd_w2t_ms: f32,
    pub silu_deriv_ms: f32,
    pub stage_ffn_bwd_w13t_ms: f32,
    pub async_ffn_bwd_w13t_plus_dw_ms: f32,
    pub rmsnorm2_bwd_ms: f32,
    pub stage_run_wot_bwd_ms: f32,
    pub stage_sdpa_bwd1_ms: f32,
    pub async_sdpa_bwd1_plus_dwo_ms: f32,
    pub read_sdpa_bwd1_ms: f32,
    pub stage_run_sdpa_bwd2_ms: f32,
    pub rope_bwd_ms: f32,
    pub stage_q_bwd_ms: f32,
    pub async_q_bwd_plus_dw_ms: f32,
    pub stage_run_kv_bwd_ms: f32,
    pub rmsnorm1_bwd_ms: f32,
    pub merge_dx_ms: f32,
    pub total_ms: f32,
}

impl BackwardTimings {
    pub fn print(&self) {
        println!("  {:<35} {:>6.2}ms", "scale dy (vDSP)", self.scale_dy_ms);
        println!(
            "  {:<35} {:>6.2}ms",
            "stage+run ffnBwdW2t (ANE)", self.stage_run_ffn_bwd_w2t_ms
        );
        println!(
            "  {:<35} {:>6.2}ms",
            "SiLU derivative (CPU)", self.silu_deriv_ms
        );
        println!(
            "  {:<35} {:>6.2}ms",
            "stage ffnBwdW13t", self.stage_ffn_bwd_w13t_ms
        );
        println!(
            "  {:<35} {:>6.2}ms",
            "async ffnBwdW13t + dW2+dW1+dW3", self.async_ffn_bwd_w13t_plus_dw_ms
        );
        println!(
            "  {:<35} {:>6.2}ms",
            "RMSNorm2 backward (CPU)", self.rmsnorm2_bwd_ms
        );
        println!(
            "  {:<35} {:>6.2}ms",
            "stage+run wotBwd (ANE)", self.stage_run_wot_bwd_ms
        );
        println!(
            "  {:<35} {:>6.2}ms",
            "stage sdpaBwd1", self.stage_sdpa_bwd1_ms
        );
        println!(
            "  {:<35} {:>6.2}ms",
            "async sdpaBwd1 + dWo", self.async_sdpa_bwd1_plus_dwo_ms
        );
        println!(
            "  {:<35} {:>6.2}ms",
            "read sdpaBwd1 output", self.read_sdpa_bwd1_ms
        );
        println!(
            "  {:<35} {:>6.2}ms",
            "stage+run sdpaBwd2 (ANE)", self.stage_run_sdpa_bwd2_ms
        );
        println!(
            "  {:<35} {:>6.2}ms",
            "RoPE backward (CPU)", self.rope_bwd_ms
        );
        println!("  {:<35} {:>6.2}ms", "stage qBwd", self.stage_q_bwd_ms);
        println!(
            "  {:<35} {:>6.2}ms",
            "async qBwd + dWq+dWk+dWv", self.async_q_bwd_plus_dw_ms
        );
        println!(
            "  {:<35} {:>6.2}ms",
            "stage+run kvBwd (ANE)", self.stage_run_kv_bwd_ms
        );
        println!(
            "  {:<35} {:>6.2}ms",
            "RMSNorm1 backward (CPU)", self.rmsnorm1_bwd_ms
        );
        println!("  {:<35} {:>6.2}ms", "merge dx (vDSP)", self.merge_dx_ms);
        println!("  {:<35} {:>6.2}ms", "TOTAL", self.total_ms);
    }
}

/// Backward pass with per-operation timing (same output as `backward`).
pub fn backward_timed(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &LayerWeights,
    cache: &ForwardCache,
    dy: &[f32],
    grads: &mut LayerGrads,
    ws: &mut BackwardWorkspace,
) -> (Vec<f32>, BackwardTimings) {
    let t_total = Instant::now();
    let dim = cfg.dim;
    let seq = cfg.seq;
    let q_dim = cfg.q_dim;
    let kv_dim = cfg.kv_dim;
    let hidden = cfg.hidden;
    let heads = cfg.heads;
    let hd = cfg.hd;
    let alpha = 1.0 / (2.0 * cfg.nlayers as f32).sqrt();

    // 1. Scale dy
    let t = Instant::now();
    vdsp::vsmul(dy, alpha, &mut ws.dffn);
    let scale_dy_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 2. ffnBwdW2t
    let t = Instant::now();
    let w2t_sp = dyn_matmul::spatial_width(seq, hidden);
    {
        let mut locked = kernels.bufs.ffn_bwd_w2t_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, dim, w2t_sp, &ws.dffn, seq, 0);
        stage_spatial(buf, dim, w2t_sp, &weights.w2, hidden, seq);
    }
    kernels
        .ffn_bwd_w2t
        .run_cached_direct(
            &[&kernels.bufs.ffn_bwd_w2t_in],
            &[&kernels.bufs.ffn_bwd_w2t_out],
        )
        .expect("ANE eval failed");
    {
        let locked = kernels.bufs.ffn_bwd_w2t_out.as_f32_slice();
        ws.dsilu_raw.copy_from_slice(&locked[..hidden * seq]);
    }
    let stage_run_ffn_bwd_w2t_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 3. SiLU derivative (vvexpf + vvrecf precompute sig, then division-free scalar loop)
    let t = Instant::now();
    let n = hidden * seq;
    {
        vdsp::vsmul(&cache.h1, -1.0, &mut ws.neg_h1);
        vdsp::expf(&ws.neg_h1, &mut ws.exp_neg);
        // Precompute sigmoid via vectorized reciprocal — eliminates scalar fdiv from hot loop
        vdsp::vsadd(&ws.exp_neg, 1.0, &mut ws.neg_h1); // neg_h1 = 1 + exp(-h1)
        vdsp::recf_inplace(&mut ws.neg_h1); // neg_h1 = sig = 1/(1+exp(-h1))
        for i in 0..n {
            let sig = ws.neg_h1[i];
            let silu_val = cache.h1[i] * sig;
            let silu_deriv = sig * (1.0 + cache.h1[i] * (1.0 - sig));
            ws.dh3[i] = ws.dsilu_raw[i] * silu_val;
            ws.dh1[i] = ws.dsilu_raw[i] * cache.h3[i] * silu_deriv;
        }
    }
    let silu_deriv_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 4. Stage ffnBwdW13t (mtrans + stage_spatial)
    let t = Instant::now();
    let w13t_sp = dyn_matmul::dual_spatial_width(seq, dim);
    {
        vdsp::mtrans(&weights.w1, hidden, &mut ws.w1t, dim, dim, hidden);
        vdsp::mtrans(&weights.w3, hidden, &mut ws.w3t, dim, dim, hidden);
        let mut locked = kernels.bufs.ffn_bwd_w13t_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, hidden, w13t_sp, &ws.dh1, seq, 0);
        stage_spatial(buf, hidden, w13t_sp, &ws.dh3, seq, seq);
        stage_spatial(buf, hidden, w13t_sp, &ws.w1t, dim, 2 * seq);
        stage_spatial(buf, hidden, w13t_sp, &ws.w3t, dim, 2 * seq + dim);
    }
    let stage_ffn_bwd_w13t_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 5. ASYNC: ANE ffnBwdW13t || CPU dW
    let t = Instant::now();
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels
                .ffn_bwd_w13t
                .run_cached_direct(
                    &[&kernels.bufs.ffn_bwd_w13t_in],
                    &[&kernels.bufs.ffn_bwd_w13t_out],
                )
                .expect("ANE eval failed");
        });
        accumulate_dw(&ws.dffn, dim, &cache.gate, hidden, seq, &mut grads.dw2);
        accumulate_dw(&cache.x2norm, dim, &ws.dh1, hidden, seq, &mut grads.dw1);
        accumulate_dw(&cache.x2norm, dim, &ws.dh3, hidden, seq, &mut grads.dw3);
        ane_handle.join().expect("ANE thread panicked");
    });
    {
        let locked = kernels.bufs.ffn_bwd_w13t_out.as_f32_slice();
        ws.dx_ffn.copy_from_slice(&locked[..dim * seq]);
    }
    let async_ffn_bwd_w13t_plus_dw_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 6. RMSNorm2 backward (channel-first, no transpose)
    let t = Instant::now();
    rmsnorm::backward_channel_first(
        &ws.dx_ffn,
        &cache.x2,
        &weights.gamma2,
        &cache.rms_inv2,
        &mut ws.dx2,
        &mut grads.dgamma2,
        dim,
        seq,
        &mut ws.rms_dot_buf,
    );
    vdsp::vadd(&ws.dx2, dy, &mut ws.dx2_tmp);
    ws.dx2.copy_from_slice(&ws.dx2_tmp);
    let rmsnorm2_bwd_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 7. wotBwd (mtrans Wo) + async pre-stage sdpaBwd1
    let t = Instant::now();
    vdsp::vsmul(&ws.dx2, alpha, &mut ws.dx2_scaled);
    let wot_sp = dyn_matmul::spatial_width(seq, q_dim);
    {
        vdsp::mtrans(&weights.wo, dim, &mut ws.wot, q_dim, q_dim, dim);
        let mut locked = kernels.bufs.wot_bwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, dim, wot_sp, &ws.dx2_scaled, seq, 0);
        stage_spatial(buf, dim, wot_sp, &ws.wot, q_dim, seq);
    }
    let bwd1_in_ch = sdpa_bwd::bwd1_input_channels(cfg);
    let bwd1_out_ch = sdpa_bwd::bwd1_output_channels(cfg);
    // ASYNC: ANE wotBwd || pre-stage 3 of 4 sdpaBwd1 inputs (from forward cache)
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels
                .wot_bwd
                .run_cached_direct(&[&kernels.bufs.wot_bwd_in], &[&kernels.bufs.wot_bwd_out])
                .expect("ANE eval failed");
        });
        {
            let mut locked = kernels.bufs.sdpa_bwd1_in.as_f32_slice_mut();
            let buf = &mut *locked;
            pack_channels(buf, bwd1_in_ch, seq, &cache.q_rope, q_dim, 0);
            pack_channels(buf, bwd1_in_ch, seq, &cache.k_rope, q_dim, q_dim);
            pack_channels(buf, bwd1_in_ch, seq, &cache.v, q_dim, 2 * q_dim);
        }
        ane_handle.join().expect("ANE thread panicked");
    });
    {
        let locked = kernels.bufs.wot_bwd_out.as_f32_slice();
        ws.da.copy_from_slice(&locked[..q_dim * seq]);
    }
    let stage_run_wot_bwd_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 8. Stage remaining sdpaBwd1 input (da depends on wotBwd output)
    let t = Instant::now();
    {
        let mut locked = kernels.bufs.sdpa_bwd1_in.as_f32_slice_mut();
        let buf = &mut *locked;
        pack_channels(buf, bwd1_in_ch, seq, &ws.da, q_dim, 3 * q_dim);
    }
    let stage_sdpa_bwd1_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 9. ASYNC: ANE sdpaBwd1 || CPU dWo
    let t = Instant::now();
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels
                .sdpa_bwd1
                .run_cached_direct(
                    &[&kernels.bufs.sdpa_bwd1_in],
                    &[&kernels.bufs.sdpa_bwd1_out],
                )
                .expect("ANE eval failed");
        });
        accumulate_dw(
            &cache.attn_out,
            q_dim,
            &ws.dx2_scaled,
            dim,
            seq,
            &mut grads.dwo,
        );
        ane_handle.join().expect("ANE thread panicked");
    });
    let async_sdpa_bwd1_plus_dwo_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 10. Read sdpaBwd1
    let t = Instant::now();
    let score_ch = heads * seq;
    {
        let locked = kernels.bufs.sdpa_bwd1_out.as_f32_slice();
        read_channels_into(&locked, bwd1_out_ch, seq, 0, q_dim, &mut ws.dv_full);
        read_channels_into(
            &locked,
            bwd1_out_ch,
            seq,
            q_dim,
            score_ch,
            &mut ws.probs_flat,
        );
        read_channels_into(
            &locked,
            bwd1_out_ch,
            seq,
            q_dim + score_ch,
            score_ch,
            &mut ws.dp_flat,
        );
    }
    let read_sdpa_bwd1_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 11. sdpaBwd2 + pre-compute wqt+wkt+wvt (overlapped with ANE)
    let t = Instant::now();
    let bwd2_in_ch = sdpa_bwd::bwd2_input_channels(cfg);
    let bwd2_out_ch = sdpa_bwd::bwd2_output_channels(cfg);
    {
        let mut locked = kernels.bufs.sdpa_bwd2_in.as_f32_slice_mut();
        let buf = &mut *locked;
        pack_channels(buf, bwd2_in_ch, seq, &ws.probs_flat, score_ch, 0);
        pack_channels(buf, bwd2_in_ch, seq, &ws.dp_flat, score_ch, score_ch);
        pack_channels(buf, bwd2_in_ch, seq, &cache.q_rope, q_dim, 2 * score_ch);
        pack_channels(
            buf,
            bwd2_in_ch,
            seq,
            &cache.k_rope,
            q_dim,
            2 * score_ch + q_dim,
        );
    }
    kernels
        .sdpa_bwd2
        .run_cached_direct(
            &[&kernels.bufs.sdpa_bwd2_in],
            &[&kernels.bufs.sdpa_bwd2_out],
        )
        .expect("ANE eval failed");
    {
        let locked = kernels.bufs.sdpa_bwd2_out.as_f32_slice();
        read_channels_into(&locked, bwd2_out_ch, seq, 0, q_dim, &mut ws.dq);
        read_channels_into(&locked, bwd2_out_ch, seq, q_dim, q_dim, &mut ws.dk);
    }
    // Pre-compute transposes (timed version runs sequentially for measurement clarity)
    vdsp::mtrans(&weights.wq, q_dim, &mut ws.wqt, dim, dim, q_dim);
    vdsp::mtrans(&weights.wk, kv_dim, &mut ws.wkt, dim, dim, kv_dim);
    vdsp::mtrans(&weights.wv, kv_dim, &mut ws.wvt, dim, dim, kv_dim);
    let stage_run_sdpa_bwd2_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 12. RoPE backward
    let t = Instant::now();
    rope_backward_inplace(&mut ws.dq, heads, hd, seq, &kernels.rope);
    rope_backward_inplace(&mut ws.dk, heads, hd, seq, &kernels.rope);
    let rope_bwd_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 12.5. Stage kvBwd early (wkt, wvt from step 11, dk post-RoPE)
    let t = Instant::now();
    let kv_bwd_sp = dyn_matmul::dual_spatial_width(seq, dim);
    {
        let mut locked = kernels.bufs.kv_bwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        for c in 0..kv_dim {
            let row = c * kv_bwd_sp;
            buf[row..row + seq].copy_from_slice(&ws.dk[c * seq..c * seq + seq]);
            buf[row + seq..row + 2 * seq].copy_from_slice(&ws.dv_full[c * seq..c * seq + seq]);
            buf[row + 2 * seq..row + 2 * seq + dim]
                .copy_from_slice(&ws.wkt[c * dim..c * dim + dim]);
            buf[row + 2 * seq + dim..row + 2 * seq + 2 * dim]
                .copy_from_slice(&ws.wvt[c * dim..c * dim + dim]);
        }
    }
    let stage_kv_bwd_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 13. Stage qBwd (wqt already computed in step 11)
    let t = Instant::now();
    let q_bwd_sp = dyn_matmul::spatial_width(seq, dim);
    {
        let mut locked = kernels.bufs.q_bwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, q_dim, q_bwd_sp, &ws.dq, seq, 0);
        stage_spatial(buf, q_dim, q_bwd_sp, &ws.wqt, dim, seq);
    }
    let stage_q_bwd_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 14. ASYNC: ANE qBwd+kvBwd || CPU dWq+dWk+dWv
    // kvBwd is already staged (step 12.5), so both kernels run back-to-back on ANE
    // while the main thread computes weight gradients.
    let t = Instant::now();
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels
                .q_bwd
                .run_cached_direct(&[&kernels.bufs.q_bwd_in], &[&kernels.bufs.q_bwd_out])
                .expect("ANE eval failed");
            kernels
                .kv_bwd
                .run_cached_direct(&[&kernels.bufs.kv_bwd_in], &[&kernels.bufs.kv_bwd_out])
                .expect("ANE eval failed");
        });
        accumulate_dw(&cache.xnorm, dim, &ws.dq, q_dim, seq, &mut grads.dwq);
        accumulate_dw(&cache.xnorm, dim, &ws.dk, kv_dim, seq, &mut grads.dwk);
        accumulate_dw(&cache.xnorm, dim, &ws.dv_full, kv_dim, seq, &mut grads.dwv);
        ane_handle.join().expect("ANE thread panicked");
    });
    {
        let locked = kernels.bufs.q_bwd_out.as_f32_slice();
        ws.dx_attn.copy_from_slice(&locked[..dim * seq]);
    }
    {
        let locked = kernels.bufs.kv_bwd_out.as_f32_slice();
        ws.dx_kv.copy_from_slice(&locked[..dim * seq]);
    }
    // Report combined time: kvBwd staging moved to step 12.5, ANE time overlapped here
    let async_q_bwd_plus_dw_ms = t.elapsed().as_secs_f32() * 1000.0;
    let stage_run_kv_bwd_ms = stage_kv_bwd_ms; // Staging only (ANE time is inside async block above)

    // 15. Merge + RMSNorm1 backward
    let t = Instant::now();
    vdsp::vadd(&ws.dx_attn, &ws.dx_kv, &mut ws.dx_merged);
    let merge_dx_ms = t.elapsed().as_secs_f32() * 1000.0;

    let t = Instant::now();
    rmsnorm::backward_channel_first(
        &ws.dx_merged,
        &cache.x,
        &weights.gamma1,
        &cache.rms_inv1,
        &mut ws.dx_rms1,
        &mut grads.dgamma1,
        dim,
        seq,
        &mut ws.rms_dot_buf,
    );
    let rmsnorm1_bwd_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 16. Final dx
    let mut dx = vec![0.0f32; dim * seq]; // only allocation — return value
    vdsp::vadd(&ws.dx_rms1, &ws.dx2, &mut dx);

    let total_ms = t_total.elapsed().as_secs_f32() * 1000.0;

    let timings = BackwardTimings {
        scale_dy_ms,
        stage_run_ffn_bwd_w2t_ms,
        silu_deriv_ms,
        stage_ffn_bwd_w13t_ms,
        async_ffn_bwd_w13t_plus_dw_ms,
        rmsnorm2_bwd_ms,
        stage_run_wot_bwd_ms,
        stage_sdpa_bwd1_ms,
        async_sdpa_bwd1_plus_dwo_ms,
        read_sdpa_bwd1_ms,
        stage_run_sdpa_bwd2_ms,
        rope_bwd_ms,
        stage_q_bwd_ms,
        async_q_bwd_plus_dw_ms,
        stage_run_kv_bwd_ms,
        rmsnorm1_bwd_ms,
        merge_dx_ms,
        total_ms,
    };

    (dx, timings)
}

// ── Backward pass ──

/// Run backward pass for one transformer layer.
/// `dy` is gradient of loss w.r.t. layer output [DIM * SEQ].
/// Returns `dx` (gradient w.r.t. layer input) and fills `grads`.
/// Uses pre-allocated workspace to eliminate ~32 vec allocations per call.
pub fn backward(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &LayerWeights,
    cache: &ForwardCache,
    dy: &[f32],
    grads: &mut LayerGrads,
    ws: &mut BackwardWorkspace,
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
    vdsp::vsmul(dy, alpha, &mut ws.dffn);

    // ── 2. ffnBwdW2t(ANE): dffn @ W2 → dsilu_raw [HIDDEN, SEQ] ──
    let w2t_sp = dyn_matmul::spatial_width(seq, hidden);
    {
        let mut locked = kernels.bufs.ffn_bwd_w2t_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, dim, w2t_sp, &ws.dffn, seq, 0);
        stage_spatial(buf, dim, w2t_sp, &weights.w2, hidden, seq);
    }
    kernels
        .ffn_bwd_w2t
        .run_cached_direct(
            &[&kernels.bufs.ffn_bwd_w2t_in],
            &[&kernels.bufs.ffn_bwd_w2t_out],
        )
        .expect("ANE eval failed");

    {
        let locked = kernels.bufs.ffn_bwd_w2t_out.as_f32_slice();
        ws.dsilu_raw.copy_from_slice(&locked[..hidden * seq]);
    }

    // ── 3. SiLU derivative (vvexpf + vvrecf precompute sig, then division-free scalar loop) ──
    let n = hidden * seq;
    {
        vdsp::vsmul(&cache.h1, -1.0, &mut ws.neg_h1);
        vdsp::expf(&ws.neg_h1, &mut ws.exp_neg);
        // Precompute sigmoid via vectorized reciprocal — eliminates scalar fdiv from hot loop
        vdsp::vsadd(&ws.exp_neg, 1.0, &mut ws.neg_h1); // neg_h1 = 1 + exp(-h1)
        vdsp::recf_inplace(&mut ws.neg_h1); // neg_h1 = sig = 1/(1+exp(-h1))
        for i in 0..n {
            let sig = ws.neg_h1[i];
            let silu_val = cache.h1[i] * sig;
            let silu_deriv = sig * (1.0 + cache.h1[i] * (1.0 - sig));
            ws.dh3[i] = ws.dsilu_raw[i] * silu_val;
            ws.dh1[i] = ws.dsilu_raw[i] * cache.h3[i] * silu_deriv;
        }
    }

    // ── 4. Stage ffnBwdW13t: mtrans weights, then stage_spatial ──
    let w13t_sp = dyn_matmul::dual_spatial_width(seq, dim);
    {
        vdsp::mtrans(&weights.w1, hidden, &mut ws.w1t, dim, dim, hidden);
        vdsp::mtrans(&weights.w3, hidden, &mut ws.w3t, dim, dim, hidden);

        let mut locked = kernels.bufs.ffn_bwd_w13t_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, hidden, w13t_sp, &ws.dh1, seq, 0);
        stage_spatial(buf, hidden, w13t_sp, &ws.dh3, seq, seq);
        stage_spatial(buf, hidden, w13t_sp, &ws.w1t, dim, 2 * seq);
        stage_spatial(buf, hidden, w13t_sp, &ws.w3t, dim, 2 * seq + dim);
    }

    // ── 5. ASYNC: ANE ffnBwdW13t || CPU dW3 accumulation ──
    // dW2 moved to step 9 (sdpaBwd1 overlap), dW1 moved to step 10 (sdpaBwd2 overlap)
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels
                .ffn_bwd_w13t
                .run_cached_direct(
                    &[&kernels.bufs.ffn_bwd_w13t_in],
                    &[&kernels.bufs.ffn_bwd_w13t_out],
                )
                .expect("ANE eval failed");
        });
        accumulate_dw(&cache.x2norm, dim, &ws.dh3, hidden, seq, &mut grads.dw3);
        ane_handle.join().expect("ANE thread panicked");
    });

    {
        let locked = kernels.bufs.ffn_bwd_w13t_out.as_f32_slice();
        ws.dx_ffn.copy_from_slice(&locked[..dim * seq]);
    }

    // ── 6. RMSNorm2 backward (CPU): channel-first, no transpose ──
    rmsnorm::backward_channel_first(
        &ws.dx_ffn,
        &cache.x2,
        &weights.gamma2,
        &cache.rms_inv2,
        &mut ws.dx2,
        &mut grads.dgamma2,
        dim,
        seq,
        &mut ws.rms_dot_buf,
    );
    // Add residual gradient: dx2 += dy (FFN residual branch, vDSP vectorized)
    vdsp::vadd(&ws.dx2, dy, &mut ws.dx2_tmp);
    ws.dx2.copy_from_slice(&ws.dx2_tmp);

    // ── 7. Scale dx2 for attention residual (vDSP vectorized) ──
    vdsp::vsmul(&ws.dx2, alpha, &mut ws.dx2_scaled);

    // ── 8. wotBwd(ANE): dx2_scaled @ Wo → da [Q_DIM, SEQ] ──
    let wot_sp = dyn_matmul::spatial_width(seq, q_dim);
    {
        vdsp::mtrans(&weights.wo, dim, &mut ws.wot, q_dim, q_dim, dim);
        let mut locked = kernels.bufs.wot_bwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, dim, wot_sp, &ws.dx2_scaled, seq, 0);
        stage_spatial(buf, dim, wot_sp, &ws.wot, q_dim, seq);
    }
    let bwd1_in_ch = sdpa_bwd::bwd1_input_channels(cfg);
    let bwd1_out_ch = sdpa_bwd::bwd1_output_channels(cfg);
    // ASYNC: ANE wotBwd || pre-stage 3 of 4 sdpaBwd1 inputs (from forward cache)
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels
                .wot_bwd
                .run_cached_direct(&[&kernels.bufs.wot_bwd_in], &[&kernels.bufs.wot_bwd_out])
                .expect("ANE eval failed");
        });
        {
            let mut locked = kernels.bufs.sdpa_bwd1_in.as_f32_slice_mut();
            let buf = &mut *locked;
            pack_channels(buf, bwd1_in_ch, seq, &cache.q_rope, q_dim, 0);
            pack_channels(buf, bwd1_in_ch, seq, &cache.k_rope, q_dim, q_dim);
            pack_channels(buf, bwd1_in_ch, seq, &cache.v, q_dim, 2 * q_dim);
        }
        ane_handle.join().expect("ANE thread panicked");
    });

    {
        let locked = kernels.bufs.wot_bwd_out.as_f32_slice();
        ws.da.copy_from_slice(&locked[..q_dim * seq]);
    }

    // ── 9. Stage remaining sdpaBwd1 input (da depends on wotBwd output) ──
    {
        let mut locked = kernels.bufs.sdpa_bwd1_in.as_f32_slice_mut();
        let buf = &mut *locked;
        pack_channels(buf, bwd1_in_ch, seq, &ws.da, q_dim, 3 * q_dim);
    }

    // ASYNC: ANE sdpaBwd1 || CPU dWo + dW2 accumulation
    // dW2 moved from step 5 to rebalance CPU load across async blocks
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels
                .sdpa_bwd1
                .run_cached_direct(
                    &[&kernels.bufs.sdpa_bwd1_in],
                    &[&kernels.bufs.sdpa_bwd1_out],
                )
                .expect("ANE eval failed");
        });
        accumulate_dw(
            &cache.attn_out,
            q_dim,
            &ws.dx2_scaled,
            dim,
            seq,
            &mut grads.dwo,
        );
        accumulate_dw(&ws.dffn, dim, &cache.gate, hidden, seq, &mut grads.dw2);
        ane_handle.join().expect("ANE thread panicked");
    });

    let score_ch = heads * seq;
    {
        let locked = kernels.bufs.sdpa_bwd1_out.as_f32_slice();
        read_channels_into(&locked, bwd1_out_ch, seq, 0, q_dim, &mut ws.dv_full);
        read_channels_into(
            &locked,
            bwd1_out_ch,
            seq,
            q_dim,
            score_ch,
            &mut ws.probs_flat,
        );
        read_channels_into(
            &locked,
            bwd1_out_ch,
            seq,
            q_dim + score_ch,
            score_ch,
            &mut ws.dp_flat,
        );
    }

    // ── 10. sdpaBwd2(ANE) || pre-compute wqt+wkt+wvt for steps 12-14 ──
    let bwd2_in_ch = sdpa_bwd::bwd2_input_channels(cfg);
    let bwd2_out_ch = sdpa_bwd::bwd2_output_channels(cfg);
    {
        let mut locked = kernels.bufs.sdpa_bwd2_in.as_f32_slice_mut();
        let buf = &mut *locked;
        pack_channels(buf, bwd2_in_ch, seq, &ws.probs_flat, score_ch, 0);
        pack_channels(buf, bwd2_in_ch, seq, &ws.dp_flat, score_ch, score_ch);
        pack_channels(buf, bwd2_in_ch, seq, &cache.q_rope, q_dim, 2 * score_ch);
        pack_channels(
            buf,
            bwd2_in_ch,
            seq,
            &cache.k_rope,
            q_dim,
            2 * score_ch + q_dim,
        );
    }
    // ASYNC: ANE sdpaBwd2 || pre-compute wqt+wkt+wvt + dW1 (moved from step 5 to rebalance)
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels
                .sdpa_bwd2
                .run_cached_direct(
                    &[&kernels.bufs.sdpa_bwd2_in],
                    &[&kernels.bufs.sdpa_bwd2_out],
                )
                .expect("ANE eval failed");
        });
        vdsp::mtrans(&weights.wq, q_dim, &mut ws.wqt, dim, dim, q_dim);
        vdsp::mtrans(&weights.wk, kv_dim, &mut ws.wkt, dim, dim, kv_dim);
        vdsp::mtrans(&weights.wv, kv_dim, &mut ws.wvt, dim, dim, kv_dim);
        accumulate_dw(&cache.x2norm, dim, &ws.dh1, hidden, seq, &mut grads.dw1);
        ane_handle.join().expect("ANE thread panicked");
    });
    {
        let locked = kernels.bufs.sdpa_bwd2_out.as_f32_slice();
        read_channels_into(&locked, bwd2_out_ch, seq, 0, q_dim, &mut ws.dq);
        read_channels_into(&locked, bwd2_out_ch, seq, q_dim, q_dim, &mut ws.dk);
    }

    // ── 11. RoPE backward in-place (CPU, cached tables) ──
    rope_backward_inplace(&mut ws.dq, heads, hd, seq, &kernels.rope);
    rope_backward_inplace(&mut ws.dk, heads, hd, seq, &kernels.rope);

    // ── 11.5. Stage kvBwd early (wkt, wvt from step 10, dk post-RoPE) ──
    let kv_bwd_sp = dyn_matmul::dual_spatial_width(seq, dim);
    {
        let mut locked = kernels.bufs.kv_bwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        for c in 0..kv_dim {
            let row = c * kv_bwd_sp;
            buf[row..row + seq].copy_from_slice(&ws.dk[c * seq..c * seq + seq]);
            buf[row + seq..row + 2 * seq].copy_from_slice(&ws.dv_full[c * seq..c * seq + seq]);
            buf[row + 2 * seq..row + 2 * seq + dim]
                .copy_from_slice(&ws.wkt[c * dim..c * dim + dim]);
            buf[row + 2 * seq + dim..row + 2 * seq + 2 * dim]
                .copy_from_slice(&ws.wvt[c * dim..c * dim + dim]);
        }
    }

    // ── 12. Stage qBwd (wqt already computed in step 10) ──
    let q_bwd_sp = dyn_matmul::spatial_width(seq, dim);
    {
        let mut locked = kernels.bufs.q_bwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, q_dim, q_bwd_sp, &ws.dq, seq, 0);
        stage_spatial(buf, q_dim, q_bwd_sp, &ws.wqt, dim, seq);
    }

    // ── 13. ASYNC: ANE qBwd+kvBwd || CPU dWq+dWk+dWv ──
    // kvBwd is already staged (step 11.5), so both kernels run back-to-back on ANE
    // while the main thread computes weight gradients.
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels
                .q_bwd
                .run_cached_direct(&[&kernels.bufs.q_bwd_in], &[&kernels.bufs.q_bwd_out])
                .expect("ANE eval failed");
            kernels
                .kv_bwd
                .run_cached_direct(&[&kernels.bufs.kv_bwd_in], &[&kernels.bufs.kv_bwd_out])
                .expect("ANE eval failed");
        });
        accumulate_dw(&cache.xnorm, dim, &ws.dq, q_dim, seq, &mut grads.dwq);
        accumulate_dw(&cache.xnorm, dim, &ws.dk, kv_dim, seq, &mut grads.dwk);
        accumulate_dw(&cache.xnorm, dim, &ws.dv_full, kv_dim, seq, &mut grads.dwv);
        ane_handle.join().expect("ANE thread panicked");
    });
    {
        let locked = kernels.bufs.q_bwd_out.as_f32_slice();
        ws.dx_attn.copy_from_slice(&locked[..dim * seq]);
    }
    {
        let locked = kernels.bufs.kv_bwd_out.as_f32_slice();
        ws.dx_kv.copy_from_slice(&locked[..dim * seq]);
    }

    // ── 14. Merge: dx_attn + dx_kv (vDSP vectorized) ──
    vdsp::vadd(&ws.dx_attn, &ws.dx_kv, &mut ws.dx_merged);

    // ── 16. RMSNorm1 backward (CPU): channel-first, no transpose ──
    rmsnorm::backward_channel_first(
        &ws.dx_merged,
        &cache.x,
        &weights.gamma1,
        &cache.rms_inv1,
        &mut ws.dx_rms1,
        &mut grads.dgamma1,
        dim,
        seq,
        &mut ws.rms_dot_buf,
    );

    // ── 17. Final: dx = dx_rms1 + dx2 (residual from attention branch, vDSP vectorized) ──
    let mut dx = vec![0.0f32; dim * seq]; // only allocation — return value
    vdsp::vadd(&ws.dx_rms1, &ws.dx2, &mut dx);

    dx
}

/// Backward pass writing dx into pre-allocated buffer (zero allocations).
pub fn backward_into(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &LayerWeights,
    cache: &ForwardCache,
    dy: &[f32],
    grads: &mut LayerGrads,
    ws: &mut BackwardWorkspace,
    dx_out: &mut [f32],
) {
    let dim = cfg.dim;
    let seq = cfg.seq;
    let q_dim = cfg.q_dim;
    let kv_dim = cfg.kv_dim;
    let hidden = cfg.hidden;
    let heads = cfg.heads;
    let hd = cfg.hd;
    let alpha = 1.0 / (2.0 * cfg.nlayers as f32).sqrt();

    // 1. Scale dy
    vdsp::vsmul(dy, alpha, &mut ws.dffn);

    // 2. ffnBwdW2t
    let w2t_sp = dyn_matmul::spatial_width(seq, hidden);
    {
        let mut locked = kernels.bufs.ffn_bwd_w2t_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, dim, w2t_sp, &ws.dffn, seq, 0);
        stage_spatial(buf, dim, w2t_sp, &weights.w2, hidden, seq);
    }
    // ASYNC: ANE ffnBwdW2t || pre-compute w1t, w3t + sigmoid(h1) for steps 3+4
    // Sigmoid chain (vsmul+expf+vsadd+recf) is the expensive part of SiLU backward.
    // Moving it here hides ~0.6ms/layer behind ANE dispatch time.
    let w13t_sp = dyn_matmul::dual_spatial_width(seq, dim);
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels
                .ffn_bwd_w2t
                .run_cached_direct(
                    &[&kernels.bufs.ffn_bwd_w2t_in],
                    &[&kernels.bufs.ffn_bwd_w2t_out],
                )
                .expect("ANE eval failed");
        });
        vdsp::mtrans(&weights.w1, hidden, &mut ws.w1t, dim, dim, hidden);
        vdsp::mtrans(&weights.w3, hidden, &mut ws.w3t, dim, dim, hidden);
        // Pre-compute sigmoid(h1) — doesn't need dsilu_raw (ANE output), safe to overlap
        vdsp::vsmul(&cache.h1, -1.0, &mut ws.neg_h1);
        vdsp::expf(&ws.neg_h1, &mut ws.exp_neg);
        vdsp::vsadd(&ws.exp_neg, 1.0, &mut ws.neg_h1); // neg_h1 = 1 + exp(-h1)
        vdsp::recf_inplace(&mut ws.neg_h1); // neg_h1 = sig = 1/(1+exp(-h1))
        ane_handle.join().expect("ANE thread panicked");
    });
    {
        let locked = kernels.bufs.ffn_bwd_w2t_out.as_f32_slice();
        ws.dsilu_raw.copy_from_slice(&locked[..hidden * seq]);
    }

    // 3. SiLU backward scalar loop (sig already in neg_h1 from step 2 overlap)
    let n = hidden * seq;
    {
        for i in 0..n {
            let sig = ws.neg_h1[i];
            let silu_val = cache.h1[i] * sig;
            let silu_deriv = sig * (1.0 + cache.h1[i] * (1.0 - sig));
            ws.dh3[i] = ws.dsilu_raw[i] * silu_val;
            ws.dh1[i] = ws.dsilu_raw[i] * cache.h3[i] * silu_deriv;
        }
    }

    // 4. Stage ffnBwdW13t — fused single-pass (w1t, w3t from step 2 overlap)
    {
        let mut locked = kernels.bufs.ffn_bwd_w13t_in.as_f32_slice_mut();
        let buf = &mut *locked;
        for c in 0..hidden {
            let row = c * w13t_sp;
            buf[row..row + seq].copy_from_slice(&ws.dh1[c * seq..c * seq + seq]);
            buf[row + seq..row + 2 * seq].copy_from_slice(&ws.dh3[c * seq..c * seq + seq]);
            buf[row + 2 * seq..row + 2 * seq + dim]
                .copy_from_slice(&ws.w1t[c * dim..c * dim + dim]);
            buf[row + 2 * seq + dim..row + 2 * seq + 2 * dim]
                .copy_from_slice(&ws.w3t[c * dim..c * dim + dim]);
        }
    }

    // 5. ASYNC: ANE ffnBwdW13t || CPU dW3 + pre-compute wot for step 7
    // dW2 moved to step 9 (sdpaBwd1 overlap), dW1 moved to step 10 (sdpaBwd2 overlap)
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels
                .ffn_bwd_w13t
                .run_cached_direct(
                    &[&kernels.bufs.ffn_bwd_w13t_in],
                    &[&kernels.bufs.ffn_bwd_w13t_out],
                )
                .expect("ANE eval failed");
        });
        accumulate_dw(&cache.x2norm, dim, &ws.dh3, hidden, seq, &mut grads.dw3);
        vdsp::mtrans(&weights.wo, dim, &mut ws.wot, q_dim, q_dim, dim);
        ane_handle.join().expect("ANE thread panicked");
    });
    {
        let locked = kernels.bufs.ffn_bwd_w13t_out.as_f32_slice();
        ws.dx_ffn.copy_from_slice(&locked[..dim * seq]);
    }

    // 6. RMSNorm2 backward
    rmsnorm::backward_channel_first(
        &ws.dx_ffn,
        &cache.x2,
        &weights.gamma2,
        &cache.rms_inv2,
        &mut ws.dx2,
        &mut grads.dgamma2,
        dim,
        seq,
        &mut ws.rms_dot_buf,
    );
    vdsp::vadd(&ws.dx2, dy, &mut ws.dx2_tmp);
    ws.dx2.copy_from_slice(&ws.dx2_tmp);

    // 7. wotBwd (wot already computed in step 5 overlap)
    vdsp::vsmul(&ws.dx2, alpha, &mut ws.dx2_scaled);
    let wot_sp = dyn_matmul::spatial_width(seq, q_dim);
    {
        let mut locked = kernels.bufs.wot_bwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, dim, wot_sp, &ws.dx2_scaled, seq, 0);
        stage_spatial(buf, dim, wot_sp, &ws.wot, q_dim, seq);
    }
    let bwd1_in_ch = sdpa_bwd::bwd1_input_channels(cfg);
    let bwd1_out_ch = sdpa_bwd::bwd1_output_channels(cfg);
    // ASYNC: ANE wotBwd || pre-stage 3 of 4 sdpaBwd1 inputs (from forward cache)
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels
                .wot_bwd
                .run_cached_direct(&[&kernels.bufs.wot_bwd_in], &[&kernels.bufs.wot_bwd_out])
                .expect("ANE eval failed");
        });
        {
            let mut locked = kernels.bufs.sdpa_bwd1_in.as_f32_slice_mut();
            let buf = &mut *locked;
            pack_channels(buf, bwd1_in_ch, seq, &cache.q_rope, q_dim, 0);
            pack_channels(buf, bwd1_in_ch, seq, &cache.k_rope, q_dim, q_dim);
            pack_channels(buf, bwd1_in_ch, seq, &cache.v, q_dim, 2 * q_dim);
        }
        ane_handle.join().expect("ANE thread panicked");
    });
    {
        let locked = kernels.bufs.wot_bwd_out.as_f32_slice();
        ws.da.copy_from_slice(&locked[..q_dim * seq]);
    }

    // 8. Stage remaining sdpaBwd1 input (da depends on wotBwd output)
    {
        let mut locked = kernels.bufs.sdpa_bwd1_in.as_f32_slice_mut();
        let buf = &mut *locked;
        pack_channels(buf, bwd1_in_ch, seq, &ws.da, q_dim, 3 * q_dim);
    }

    // 9. ASYNC: ANE sdpaBwd1 || CPU dWo + dW2 (moved from step 5 to rebalance CPU load)
    // Step 5 was CPU-bound at ~2.3ms (3 sgemm); step 9 had ~0.6ms ANE headroom.
    // dW2 = dffn @ gate^T — both available since step 1 (dffn) and forward cache (gate).
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels
                .sdpa_bwd1
                .run_cached_direct(
                    &[&kernels.bufs.sdpa_bwd1_in],
                    &[&kernels.bufs.sdpa_bwd1_out],
                )
                .expect("ANE eval failed");
        });
        accumulate_dw(
            &cache.attn_out,
            q_dim,
            &ws.dx2_scaled,
            dim,
            seq,
            &mut grads.dwo,
        );
        accumulate_dw(&ws.dffn, dim, &cache.gate, hidden, seq, &mut grads.dw2);
        ane_handle.join().expect("ANE thread panicked");
    });

    let score_ch = heads * seq;
    {
        let locked = kernels.bufs.sdpa_bwd1_out.as_f32_slice();
        read_channels_into(&locked, bwd1_out_ch, seq, 0, q_dim, &mut ws.dv_full);
        read_channels_into(
            &locked,
            bwd1_out_ch,
            seq,
            q_dim,
            score_ch,
            &mut ws.probs_flat,
        );
        read_channels_into(
            &locked,
            bwd1_out_ch,
            seq,
            q_dim + score_ch,
            score_ch,
            &mut ws.dp_flat,
        );
    }

    // 10. sdpaBwd2
    let bwd2_in_ch = sdpa_bwd::bwd2_input_channels(cfg);
    let bwd2_out_ch = sdpa_bwd::bwd2_output_channels(cfg);
    {
        let mut locked = kernels.bufs.sdpa_bwd2_in.as_f32_slice_mut();
        let buf = &mut *locked;
        pack_channels(buf, bwd2_in_ch, seq, &ws.probs_flat, score_ch, 0);
        pack_channels(buf, bwd2_in_ch, seq, &ws.dp_flat, score_ch, score_ch);
        pack_channels(buf, bwd2_in_ch, seq, &cache.q_rope, q_dim, 2 * score_ch);
        pack_channels(
            buf,
            bwd2_in_ch,
            seq,
            &cache.k_rope,
            q_dim,
            2 * score_ch + q_dim,
        );
    }
    // ASYNC: ANE sdpaBwd2 || pre-compute wqt+wkt+wvt + dW1 (moved from step 5 to rebalance)
    // Step 5 was CPU-bound (~1.44ms for dW1+dW3+wot vs ANE ~0.5ms). Moving dW1 here
    // reduces step 5 CPU to ~0.91ms. sdpaBwd2 ANE time absorbs the extra dW1 sgemm.
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels
                .sdpa_bwd2
                .run_cached_direct(
                    &[&kernels.bufs.sdpa_bwd2_in],
                    &[&kernels.bufs.sdpa_bwd2_out],
                )
                .expect("ANE eval failed");
        });
        vdsp::mtrans(&weights.wq, q_dim, &mut ws.wqt, dim, dim, q_dim);
        vdsp::mtrans(&weights.wk, kv_dim, &mut ws.wkt, dim, dim, kv_dim);
        vdsp::mtrans(&weights.wv, kv_dim, &mut ws.wvt, dim, dim, kv_dim);
        accumulate_dw(&cache.x2norm, dim, &ws.dh1, hidden, seq, &mut grads.dw1);
        ane_handle.join().expect("ANE thread panicked");
    });
    {
        let locked = kernels.bufs.sdpa_bwd2_out.as_f32_slice();
        read_channels_into(&locked, bwd2_out_ch, seq, 0, q_dim, &mut ws.dq);
        read_channels_into(&locked, bwd2_out_ch, seq, q_dim, q_dim, &mut ws.dk);
    }

    // 11. RoPE backward
    rope_backward_inplace(&mut ws.dq, heads, hd, seq, &kernels.rope);
    rope_backward_inplace(&mut ws.dk, heads, hd, seq, &kernels.rope);

    // 11.5. Stage kvBwd early (wkt, wvt from step 10 overlap, dk post-RoPE)
    let kv_bwd_sp = dyn_matmul::dual_spatial_width(seq, dim);
    {
        let mut locked = kernels.bufs.kv_bwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        for c in 0..kv_dim {
            let row = c * kv_bwd_sp;
            buf[row..row + seq].copy_from_slice(&ws.dk[c * seq..c * seq + seq]);
            buf[row + seq..row + 2 * seq].copy_from_slice(&ws.dv_full[c * seq..c * seq + seq]);
            buf[row + 2 * seq..row + 2 * seq + dim]
                .copy_from_slice(&ws.wkt[c * dim..c * dim + dim]);
            buf[row + 2 * seq + dim..row + 2 * seq + 2 * dim]
                .copy_from_slice(&ws.wvt[c * dim..c * dim + dim]);
        }
    }

    // 12. Stage qBwd (wqt already computed in step 10 overlap)
    let q_bwd_sp = dyn_matmul::spatial_width(seq, dim);
    {
        let mut locked = kernels.bufs.q_bwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, q_dim, q_bwd_sp, &ws.dq, seq, 0);
        stage_spatial(buf, q_dim, q_bwd_sp, &ws.wqt, dim, seq);
    }

    // 13. ASYNC: ANE qBwd+kvBwd || CPU dWq+dWk+dWv
    // kvBwd is already staged (step 11.5), so both kernels run back-to-back on ANE
    // while the main thread computes weight gradients. This overlaps kvBwd ANE time
    // (~0.5ms) with CPU dW time instead of running kvBwd fully sequentially.
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels
                .q_bwd
                .run_cached_direct(&[&kernels.bufs.q_bwd_in], &[&kernels.bufs.q_bwd_out])
                .expect("ANE eval failed");
            kernels
                .kv_bwd
                .run_cached_direct(&[&kernels.bufs.kv_bwd_in], &[&kernels.bufs.kv_bwd_out])
                .expect("ANE eval failed");
        });
        accumulate_dw(&cache.xnorm, dim, &ws.dq, q_dim, seq, &mut grads.dwq);
        accumulate_dw(&cache.xnorm, dim, &ws.dk, kv_dim, seq, &mut grads.dwk);
        accumulate_dw(&cache.xnorm, dim, &ws.dv_full, kv_dim, seq, &mut grads.dwv);
        ane_handle.join().expect("ANE thread panicked");
    });
    {
        let locked = kernels.bufs.q_bwd_out.as_f32_slice();
        ws.dx_attn.copy_from_slice(&locked[..dim * seq]);
    }
    {
        let locked = kernels.bufs.kv_bwd_out.as_f32_slice();
        ws.dx_kv.copy_from_slice(&locked[..dim * seq]);
    }

    // 15. Merge + RMSNorm1 backward
    vdsp::vadd(&ws.dx_attn, &ws.dx_kv, &mut ws.dx_merged);
    rmsnorm::backward_channel_first(
        &ws.dx_merged,
        &cache.x,
        &weights.gamma1,
        &cache.rms_inv1,
        &mut ws.dx_rms1,
        &mut grads.dgamma1,
        dim,
        seq,
        &mut ws.rms_dot_buf,
    );

    // 16. Final dx into pre-allocated buffer
    vdsp::vadd(&ws.dx_rms1, &ws.dx2, dx_out);
}

// ── CPU helpers ──

/// Accumulate weight gradient via BLAS: dW[a_ch, b_ch] += A[a_ch, seq] @ B[b_ch, seq]^T
/// `a` is [a_ch * seq] row-major, `b` is [b_ch * seq] row-major, `dw` is [a_ch * b_ch].
fn accumulate_dw(a: &[f32], a_ch: usize, b: &[f32], b_ch: usize, seq: usize, dw: &mut [f32]) {
    vdsp::sgemm_at(a, a_ch, seq, b, b_ch, dw);
}

/// Pack activation data into the channel dimension of an IOSurface.
/// Uses copy_from_slice for vectorized memcpy on inner dimension.
fn pack_channels(
    dst: &mut [f32],
    _total_ch: usize,
    seq: usize,
    src: &[f32],
    src_ch: usize,
    ch_offset: usize,
) {
    for c in 0..src_ch {
        let d = (ch_offset + c) * seq;
        let s = c * seq;
        dst[d..d + seq].copy_from_slice(&src[s..s + seq]);
    }
}

/// RoPE backward: inverse rotation applied in-place.
/// Uses cached cos/sin tables from CompiledKernels (computed once at init).
fn rope_backward_inplace(dx: &mut [f32], heads: usize, hd: usize, seq: usize, rope: &RopeTable) {
    let pairs = hd / 2;
    for h in 0..heads {
        for i in 0..pairs {
            let base0 = (h * hd + 2 * i) * seq;
            let base1 = (h * hd + 2 * i + 1) * seq;
            let tbase = i * seq;
            for p in 0..seq {
                let c = rope.cos[tbase + p];
                let s = rope.sin[tbase + p];
                let d0 = dx[base0 + p];
                let d1 = dx[base1 + p];
                dx[base0 + p] = c * d0 + s * d1;
                dx[base1 + p] = -s * d0 + c * d1;
            }
        }
    }
}
