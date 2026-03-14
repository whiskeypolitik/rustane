//! Full transformer model: embedding → NL layers → final norm → logits → loss.
//!
//! gpt_karpathy: 6 layers, 48.8M params, SwiGLU, MHA, RoPE.
//! Tied embedding weights (embedding table = output projection transposed).

use crate::cpu::{rmsnorm, cross_entropy, embedding, vdsp};
use crate::layer::{self, CompiledKernels, LayerWeights, LayerGrads, ForwardCache, BackwardWorkspace};
use crate::model::ModelConfig;
use crate::training::LayerOptState;
use crate::metal_adam::MetalAdam;

/// Full model weights.
pub struct ModelWeights {
    pub embed: Vec<f32>,       // [VOCAB * DIM]
    pub layers: Vec<LayerWeights>,
    pub gamma_final: Vec<f32>, // [DIM]
}

/// Full model gradients.
pub struct ModelGrads {
    pub dembed: Vec<f32>,      // [VOCAB * DIM]
    pub layers: Vec<LayerGrads>,
    pub dgamma_final: Vec<f32>,
}

/// Full model optimizer state.
pub struct ModelOptState {
    pub embed_m: Vec<f32>, pub embed_v: Vec<f32>,
    pub layers: Vec<LayerOptState>,
    pub gamma_final_m: Vec<f32>, pub gamma_final_v: Vec<f32>,
}

/// Training hyperparameters (from Obj-C reference).
pub struct TrainConfig {
    pub max_lr: f32,
    pub min_lr_frac: f32,
    pub warmup_steps: u32,
    pub total_steps: u32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub matrix_lr_scale: f32,
    pub embed_lr_scale: f32,
    pub loss_scale: f32,
    pub accum_steps: u32,
    pub grad_clip: f32,
    pub softcap: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            max_lr: 3e-4,
            min_lr_frac: 0.1,
            warmup_steps: 100,
            total_steps: 10000,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.1,
            matrix_lr_scale: 0.05,
            embed_lr_scale: 5.0,
            loss_scale: 256.0,
            accum_steps: 10,
            grad_clip: 1.0,
            softcap: 15.0,
        }
    }
}

/// Everything needed from forward pass for backward.
pub struct ForwardResult {
    pub loss: f32,
    pub caches: Vec<ForwardCache>,
    pub dlogits: Vec<f32>,        // [SEQ * VOCAB]
    pub x_final: Vec<f32>,        // [DIM * SEQ] channel-first, post-norm
    pub x_prenorm: Vec<f32>,      // [DIM * SEQ] channel-first, pre-final-norm
    pub rms_inv_final: Vec<f32>,  // [SEQ]
    pub logits_capped: Vec<f32>,  // [SEQ * VOCAB] post-softcap (empty if no softcap)
}

impl ModelWeights {
    pub fn random(cfg: &ModelConfig) -> Self {
        Self {
            embed: random_vec(cfg.vocab * cfg.dim, 0.005), // Obj-C E50 value
            layers: (0..cfg.nlayers).map(|_| LayerWeights::random(cfg)).collect(),
            gamma_final: vec![1.0; cfg.dim],
        }
    }
}

impl ModelGrads {
    pub fn zeros(cfg: &ModelConfig) -> Self {
        Self {
            dembed: vec![0.0; cfg.vocab * cfg.dim],
            layers: (0..cfg.nlayers).map(|_| LayerGrads::zeros(cfg)).collect(),
            dgamma_final: vec![0.0; cfg.dim],
        }
    }

    pub fn zero_out(&mut self) {
        self.dembed.fill(0.0);
        self.dgamma_final.fill(0.0);
        for lg in &mut self.layers {
            lg.zero_out();
        }
    }
}

impl ModelOptState {
    pub fn zeros(cfg: &ModelConfig) -> Self {
        Self {
            embed_m: vec![0.0; cfg.vocab * cfg.dim],
            embed_v: vec![0.0; cfg.vocab * cfg.dim],
            layers: (0..cfg.nlayers).map(|_| LayerOptState::zeros(cfg)).collect(),
            gamma_final_m: vec![0.0; cfg.dim],
            gamma_final_v: vec![0.0; cfg.dim],
        }
    }
}

/// Pre-allocated workspace for full model backward pass.
/// Eliminates ~1.2 GB of per-call allocations (dl, dx_final, dy, rms_dot_buf, layer workspace).
pub struct ModelBackwardWorkspace {
    pub dl: Vec<f32>,            // [SEQ * VOCAB]
    pub dx_final: Vec<f32>,      // [DIM * SEQ]
    pub dy: Vec<f32>,            // [DIM * SEQ]
    pub dy_buf: Vec<f32>,        // [DIM * SEQ] swap buffer for backward_into
    pub rms_dot_buf: Vec<f32>,   // [SEQ]
    pub layer_ws: BackwardWorkspace,
}

impl ModelBackwardWorkspace {
    pub fn new(cfg: &ModelConfig) -> Self {
        Self {
            dl: vec![0.0; cfg.seq * cfg.vocab],
            dx_final: vec![0.0; cfg.dim * cfg.seq],
            dy: vec![0.0; cfg.dim * cfg.seq],
            dy_buf: vec![0.0; cfg.dim * cfg.seq],
            rms_dot_buf: vec![0.0; cfg.seq],
            layer_ws: BackwardWorkspace::new(cfg),
        }
    }
}

/// Pre-allocated workspace for full model forward pass.
/// Eliminates ~144MB of per-step alloc/zero/free for layer caches +
/// ~37MB for model-level buffers (logits, dlogits, etc.).
pub struct ModelForwardWorkspace {
    pub caches: Vec<ForwardCache>,
    pub x_row: Vec<f32>,        // [SEQ * DIM]
    pub x_buf: Vec<f32>,        // [DIM * SEQ] current layer input
    pub x_next_buf: Vec<f32>,   // [DIM * SEQ] current layer output
    pub x_final: Vec<f32>,      // [DIM * SEQ] post final rmsnorm
    pub x_prenorm: Vec<f32>,    // [DIM * SEQ] pre final rmsnorm (= last layer output)
    pub rms_inv_final: Vec<f32>,// [SEQ]
    pub x_final_row: Vec<f32>,  // [SEQ * DIM]
    pub logits: Vec<f32>,       // [SEQ * VOCAB]
    pub logits_capped: Vec<f32>,// [SEQ * VOCAB] (for softcap backward)
    pub dlogits: Vec<f32>,      // [SEQ * VOCAB]
}

impl ModelForwardWorkspace {
    pub fn new(cfg: &ModelConfig) -> Self {
        let dim = cfg.dim;
        let seq = cfg.seq;
        let vocab = cfg.vocab;
        Self {
            caches: (0..cfg.nlayers).map(|_| ForwardCache::new(cfg)).collect(),
            x_row: vec![0.0; seq * dim],
            x_buf: vec![0.0; dim * seq],
            x_next_buf: vec![0.0; dim * seq],
            x_final: vec![0.0; dim * seq],
            x_prenorm: vec![0.0; dim * seq],
            rms_inv_final: vec![0.0; seq],
            x_final_row: vec![0.0; seq * dim],
            logits: vec![0.0; seq * vocab],
            logits_capped: vec![0.0; seq * vocab],
            dlogits: vec![0.0; seq * vocab],
        }
    }
}

/// Forward pass using pre-allocated workspace (zero heap allocations in steady state).
/// Returns loss. All intermediate data lives in `ws` for backward_ws to read.
pub fn forward_ws(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &ModelWeights,
    tokens: &[u32],
    targets: &[u32],
    softcap: f32,
    ws: &mut ModelForwardWorkspace,
) -> f32 {
    let dim = cfg.dim;
    let seq = cfg.seq;
    let vocab = cfg.vocab;

    // 1. Embedding lookup → x_buf [DIM, SEQ]
    embedding::forward(&weights.embed, dim, tokens, &mut ws.x_row);
    vdsp::mtrans(&ws.x_row, dim, &mut ws.x_buf, seq, seq, dim);

    // 2. Forward through NL layers (pre-allocated caches, zero allocs)
    for l in 0..cfg.nlayers {
        layer::forward_into(cfg, kernels, &weights.layers[l], &ws.x_buf, &mut ws.caches[l], &mut ws.x_next_buf);
        std::mem::swap(&mut ws.x_buf, &mut ws.x_next_buf);
    }

    // 3. x_buf now holds the last layer's output = x_prenorm
    ws.x_prenorm.copy_from_slice(&ws.x_buf);

    // 4. Final RMSNorm
    rmsnorm::forward_channel_first(&ws.x_prenorm, &weights.gamma_final, &mut ws.x_final, &mut ws.rms_inv_final, dim, seq);

    // 5. Logits: x_final^T @ embed^T → [SEQ, VOCAB]
    vdsp::mtrans(&ws.x_final, seq, &mut ws.x_final_row, dim, dim, seq);
    vdsp::sgemm_at(&ws.x_final_row, seq, dim, &weights.embed, vocab, &mut ws.logits);

    // 6. Softcap: logits = softcap * tanh(logits / softcap)
    //    Store unscaled tanh in logits_capped for backward (avoids extra pass + simplifies derivative).
    let has_softcap = softcap > 0.0;
    if has_softcap {
        vdsp::sscal(&mut ws.logits, 1.0 / softcap);                      // pass 1: logits /= softcap
        vdsp::tanhf(&ws.logits, &mut ws.logits_capped);                   // pass 2: logits_capped = tanh(logits/softcap)
        vdsp::vsmul(&ws.logits_capped, softcap, &mut ws.logits);          // pass 3: logits = softcap * tanh(...)
    }

    // 7. Cross-entropy
    let total_loss = cross_entropy::forward_backward_batch(
        &ws.logits, targets, vocab, &mut ws.dlogits, 1.0 / seq as f32,
    );

    total_loss / seq as f32
}

/// Backward pass using pre-allocated workspaces (reads from fwd_ws, writes grads).
pub fn backward_ws(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &ModelWeights,
    fwd_ws: &ModelForwardWorkspace,
    tokens: &[u32],
    softcap: f32,
    loss_scale: f32,
    grads: &mut ModelGrads,
    bwd_ws: &mut ModelBackwardWorkspace,
) {
    let dim = cfg.dim;
    let seq = cfg.seq;
    let vocab = cfg.vocab;

    // 1+2. Fused scale + softcap backward
    //    logits_capped stores tanh(logits/softcap) — unscaled, so sech^2 = 1 - t^2
    if softcap > 0.0 {
        let dl = &mut bwd_ws.dl;
        let dlog = &fwd_ws.dlogits;
        let t = &fwd_ws.logits_capped; // tanh(logits/softcap), NOT scaled by softcap
        for i in 0..dl.len() {
            dl[i] = dlog[i] * loss_scale * (1.0 - t[i] * t[i]);
        }
    } else {
        vdsp::vsmul(&fwd_ws.dlogits, loss_scale, &mut bwd_ws.dl);
    }

    // 3a. dx_final = embed^T @ dl (needed for RMSNorm backward, sequential)
    unsafe {
        vdsp::cblas_sgemm(
            101, 112, 112,
            dim as i32, seq as i32, vocab as i32,
            1.0,
            weights.embed.as_ptr(), dim as i32,
            bwd_ws.dl.as_ptr(), vocab as i32,
            0.0,
            bwd_ws.dx_final.as_mut_ptr(), seq as i32,
        );
    }

    // 3b+4+5. ASYNC: dembed sgemm overlaps with RMSNorm backward + layer backward.
    // Safety: dembed is not accessed by rmsnorm backward (writes dy, dgamma_final)
    // or layer backward (writes grads.layers[l]) — disjoint struct fields.
    // Using usize casts for Send safety across thread boundary.
    let dl_addr = bwd_ws.dl.as_ptr() as usize;
    let xf_addr = fwd_ws.x_final.as_ptr() as usize;
    let de_addr = grads.dembed.as_mut_ptr() as usize;
    let (sv, sd, ss) = (vocab as i32, dim as i32, seq as i32);
    std::thread::scope(|s| {
        let sgemm_handle = s.spawn(move || {
            unsafe {
                vdsp::cblas_sgemm(
                    101, 112, 112,
                    sv, sd, ss,
                    1.0,
                    dl_addr as *const f32, sv,
                    xf_addr as *const f32, ss,
                    1.0,
                    de_addr as *mut f32, sd,
                );
            }
        });

        // 4. Final RMSNorm backward (main thread)
        rmsnorm::backward_channel_first(
            &bwd_ws.dx_final, &fwd_ws.x_prenorm, &weights.gamma_final, &fwd_ws.rms_inv_final,
            &mut bwd_ws.dy, &mut grads.dgamma_final, dim, seq, &mut bwd_ws.rms_dot_buf,
        );

        // 5. Backward through NL layers (main thread)
        for l in (0..cfg.nlayers).rev() {
            layer::backward_into(cfg, kernels, &weights.layers[l], &fwd_ws.caches[l], &bwd_ws.dy, &mut grads.layers[l], &mut bwd_ws.layer_ws, &mut bwd_ws.dy_buf);
            std::mem::swap(&mut bwd_ws.dy, &mut bwd_ws.dy_buf);
        }

        sgemm_handle.join().expect("dembed sgemm thread panicked");
    });

    // 6. Embedding backward (after dembed sgemm completes)
    embedding::backward_channel_first(&bwd_ws.dy, dim, tokens, &mut grads.dembed);
}

/// Cosine LR schedule with linear warmup.
pub fn learning_rate(step: u32, tc: &TrainConfig) -> f32 {
    if step < tc.warmup_steps {
        tc.max_lr * (step + 1) as f32 / tc.warmup_steps as f32
    } else {
        let decay = (step - tc.warmup_steps) as f32 / (tc.total_steps - tc.warmup_steps) as f32;
        let min_lr = tc.max_lr * tc.min_lr_frac;
        min_lr + 0.5 * (1.0 + (std::f32::consts::PI * decay).cos()) * (tc.max_lr - min_lr)
    }
}

/// Full forward pass: embedding → NL layers → final norm → logits → loss.
pub fn forward(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &ModelWeights,
    tokens: &[u32],   // [SEQ] input token IDs
    targets: &[u32],  // [SEQ] target token IDs
    softcap: f32,
) -> ForwardResult {
    let dim = cfg.dim;
    let seq = cfg.seq;
    let vocab = cfg.vocab;

    // 1. Embedding lookup → x [DIM, SEQ] (channel-first)
    let mut x_row = vec![0.0f32; seq * dim]; // [SEQ, DIM] row-major
    embedding::forward(&weights.embed, dim, tokens, &mut x_row);
    // Transpose to channel-first [DIM, SEQ] via vDSP_mtrans
    let mut x = vec![0.0f32; dim * seq];
    vdsp::mtrans(&x_row, dim, &mut x, seq, seq, dim);

    // 2. Forward through NL layers
    let mut caches = Vec::with_capacity(cfg.nlayers);
    for l in 0..cfg.nlayers {
        let (x_next, cache) = layer::forward(cfg, kernels, &weights.layers[l], &x);
        caches.push(cache);
        x = x_next;
    }

    // 3. Final RMSNorm (CPU) — channel-first, no transpose
    let x_prenorm = x;
    let mut x_final = vec![0.0f32; dim * seq];
    let mut rms_inv_final = vec![0.0f32; seq];
    rmsnorm::forward_channel_first(&x_prenorm, &weights.gamma_final, &mut x_final, &mut rms_inv_final, dim, seq);

    // 4. Logits: x_final^T @ embed^T → [SEQ, VOCAB]
    let mut x_final_row = vec![0.0f32; seq * dim];
    vdsp::mtrans(&x_final, seq, &mut x_final_row, dim, dim, seq);
    let mut logits = vec![0.0f32; seq * vocab];
    vdsp::sgemm_at(&x_final_row, seq, dim, &weights.embed, vocab, &mut logits);

    // 5. Logit softcap: logits = softcap * tanh(logits / softcap)
    //    Store unscaled tanh in logits_capped for backward (sech^2 = 1 - t^2).
    let has_softcap = softcap > 0.0;
    let mut logits_capped = Vec::new();
    if has_softcap {
        logits_capped = vec![0.0f32; seq * vocab];
        vdsp::sscal(&mut logits, 1.0 / softcap);
        vdsp::tanhf(&logits, &mut logits_capped);  // logits_capped = tanh(logits/softcap)
        vdsp::vsmul(&logits_capped, softcap, &mut logits); // logits = softcap * tanh(...)
    }

    // 6. Cross-entropy loss (batched — single alloc for all positions)
    let mut dlogits = vec![0.0f32; seq * vocab];
    let total_loss = cross_entropy::forward_backward_batch(
        &logits, targets, vocab, &mut dlogits, 1.0 / seq as f32,
    );

    ForwardResult {
        loss: total_loss / seq as f32,
        caches,
        dlogits,
        x_final,
        x_prenorm,
        rms_inv_final,
        logits_capped,
    }
}

/// Full backward pass: logits grad → softcap → final norm → NL layers (reverse) → embedding.
/// Takes a pre-allocated workspace to avoid per-call allocations (~1.2 GB churn).
pub fn backward(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &ModelWeights,
    fwd: &ForwardResult,
    tokens: &[u32],        // input token IDs (for embedding backward)
    softcap: f32,
    loss_scale: f32,
    grads: &mut ModelGrads,
    ws: &mut ModelBackwardWorkspace,
) {
    let dim = cfg.dim;
    let seq = cfg.seq;
    let vocab = cfg.vocab;

    // 1+2. Fused scale + softcap backward
    //    logits_capped stores tanh(logits/softcap) — unscaled, so sech^2 = 1 - t^2
    if softcap > 0.0 && !fwd.logits_capped.is_empty() {
        let dl = &mut ws.dl;
        let dlog = &fwd.dlogits;
        let t = &fwd.logits_capped;
        for i in 0..dl.len() {
            dl[i] = dlog[i] * loss_scale * (1.0 - t[i] * t[i]);
        }
    } else {
        vdsp::vsmul(&fwd.dlogits, loss_scale, &mut ws.dl);
    }

    // 3. Output projection gradients (tied embedding weights)
    //    dembed += dl^T @ x_final^T: use BLAS trans flags, no mtrans needed
    unsafe {
        vdsp::cblas_sgemm(
            101, 112, 112, // row-major, transA(dl), transB(x_final)
            vocab as i32, dim as i32, seq as i32,
            1.0,
            ws.dl.as_ptr(), vocab as i32,
            fwd.x_final.as_ptr(), seq as i32,
            1.0,
            grads.dembed.as_mut_ptr(), dim as i32,
        );
    }
    //    dx_final = embed^T @ dl^T → directly in [dim,seq] channel-first, no mtrans
    unsafe {
        vdsp::cblas_sgemm(
            101, 112, 112,
            dim as i32, seq as i32, vocab as i32,
            1.0,
            weights.embed.as_ptr(), dim as i32,
            ws.dl.as_ptr(), vocab as i32,
            0.0,
            ws.dx_final.as_mut_ptr(), seq as i32,
        );
    }

    // 4. Final RMSNorm backward (CPU) — channel-first, no transpose
    rmsnorm::backward_channel_first(
        &ws.dx_final, &fwd.x_prenorm, &weights.gamma_final, &fwd.rms_inv_final,
        &mut ws.dy, &mut grads.dgamma_final, dim, seq, &mut ws.rms_dot_buf,
    );

    // 5. Backward through NL layers (reverse order, pre-allocated workspace)
    for l in (0..cfg.nlayers).rev() {
        // layer::backward returns dx into a new Vec; copy into ws.dy for next layer
        let dx = layer::backward(cfg, kernels, &weights.layers[l], &fwd.caches[l], &ws.dy, &mut grads.layers[l], &mut ws.layer_ws);
        ws.dy = dx;
    }

    // 6. Input embedding backward: channel-first scatter-add, no mtrans
    embedding::backward_channel_first(&ws.dy, dim, tokens, &mut grads.dembed);
}

/// Apply Adam to all model weights with split learning rates.
/// Batches all 56 parameter updates into a single Metal GPU command buffer.
/// `grad_scale` is applied to gradients inline on GPU (fused descale+clip).
/// Pass 1.0 for no scaling, or the combined descale/clip factor.
pub fn update_weights(
    cfg: &ModelConfig,
    weights: &mut ModelWeights,
    grads: &ModelGrads,
    opt: &mut ModelOptState,
    t: u32,
    lr: f32,
    tc: &TrainConfig,
    metal_adam: &MetalAdam,
    grad_scale: f32,
) {
    let wd = tc.weight_decay;
    let matrix_lr = lr * tc.matrix_lr_scale;
    let embed_lr = lr * tc.embed_lr_scale;

    let (b1, b2, eps) = (tc.beta1, tc.beta2, tc.eps);
    // begin_batch pre-creates 6 shared scalar buffers (beta1,beta2,eps,bc1,bc2,grad_scale).
    // add() only allocates 2 per-tensor scalars (lr, wd) — down from 8 per call.
    let mut batch = metal_adam.begin_batch(t, b1, b2, eps, grad_scale);

    // Embedding (no weight decay)
    batch.add(&mut weights.embed, &grads.dembed, &mut opt.embed_m, &mut opt.embed_v, embed_lr, 0.0);

    // Final RMSNorm (no weight decay)
    batch.add(&mut weights.gamma_final, &grads.dgamma_final, &mut opt.gamma_final_m, &mut opt.gamma_final_v, lr, 0.0);

    // Per-layer
    for l in 0..cfg.nlayers {
        let w = &mut weights.layers[l];
        let g = &grads.layers[l];
        let o = &mut opt.layers[l];

        // Weight matrices
        batch.add(&mut w.wq, &g.dwq, &mut o.m_wq, &mut o.v_wq, matrix_lr, wd);
        batch.add(&mut w.wk, &g.dwk, &mut o.m_wk, &mut o.v_wk, matrix_lr, wd);
        batch.add(&mut w.wv, &g.dwv, &mut o.m_wv, &mut o.v_wv, matrix_lr, wd);
        batch.add(&mut w.wo, &g.dwo, &mut o.m_wo, &mut o.v_wo, matrix_lr, wd);
        batch.add(&mut w.w1, &g.dw1, &mut o.m_w1, &mut o.v_w1, matrix_lr, wd);
        batch.add(&mut w.w3, &g.dw3, &mut o.m_w3, &mut o.v_w3, matrix_lr, wd);
        batch.add(&mut w.w2, &g.dw2, &mut o.m_w2, &mut o.v_w2, matrix_lr, wd);

        // RMSNorm scales (no weight decay)
        batch.add(&mut w.gamma1, &g.dgamma1, &mut o.m_gamma1, &mut o.v_gamma1, lr, 0.0);
        batch.add(&mut w.gamma2, &g.dgamma2, &mut o.m_gamma2, &mut o.v_gamma2, lr, 0.0);
    }

    batch.execute();
}


/// Global gradient L2 norm (uses vDSP_svesq — single call per tensor, no scratch).
pub fn grad_norm(grads: &ModelGrads) -> f32 {
    let mut sum = 0.0f32;
    sum += vdsp::svesq(&grads.dembed);
    sum += vdsp::svesq(&grads.dgamma_final);
    for lg in &grads.layers {
        for g in [&lg.dwq, &lg.dwk, &lg.dwv, &lg.dwo, &lg.dw1, &lg.dw3, &lg.dw2, &lg.dgamma1, &lg.dgamma2] {
            sum += vdsp::svesq(g);
        }
    }
    sum.sqrt()
}

/// Clip all gradients by global L2 norm (in-place cblas_sscal, no scratch allocs).
pub fn clip_grads(grads: &mut ModelGrads, max_norm: f32) {
    let norm = grad_norm(grads);
    if norm > max_norm {
        let scale = max_norm / norm;
        vdsp::sscal(&mut grads.dembed, scale);
        vdsp::sscal(&mut grads.dgamma_final, scale);
        for lg in &mut grads.layers {
            vdsp::sscal(&mut lg.dwq, scale);
            vdsp::sscal(&mut lg.dwk, scale);
            vdsp::sscal(&mut lg.dwv, scale);
            vdsp::sscal(&mut lg.dwo, scale);
            vdsp::sscal(&mut lg.dw1, scale);
            vdsp::sscal(&mut lg.dw3, scale);
            vdsp::sscal(&mut lg.dw2, scale);
            vdsp::sscal(&mut lg.dgamma1, scale);
            vdsp::sscal(&mut lg.dgamma2, scale);
        }
    }
}

/// Simple LCG pseudo-random (same as layer.rs).
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

/// Run one full training step with gradient accumulation.
/// Returns average loss across the accumulation microbatches.
pub fn train_step(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &mut ModelWeights,
    grads: &mut ModelGrads,
    opt: &mut ModelOptState,
    data: &[u16],          // all training tokens
    step: u32,
    tc: &TrainConfig,
    metal_adam: &MetalAdam,
    fwd_ws: &mut ModelForwardWorkspace,
    bwd_ws: &mut ModelBackwardWorkspace,
) -> f32 {
    let seq = cfg.seq;
    grads.zero_out();

    let mut total_loss = 0.0f32;
    let max_pos = data.len() - seq - 1;

    for _micro in 0..tc.accum_steps {
        // Random position (simple LCG)
        let pos = ((step as u64 * 7919 + _micro as u64 * 104729) % max_pos as u64) as usize;
        let input_tokens: Vec<u32> = data[pos..pos + seq].iter().map(|&t| t as u32).collect();
        let target_tokens: Vec<u32> = data[pos + 1..pos + seq + 1].iter().map(|&t| t as u32).collect();

        let loss = forward_ws(
            cfg, kernels, weights, &input_tokens, &target_tokens, tc.softcap, fwd_ws,
        );
        total_loss += loss;

        backward_ws(
            cfg, kernels, weights, fwd_ws, &input_tokens, tc.softcap, tc.loss_scale, grads, bwd_ws,
        );
    }

    // Compute grad norm for clip decision (single read pass over ~168MB)
    let gsc = 1.0 / (tc.accum_steps as f32 * tc.loss_scale);
    let raw_norm = grad_norm(grads);
    let scaled_norm = raw_norm * gsc;
    let combined_scale = if scaled_norm > tc.grad_clip {
        tc.grad_clip / raw_norm
    } else {
        gsc
    };

    // LR schedule
    let lr = learning_rate(step, tc);

    // Weight update with fused grad scaling (GPU applies grad * combined_scale inline)
    // Eliminates separate CPU sscal pass over ~168MB
    update_weights(cfg, weights, grads, opt, step + 1, lr, tc, metal_adam, combined_scale);

    total_loss / tc.accum_steps as f32
}

/// Forward-only pass returning per-token losses (for val_bpb computation).
pub fn forward_losses(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &ModelWeights,
    tokens: &[u32],
    targets: &[u32],
    softcap: f32,
) -> Vec<f32> {
    let dim = cfg.dim;
    let seq = cfg.seq;
    let vocab = cfg.vocab;

    // Embedding
    let mut x_row = vec![0.0f32; seq * dim];
    embedding::forward(&weights.embed, dim, tokens, &mut x_row);
    let mut x = vec![0.0f32; dim * seq];
    vdsp::mtrans(&x_row, dim, &mut x, seq, seq, dim);

    // Layers
    for l in 0..cfg.nlayers {
        let (x_next, _) = layer::forward(cfg, kernels, &weights.layers[l], &x);
        x = x_next;
    }

    // Final RMSNorm — channel-first, no transpose
    let mut x_final = vec![0.0f32; dim * seq];
    let mut rms_inv = vec![0.0f32; seq];
    rmsnorm::forward_channel_first(&x, &weights.gamma_final, &mut x_final, &mut rms_inv, dim, seq);

    // Logits
    let mut x_final_row = vec![0.0f32; seq * dim];
    vdsp::mtrans(&x_final, seq, &mut x_final_row, dim, dim, seq);
    let mut logits = vec![0.0f32; seq * vocab];
    vdsp::sgemm_at(&x_final_row, seq, dim, &weights.embed, vocab, &mut logits);

    // Softcap
    if softcap > 0.0 {
        vdsp::sscal(&mut logits, 1.0 / softcap);
        vdsp::tanhf_inplace(&mut logits);
        vdsp::sscal(&mut logits, softcap);
    }

    // Per-token cross-entropy
    let mut losses = vec![0.0f32; seq];
    for s in 0..seq {
        let tok_logits = &logits[s * vocab..(s + 1) * vocab];
        let (loss, _) = cross_entropy::forward(tok_logits, targets[s] as usize);
        losses[s] = loss;
    }
    losses
}
