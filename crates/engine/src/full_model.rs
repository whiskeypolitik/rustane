//! Full transformer model: embedding → NL layers → final norm → logits → loss.
//!
//! gpt_karpathy: 6 layers, 48.8M params, SwiGLU, MHA, RoPE.
//! Tied embedding weights (embedding table = output projection transposed).

use crate::cpu::{rmsnorm, cross_entropy, embedding, vdsp};
use crate::layer::{self, CompiledKernels, LayerWeights, LayerGrads, ForwardCache};
use crate::model::ModelConfig;
use crate::training::LayerOptState;
use crate::cpu::adam::{self, AdamConfig};

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
    // Transpose to channel-first: x[c, s] = x_row[s * dim + c]
    let mut x = vec![0.0f32; dim * seq];
    for s in 0..seq {
        for c in 0..dim {
            x[c * seq + s] = x_row[s * dim + c];
        }
    }

    // 2. Forward through NL layers
    let mut caches = Vec::with_capacity(cfg.nlayers);
    for l in 0..cfg.nlayers {
        let (x_next, cache) = layer::forward(cfg, kernels, &weights.layers[l], &x);
        caches.push(cache);
        x = x_next;
    }

    // 3. Final RMSNorm (CPU) — save pre-norm x for backward
    let x_prenorm = x; // move (layer loop is done, x not needed after)
    let mut x_final = vec![0.0f32; dim * seq];
    let mut rms_inv_final = vec![0.0f32; seq];
    let mut x_pos = vec![0.0f32; dim];
    let mut out_pos = vec![0.0f32; dim];
    for s in 0..seq {
        for c in 0..dim { x_pos[c] = x_prenorm[c * seq + s]; }
        rms_inv_final[s] = rmsnorm::forward(&x_pos, &weights.gamma_final, &mut out_pos);
        for c in 0..dim { x_final[c * seq + s] = out_pos[c]; }
    }

    // 4. Logits: x_final^T @ embed^T → [SEQ, VOCAB]
    let mut x_final_row = vec![0.0f32; seq * dim];
    for s in 0..seq {
        for c in 0..dim { x_final_row[s * dim + c] = x_final[c * seq + s]; }
    }
    let mut logits = vec![0.0f32; seq * vocab];
    vdsp::sgemm_at(&x_final_row, seq, dim, &weights.embed, vocab, &mut logits);

    // 5. Logit softcap: logits = softcap * tanh(logits / softcap)
    let logits_capped = if softcap > 0.0 {
        let inv_cap = 1.0 / softcap;
        let mut scaled = vec![0.0f32; seq * vocab];
        vdsp::vsmul(&logits, inv_cap, &mut scaled);
        vdsp::tanhf(&scaled, &mut logits);
        let mut capped = vec![0.0f32; seq * vocab];
        vdsp::vsmul(&logits, softcap, &mut capped);
        logits.copy_from_slice(&capped);
        logits.clone() // save capped logits for softcap backward
    } else {
        Vec::new()
    };

    // 6. Cross-entropy loss (per-token, averaged)
    let mut total_loss = 0.0f32;
    let mut dlogits = vec![0.0f32; seq * vocab];
    for s in 0..seq {
        let tok_logits = &logits[s * vocab..(s + 1) * vocab];
        let (loss, log_sm) = cross_entropy::forward(tok_logits, targets[s] as usize);
        total_loss += loss;
        let mut d_tok = vec![0.0f32; vocab];
        cross_entropy::backward(&log_sm, targets[s] as usize, &mut d_tok);
        for v in 0..vocab {
            dlogits[s * vocab + v] = d_tok[v] / seq as f32;
        }
    }

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
pub fn backward(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &ModelWeights,
    fwd: &ForwardResult,
    tokens: &[u32],        // input token IDs (for embedding backward)
    softcap: f32,
    loss_scale: f32,
    grads: &mut ModelGrads,
) {
    let dim = cfg.dim;
    let seq = cfg.seq;
    let vocab = cfg.vocab;

    // 1. Scale dlogits by loss_scale
    let mut dl = vec![0.0f32; seq * vocab];
    vdsp::vsmul(&fwd.dlogits, loss_scale, &mut dl);

    // 2. Softcap backward: dl *= (1 - tanh²(raw/softcap))
    //    tanh(raw/softcap) = capped/softcap
    if softcap > 0.0 && !fwd.logits_capped.is_empty() {
        let inv_cap = 1.0 / softcap;
        for i in 0..dl.len() {
            let t = fwd.logits_capped[i] * inv_cap; // tanh value
            dl[i] *= 1.0 - t * t;
        }
    }

    // 3. Output projection gradients (tied embedding weights)
    //    dembed += dl^T @ x_final_row: [VOCAB, SEQ]^T @ [SEQ, DIM] → [VOCAB, DIM]
    //    dx_final_row = dl @ embed: [SEQ, VOCAB] @ [VOCAB, DIM] → [SEQ, DIM]
    let mut x_final_row = vec![0.0f32; seq * dim];
    for s in 0..seq {
        for c in 0..dim { x_final_row[s * dim + c] = fwd.x_final[c * seq + s]; }
    }
    // dembed += dl^T @ x_final_row (accumulate with beta=1.0)
    unsafe {
        vdsp::cblas_sgemm(
            101, 112, 111, // row-major, transA, no-transB
            vocab as i32, dim as i32, seq as i32,
            1.0,
            dl.as_ptr(), vocab as i32,
            x_final_row.as_ptr(), dim as i32,
            1.0, // accumulate into dembed
            grads.dembed.as_mut_ptr(), dim as i32,
        );
    }
    // dx_final_row = dl @ embed
    let mut dx_final_row = vec![0.0f32; seq * dim];
    unsafe {
        vdsp::cblas_sgemm(
            101, 111, 111,
            seq as i32, dim as i32, vocab as i32,
            1.0,
            dl.as_ptr(), vocab as i32,
            weights.embed.as_ptr(), dim as i32,
            0.0,
            dx_final_row.as_mut_ptr(), dim as i32,
        );
    }

    // Transpose dx_final to channel-first
    let mut dx_final = vec![0.0f32; dim * seq];
    for s in 0..seq {
        for c in 0..dim { dx_final[c * seq + s] = dx_final_row[s * dim + c]; }
    }

    // 4. Final RMSNorm backward (CPU)
    let mut dy = vec![0.0f32; dim * seq];
    let mut dx_pos_buf = vec![0.0f32; dim];
    let mut x_pos_buf = vec![0.0f32; dim];
    let mut dy_pos = vec![0.0f32; dim];
    for s in 0..seq {
        for c in 0..dim {
            dx_pos_buf[c] = dx_final[c * seq + s];
            x_pos_buf[c] = fwd.x_prenorm[c * seq + s];
        }
        rmsnorm::backward(
            &dx_pos_buf, &x_pos_buf, &weights.gamma_final,
            fwd.rms_inv_final[s], &mut dy_pos, &mut grads.dgamma_final,
        );
        for c in 0..dim { dy[c * seq + s] = dy_pos[c]; }
    }

    // 5. Backward through NL layers (reverse order)
    for l in (0..cfg.nlayers).rev() {
        dy = layer::backward(cfg, kernels, &weights.layers[l], &fwd.caches[l], &dy, &mut grads.layers[l]);
    }

    // 6. Input embedding backward: scatter-add dy into dembed
    let mut dy_row = vec![0.0f32; seq * dim];
    for s in 0..seq {
        for c in 0..dim { dy_row[s * dim + c] = dy[c * seq + s]; }
    }
    embedding::backward(&dy_row, dim, tokens, &mut grads.dembed);
}

/// Apply Adam to all model weights with split learning rates.
pub fn update_weights(
    cfg: &ModelConfig,
    weights: &mut ModelWeights,
    grads: &ModelGrads,
    opt: &mut ModelOptState,
    t: u32,
    lr: f32,
    tc: &TrainConfig,
) {
    let matrix_cfg = AdamConfig {
        lr: lr * tc.matrix_lr_scale,
        beta1: tc.beta1, beta2: tc.beta2, eps: tc.eps,
        weight_decay: tc.weight_decay,
    };
    let norm_cfg = AdamConfig {
        lr,
        beta1: tc.beta1, beta2: tc.beta2, eps: tc.eps,
        weight_decay: 0.0, // no weight decay on norms
    };
    let embed_cfg = AdamConfig {
        lr: lr * tc.embed_lr_scale,
        beta1: tc.beta1, beta2: tc.beta2, eps: tc.eps,
        weight_decay: 0.0, // no weight decay on embeddings
    };

    // Embedding
    adam::step(&mut weights.embed, &grads.dembed, &mut opt.embed_m, &mut opt.embed_v, t, &embed_cfg);

    // Final RMSNorm
    adam::step(&mut weights.gamma_final, &grads.dgamma_final, &mut opt.gamma_final_m, &mut opt.gamma_final_v, t, &norm_cfg);

    // Per-layer
    for l in 0..cfg.nlayers {
        let w = &mut weights.layers[l];
        let g = &grads.layers[l];
        let o = &mut opt.layers[l];

        // Weight matrices
        adam::step(&mut w.wq, &g.dwq, &mut o.m_wq, &mut o.v_wq, t, &matrix_cfg);
        adam::step(&mut w.wk, &g.dwk, &mut o.m_wk, &mut o.v_wk, t, &matrix_cfg);
        adam::step(&mut w.wv, &g.dwv, &mut o.m_wv, &mut o.v_wv, t, &matrix_cfg);
        adam::step(&mut w.wo, &g.dwo, &mut o.m_wo, &mut o.v_wo, t, &matrix_cfg);
        adam::step(&mut w.w1, &g.dw1, &mut o.m_w1, &mut o.v_w1, t, &matrix_cfg);
        adam::step(&mut w.w3, &g.dw3, &mut o.m_w3, &mut o.v_w3, t, &matrix_cfg);
        adam::step(&mut w.w2, &g.dw2, &mut o.m_w2, &mut o.v_w2, t, &matrix_cfg);

        // RMSNorm scales (no weight decay)
        adam::step(&mut w.gamma1, &g.dgamma1, &mut o.m_gamma1, &mut o.v_gamma1, t, &norm_cfg);
        adam::step(&mut w.gamma2, &g.dgamma2, &mut o.m_gamma2, &mut o.v_gamma2, t, &norm_cfg);
    }
}

fn sum_sq(v: &[f32], scratch: &mut Vec<f32>) -> f32 {
    if scratch.len() < v.len() { scratch.resize(v.len(), 0.0); }
    vdsp::vmul(v, v, &mut scratch[..v.len()]);
    vdsp::sve(&scratch[..v.len()])
}

/// Global gradient L2 norm.
pub fn grad_norm(grads: &ModelGrads) -> f32 {
    let mut scratch = Vec::new();
    let mut sum = 0.0f32;
    sum += sum_sq(&grads.dembed, &mut scratch);
    sum += sum_sq(&grads.dgamma_final, &mut scratch);
    for lg in &grads.layers {
        for g in [&lg.dwq, &lg.dwk, &lg.dwv, &lg.dwo, &lg.dw1, &lg.dw3, &lg.dw2, &lg.dgamma1, &lg.dgamma2] {
            sum += sum_sq(g, &mut scratch);
        }
    }
    sum.sqrt()
}

/// Clip all gradients by global L2 norm.
pub fn clip_grads(grads: &mut ModelGrads, max_norm: f32) {
    let norm = grad_norm(grads);
    if norm > max_norm {
        let scale = max_norm / norm;
        let mut scratch = Vec::new();
        scale_vec(&mut grads.dembed, scale, &mut scratch);
        scale_vec(&mut grads.dgamma_final, scale, &mut scratch);
        for lg in &mut grads.layers {
            scale_vec(&mut lg.dwq, scale, &mut scratch);
            scale_vec(&mut lg.dwk, scale, &mut scratch);
            scale_vec(&mut lg.dwv, scale, &mut scratch);
            scale_vec(&mut lg.dwo, scale, &mut scratch);
            scale_vec(&mut lg.dw1, scale, &mut scratch);
            scale_vec(&mut lg.dw3, scale, &mut scratch);
            scale_vec(&mut lg.dw2, scale, &mut scratch);
            scale_vec(&mut lg.dgamma1, scale, &mut scratch);
            scale_vec(&mut lg.dgamma2, scale, &mut scratch);
        }
    }
}

fn scale_vec(v: &mut [f32], s: f32, scratch: &mut Vec<f32>) {
    if scratch.len() < v.len() {
        scratch.resize(v.len(), 0.0);
    }
    vdsp::vsmul(v, s, &mut scratch[..v.len()]);
    v.copy_from_slice(&scratch[..v.len()]);
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

        let fwd = forward(
            cfg, kernels, weights, &input_tokens, &target_tokens, tc.softcap,
        );
        total_loss += fwd.loss;

        backward(
            cfg, kernels, weights, &fwd, &input_tokens, tc.softcap, tc.loss_scale, grads,
        );
    }

    // Scale gradients: 1 / (accum_steps * loss_scale)
    let gsc = 1.0 / (tc.accum_steps as f32 * tc.loss_scale);
    let mut scratch = Vec::new();
    scale_vec(&mut grads.dembed, gsc, &mut scratch);
    scale_vec(&mut grads.dgamma_final, gsc, &mut scratch);
    for lg in &mut grads.layers {
        scale_vec(&mut lg.dwq, gsc, &mut scratch);
        scale_vec(&mut lg.dwk, gsc, &mut scratch);
        scale_vec(&mut lg.dwv, gsc, &mut scratch);
        scale_vec(&mut lg.dwo, gsc, &mut scratch);
        scale_vec(&mut lg.dw1, gsc, &mut scratch);
        scale_vec(&mut lg.dw3, gsc, &mut scratch);
        scale_vec(&mut lg.dw2, gsc, &mut scratch);
        scale_vec(&mut lg.dgamma1, gsc, &mut scratch);
        scale_vec(&mut lg.dgamma2, gsc, &mut scratch);
    }

    // Gradient clipping
    clip_grads(grads, tc.grad_clip);

    // LR schedule
    let lr = learning_rate(step, tc);

    // Weight update
    update_weights(cfg, weights, grads, opt, step + 1, lr, tc);

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
    for s in 0..seq {
        for c in 0..dim { x[c * seq + s] = x_row[s * dim + c]; }
    }

    // Layers
    for l in 0..cfg.nlayers {
        let (x_next, _) = layer::forward(cfg, kernels, &weights.layers[l], &x);
        x = x_next;
    }

    // Final RMSNorm
    let mut x_final = vec![0.0f32; dim * seq];
    let mut x_pos = vec![0.0f32; dim];
    let mut out_pos = vec![0.0f32; dim];
    for s in 0..seq {
        for c in 0..dim { x_pos[c] = x[c * seq + s]; }
        rmsnorm::forward(&x_pos, &weights.gamma_final, &mut out_pos);
        for c in 0..dim { x_final[c * seq + s] = out_pos[c]; }
    }

    // Logits
    let mut x_final_row = vec![0.0f32; seq * dim];
    for s in 0..seq {
        for c in 0..dim { x_final_row[s * dim + c] = x_final[c * seq + s]; }
    }
    let mut logits = vec![0.0f32; seq * vocab];
    vdsp::sgemm_at(&x_final_row, seq, dim, &weights.embed, vocab, &mut logits);

    // Softcap
    if softcap > 0.0 {
        let inv_cap = 1.0 / softcap;
        let mut scaled = vec![0.0f32; seq * vocab];
        vdsp::vsmul(&logits, inv_cap, &mut scaled);
        vdsp::tanhf(&scaled, &mut logits);
        let mut capped = vec![0.0f32; seq * vocab];
        vdsp::vsmul(&logits, softcap, &mut capped);
        logits.copy_from_slice(&capped);
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
