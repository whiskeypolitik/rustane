//! Model configuration for transformer architectures.

/// Transformer model config. All 10 ANE kernels are parameterized by this.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub dim: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub hd: usize,        // dim / heads
    pub seq: usize,
    pub nlayers: usize,
    pub vocab: usize,
    pub q_dim: usize,     // heads * hd
    pub kv_dim: usize,    // kv_heads * hd
    pub gqa_ratio: usize, // heads / kv_heads
}

impl ModelConfig {
    /// GPT-Karpathy: NL=6, DIM=768, HEADS=6, MHA, climbmix-400B
    pub fn gpt_karpathy() -> Self {
        Self {
            dim: 768,
            hidden: 2048,
            heads: 6,
            kv_heads: 6,
            hd: 128,
            seq: 512,
            nlayers: 6,
            vocab: 8192,
            q_dim: 768,
            kv_dim: 768,
            gqa_ratio: 1,
        }
    }

    /// GPT-1024: NL=8, DIM=1024, HEADS=8, MHA — ~110M params
    pub fn gpt_1024() -> Self {
        Self {
            dim: 1024,
            hidden: 2816,
            heads: 8,
            kv_heads: 8,
            hd: 128,
            seq: 512,
            nlayers: 8,
            vocab: 8192,
            q_dim: 1024,
            kv_dim: 1024,
            gqa_ratio: 1,
        }
    }
}
