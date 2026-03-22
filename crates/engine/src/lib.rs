//! Hybrid inference engine.
//!
//! ANE for prefill + fused FFN mega-kernels.
//! Metal GPU for single-token decode.
//! CPU fallback for unsupported ops.

pub mod cpu;
pub mod model;
pub mod kernels;
pub mod layer;
pub mod training;
pub mod full_model;
pub mod metal_adam;
pub mod data;
pub mod bench_result;
