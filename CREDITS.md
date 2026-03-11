# Credits

Rustane builds on the work of many researchers and open-source projects. This documents every significant source that informed the architecture, research, and implementation.

## ANE Reverse Engineering

| Project | Author | What We Learned |
|---------|--------|-----------------|
| [maderix/ANE](https://github.com/maderix/ANE) | maderix | ANE private API reverse engineering (C++). Dynamic weight pipeline, 1x1 conv discovery, mega-kernel fusion, IOSurface weight staging, INT8 W8A8 training. The foundational RE work everything else builds on. |
| [Anemll](https://github.com/Anemll/Anemll) | Anemll team | ANE inference tricks: Conv2d workaround for matmul, doubled RMSNorm for stability, in-model argmax, ring buffer KV cache, LUT quantization limits, SRAM bandwidth analysis. |
| [ANEgpt](https://github.com/vipuldivyanshu92/ANEgpt) | Vipul Divyanshu | ANE classifier as 1x1 conv (10.2x speedup over CPU sgemm). Bridge API patterns for _ANEClient. |

## Rust + ANE + Metal

| Project | Author | What We Learned |
|---------|--------|-----------------|
| [ane-infer](https://github.com/thebasedcapital/ane-infer) | thebasedcapital | First Rust + ANE + Metal hybrid prototype. 13 custom Metal shaders, single-command-buffer decode (zero allocations), `doEvaluateDirectWithModel` chaining, fused FFN mega-kernels (3.6 TFLOPS), IOKit H11ANE kernel access. Benchmarks: 32 tok/s Q8 on M5. Proves the Rust+ANE+Metal stack is memory-safe and viable. |
| [ane crate](https://lib.rs/crates/ane) | computer-graphics-tools | Clean Rust FFI to private AppleNeuralEngine.framework via objc2. GPT-2 inference example. 2,567 LOC across 5 key files. Our base for ane-bridge (will vendor and extend for training). |

## Rust ML / Metal Inference

| Project | Author | What We Learned |
|---------|--------|-----------------|
| [uzu](https://github.com/trymirai/uzu) | trymirai | Pure Rust Metal LLM engine. 40+ Metal shaders, fused MLP epilogue, quantized dispatch hierarchy, speculative decoding agent module. Closest to production Rust edge AI on M-series (no ANE). Informs our metal-decode crate design. |
| [candle](https://github.com/huggingface/candle) | Hugging Face | Rust ML framework with Metal + CUDA backends. Hardcoded 3-variant Storage/Device enums make direct ANE integration impractical — but CUDA backend via cudarc is our Jetson deployment path. SafeTensors + GGML quantization support. |

## Agent / Inference Benchmarks

| Project | Author | What We Learned |
|---------|--------|-----------------|
| [RCLI](https://github.com/RunanywhereAI/RCLI) | RunanywhereAI | Swift + MetalRT inference CLI. 658 tok/s MetalRT benchmarks on M-series. Voice/agent pipeline. Useful for comparing our ANE inference tok/s against GPU-only approaches on the same hardware. |
| [runanywhere-sdks](https://github.com/RunanywhereAI/runanywhere-sdks) | RunanywhereAI | Production agent SDKs (iOS/Android). Screen reindexing, /no_think prompting, inference guards. Language-agnostic patterns we'll port to Rust for the agent loop — especially relevant for Jetson drone/sat deployment. |

## Training Foundations

| Project | Author | What We Learned |
|---------|--------|-----------------|
| [autoresearch](https://github.com/karpathy/autoresearch) | Andrej Karpathy | Autonomous research framework, climbmix-400B dataset, rustbpe tokenizer. Our gpt_karpathy Phase 1 config derives from this. |
| [autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) | trevin-creator | MLX port of autoresearch for Apple Silicon. Muon+AdamW optimizer, 241 autonomous experiments, val_bpb 1.664→1.266. Provides our validation baseline and architecture exploration data. |

## Papers

| Paper | Authors | Relevance |
|-------|---------|-----------|
| [Orion (arXiv 2603.06728)](https://arxiv.org/abs/2603.06728) | — | ANE training/inference on M4 Max. 110M model trains 1000 steps in 22min. 8.5x speedup from weight-reload optimization. Validates dynamic weight pipeline approach. |

## Frameworks and Libraries

| Crate / Framework | Use |
|-------------------|-----|
| [objc2](https://docs.rs/objc2) | Safe Rust Obj-C FFI. All ANE private API calls go through this. |
| [objc2-foundation](https://docs.rs/objc2-foundation) | NSString, NSDictionary, NSData — needed for ANE model loading. |
| [objc2-io-surface](https://docs.rs/objc2-io-surface) | IOSurface creation and locking for ANE weight staging. |
| [half](https://docs.rs/half) | f16 type for CPU-side weight manipulation. |
| [safetensors](https://docs.rs/safetensors) | Weight interchange format (MLX ↔ Rustane ↔ candle). |
| [MLX](https://github.com/ml-explore/mlx) | Apple's GPU ML framework. Architecture exploration baseline. |
| Accelerate.framework | vDSP (vectorized f32 ops) + cblas_sgemm (CPU matmul) for non-ANE training ops. |
