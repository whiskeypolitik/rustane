# Rustane — common commands
#
# Run `make help` to see all available targets.

.PHONY: help build test sweep-600m sweep-1b sweep-3b sweep-5b sweep-full \
	forward-ladder forward-ceiling forward-7b forward-10b train-600m

help: ## Show this help
	@echo "  Rustane — available commands:"
	@echo ""
	@echo "  make build                Build all crates"
	@echo "  make test                 Run all unit + integration tests"
	@echo ""
	@echo "  Training validation:"
	@echo "  make sweep-600m           Validate at 600M (~17s)"
	@echo "  make sweep-1b             Validate at 1B (~35s)"
	@echo "  make sweep-3b             Validate at 3B (~80s, needs 55GB)"
	@echo "  make sweep-5b             Validate at 5B (~150s, needs 85GB)"
	@echo "  make sweep-full           All 25 configs, 600M-5B (~60 min, needs 85GB)"
	@echo ""
	@echo "  Forward-only probes:"
	@echo "  make forward-7b           Forward pass at 7B (~30s, needs 31GB)"
	@echo "  make forward-10b          Forward pass at 10B (~45s, needs 46GB)"
	@echo "  make forward-ladder       Forward pass 5B to 20B (~8 min, needs 93GB)"
	@echo "  make forward-ceiling      Forward pass 25B/30B (~10 min, needs 130GB)"
	@echo ""
	@echo "  Training on real data:"
	@echo "  make train-600m DATA=/path/to/train.bin"

build: ## Build all crates
	cargo build

test: ## Run all unit + integration tests
	cargo test -p engine --release

# ── Training validation ─────────────────────────────────────────────

sweep-600m: ## Validate training pipeline at 600M (~17s)
	cargo test -p engine --test bench_param_sweep --release -- --ignored --nocapture sweep_600m_a

sweep-1b: ## Validate training pipeline at 1B (~35s)
	cargo test -p engine --test bench_param_sweep --release -- --ignored --nocapture sweep_1b_a

sweep-3b: ## Validate training pipeline at 3B (~80s, needs 55GB)
	cargo test -p engine --test bench_param_sweep --release -- --ignored --nocapture sweep_3b_a

sweep-5b: ## Validate training pipeline at 5B (~150s, needs 85GB)
	cargo test -p engine --test bench_param_sweep --release -- --ignored --nocapture sweep_5b_a

sweep-full: ## Full parameter sweep, 25 configs, 600M-5B (~60 min, needs 85GB)
	cargo test -p engine --test bench_param_sweep --release -- --ignored --nocapture sweep_full

# ── Forward-only scale probes ────────────────────────────────────────

forward-ladder: ## Forward pass 5B to 20B (~8 min, needs 93GB)
	cargo test -p engine --test bench_fwd_only_scale --release -- --ignored --nocapture fwd_scale_ladder

forward-ceiling: ## Push forward pass to 25B/30B (~10 min, needs 130GB)
	cargo test -p engine --test bench_fwd_only_scale --release -- --ignored --nocapture fwd_find_ceiling

forward-7b: ## Single forward pass at 7B (~30s, needs 31GB)
	cargo test -p engine --test bench_fwd_only_scale --release -- --ignored --nocapture fwd_7b

forward-10b: ## Single forward pass at 10B (~45s, needs 46GB)
	cargo test -p engine --test bench_fwd_only_scale --release -- --ignored --nocapture fwd_10b

# ── Real data training ───────────────────────────────────────────────

train-600m: ## Train 600M on real data (needs climbmix-400B data file)
	cargo run -p engine --release --bin train -- \
		--model custom:1536,4096,20,512 --data $(DATA) \
		--lr 3e-4 --accum 1 --warmup 100 \
		--embed-lr 1.0 --beta2 0.99 \
		--loss-scale 1 --grad-clip 1 \
		--steps 72000
