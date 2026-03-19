#!/bin/bash
# Target 600M: Full sweep suite (post logits-zeroing fix)
# Sweep 1: Hyperparameter alignment (1A-1J)
# Sweep 2: Stability frontier at 600M (2A-2K)
set -uo pipefail

DATA="/Users/dan/Dev/autoresearch-ANE/native/data/train_karpathy.bin"
TSV="/Users/dan/Dev/rustane/checkpoints/target_600m.tsv"
BIN="/Users/dan/Dev/rustane/target/release/train"
DATE=$(date +%Y-%m-%d)

run_test() {
    local experiment=$1 model=$2 steps=$3
    shift 3
    local extra_args="$@"

    printf ">>> %-35s " "$experiment"

    local output
    output=$($BIN --model "$model" --data "$DATA" --steps "$steps" \
        --val-interval 999999 --ckpt-interval 999999 \
        $extra_args 2>&1)
    local rc=$?

    if [ $rc -ne 0 ] || echo "$output" | grep -q "panicked\|SIGABRT"; then
        echo "CRASH"
        local params_m=$(echo "$output" | grep "params:" | head -1 | grep -o "[0-9.]*M" | grep -o "[0-9.]*" || echo "-")
        printf "%s\t%s\t%s\t%s\t-\t-\t-\t-\t-\t-\t-\t-\t%s\t-\t-\t-\tCRASH\t-\t-\t\n" \
            "$DATE" "$experiment" "$model" "$params_m" "$steps" >> "$TSV"
        return 1
    fi

    # Parse params
    local params_m=$(echo "$output" | grep "params:" | head -1 | grep -o "[0-9.]*M" | grep -o "[0-9.]*")

    # Parse config values from printout (single-line grep)
    local accum=$(echo "$output" | grep "accum:" | head -1 | sed 's/.*accum: \([0-9]*\).*/\1/')
    local warmup=$(echo "$output" | grep "warmup" | head -1 | sed 's/.*warmup \([0-9]*\).*/\1/')
    local beta2=$(echo "$output" | grep "beta2:" | head -1 | sed 's/.*beta2: \([0-9.]*\).*/\1/')
    local eps=$(echo "$output" | grep "eps:" | head -1 | sed 's/.*eps: \([0-9.e+-]*\).*/\1/')
    local wd=$(echo "$output" | grep "weight_decay:" | head -1 | sed 's/.*weight_decay: \([0-9.]*\).*/\1/')
    local embed_lr=$(echo "$output" | grep "embed_lr_scale:" | head -1 | sed 's/.*embed_lr_scale: \([0-9.]*\).*/\1/')
    local min_lr_frac=$(echo "$output" | grep "min_lr_frac:" | head -1 | sed 's/.*min_lr_frac: \([0-9.]*\).*/\1/')
    local lr=$(echo "$output" | grep "^  lr:" | head -1 | sed 's/.*lr: \([0-9.e+-]*\).*/\1/')

    # Get first and last loss
    local loss_0=$(echo "$output" | grep "^step " | head -1 | sed 's/.*loss = \([0-9.]*\).*/\1/')
    local loss_end=$(echo "$output" | grep "^step " | tail -1 | sed 's/.*loss = \([0-9.]*\).*/\1/')
    local step_time=$(echo "$output" | grep "^step " | tail -1 | sed 's/.*time = \([0-9.]*\)s.*/\1/')

    if [ -z "$loss_0" ] || [ -z "$loss_end" ]; then
        echo "CRASH (no output)"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t-\t-\t-\tCRASH\t-\t-\tno output\n" \
            "$DATE" "$experiment" "$model" "$params_m" "$accum" "$lr" "$warmup" "$beta2" "$eps" "$wd" "$embed_lr" "$min_lr_frac" "$steps" >> "$TSV"
        return 1
    fi

    # Check for NaN
    if echo "$output" | grep -q "loss = NaN\|loss = nan\|loss = inf"; then
        echo "CRASH (NaN)"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t-\t-\tCRASH\t-\t-\tNaN\n" \
            "$DATE" "$experiment" "$model" "$params_m" "$accum" "$lr" "$warmup" "$beta2" "$eps" "$wd" "$embed_lr" "$min_lr_frac" "$steps" "$loss_0" >> "$TSV"
        return 1
    fi

    local delta=$(python3 -c "print(f'{$loss_end - $loss_0:.4f}')")
    local ms_step=$(python3 -c "print(f'{$step_time * 1000:.0f}')" 2>/dev/null || echo "-")

    # Determine status
    local status=$(python3 -c "
d = $loss_end - $loss_0
if d < -0.05: print('LEARN')
elif abs(d) <= 0.05: print('OK')
elif d < 0.5: print('DRIFT')
else: print('BAD')
")

    echo "loss: $loss_0 → $loss_end (${delta}) ${status} ${ms_step}ms"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t-\t\n" \
        "$DATE" "$experiment" "$model" "$params_m" "$accum" "$lr" "$warmup" "$beta2" "$eps" "$wd" "$embed_lr" "$min_lr_frac" "$steps" "$loss_0" "$loss_end" "$delta" "$status" "$ms_step" >> "$TSV"
    return 0
}

echo "============================================"
echo "  SWEEP 1 — Hyperparam Alignment"
echo "  (post logits-zeroing fix)"
echo "============================================"
echo ""

# BASELINE: actual TrainConfig defaults
# beta2=0.95, embed_lr=5.0, eps=1e-8, wd=0.1, min_lr_frac=0.1, warmup=auto
echo "── Baseline (TrainConfig defaults) ──"
run_test "baseline_defaults" gpt_karpathy 50 \
    --lr 3e-4 --accum 10 --warmup 5

echo ""
echo "── 1A: embed_lr_scale = 1.0 (lower, CURRENT.md) ──"
run_test "1A_embed_lr_1" gpt_karpathy 50 \
    --lr 3e-4 --accum 10 --warmup 5 --embed-lr 1.0

echo ""
echo "── 1B: eps = 1e-10 (Obj-C value) ──"
run_test "1B_eps_1e10" gpt_karpathy 50 \
    --lr 3e-4 --accum 10 --warmup 5 --eps 1e-10

echo ""
echo "── 1C: weight_decay = 0.0 (Obj-C value) ──"
run_test "1C_wd_0" gpt_karpathy 50 \
    --lr 3e-4 --accum 10 --warmup 5 --wd 0.0

echo ""
echo "── 1D: accum = 2 (Obj-C value) ──"
run_test "1D_accum_2" gpt_karpathy 50 \
    --lr 3e-4 --accum 2 --warmup 5

echo ""
echo "── 1E: min_lr_frac = 0.0 (Obj-C value) ──"
run_test "1E_min_lr_0" gpt_karpathy 50 \
    --lr 3e-4 --accum 10 --warmup 5 --min-lr-frac 0.0

echo ""
echo "── 1F: beta2 = 0.99 (Obj-C value) ──"
run_test "1F_beta2_099" gpt_karpathy 50 \
    --lr 3e-4 --accum 10 --warmup 5 --beta2 0.99

echo ""
echo "── 1G: beta2 = 0.999 (CURRENT.md "New") ──"
run_test "1G_beta2_0999" gpt_karpathy 50 \
    --lr 3e-4 --accum 10 --warmup 5 --beta2 0.999

echo ""
echo "============================================"
echo "  1H: Full Obj-C E68 config"
echo "============================================"
run_test "1H_objc_full" gpt_karpathy 200 \
    --lr 3e-4 --accum 2 --warmup 6 --embed-lr 5.0 --beta2 0.99 --eps 1e-10 --wd 0.0 --min-lr-frac 0.0

echo ""
echo "── 1H2: Obj-C config but accum=10 ──"
run_test "1H2_objc_accum10" gpt_karpathy 200 \
    --lr 3e-4 --accum 10 --warmup 20 --embed-lr 5.0 --beta2 0.99 --eps 1e-10 --wd 0.0 --min-lr-frac 0.0

echo ""
echo "============================================"
echo "  1I: Combined at 1024/8L/s512 (~111M)"
echo "============================================"
run_test "1I_1024_8L_objc" gpt_1024 50 \
    --lr 3e-4 --accum 2 --warmup 2 --embed-lr 5.0 --beta2 0.99 --eps 1e-10 --wd 0.0 --min-lr-frac 0.0

echo ""
echo "── 1I2: 1024/8L with defaults ──"
run_test "1I2_1024_8L_defaults" gpt_1024 50 \
    --lr 3e-4 --accum 2 --warmup 2

echo ""
echo "============================================"
echo "  1J: Combined at 1536/4096/12L/s256 (~352M)"
echo "============================================"
run_test "1J_1536_12L_objc" "custom:1536,4096,12,256" 20 \
    --lr 3e-4 --accum 2 --warmup 1 --embed-lr 5.0 --beta2 0.99 --eps 1e-10 --wd 0.0 --min-lr-frac 0.0

echo ""
echo "── 1J2: 1536/12L with defaults ──"
run_test "1J2_1536_12L_defaults" "custom:1536,4096,12,256" 20 \
    --lr 3e-4 --accum 2 --warmup 1

echo ""
echo ""
echo "============================================"
echo "  SWEEP 2 — Stability Frontier at 600M"
echo "============================================"
echo ""

# All Sweep 2 tests use Obj-C E68 params (best from Sweep 1 expected)
echo "── 2A: 579M, accum=1, lr=1e-4 (conservative) ──"
run_test "2A_579M_a1_lr1e4" "custom:1536,4096,20,256" 20 \
    --lr 1e-4 --accum 1 --warmup 1 --embed-lr 5.0 --beta2 0.99 --eps 1e-10 --wd 0.0 --min-lr-frac 0.0

echo ""
echo "── 2B: 579M, accum=1, lr=3e-4 (standard) ──"
run_test "2B_579M_a1_lr3e4" "custom:1536,4096,20,256" 20 \
    --lr 3e-4 --accum 1 --warmup 1 --embed-lr 5.0 --beta2 0.99 --eps 1e-10 --wd 0.0 --min-lr-frac 0.0

echo ""
echo "── 2C: 579M, accum=1, lr=5e-4 (aggressive) ──"
run_test "2C_579M_a1_lr5e4" "custom:1536,4096,20,256" 20 \
    --lr 5e-4 --accum 1 --warmup 1 --embed-lr 5.0 --beta2 0.99 --eps 1e-10 --wd 0.0 --min-lr-frac 0.0

echo ""
echo "── 2D: 579M, accum=2, lr=1e-4 ──"
run_test "2D_579M_a2_lr1e4" "custom:1536,4096,20,256" 20 \
    --lr 1e-4 --accum 2 --warmup 1 --embed-lr 5.0 --beta2 0.99 --eps 1e-10 --wd 0.0 --min-lr-frac 0.0

echo ""
echo "── 2E: 579M, accum=2, lr=5e-5 (very conservative) ──"
run_test "2E_579M_a2_lr5e5" "custom:1536,4096,20,256" 20 \
    --lr 5e-5 --accum 2 --warmup 1 --embed-lr 5.0 --beta2 0.99 --eps 1e-10 --wd 0.0 --min-lr-frac 0.0

echo ""
echo "── 2F: 579M, accum=4, lr=3e-5 (high accum, low lr) ──"
run_test "2F_579M_a4_lr3e5" "custom:1536,4096,20,256" 20 \
    --lr 3e-5 --accum 4 --warmup 1 --embed-lr 5.0 --beta2 0.99 --eps 1e-10 --wd 0.0 --min-lr-frac 0.0

echo ""
echo "── Alternative configs ──"
echo ""
echo "── 2G: mha_28l (1024/3072/28L, ~390M) ──"
run_test "2G_mha28l" mha_28l 20 \
    --lr 1e-4 --accum 2 --warmup 1 --embed-lr 5.0 --beta2 0.99 --eps 1e-10 --wd 0.0 --min-lr-frac 0.0

echo ""
echo "── 2H: 1536/4096/16L (~466M) ──"
run_test "2H_466M" "custom:1536,4096,16,256" 20 \
    --lr 1e-4 --accum 2 --warmup 1 --embed-lr 5.0 --beta2 0.99 --eps 1e-10 --wd 0.0 --min-lr-frac 0.0

echo ""
echo "── 2I: 1536/4096/12L (~352M, conservative scale) ──"
run_test "2I_352M" "custom:1536,4096,12,256" 20 \
    --lr 1e-4 --accum 2 --warmup 1 --embed-lr 5.0 --beta2 0.99 --eps 1e-10 --wd 0.0 --min-lr-frac 0.0

echo ""
echo "── Extended validation ──"
echo ""
echo "── 2J: 579M, 100 steps (accum=1, lr=1e-4) ──"
run_test "2J_579M_100step" "custom:1536,4096,20,256" 100 \
    --lr 1e-4 --accum 1 --warmup 3 --embed-lr 5.0 --beta2 0.99 --eps 1e-10 --wd 0.0 --min-lr-frac 0.0

echo ""
echo "── 2K: 352M, 100 steps (accum=2, lr=1e-4) ──"
run_test "2K_352M_100step" "custom:1536,4096,12,256" 100 \
    --lr 1e-4 --accum 2 --warmup 3 --embed-lr 5.0 --beta2 0.99 --eps 1e-10 --wd 0.0 --min-lr-frac 0.0

echo ""
echo "── 2L: 579M, 100 steps with defaults ──"
run_test "2L_579M_100step_defaults" "custom:1536,4096,20,256" 100 \
    --lr 1e-4 --accum 1 --warmup 3

echo ""
echo ""
echo "============================================"
echo "  ALL SWEEPS COMPLETE"
echo "============================================"
echo ""
cat "$TSV"
