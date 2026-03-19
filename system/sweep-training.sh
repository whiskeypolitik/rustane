#!/bin/bash
# Sweep training configs to find what works
# Each test: 20 steps, check if loss stays stable
set -uo pipefail

DATA="/Users/dan/Dev/autoresearch-ANE/native/data/train_karpathy.bin"
RESULTS="checkpoints/sweep_results.tsv"
BIN="cargo run -p engine --release --bin train --"

echo -e "model\taccum\tloss_scale\tgrad_clip\tlr\twarmup\tloss_start\tloss_end\tstable\tms_per_step" > "$RESULTS"

run_test() {
    local model=$1 accum=$2 ls=$3 gc=$4 lr=$5 warmup=$6
    local label="${model}/a${accum}/ls${ls}/gc${gc}/lr${lr}/w${warmup}"

    output=$($BIN --model "$model" --data "$DATA" --steps 20 \
        --accum "$accum" --loss-scale "$ls" --grad-clip "$gc" \
        --lr "$lr" --warmup "$warmup" 2>&1)

    first_loss=$(echo "$output" | grep "^step     0:" | grep -o "loss = [0-9.]*" | grep -o "[0-9.]*")
    last_loss=$(echo "$output" | grep "^step    19:" | grep -o "loss = [0-9.]*" | grep -o "[0-9.]*")
    step_time=$(echo "$output" | grep "^step    19:" | grep -o "time = [0-9.]*" | grep -o "[0-9.]*")

    if [ -z "$first_loss" ] || [ -z "$last_loss" ]; then
        echo -e "${model}\t${accum}\t${ls}\t${gc}\t${lr}\t${warmup}\t-\t-\tCRASH\t-" >> "$RESULTS"
        printf "  %-50s CRASH\n" "$label"
        return
    fi

    # Check stability: loss_end < loss_start * 1.2 = stable
    stable=$(python3 -c "
f,l = $first_loss, $last_loss
if l < f * 0.95: print('LEARNING')
elif l < f * 1.1: print('STABLE')
elif l < f * 1.5: print('DRIFTING')
else: print('DIVERGED')
")

    echo -e "${model}\t${accum}\t${ls}\t${gc}\t${lr}\t${warmup}\t${first_loss}\t${last_loss}\t${stable}\t${step_time}" >> "$RESULTS"
    printf "  %-50s %s\t%.4f → %.4f\t%ss\n" "$label" "$stable" "$first_loss" "$last_loss" "$step_time"
}

echo "=== Training Config Sweep ==="
echo "Running 20-step tests across configurations..."
echo ""

# ── GROUP 1: 768/6L — find working accum settings ──
echo "── GROUP 1: gpt_karpathy (768/6L/s512, 48.8M) ──"
for accum in 1 2 4 10; do
    for ls in 1 256; do
        for gc in 1 100000; do
            for lr in 1e-4 3e-4; do
                run_test gpt_karpathy $accum $ls $gc $lr 100
            done
        done
    done
done

# ── GROUP 2: 1024/8L — same sweep ──
echo ""
echo "── GROUP 2: gpt_1024 (1024/8L/s512, 111M) ──"
for accum in 1 2 4 10; do
    for ls in 1 256; do
        for gc in 1 100000; do
            run_test gpt_1024 $accum $ls $gc 1e-4 100
        done
    done
done

# ── GROUP 3: 28L — targeted sweep on what works from groups 1-2 ──
echo ""
echo "── GROUP 3: mha_28l (1024/28L/s256, 390M) ──"
for accum in 1 2 4 10; do
    for ls in 1 256; do
        for gc in 1 100000; do
            run_test mha_28l $accum $ls $gc 1e-4 100
        done
    done
done

# ── GROUP 4: LR sweep on promising configs ──
echo ""
echo "── GROUP 4: LR sweep on accum=1/ls=1 (known stable) ──"
for model in gpt_karpathy mha_28l; do
    for lr in 1e-5 3e-5 1e-4 3e-4 1e-3; do
        run_test $model 1 1 100000 $lr 0
    done
done

# ── GROUP 5: warmup sweep ──
echo ""
echo "── GROUP 5: Warmup sweep ──"
for warmup in 0 10 100 1000; do
    run_test mha_28l 10 1 100000 1e-4 $warmup
    run_test mha_28l 10 1 1 1e-4 $warmup
done

echo ""
echo "=== SWEEP COMPLETE ==="
echo "Results: $RESULTS"
echo ""
column -t -s $'\t' "$RESULTS"
