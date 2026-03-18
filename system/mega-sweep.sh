#!/bin/bash
# Mega sweep: test every viable config combination in 30 min
# Each test: 10 steps (minimal), extract loss trend + ms/step
set -uo pipefail

DATA="/Users/dan/Dev/autoresearch-ANE/native/data/train_karpathy.bin"
RESULTS="checkpoints/mega_sweep.tsv"
BIN="cargo run -p engine --release --bin train --"

echo -e "dim\thidden\tnlayers\tseq\theads\tparams_M\taccum\tlr\tloss_start\tloss_end\tdelta\tstable\tms_step\ttok_per_s\t8h_tokens_M" > "$RESULTS"

# We need dynamic model configs. Add a generic model flag.
# For now, create configs on the fly by editing model.rs, rebuilding, and testing.
# FASTER: use the existing train binary with a --dim/--hidden/--nlayers/--seq override.
# But we don't have that. So let's create all configs in model.rs first.

MODELRS="/Users/dan/Dev/rustane/crates/engine/src/model.rs"

# Save original
cp "$MODELRS" "${MODELRS}.bak"

# Generate all configs as model functions
cat > /tmp/sweep_configs.txt << 'CONFIGS'
sweep_6L_768_2048_128 768 2048 6 128 6
sweep_6L_768_2048_256 768 2048 6 256 6
sweep_6L_768_2048_512 768 2048 6 512 6
sweep_8L_768_2048_128 768 2048 8 128 6
sweep_8L_768_2048_256 768 2048 8 256 6
sweep_8L_768_2048_512 768 2048 8 512 6
sweep_12L_768_2048_128 768 2048 12 128 6
sweep_12L_768_2048_256 768 2048 12 256 6
sweep_12L_768_2048_512 768 2048 12 512 6
sweep_8L_1024_2816_128 1024 2816 8 128 8
sweep_8L_1024_2816_256 1024 2816 8 256 8
sweep_8L_1024_2816_512 1024 2816 8 512 8
sweep_12L_1024_2816_128 1024 2816 12 128 8
sweep_12L_1024_2816_256 1024 2816 12 256 8
sweep_12L_1024_2816_512 1024 2816 12 512 8
sweep_16L_1024_2816_128 1024 2816 16 128 8
sweep_16L_1024_2816_256 1024 2816 16 256 8
sweep_20L_1024_2816_128 1024 2816 20 128 8
sweep_20L_1024_2816_256 1024 2816 20 256 8
sweep_24L_1024_2816_128 1024 2816 24 128 8
sweep_24L_1024_2816_256 1024 2816 24 256 8
sweep_28L_1024_2816_128 1024 2816 28 128 8
sweep_28L_1024_2816_256 1024 2816 28 256 8
sweep_28L_1024_3072_256 1024 3072 28 256 8
sweep_12L_1536_4096_128 1536 4096 12 128 12
sweep_12L_1536_4096_256 1536 4096 12 256 12
sweep_16L_1536_4096_128 1536 4096 16 128 12
sweep_16L_1536_4096_256 1536 4096 16 256 12
sweep_8L_1536_4096_128 1536 4096 8 128 12
sweep_8L_1536_4096_256 1536 4096 8 256 12
sweep_12L_2048_5632_128 2048 5632 12 128 16
sweep_8L_2048_5632_128 2048 5632 8 128 16
sweep_6L_2048_5632_128 2048 5632 6 128 16
sweep_4L_2048_5632_128 2048 5632 4 128 16
sweep_6L_1280_3584_256 1280 3584 6 256 10
sweep_8L_1280_3584_256 1280 3584 8 256 10
sweep_12L_1280_3584_256 1280 3584 12 256 10
sweep_16L_1280_3584_128 1280 3584 16 128 10
sweep_12L_1280_3584_128 1280 3584 12 128 10
CONFIGS

# Generate Rust code for all sweep configs
SWEEP_CODE=""
MATCH_ARMS=""
while IFS=' ' read -r name dim hidden nl seq heads; do
    hd=128
    kv_heads=$heads
    q_dim=$dim
    kv_dim=$dim
    gqa=1
    SWEEP_CODE+="
    pub fn ${name}() -> Self {
        Self { dim: ${dim}, hidden: ${hidden}, heads: ${heads}, kv_heads: ${kv_heads}, hd: ${hd}, seq: ${seq}, nlayers: ${nl}, vocab: 8192, q_dim: ${q_dim}, kv_dim: ${kv_dim}, gqa_ratio: ${gqa} }
    }"
    MATCH_ARMS+="
        \"${name}\" => ModelConfig::${name}(),"
done < /tmp/sweep_configs.txt

# Patch model.rs: add sweep configs before final }
# Find the closing brace of impl ModelConfig
sed -i.bak2 '/^}$/i\
'"$(echo "$SWEEP_CODE" | sed 's/$/\\/' | sed '$ s/\\$//')"'
' "$MODELRS" 2>/dev/null || true

# Actually, let's do this more carefully with a temp file approach
cat "${MODELRS}.bak" | sed '/^}$/d' > /tmp/model_sweep.rs
echo "$SWEEP_CODE" >> /tmp/model_sweep.rs
echo "}" >> /tmp/model_sweep.rs
cp /tmp/model_sweep.rs "$MODELRS"

# Patch train.rs to add all sweep model names
TRAINRS="/Users/dan/Dev/rustane/crates/engine/src/bin/train.rs"
cp "$TRAINRS" "${TRAINRS}.bak"

# Add match arms before the "other =>" line
while IFS=' ' read -r name dim hidden nl seq heads; do
    sed -i '' "s|other => { eprintln|\"${name}\" => ModelConfig::${name}(),\n        other => { eprintln|" "$TRAINRS"
done < /tmp/sweep_configs.txt

echo "Building with all sweep configs..."
if ! cargo build -p engine --release 2>&1 | tail -3; then
    echo "BUILD FAILED"
    cp "${MODELRS}.bak" "$MODELRS"
    cp "${TRAINRS}.bak" "$TRAINRS"
    exit 1
fi

echo "Build successful. Running sweep..."
echo ""

run_test() {
    local name=$1 dim=$2 hidden=$3 nl=$4 seq=$5 heads=$6 accum=$7 lr=$8

    # Calculate params
    local params=$(python3 -c "
d,h,n,v=$dim,$hidden,$nl,8192
per_layer = 4*d*d + 3*d*h + 2*d
total = v*d + n*per_layer + d
print(f'{total/1e6:.1f}')
")

    output=$($BIN --model "$name" --data "$DATA" --steps 10 \
        --accum "$accum" --loss-scale 1 --grad-clip 100000 \
        --lr "$lr" --warmup 0 2>&1)

    first_loss=$(echo "$output" | grep "^step     0:" | grep -o "loss = [0-9.]*" | grep -o "[0-9.]*")
    last_line=$(echo "$output" | grep "^step     9:" || echo "$output" | grep "^step" | tail -1)
    last_loss=$(echo "$last_line" | grep -o "loss = [0-9.]*" | grep -o "[0-9.]*")
    step_time=$(echo "$last_line" | grep -o "time = [0-9.]*" | grep -o "[0-9.]*")

    if [ -z "$first_loss" ] || [ -z "$last_loss" ]; then
        echo -e "${dim}\t${hidden}\t${nl}\t${seq}\t${heads}\t${params}\t${accum}\t${lr}\t-\t-\t-\tCRASH\t-\t-\t-" >> "$RESULTS"
        printf "  %-45s  CRASH\n" "${nl}L/${dim}d/${hidden}h/s${seq}/a${accum}"
        return
    fi

    local delta=$(python3 -c "print(f'{$last_loss - $first_loss:.4f}')")
    local stable=$(python3 -c "
f,l = $first_loss, $last_loss
if l < f * 0.95: print('LEARNING')
elif l < f * 1.05: print('STABLE')
elif l < f * 1.2: print('DRIFTING')
else: print('DIVERGED')
")
    local tok_s=$(python3 -c "print(f'{$seq * $accum / $step_time:.0f}')" 2>/dev/null || echo "0")
    local tokens_8h=$(python3 -c "print(f'{$seq * $accum * 28800 / $step_time / 1e6:.1f}')" 2>/dev/null || echo "0")

    echo -e "${dim}\t${hidden}\t${nl}\t${seq}\t${heads}\t${params}\t${accum}\t${lr}\t${first_loss}\t${last_loss}\t${delta}\t${stable}\t${step_time}\t${tok_s}\t${tokens_8h}" >> "$RESULTS"
    printf "  %-45s  %-9s  %.4f→%.4f  %4s tok/s  %5sM/8h\n" "${nl}L/${dim}d/${hidden}h/s${seq}/a${accum}" "$stable" "$first_loss" "$last_loss" "$tok_s" "$tokens_8h"
}

echo "=== MEGA SWEEP: Architecture × Accumulation ==="
echo ""

# Run all configs with accum=1 first (known stable, fast to test)
echo "── Phase 1: All configs at accum=1, lr=1e-4 ──"
while IFS=' ' read -r name dim hidden nl seq heads; do
    run_test "$name" "$dim" "$hidden" "$nl" "$seq" "$heads" 1 1e-4
done < /tmp/sweep_configs.txt

# Then test accum=2 and accum=4 on all configs
echo ""
echo "── Phase 2: All configs at accum=2, lr=1e-4 ──"
while IFS=' ' read -r name dim hidden nl seq heads; do
    run_test "$name" "$dim" "$hidden" "$nl" "$seq" "$heads" 2 1e-4
done < /tmp/sweep_configs.txt

echo ""
echo "── Phase 3: All configs at accum=4, lr=1e-4 ──"
while IFS=' ' read -r name dim hidden nl seq heads; do
    run_test "$name" "$dim" "$hidden" "$nl" "$seq" "$heads" 4 1e-4
done < /tmp/sweep_configs.txt

# Restore original files
cp "${MODELRS}.bak" "$MODELRS"
cp "${TRAINRS}.bak" "$TRAINRS"
cargo build -p engine --release 2>&1 > /dev/null

echo ""
echo "=== SWEEP COMPLETE ==="
echo "Results: $RESULTS"
echo ""
echo "── TOP CONFIGS BY 8h TOKEN COUNT (learning or stable only) ──"
grep -v "CRASH\|DIVERGED\|DRIFTING" "$RESULTS" | sort -t$'\t' -k15 -rn | head -20 | column -t -s $'\t'
