#!/bin/bash
set -uo pipefail
DATA="/Users/dan/Dev/autoresearch-ANE/native/data/train_karpathy.bin"
RESULTS="checkpoints/sweep_gaps.tsv"

echo -e "dim\thidden\tnlayers\tseq\tparams_M\taccum\tlr\tsteps\tloss_0\tloss_end\tdelta\tstatus\tms_step\ttok_s\t8h_M" > "$RESULTS"

run() {
    local dim=$1 hidden=$2 nl=$3 seq=$4 accum=$5 lr=$6 nsteps=$7
    local model="custom:${dim},${hidden},${nl},${seq}"
    local params=$(python3 -c "d,h,n=$dim,$hidden,$nl; print(f'{(8192*d+n*(4*d*d+3*d*h+2*d)+d)/1e6:.1f}')")

    out=$(cargo run -p engine --release --bin train -- \
        --model "$model" --data "$DATA" --steps "$nsteps" \
        --accum "$accum" --loss-scale 1 --grad-clip 100000 \
        --lr "$lr" --warmup 0 2>&1)

    local l0=$(echo "$out" | grep "^step     0:" | grep -o "loss = [0-9.]*" | grep -o "[0-9.]*")
    local last_step=$((nsteps - 1))
    local lN=$(echo "$out" | grep "^step" | tail -1 | grep -o "loss = [0-9.]*" | grep -o "[0-9.]*")
    local ms=$(echo "$out" | grep "^step" | tail -1 | grep -o "time = [0-9.]*" | grep -o "[0-9.]*")

    if [ -z "$l0" ] || [ -z "$lN" ]; then
        echo -e "${dim}\t${hidden}\t${nl}\t${seq}\t${params}\t${accum}\t${lr}\t${nsteps}\t-\t-\t-\tCRASH\t-\t-\t-" >> "$RESULTS"
        printf "  %-55s CRASH\n" "${nl}L/${dim}/${hidden}/s${seq}/a${accum}/lr${lr}/${nsteps}st"
        return
    fi

    local info=$(python3 -c "
f,l,ms,seq,acc=$l0,$lN,$ms,$seq,$accum
d=l-f
s='LEARN' if l<f*0.98 else ('OK' if l<f*1.05 else ('DRIFT' if l<f*1.2 else 'BAD'))
ts=seq*acc/ms
m8=ts*28800/1e6
print(f'{d:.4f}\t{s}\t{ms}\t{ts:.0f}\t{m8:.1f}')
")
    echo -e "${dim}\t${hidden}\t${nl}\t${seq}\t${params}\t${accum}\t${lr}\t${nsteps}\t${l0}\t${lN}\t${info}" >> "$RESULTS"
    local status=$(echo "$info" | cut -f2)
    local tok_s=$(echo "$info" | cut -f4)
    local m8=$(echo "$info" | cut -f5)
    printf "  %-55s %-5s  %.4f→%.4f  %5s tok/s  %5sM/8h\n" "${nl}L/${dim}/${hidden}/s${seq}/a${accum}/lr${lr}/${nsteps}st" "$status" "$l0" "$lN" "$tok_s" "$m8"
}

echo "=== GAP SWEEP ==="
echo ""

# ── GAP 1: Accum boundary (5,6,7,8,9) on 768/6L and 1024/12L ──
echo "── Accum boundary: where does it break? ──"
for accum in 5 6 7 8 9; do
    run 768 2048 6 512 $accum 1e-4 20
    run 1024 2816 12 256 $accum 1e-4 20
done

# ── GAP 2: 2048 dim with more steps + different LR ──
echo ""
echo "── 2048 dim: more steps, different LR ──"
for lr in 3e-4 1e-3 3e-3; do
    run 2048 5632 6 128 1 $lr 30
    run 2048 5632 8 128 1 $lr 30
    run 2048 5632 12 128 1 $lr 30
done
# 2048 with smaller hidden (less staging overhead)
for hidden in 4096 3072; do
    run 2048 $hidden 6 128 1 1e-3 30
    run 2048 $hidden 8 128 1 1e-3 30
    run 2048 $hidden 12 128 1 1e-3 30
done
# 2048 with 50 steps at high LR
run 2048 5632 6 128 1 1e-3 50
run 2048 5632 8 128 1 1e-3 50

# ── GAP 3: LR sweep for 1024/28L and 1280/16L (the big models) ──
echo ""
echo "── LR sweep for biggest viable models ──"
for lr in 3e-5 1e-4 3e-4 1e-3; do
    run 1024 2816 28 128 4 $lr 10
    run 1024 2816 28 256 4 $lr 10
    run 1280 3584 16 128 4 $lr 10
done

# ── GAP 4: Different hidden ratios at 1024 dim ──
echo ""
echo "── Hidden ratio sweep at 1024 dim ──"
for hidden in 2048 2816 3072 4096; do
    run 1024 $hidden 12 256 4 1e-4 10
    run 1024 $hidden 20 128 4 1e-4 10
done

# ── GAP 5: Cross-test accum × dim × seq ──
echo ""
echo "── Cross: accum × bigger models ──"
for accum in 1 2 3 4; do
    run 1536 4096 12 128 $accum 3e-4 20
    run 1536 4096 8 256 $accum 3e-4 20
    run 2048 5632 6 128 $accum 3e-4 20
done

# ── GAP 6: 1536 with different LR (it was marginal) ──
echo ""
echo "── 1536 LR sweep ──"
for lr in 1e-4 3e-4 1e-3; do
    run 1536 4096 12 128 2 $lr 20
    run 1536 4096 12 256 2 $lr 20
    run 1536 4096 8 128 4 $lr 20
done

# ── GAP 7: Scale to 1.5B — big dim, big layers ──
echo ""
echo "── Scaling to 1B+ ──"
# ~500M configs
run 1536 4096 16 128 1 1e-3 20
run 1536 4096 16 256 1 3e-4 20
run 1536 4096 20 128 1 1e-3 20
run 1536 4096 24 128 1 1e-3 20
run 1280 3584 24 128 1 1e-3 20
run 1280 3584 28 128 1 1e-3 20
run 1280 3584 32 128 1 1e-3 20
# ~700M-1B configs
run 2048 5632 16 128 1 1e-3 30
run 2048 5632 20 128 1 1e-3 30
run 2048 5632 24 128 1 1e-3 30
run 2048 5632 28 128 1 1e-3 20
run 2048 5632 32 128 1 1e-3 20
run 1536 4096 28 128 1 1e-3 20
run 1536 4096 32 128 1 1e-3 20
# ~1.2-1.5B configs
run 2048 5632 36 128 1 1e-3 20
run 2048 5632 40 128 1 1e-3 20
run 2560 7168 16 128 1 1e-3 20
run 2560 7168 20 128 1 1e-3 20
run 2560 7168 24 128 1 1e-3 20
run 3072 8192 12 128 1 1e-3 20
run 3072 8192 16 128 1 1e-3 20
# accum=2 on promising big configs
run 2048 5632 16 128 2 1e-3 20
run 2048 5632 24 128 2 1e-3 20
run 1536 4096 24 128 2 1e-3 20
run 2560 7168 16 128 2 1e-3 20

echo ""
echo "=== GAP SWEEP COMPLETE ==="
echo "Results: $RESULTS"
echo ""
echo "── ALL RESULTS ──"
column -t -s $'\t' "$RESULTS"
