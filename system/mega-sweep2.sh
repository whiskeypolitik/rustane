#!/bin/bash
set -uo pipefail
DATA="/Users/dan/Dev/autoresearch-ANE/native/data/train_karpathy.bin"
RESULTS="checkpoints/mega_sweep.tsv"

echo -e "dim\thidden\tnlayers\tseq\tparams_M\taccum\tloss_0\tloss_9\tdelta\tstatus\tms_step\ttok_s\t8h_M" > "$RESULTS"

run() {
    local dim=$1 hidden=$2 nl=$3 seq=$4 accum=$5
    local model="custom:${dim},${hidden},${nl},${seq}"
    local params=$(python3 -c "d,h,n=$dim,$hidden,$nl; print(f'{(8192*d+n*(4*d*d+3*d*h+2*d)+d)/1e6:.1f}')")

    out=$(cargo run -p engine --release --bin train -- \
        --model "$model" --data "$DATA" --steps 10 \
        --accum "$accum" --loss-scale 1 --grad-clip 100000 \
        --lr 1e-4 --warmup 0 2>&1)

    local l0=$(echo "$out" | grep "^step     0:" | grep -o "loss = [0-9.]*" | grep -o "[0-9.]*")
    local l9=$(echo "$out" | grep "^step     9:" | grep -o "loss = [0-9.]*" | grep -o "[0-9.]*")
    local ms=$(echo "$out" | grep "^step     9:" | grep -o "time = [0-9.]*" | grep -o "[0-9.]*")

    if [ -z "$l0" ] || [ -z "$l9" ]; then
        echo -e "${dim}\t${hidden}\t${nl}\t${seq}\t${params}\t${accum}\t-\t-\t-\tCRASH\t-\t-\t-" >> "$RESULTS"
        printf "  %-40s CRASH\n" "${nl}L/${dim}/${hidden}/s${seq}/a${accum}"
        return
    fi

    local info=$(python3 -c "
f,l,ms,seq,acc=$l0,$l9,$ms,$seq,$accum
d=l-f
s='LEARN' if l<f*0.98 else ('OK' if l<f*1.05 else ('DRIFT' if l<f*1.2 else 'BAD'))
ts=seq*acc/ms
m8=ts*28800/1e6
print(f'{d:.4f}\t{s}\t{ms}\t{ts:.0f}\t{m8:.1f}')
")

    echo -e "${dim}\t${hidden}\t${nl}\t${seq}\t${params}\t${accum}\t${l0}\t${l9}\t${info}" >> "$RESULTS"
    local status=$(echo "$info" | cut -f2)
    local tok_s=$(echo "$info" | cut -f4)
    local m8=$(echo "$info" | cut -f5)
    printf "  %-40s %-5s  %.4f→%.4f  %5s tok/s  %5sM/8h\n" "${nl}L/${dim}/${hidden}/s${seq}/a${accum}" "$status" "$l0" "$l9" "$tok_s" "$m8"
}

echo "=== MEGA SWEEP: 39 architectures × 3 accum = 117 tests ==="
echo "=== 10 steps each, ls=1, gc=100K, lr=1e-4, warmup=0 ==="
echo ""

# All configs: dim,hidden,nlayers,seq
CONFIGS=(
    # 768 dim variants
    "768,2048,4,128"  "768,2048,4,256"  "768,2048,4,512"
    "768,2048,6,128"  "768,2048,6,256"  "768,2048,6,512"
    "768,2048,8,128"  "768,2048,8,256"  "768,2048,8,512"
    "768,2048,12,128" "768,2048,12,256" "768,2048,12,512"
    # 1024 dim variants
    "1024,2816,6,128"  "1024,2816,6,256"  "1024,2816,6,512"
    "1024,2816,8,128"  "1024,2816,8,256"  "1024,2816,8,512"
    "1024,2816,12,128" "1024,2816,12,256" "1024,2816,12,512"
    "1024,2816,16,128" "1024,2816,16,256"
    "1024,2816,20,128" "1024,2816,20,256"
    "1024,2816,24,128" "1024,2816,24,256"
    "1024,2816,28,128" "1024,2816,28,256"
    "1024,3072,28,256"
    # 1280 dim
    "1280,3584,8,128"  "1280,3584,8,256"
    "1280,3584,12,128" "1280,3584,12,256"
    "1280,3584,16,128"
    # 1536 dim
    "1536,4096,8,128"  "1536,4096,8,256"
    "1536,4096,12,128" "1536,4096,12,256"
    # 2048 dim
    "2048,5632,4,128"  "2048,5632,6,128"  "2048,5632,8,128"
    "2048,5632,12,128"
)

for accum in 1 2 4; do
    echo "── accum=${accum} ──"
    for cfg in "${CONFIGS[@]}"; do
        IFS=',' read -r dim hidden nl seq <<< "$cfg"
        run "$dim" "$hidden" "$nl" "$seq" "$accum"
    done
    echo ""
done

echo "=== COMPLETE ==="
echo ""
echo "── TOP 20 BY 8h TOKEN THROUGHPUT (stable/learning only) ──"
grep -v "CRASH\|BAD\|DRIFT" "$RESULTS" | sort -t$'\t' -k13 -rn | head -21 | column -t -s $'\t'
echo ""
echo "── TOP 20 BY PARAMS × STABLE ──"
grep -v "CRASH\|BAD\|DRIFT" "$RESULTS" | sort -t$'\t' -k5 -rn | head -21 | column -t -s $'\t'
