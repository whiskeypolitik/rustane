#!/bin/bash
# Train monitor: watches training log, restarts on crash, plots loss curve
# Usage: ./system/train-monitor.sh checkpoints/mha_28l_overnight/train.log

set -euo pipefail

LOG="${1:?Usage: $0 <train.log>}"
DIR="$(dirname "$LOG")"
PID_FILE="$DIR/train.pid"
PLOT_SCRIPT="$DIR/plot_loss.py"
PLOT_INTERVAL=120  # regenerate plot every 2 min

# Extract the training command from the log header
TRAIN_CMD="cargo run -p engine --release --bin train"

echo "=== Train Monitor ==="
echo "Log: $LOG"
echo "Plot interval: ${PLOT_INTERVAL}s"
echo ""

# Write the plot script
cat > "$PLOT_SCRIPT" << 'PYEOF'
import sys, re, os
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not available, skipping plot")
    sys.exit(0)

log_path = sys.argv[1]
out_path = os.path.join(os.path.dirname(log_path), "loss_curve.png")

steps, losses, val_steps, val_bpbs = [], [], [], []

with open(log_path) as f:
    for line in f:
        m = re.match(r'step\s+(\d+):\s+loss\s*=\s*([\d.]+)', line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
        m = re.search(r'val_loss\s*=\s*[\d.]+,\s*val_bpb\s*=\s*([\d.]+)', line)
        if m and steps:
            val_steps.append(steps[-1])
            val_bpbs.append(float(m.group(1)))

if len(steps) < 2:
    sys.exit(0)

c = {'bg': '#0c0c14', 'grid': '#181826', 'ane': '#79c0ff', 'accent': '#e3b341',
     'white': '#c9d1d9', 'dim': '#8b949e', 'dimmer': '#484f58', 'green': '#56d364'}

fig, ax1 = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(c['bg'])
ax1.set_facecolor(c['bg'])
ax1.plot(steps, losses, color=c['ane'], linewidth=1.5, alpha=0.7, label='train loss')
if val_bpbs:
    ax1.plot(val_steps, val_bpbs, 'o-', color=c['accent'], linewidth=2, markersize=4, label='val_bpb')
ax1.set_xlabel('step', fontsize=9, color=c['dim'], fontfamily='monospace')
ax1.set_ylabel('loss / bpb', fontsize=9, color=c['dim'], fontfamily='monospace')
ax1.set_title(f'390M ANE Training — step {steps[-1]}, loss {losses[-1]:.4f}',
              fontsize=13, color=c['white'], fontfamily='monospace', fontweight='bold')
ax1.tick_params(colors=c['dim'], labelsize=8)
ax1.spines['bottom'].set_color(c['dimmer'])
ax1.spines['left'].set_color(c['dimmer'])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='y', color=c['grid'], linewidth=0.5)
ax1.legend(fontsize=8, facecolor=c['bg'], edgecolor=c['dimmer'], labelcolor=c['white'])
fig.text(0.98, 0.02, '@danpacary', fontsize=8, color=c['dimmer'],
         fontfamily='monospace', ha='right', va='bottom')
plt.tight_layout()
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=c['bg'])
plt.close()
print(f"  plot saved: {out_path}")
PYEOF

# Monitor loop
last_plot=0
while true; do
    # Check if training process is alive
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if ! kill -0 "$pid" 2>/dev/null; then
            echo ""
            echo "!!! Training process $pid DIED !!!"
            # Check for NaN
            if grep -q "NaN\|nan\|inf" "$LOG" 2>/dev/null; then
                echo "  CAUSE: NaN/inf detected in log"
                echo "  Last 5 lines:"
                tail -5 "$LOG"
                echo ""
                echo "  NOT restarting (NaN = hyperparameter issue, needs manual fix)"
            else
                echo "  CAUSE: unknown crash"
                echo "  Last 5 lines:"
                tail -5 "$LOG"
            fi
            break
        fi
    else
        # Try to find PID from process list
        pid=$(pgrep -f "train.*mha_28l" 2>/dev/null | head -1 || true)
        if [ -z "$pid" ]; then
            echo "  No training process found. Exiting monitor."
            break
        fi
        echo "$pid" > "$PID_FILE"
        echo "  Found training PID: $pid"
    fi

    # Get latest stats
    last_line=$(grep "^step" "$LOG" 2>/dev/null | tail -1 || true)
    if [ -n "$last_line" ]; then
        step=$(echo "$last_line" | grep -o 'step\s*[0-9]*' | grep -o '[0-9]*')
        loss=$(echo "$last_line" | grep -o 'loss = [0-9.]*' | grep -o '[0-9.]*')
        time_per=$(echo "$last_line" | grep -o 'time = [0-9.]*' | grep -o '[0-9.]*')
        elapsed=$(echo "$last_line" | grep -o 'elapsed = [0-9]*' | grep -o '[0-9]*')

        # Progress bar
        pct=$((step * 100 / 6000))
        bar_len=30
        filled=$((pct * bar_len / 100))
        empty=$((bar_len - filled))
        bar=$(printf '%*s' "$filled" | tr ' ' '#')$(printf '%*s' "$empty" | tr ' ' '.')

        # ETA
        if [ -n "$elapsed" ] && [ "$step" -gt 0 ]; then
            eta_s=$(( (6000 - step) * elapsed / step ))
            eta_h=$((eta_s / 3600))
            eta_m=$(( (eta_s % 3600) / 60 ))
            eta="${eta_h}h${eta_m}m"
        else
            eta="?"
        fi

        printf "\r  [%s] %d/6000 (%d%%)  loss=%.4f  %ss/step  ETA: %s  " \
               "$bar" "$step" "$pct" "$loss" "$time_per" "$eta"
    fi

    # Regenerate plot periodically
    now=$(date +%s)
    if (( now - last_plot >= PLOT_INTERVAL )); then
        python3 "$PLOT_SCRIPT" "$LOG" 2>/dev/null || true
        last_plot=$now
    fi

    sleep 10
done

echo ""
echo "=== Monitor exited ==="
