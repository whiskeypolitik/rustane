#!/bin/bash
# optimize-loop.sh — Autonomous optimization loop for rustane
#
# Runs Claude in headless mode in a git worktree. One experiment per iteration.
# State persists in filesystem (system/experiments.tsv, gossip file, git commits).
# Each Claude invocation is stateless — reads context from files.
#
# Usage:
#   ./system/optimize-loop.sh                        # 20 iters, sonnet
#   ./system/optimize-loop.sh --iters 1              # single iteration
#   ./system/optimize-loop.sh --dry-run              # print prompt, don't run
#   ./system/optimize-loop.sh --model opus --iters 3 # 3 iters with opus
#   ./system/optimize-loop.sh --id beta              # second agent (gossip-aware)
#
# In tmux:
#   tmux new -s opt-alpha './system/optimize-loop.sh --id alpha'
#   tmux new -s opt-beta  './system/optimize-loop.sh --id beta'
#
# Stop:  Ctrl-C | touch /tmp/rustane-opt-STOP | tmux kill-session -t opt-alpha
# Logs:  tail -f /tmp/rustane-opt-alpha.log

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# --- Config ---
TARGET_MS=89
MAX_ITERS=20
COOLDOWN=10
MODEL="claude-sonnet-4-6"
AGENT_ID="alpha"
DRY_RUN=false
WORKTREE_BASE="/tmp/rustane-opt"
GOSSIP_FILE="/tmp/rustane-gossip.md"

# --- Parse Args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=true; shift ;;
        --model)    MODEL="$2"; shift 2 ;;
        --iters)    MAX_ITERS="$2"; shift 2 ;;
        --target)   TARGET_MS="$2"; shift 2 ;;
        --id)       AGENT_ID="$2"; shift 2 ;;
        *)          echo "Unknown arg: $1"; exit 1 ;;
    esac
done

BRANCH="phase5/auto-opt-${AGENT_ID}"
WORKTREE="${WORKTREE_BASE}-${AGENT_ID}"
LOGFILE="/tmp/rustane-opt-${AGENT_ID}.log"
STOPFILE="/tmp/rustane-opt-STOP"

# --- Functions ---
log() {
    echo "[$(date '+%H:%M:%S')] [${AGENT_ID}] $*" | tee -a "$LOGFILE"
}

get_latest_ms() {
    # Extract the most recent numeric ms/step from experiments.tsv
    # Skips rows with '-', '~', 'PLANNED', non-numeric values
    tail -20 system/experiments.tsv 2>/dev/null \
        | grep -v '^-' \
        | grep -v 'PLANNED' \
        | awk -F'\t' '$7 ~ /^[0-9]+$/ { ms=$7 } END { print ms }' \
        2>/dev/null
}

gossip_write() {
    echo "[$(date '+%H:%M:%S')] [${AGENT_ID}] $*" >> "$GOSSIP_FILE"
}

gossip_read() {
    tail -30 "$GOSSIP_FILE" 2>/dev/null || echo "(no gossip yet)"
}

cleanup_worktree() {
    log "Cleaning up worktree at ${WORKTREE}"
    cd "$REPO_ROOT"
    git worktree remove --force "$WORKTREE" 2>/dev/null || true
}

# --- Signal Handling ---
INTERRUPTED=false
on_signal() {
    INTERRUPTED=true
    log "Signal received — finishing current iteration then exiting."
}
trap on_signal SIGINT SIGTERM

# --- Setup Worktree ---
log "=== optimize-loop.sh starting ==="
log "Model: ${MODEL} | Target: ${TARGET_MS}ms | Max: ${MAX_ITERS} iters | Agent: ${AGENT_ID}"

# Fetch latest from origin so we branch from up-to-date master
log "Fetching origin..."
git -C "$REPO_ROOT" fetch origin --quiet

# Clean up any stale worktree from a previous run
if [ -d "$WORKTREE" ]; then
    log "Removing stale worktree at ${WORKTREE}"
    cd "$REPO_ROOT"
    git worktree remove --force "$WORKTREE" 2>/dev/null || rm -rf "$WORKTREE"
fi

# Create or rebase branch onto latest origin/master
if ! git -C "$REPO_ROOT" show-ref --quiet "refs/heads/${BRANCH}"; then
    git -C "$REPO_ROOT" branch "$BRANCH" origin/master
    log "Created branch ${BRANCH} from origin/master"
else
    log "Rebasing existing branch ${BRANCH} onto origin/master"
    git -C "$REPO_ROOT" checkout "$BRANCH" --quiet
    git -C "$REPO_ROOT" rebase origin/master --quiet || {
        log "WARNING: rebase failed, resetting to origin/master"
        git -C "$REPO_ROOT" rebase --abort 2>/dev/null || true
        git -C "$REPO_ROOT" reset --hard origin/master --quiet
    }
    git -C "$REPO_ROOT" checkout - --quiet 2>/dev/null || true
fi

# Create the worktree
git -C "$REPO_ROOT" worktree add "$WORKTREE" "$BRANCH"
log "Worktree created at ${WORKTREE}"

# Copy dev/ context files into worktree (gitignored, not in worktree otherwise)
# These are read-only context for the agent. Mutable state is in system/experiments.tsv (tracked).
if [ -d "${REPO_ROOT}/dev" ]; then
    mkdir -p "${WORKTREE}/dev"
    for f in CURRENT.md METHODOLOGY.md; do
        [ -f "${REPO_ROOT}/dev/$f" ] && cp "${REPO_ROOT}/dev/$f" "${WORKTREE}/dev/$f"
    done
    [ -d "${REPO_ROOT}/dev/sessions" ] && cp -a "${REPO_ROOT}/dev/sessions" "${WORKTREE}/dev/sessions"
    log "Copied dev/ context files into worktree"
fi

# Work inside the worktree from now on
cd "$WORKTREE"
log "Working directory: $(pwd)"

# Initialize gossip file if it doesn't exist
if [ ! -f "$GOSSIP_FILE" ]; then
    cat > "$GOSSIP_FILE" << 'EOF'
# Rustane Optimization Gossip File
# Shared between parallel optimization agents.
# Format: [HH:MM:SS] [agent-id] message
# Agents read this to coordinate and avoid duplicate work.
EOF
    log "Created gossip file at ${GOSSIP_FILE}"
fi

gossip_write "ONLINE — starting optimization loop (model=${MODEL}, target=${TARGET_MS}ms)"

STARTING_MS=$(get_latest_ms)
log "Starting ms/step: ${STARTING_MS:-unknown}"

# --- The Prompt ---
read -r -d '' PROMPT << 'PROMPT_END' || true
You are an optimization agent for the rustane training engine on Apple M4 Max.
Your agent ID is: %%AGENT_ID%%
Goal: reduce ms/step to sub-89ms. Maderix Obj-C achieves 89ms.
Current state: ~125ms/step single-step (with accum=10: ~960ms/step estimated).
ANY reduction in ms/step is valuable and worth committing — even 2-3ms wins compound.

STEP 1 — READ CONTEXT (do this first, do not skip):
  - dev/CURRENT.md (current project state, what works, what's broken)
  - system/experiments.tsv (every experiment tried, results, verdicts)
  - dev/sessions/2026-03-12_perf-optimization.md (what worked, what failed, why)
  - dev/METHODOLOGY.md (rules: one variable, verify correctness, log everything)
  - results/rust_vs_objc_deep_comparison.md (root cause analysis of the perf gap)
  - CREDITS.md (reference implementations — especially Espresso for fused kernels,
    maderix/ANE for async dispatch, ane-infer for Metal decode)

STEP 1.5 — READ GOSSIP FILE:
  Read /tmp/rustane-gossip.md to see what other agents are working on or have tried.
  This file is shared between parallel optimization agents. Use it to:
  - Avoid picking an experiment another agent already claimed
  - Learn from other agents' findings (what worked, what's blocked, insights)
  - See if another agent found a blocker that affects your planned experiment

STEP 2 — PICK ONE EXPERIMENT:
Read system/experiments.tsv carefully. Choose the single highest-impact thing to try next.
Priority order (remaining opportunities from root cause analysis):
  1. Fused multi-layer ANE kernels — 2 layers per program, fewer dispatches (PLANNED, biggest remaining win)
  2. INT4 quantized classifier — packed nibble matvec, skip full logit buffer (PLANNED)
  3. Vectorize SiLU derivative — currently scalar loop ~0.8ms/layer, SIMD should be ~0.1ms
  4. Reduce IOSurface staging overhead — channel-interleaved layout eliminates per-channel copy loops
  5. Metal compute for backward CPU ops — move dW accumulation or RoPE to GPU
Already DONE (do not re-attempt): async ANE dispatch, pre-stage weights, allocation churn,
  softcap removal, parallel pre-staging, workspace path optimization.
Do NOT re-run experiments already marked PASS, IMPROVED, or REVERTED in the TSV.
Do NOT duplicate work another agent claimed in the gossip file.
If all high-priority items are blocked or done, pick the next logical thing from profiling data.

After picking, write your claim to the gossip file:
  echo "[$(date '+%H:%M:%S')] [%%AGENT_ID%%] CLAIMED: <experiment_name> — <one-line hypothesis>" >> /tmp/rustane-gossip.md

STEP 3 — IMPLEMENT:
  - Change ONE variable. Read the code you're modifying first.
  - Keep changes minimal and focused.
  - Do not refactor unrelated code.

STEP 3.5 — DESIGN A TEST FOR THIS SPECIFIC CHANGE:
  Before verifying correctness, write a targeted test that validates YOUR specific
  optimization is semantically equivalent to the original code. This is critical —
  the existing test suite was written before your change existed and may not cover it.

  Your test goes in: crates/engine/tests/auto_<experiment_name>.rs
  (e.g., auto_vectorize_silu.rs, auto_fused_2layer.rs)

  The test MUST:
  a) Capture the output (numerical values, gradients, or loss) of the ORIGINAL
     code path BEFORE your optimization runs. If you replaced code, keep the old
     version as a reference function inside the test.
  b) Run the OPTIMIZED code path on the same input.
  c) Assert they match within tolerance. Choose tolerance based on what changed:
     - Pure refactor (no math change): exact match (tolerance 0.0)
     - Reordered floating-point ops: tolerance 1e-5 (associativity differences)
     - Changed precision (f32↔f16 boundary): tolerance 1e-3
     - Algorithmic change (different formula, same result): tolerance 1e-2, justify in comment
  d) Test at least 2 edge cases specific to your change. Examples:
     - Vectorized a loop? Test with input size that's NOT a multiple of SIMD width.
     - Changed buffer reuse? Test with back-to-back calls (stale data check).
     - Changed dispatch order? Test that all outputs are still written.
     - Changed accumulation? Test with accum_steps=1 AND accum_steps=10.
  e) Run in under 10 seconds. It's a unit test, not a training run.
  f) Include a doc comment at the top explaining: what was optimized, what invariant
     this test checks, and what a failure means.

  Run your new test: cargo test -p engine --test auto_<name> --release -- --nocapture
  If YOUR OWN TEST fails: revert changes, log as BROKEN, skip to STEP 6.

STEP 4 — VERIFY CORRECTNESS (existing test suite — all must pass or REVERT):
  4a. Your new test from STEP 3.5 (already ran above, but confirm it's in the suite)
  4b. Phase 3 kernel tests — every ANE kernel compiles and runs:
      cargo test -p engine --test phase3_kernels --release
  4c. Integration — full model still learns:
      cargo test -p engine --test phase4_training --release
  4d. Full unit test suite:
      cargo test -p engine --release
  If ANY test fails: revert ALL changes INCLUDING your new test file
  (git checkout -- .) and skip to STEP 6.

STEP 5 — MEASURE PERFORMANCE:
  Run: cargo test -p engine --test profile_single_layer --release -- --ignored --nocapture
  Record: per-layer fwd_ms, bwd_ms, and total from output.
  Also run: cargo test -p engine --test bench_step_time --release -- --ignored --nocapture
  Record: ms/step, fwd_ms, bwd_ms, adam_ms from output.
  Compare against the baseline in system/experiments.tsv.

STEP 6 — LOG RESULTS:
  Append exactly ONE row to system/experiments.tsv matching the existing column format:
    date<TAB>experiment<TAB>variable<TAB>baseline<TAB>result<TAB>verdict<TAB>ms/step<TAB>fwd_ms<TAB>bwd_ms<TAB>adam_ms<TAB>loss_start<TAB>loss_end<TAB>val_bpb<TAB>notes
  Verdict must be one of: IMPROVED, REVERTED, NO EFFECT, BROKEN, INTERESTING
  Always log, even if the experiment failed — failed experiments are valuable data.
  IMPORTANT: ms/step column (column 7) must be a plain integer when available (e.g. 125),
  not prefixed with ~ or left as -. Use bench_step_time output for this value.

STEP 6.5 — UPDATE GOSSIP:
  Write your result to the gossip file:
    echo "[$(date '+%H:%M:%S')] [%%AGENT_ID%%] RESULT: <experiment> — <verdict>, <ms/step>ms. <one-line insight>" >> /tmp/rustane-gossip.md
  If you discovered something that would help other agents (a blocker, a surprising
  finding, a technique that worked), add a INSIGHT line:
    echo "[$(date '+%H:%M:%S')] [%%AGENT_ID%%] INSIGHT: <what you learned>" >> /tmp/rustane-gossip.md

STEP 7 — COMMIT OR REVERT:
  If IMPROVED and tests pass:
    - git add the changed source files + the new test file + system/experiments.tsv
    - git commit with a descriptive message: "Phase 5: <what changed> — <old>ms → <new>ms (<X>% faster)"
    - The test file is ALWAYS committed (even if the optimization is marginal) —
      it permanently protects the invariant for future iterations.
  If REVERTED/BROKEN/NO EFFECT:
    - git checkout -- . (revert code changes AND the test file)
    - git add system/experiments.tsv && git commit -m "Log experiment: <name> (REVERTED/NO EFFECT/etc)"
    - Exception: if the test revealed a pre-existing bug (not caused by your change),
      KEEP the test file, fix the bug, and log as a separate experiment.

STEP 8 — UPDATE STATE:
  If the experiment was significant (>5% improvement), update dev/CURRENT.md "What's Next" section.

RULES:
  - NEVER change test assertions to make tests pass
  - NEVER skip the correctness check (all tests in STEP 4)
  - NEVER change more than one variable per iteration
  - NEVER modify experiments.tsv rows that already exist — only append
  - If you're stuck or unsure, log a PLANNED row with your hypothesis and exit
  - Optimizations must be SEMANTICALLY equivalent — same math, fewer cycles.
    If your change alters numerical output (even slightly), it must be justified
    and noted in the system/experiments.tsv notes column. Precision changes compound:
    0.1% gradient error per layer x 6 layers x 1000 steps = training divergence.
  - The existing tests are a FAST GATE, not proof of correctness. That's why
    STEP 3.5 exists — your custom test is the real proof that YOUR change is safe.
    If your change touches backward math (gradients, accumulation, scaling),
    your custom test MUST compare gradient values, not just "does loss decrease."
  - Before writing your test, read existing auto_*.rs tests in crates/engine/tests/
    to see what patterns previous iterations used and what invariants are already covered.
    Build on them, don't duplicate them.
PROMPT_END

# Inject agent ID into prompt
PROMPT="${PROMPT//%%AGENT_ID%%/$AGENT_ID}"

# --- Main Loop ---
for i in $(seq 1 "$MAX_ITERS"); do
    log "--- Iteration $i / $MAX_ITERS ---"

    if $DRY_RUN; then
        echo ""
        echo "=== PROMPT THAT WOULD BE SENT ==="
        echo "$PROMPT"
        echo "================================="
        echo ""
        echo "Worktree: ${WORKTREE}"
        echo "Branch:   ${BRANCH}"
        echo "Gossip:   ${GOSSIP_FILE}"
        echo "Log:      ${LOGFILE}"
        echo "dev/ context: $(ls "${WORKTREE}/dev/" 2>/dev/null | tr '\n' ' ' || echo 'MISSING')"
        cleanup_worktree
        exit 0
    fi

    # Check stop conditions
    if $INTERRUPTED; then
        log "Exiting due to interrupt."
        break
    fi
    if [[ -f "$STOPFILE" ]]; then
        log "STOP file found (${STOPFILE}). Exiting."
        rm -f "$STOPFILE"
        break
    fi

    gossip_write "ITERATION $i starting"

    # Run Claude headless in the worktree directory
    log "Launching claude -p --model ${MODEL} ..."
    set +e
    CLAUDE_OUTPUT=$(cd "$WORKTREE" && claude -p \
        --dangerously-skip-permissions \
        --model "$MODEL" \
        "$PROMPT" 2>&1)
    CLAUDE_EXIT=$?
    set -e

    # Log Claude's output
    echo "--- Iteration $i output ---" >> "$LOGFILE"
    echo "$CLAUDE_OUTPUT" >> "$LOGFILE"
    echo "--- end ---" >> "$LOGFILE"

    if [ $CLAUDE_EXIT -ne 0 ]; then
        log "Claude exited with code $CLAUDE_EXIT. Pausing 30s."
        gossip_write "ERROR: claude exited with code $CLAUDE_EXIT"
        sleep 30
        continue
    fi

    # Sync experiments.tsv back to dev/ (keep local copy in sync)
    if [ -f "${WORKTREE}/system/experiments.tsv" ]; then
        cp "${WORKTREE}/system/experiments.tsv" "${REPO_ROOT}/dev/experiments.tsv" 2>/dev/null || true
    fi

    # Check current ms/step
    CURRENT_MS=$(get_latest_ms)
    log "After iteration $i: ms/step = ${CURRENT_MS:-unknown}"
    gossip_write "ITERATION $i done — ms/step = ${CURRENT_MS:-unknown}"

    # Check if we hit target
    if [[ -n "${CURRENT_MS}" ]] && [[ "${CURRENT_MS}" =~ ^[0-9]+$ ]] && [ "${CURRENT_MS}" -le "$TARGET_MS" ]; then
        log "TARGET HIT: ${CURRENT_MS}ms <= ${TARGET_MS}ms"
        gossip_write "TARGET HIT: ${CURRENT_MS}ms! Stopping."
        break
    fi

    log "Cooling down ${COOLDOWN}s..."
    sleep "$COOLDOWN"
done

gossip_write "OFFLINE — loop complete after $i iterations"
log "=== optimize-loop.sh complete ==="
log "Final ms/step: $(get_latest_ms)"
log "Worktree preserved at: ${WORKTREE}"
log "Branch: ${BRANCH}"
log "To merge wins: git merge ${BRANCH}"
log "To clean up:   git worktree remove ${WORKTREE}"
