# Bench Result Queue — Implementation Plan

## Goal

Replace single-file overwrite (`target/bench-result.json`) with directory queue (`target/bench-results/`). `make submit` submits all pending results in order, deletes each only after successful POST.

---

## Changes

### [MODIFY] [bench_result.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/bench_result.rs)

Change `write_result` (L96):
- Filename: `<YYYYMMDD_HHMMSS>_<benchmark-slug>_<fingerprint[..8]>.json` (e.g. `20260326_111500_train-600m_a3f7c012.json`). The 8-char fingerprint prefix guarantees uniqueness even for same-second results.
- **Atomic write:** write to a temp file (`.tmp` extension) in the same directory, then `std::fs::rename` into the final `.json` name. A crash mid-write leaves only a `.tmp` file, which `make submit` ignores (only globs `*.json`).
- `create_dir_all("target/bench-results/")`
- Print: `"📊 Result queued in target/bench-results/  (N pending)"`

No changes to `BenchResult` struct, callers, fingerprint, or any other function. The 5 existing call sites (`bench_fwd_only_scale.rs`, `bench_forward_multistream.rs`, `bench_param_sweep.rs`, `theory_runner.rs`) call `write_result` unchanged.

---

### [MODIFY] [Makefile](file:///Users/USER/RustRover-Projects/rustane/Makefile) — `submit` target (L111–149)

Replace the current single-file submit with a loop:

```makefile
submit:
	@if [ -z "$$(ls target/bench-results/*.json 2>/dev/null)" ]; then \
		echo "No pending results. Run a benchmark first."; exit 1; \
	fi
	@COUNT=$$(ls target/bench-results/*.json | wc -l | tr -d ' '); \
	echo ""; \
	echo "$$COUNT pending result(s) in target/bench-results/"; \
	echo ""; \
	read -p "Your name: " NAME; \
	read -p "X handle (optional): " XHANDLE; \
	echo ""; \
	SUBMITTED=0; \
	for f in $$(ls target/bench-results/*.json | sort); do \
		BENCH=$$(jq -r '.benchmark' "$$f"); \
		echo "Submitting $$BENCH ($$f)..."; \
		TMPFILE=$$(mktemp); \
		jq --arg name "$$NAME" --arg x "$$XHANDLE" \
			'.submitter = {name: $$name, x_handle: $$x}' "$$f" > "$$TMPFILE"; \
		RESPONSE=$$(curl -s -X POST $(LEADERBOARD_API)/api/submit \
			-H "Content-Type: application/json" -d @"$$TMPFILE"); \
		rm -f "$$TMPFILE"; \
		RESULT_ID=$$(echo "$$RESPONSE" | jq -r '.id // empty'); \
		if [ -n "$$RESULT_ID" ]; then \
			echo "  ✅ $$BENCH → https://bench.rustane.org/?id=$$RESULT_ID"; \
			rm -f "$$f"; \
			SUBMITTED=$$((SUBMITTED + 1)); \
		else \
			echo "  ❌ $$BENCH failed:"; \
			echo "$$RESPONSE" | jq .; \
			echo "  Stopping. $$SUBMITTED submitted, remaining files kept."; \
			exit 1; \
		fi; \
	done; \
	echo ""; \
	echo "$$SUBMITTED result(s) submitted successfully."
```

Key behaviors:
- **Submitter prompt once**, reused for all files
- **Sorted by filename** (timestamp prefix = chronological order)
- **Delete each file only after successful POST** (`rm -f "$f"` inside success branch)
- **Stop on first failure**, keep remaining files for retry
- **X share prompt removed** from loop (too noisy for batch). Add back as a separate `make share` if wanted.

---

## Backward compat

- Old `target/bench-result.json` is no longer written. If it exists from a prior run, `make submit` won't see it (only reads from `bench-results/`). Add a one-time migration note to README or print a warning if the old file exists.
- `.gitignore` already covers `target/` so no change needed.

## Verify

- Run `make sweep-600m` twice → two files in `target/bench-results/`
- `make submit` submits both in order, directory empty after
- Kill network mid-submit → first file deleted, second file survives
- `make submit` with no files → clean error message
