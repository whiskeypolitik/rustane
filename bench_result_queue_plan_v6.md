# Bench Result Queue — Implementation Plan

## Goal

Replace single-file overwrite (`target/bench-result.json`) with directory queue (`target/bench-results/`). `make submit` submits all pending results in order, deletes each only after successful POST.

---

## Changes

### [MODIFY] [bench_result.rs](file:///Users/andrewgordon/RustRover-Projects/rustane/crates/engine/src/bench_result.rs)

Change `write_result` (L96):
- Filename: `<YYYYMMDD_HHMMSS>_<benchmark-slug>_<fingerprint[..8]>.json` (e.g. `20260326_111500_train-600m_a3f7c012.json`). The 8-char fingerprint prefix guarantees uniqueness even for same-second results.
- **Slug sanitization:** `benchmark` field mapped to `[A-Za-z0-9._-]` with all other characters replaced by `_`. Applied in `write_result` before constructing the filename.
- **Atomic write:** write to a temp file (`.tmp` extension) in the same directory, then `std::fs::rename` into the final `.json` name. A crash mid-write leaves only a `.tmp` file, which `make submit` ignores (only globs `*.json`).
- `create_dir_all("target/bench-results/")`
- Print: `"📊 Result queued in target/bench-results/  (N pending)"`

No changes to `BenchResult` struct, callers, fingerprint, or any other function. The 5 existing call sites (`bench_fwd_only_scale.rs`, `bench_forward_multistream.rs`, `bench_param_sweep.rs`, `theory_runner.rs`) call `write_result` unchanged.

---

### [MODIFY] [Makefile](file:///Users/andrewgordon/RustRover-Projects/rustane/Makefile) — `submit` target (L111–149)

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
	SUBMITTED=0; SKIPPED=0; \
	for f in $$(ls target/bench-results/*.json | sort); do \
		BENCH=$$(jq -r '.benchmark' "$$f"); \
		FP=$$(jq -r '.fingerprint' "$$f"); \
		CHECK_BODY=$$(curl -sf "$(LEADERBOARD_API)/api/check?fingerprint=$$FP" 2>/dev/null) && \
		CHECK_EXISTS=$$(echo "$$CHECK_BODY" | jq -r '.exists // false') || \
		CHECK_EXISTS="false"; \
		if [ "$$CHECK_EXISTS" = "true" ]; then \
			echo "  ⏭  $$BENCH already submitted (fingerprint match), removing"; \
			rm -f "$$f"; \
			SKIPPED=$$((SKIPPED + 1)); \
			continue; \
		fi; \
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
			echo "  Stopping. $$SUBMITTED submitted, $$SKIPPED skipped, remaining kept."; \
			exit 1; \
		fi; \
	done; \
	echo ""; \
	echo "$$SUBMITTED submitted, $$SKIPPED skipped (already on server)."
```

Key behaviors:
- **Submitter prompt once**, reused for all files
- **Sorted by filename** (timestamp-grouped, deterministic within same second — not strict enqueue order)
- **Delete each file only after successful POST** (`rm -f "$f"` inside success branch)
- **Stop on first POST failure**, keep remaining files for retry
- **Fingerprint preflight:** before POSTing, queries `/api/check?fingerprint=<fp>`. Endpoint contract: returns `{"exists": true}` or `{"exists": false}`. Client behavior:
  - Response parsed, `exists == true` → skip file and delete locally
  - `curl` fails (`-sf` returns non-zero), endpoint absent (404), timeout, 5xx, invalid JSON, or `exists` field missing → `CHECK_EXISTS` defaults to `"false"`, proceed with POST. Preflight is best-effort, not a gate — server-side dedup is the backstop.
- **Server-side prerequisite:** the leaderboard API should reject duplicate fingerprints on POST. Add before rolling out batch submit.
- X share prompt removed from loop (too noisy for batch)

---

## Backward compat

- Old `target/bench-result.json` is no longer written. If it exists from a prior run, `make submit` won't see it (only reads from `bench-results/`). Add a one-time migration note to README or print a warning if the old file exists.
- `.gitignore` already covers `target/` so no change needed.

## Verify

- Run `make sweep-600m` twice → two files in `target/bench-results/`
- `make submit` submits both in order, directory empty after
- Kill network mid-submit → first file deleted (if response received), second survives
- Re-submit after network restore → server deduplicates by fingerprint if first was already accepted
- `make submit` with no files → clean error message
- Benchmark name with spaces/special chars → slug sanitized in filename
