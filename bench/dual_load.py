#!/usr/bin/env python3
"""Phase 0.5.4: Memory bandwidth under dual ANE + GPU load.

Measures whether ANE and GPU compete for memory bandwidth on M4 Max.

Usage:
    python3 bench/dual_load.py

Requires: mlx installed, cargo + ane-bridge tests compiled (release mode).
"""

import subprocess
import threading
import time
import os

# Ensure we can find cargo
os.environ["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + os.environ["PATH"]

def gpu_matmul_loop(duration_sec: float, results: dict):
    """Run MLX matmul in a tight loop for `duration_sec` seconds."""
    try:
        import mlx.core as mx
    except ImportError:
        results["gpu_error"] = "mlx not installed"
        return

    m, n, k = 4096, 4096, 4096
    a = mx.random.normal((m, k)).astype(mx.float16)
    b = mx.random.normal((k, n)).astype(mx.float16)
    mx.eval(a, b)

    # Warmup
    for _ in range(5):
        c = a @ b
        mx.eval(c)

    count = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration_sec:
        c = a @ b
        mx.eval(c)
        count += 1

    elapsed = time.perf_counter() - t0
    flops = 2 * m * n * k
    tflops = flops * count / elapsed / 1e12
    results["gpu_tflops"] = tflops
    results["gpu_iters"] = count
    results["gpu_sec"] = elapsed


def ane_benchmark(results: dict):
    """Run the ANE conv1x1 benchmark and parse output."""
    try:
        proc = subprocess.run(
            ["cargo", "test", "-p", "ane-bridge", "--test", "bench_ane_tflops",
             "--release", "--", "--include-ignored", "--nocapture"],
            capture_output=True, text=True, timeout=120,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        output = proc.stdout + proc.stderr
        results["ane_output"] = output

        # Parse the FFN up line (768→3072)
        for line in output.split("\n"):
            if "768→3072" in line and "FFN up" in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    try:
                        val = float(p)
                        if 0.001 < val < 100:  # TFLOPS range
                            results["ane_tflops"] = val
                            break
                    except ValueError:
                        continue
    except Exception as e:
        results["ane_error"] = str(e)


def main():
    print("\n=== Phase 0.5.4: Dual Load (ANE + GPU) Bandwidth Test ===\n")

    # ── 1. ANE standalone ──────────────────────────────────────────
    print("1. Running ANE benchmark (standalone)...")
    ane_solo = {}
    ane_benchmark(ane_solo)
    ane_solo_tflops = ane_solo.get("ane_tflops", 0)
    if "ane_error" in ane_solo:
        print(f"   ANE ERROR: {ane_solo['ane_error']}")
        return
    print(f"   ANE standalone: {ane_solo_tflops:.3f} TFLOPS (768→3072)")
    print()

    # ── 2. GPU standalone ──────────────────────────────────────────
    print("2. Running GPU benchmark (standalone, 5s)...")
    gpu_solo = {}
    gpu_matmul_loop(5.0, gpu_solo)
    if "gpu_error" in gpu_solo:
        print(f"   GPU ERROR: {gpu_solo['gpu_error']}")
        return
    print(f"   GPU standalone: {gpu_solo['gpu_tflops']:.3f} TFLOPS (4K×4K fp16)")
    print()

    # ── 3. Dual load ──────────────────────────────────────────────
    print("3. Running BOTH simultaneously...")
    gpu_dual = {}
    ane_dual = {}

    gpu_thread = threading.Thread(target=gpu_matmul_loop, args=(30.0, gpu_dual))
    gpu_thread.start()

    # Small delay to let GPU saturate
    time.sleep(1.0)

    # Run ANE benchmark while GPU is active
    ane_benchmark(ane_dual)

    gpu_thread.join()

    ane_dual_tflops = ane_dual.get("ane_tflops", 0)
    gpu_dual_tflops = gpu_dual.get("gpu_tflops", 0)

    print(f"   ANE under load: {ane_dual_tflops:.3f} TFLOPS")
    print(f"   GPU under load: {gpu_dual_tflops:.3f} TFLOPS")
    print()

    # ── 4. Summary ─────────────────────────────────────────────────
    print("=== Results ===")
    print(f"{'':30} {'Standalone':>12} {'Dual Load':>12} {'Degradation':>12}")
    print("-" * 70)

    if ane_solo_tflops > 0 and ane_dual_tflops > 0:
        ane_deg = (1 - ane_dual_tflops / ane_solo_tflops) * 100
        print(f"{'ANE (768→3072 conv1x1)':30} {ane_solo_tflops:>12.3f} {ane_dual_tflops:>12.3f} {ane_deg:>11.1f}%")

    if gpu_solo["gpu_tflops"] > 0 and gpu_dual_tflops > 0:
        gpu_deg = (1 - gpu_dual_tflops / gpu_solo["gpu_tflops"]) * 100
        print(f"{'GPU (4K×4K fp16 matmul)':30} {gpu_solo['gpu_tflops']:>12.3f} {gpu_dual_tflops:>12.3f} {gpu_deg:>11.1f}%")

    print()
    print("If degradation is <5%, ANE and GPU can run concurrently without contention.")
    print("If >10%, shared memory bandwidth is the bottleneck.")
    print()


if __name__ == "__main__":
    main()
