//! Tiny isolated ANE perf-stats probe using the public `PerfStats` API.
//!
//! Run manually:
//!   cargo test -p engine --test bench_ane_perf_stats_probe --release -- --ignored --nocapture probe_stats_768
//!   cargo test -p engine --test bench_ane_perf_stats_probe --release -- --ignored --nocapture probe_stats_1536

use ane_bridge::ane::{Executable, PerfStats, Shape, TensorData};
use engine::kernels::dyn_matmul;
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

const WARMUP_RUNS: usize = 5;
const SAMPLE_RUNS: usize = 20;

#[derive(Clone, Copy)]
struct KernelSpec {
    name: &'static str,
    ic: usize,
    oc: usize,
    seq: usize,
}

fn make_io(spec: KernelSpec) -> (TensorData, TensorData) {
    let sp = dyn_matmul::spatial_width(spec.seq, spec.oc);
    let input_shape = Shape {
        batch: 1,
        channels: spec.ic,
        height: 1,
        width: sp,
    };
    let output_shape = Shape {
        batch: 1,
        channels: spec.oc,
        height: 1,
        width: spec.seq,
    };

    let input_data: Vec<f32> = (0..input_shape.total_elements())
        .map(|i| (i % 100) as f32 * 0.001)
        .collect();

    (
        TensorData::with_f32(&input_data, input_shape),
        TensorData::new(output_shape),
    )
}

fn compile_kernel(spec: KernelSpec) -> Executable {
    dyn_matmul::build(spec.ic, spec.oc, spec.seq)
        .compile(NSQualityOfService::UserInteractive)
        .expect("ANE compile failed")
}

fn warmup_cached(exe: &Executable, input: &TensorData, output: &TensorData) {
    for _ in 0..WARMUP_RUNS {
        exe.run_cached(&[input], &[output])
            .expect("cached warmup failed");
    }
}

fn warmup_perf(exe: &Executable, input: &TensorData, output: &TensorData) {
    for _ in 0..WARMUP_RUNS {
        let _ = exe
            .run_cached_with_perf_stats(&[input], &[output])
            .expect("perf warmup failed");
    }
}

fn collect_hw_only_samples(
    exe: &Executable,
    input: &TensorData,
    output: &TensorData,
) -> (Vec<u64>, Vec<f64>) {
    let mut hw_ns = Vec::with_capacity(SAMPLE_RUNS);
    let mut wall_us = Vec::with_capacity(SAMPLE_RUNS);
    for _ in 0..SAMPLE_RUNS {
        let t0 = Instant::now();
        let hw = exe
            .run_cached_with_stats(&[input], &[output])
            .expect("hw-only sample failed");
        wall_us.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
        hw_ns.push(hw);
    }
    (hw_ns, wall_us)
}

fn collect_perf_samples(
    exe: &Executable,
    input: &TensorData,
    output: &TensorData,
) -> (Vec<u64>, Vec<f64>, Option<PerfStats>) {
    let mut hw_ns = Vec::with_capacity(SAMPLE_RUNS);
    let mut wall_us = Vec::with_capacity(SAMPLE_RUNS);
    let mut sample = None;
    for _ in 0..SAMPLE_RUNS {
        let t0 = Instant::now();
        let perf = exe
            .run_cached_with_perf_stats(&[input], &[output])
            .expect("perf sample failed");
        wall_us.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
        hw_ns.push(perf.hw_execution_time_ns);
        if sample.is_none() {
            sample = Some(perf);
        }
    }
    (hw_ns, wall_us, sample)
}

fn summarize(values: &[u64]) -> (usize, u64, u64) {
    let nonzero = values.iter().filter(|&&v| v != 0).count();
    let min = values.iter().copied().min().unwrap_or(0);
    let max = values.iter().copied().max().unwrap_or(0);
    (nonzero, min, max)
}

fn mean_wall_us(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn format_top_counters(perf: &PerfStats) -> String {
    let mut counters = perf
        .counters
        .iter()
        .filter(|(_, value)| **value != 0)
        .map(|(name, value)| (name.as_str(), *value))
        .collect::<Vec<_>>();
    counters.sort_by(|lhs, rhs| rhs.1.cmp(&lhs.1).then_with(|| lhs.0.cmp(rhs.0)));
    counters
        .into_iter()
        .take(8)
        .map(|(name, value)| format!("{name}={value}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn print_phase(label: &str, hw_ns: &[u64], wall_us: &[f64]) {
    let (nonzero, min, max) = summarize(hw_ns);
    let first_ten = hw_ns.iter().take(10).copied().collect::<Vec<_>>();
    println!(
        "  {label}: nonzero={nonzero}/{total} min={min}ns max={max}ns mean_wall={wall:.1}us first10={first_ten:?}",
        total = hw_ns.len(),
        wall = mean_wall_us(wall_us),
    );
}

fn print_perf_snapshot(perf: &PerfStats) {
    println!(
        "  perf snapshot: hw={}ns bytes={} words={} named_counters={}",
        perf.hw_execution_time_ns,
        perf.counter_bytes.len(),
        perf.counter_words.len(),
        perf.counters.len()
    );
    let top = format_top_counters(perf);
    if !top.is_empty() {
        println!("  top counters: {top}");
    }
}

fn probe_kernel(spec: KernelSpec) {
    println!(
        "\n=== {} ({}x{}, seq={}) ===",
        spec.name, spec.ic, spec.oc, spec.seq
    );

    let (input_a, output_a) = make_io(spec);
    let exe_a = compile_kernel(spec);
    warmup_cached(&exe_a, &input_a, &output_a);
    let (hw_a, wall_a) = collect_hw_only_samples(&exe_a, &input_a, &output_a);
    print_phase("cached warmup -> hw-only", &hw_a, &wall_a);

    let (input_b, output_b) = make_io(spec);
    let exe_b = compile_kernel(spec);
    warmup_perf(&exe_b, &input_b, &output_b);
    let (hw_b, wall_b, perf_b) = collect_perf_samples(&exe_b, &input_b, &output_b);
    print_phase("perf warmup -> perf", &hw_b, &wall_b);
    if let Some(perf) = perf_b.as_ref() {
        print_perf_snapshot(perf);
    }

    let (input_c, output_c) = make_io(spec);
    let exe_c = compile_kernel(spec);
    let (hw_c, wall_c, perf_c) = collect_perf_samples(&exe_c, &input_c, &output_c);
    print_phase("perf only fresh", &hw_c, &wall_c);
    if let Some(perf) = perf_c.as_ref() {
        print_perf_snapshot(perf);
    }
}

#[test]
#[ignore]
fn probe_stats_768() {
    probe_kernel(KernelSpec {
        name: "woFwd",
        ic: 768,
        oc: 768,
        seq: 512,
    });
    probe_kernel(KernelSpec {
        name: "ffnBwdW2t",
        ic: 768,
        oc: 2048,
        seq: 512,
    });
}

#[test]
#[ignore]
fn probe_stats_1536() {
    probe_kernel(KernelSpec {
        name: "woFwd",
        ic: 1536,
        oc: 1536,
        seq: 512,
    });
    probe_kernel(KernelSpec {
        name: "ffnBwdW2t",
        ic: 1536,
        oc: 4096,
        seq: 512,
    });
}
