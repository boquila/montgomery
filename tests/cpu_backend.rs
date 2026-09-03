//! Experimental CPU-backend measurements.
//!
//! Two configurations were measured against the default Flex backend with the same methodology
//! (zeros `[1, 3, 640, 640]` input, 3 warmup runs + 10 timed runs of `model.forward` — body,
//! head decode, and result sums forced to completion — median and minimum, sequential execution):
//!
//! 1. `cpu-simd` — Flex with the gemm crate's AVX-512 codegen paths (`burn-flex/x86-v4`).
//!    Numerically sound (all in-tree Flex golden tests pass) but consistently ~2-4% slower. It
//!    needs no extra tests: the in-tree Flex latency and golden tests cover it.
//! 2. `cpu-cubecl` — the alternative `burn-cpu` (CubeCL CPU) backend via `burn/cpu`. Measured
//!    here. Verdict: **numerically unsound on this graph** on burn 0.21.0-pre.4.
//!
//! Root cause of the CubeCL mismatch (element-wise probes against Flex, kept below as
//! [`cubecl_cpu_reshape_after_conv_is_unsound`]): the CubeCL CPU backend mis-handles
//! `reshape` on a conv output tensor. Every detection tower's final 1x1 conv output is
//! reshaped from `[batch, channels, height, width]` to `[batch, channels, anchors]` inside
//! the detection heads; after that reshape the values no longer match Flex (every op and
//! every tower matches when the output is read without the reshape). This scrambles the raw
//! predictions of all three Ultralytics families, so the golden parity tests fail far beyond
//! tolerance (e.g. yolo11n raw scores: Flex mean -15.16 vs CubeCL +1.70). Re-test these tests
//! when upgrading Burn; do not enable `cpu-cubecl` for inference until they pass.
//!
//! Run with:
//!
//! ```console
//! cargo test --release --features cpu-cubecl cubecl_cpu -- --ignored --nocapture --test-threads 1
//! ```
#![cfg(feature = "cpu-cubecl")]

use std::collections::BTreeMap;
use std::time::Instant;

use burn::backend::cpu::Cpu;
use burn::tensor::module;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Device, ElementConversion, Tensor, TensorData, backend::Backend};
use burn_flex::Flex;
use montgomery::models::yolo11::{
    Yolo11N, Yolo11NConfig, Yolo11S, Yolo11SConfig, Yolo11X, Yolo11XConfig,
};
use montgomery::models::yolo26::{
    Yolo26N, Yolo26NConfig, Yolo26S, Yolo26SConfig, Yolo26X, Yolo26XConfig,
};
use montgomery::models::yolov10::{
    Yolov10N, Yolov10NConfig, Yolov10S, Yolov10SConfig, Yolov10X, Yolov10XConfig,
};
use serde::Deserialize;

const WARMUP_RUNS: usize = 3;
const TIMED_RUNS: usize = 10;

/// The one operation every latency-harness model must expose: `forward` split into its
/// box and score outputs so the result can be synchronized like the Flex harness does.
trait LatencyForward<B: Backend> {
    fn forward_latency(&self, input: Tensor<B, 4>) -> (Tensor<B, 3>, Tensor<B, 3>);
}

macro_rules! latency_forward {
    ($($model:ident),* $(,)?) => {
        $(
            impl<B: Backend> LatencyForward<B> for $model<B> {
                fn forward_latency(&self, input: Tensor<B, 4>) -> (Tensor<B, 3>, Tensor<B, 3>) {
                    let output = self.forward(input);
                    (output.boxes, output.scores)
                }
            }
        )*
    };
}

latency_forward!(
    Yolo11N, Yolo11S, Yolo11X, Yolov10N, Yolov10S, Yolov10X, Yolo26N, Yolo26S, Yolo26X,
);

fn artifact(id: &str) -> std::path::PathBuf {
    let checkpoint = std::path::PathBuf::from(format!("target/{id}.bpk"));
    assert!(
        checkpoint.exists(),
        "pack the {id} artifact with pack-weights first"
    );
    checkpoint
}

/// Run the body of a test on a worker thread with the same 64 MB stack the in-tree tests use.
fn with_big_stack<T: Send + 'static>(body: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(body)
        .unwrap()
        .join()
        .unwrap()
}

/// The shared latency harness: first-run cost (JIT/autotune on this backend), then 3 warmups
/// and 10 timed runs with the sums of both outputs forced to completion, median + min printed.
fn measure_latency<B: Backend, M: LatencyForward<B>>(id: &str, model: &M, device: &Device<B>) {
    let input = Tensor::<B, 4>::zeros([1, 3, 640, 640], device);

    let run = |input: Tensor<B, 4>| {
        let (boxes, scores) = model.forward_latency(input);
        let _ = boxes.sum().into_data();
        let _ = scores.sum().into_data();
    };

    let started = Instant::now();
    run(input.clone());
    let first_run = started.elapsed();

    for _ in 0..WARMUP_RUNS {
        run(input.clone());
    }
    let mut samples = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        let started = Instant::now();
        run(input.clone());
        samples.push(started.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(|a, b| a.total_cmp(&b));
    let median = samples[samples.len() / 2];
    let min = samples[0];
    println!(
        "{id:>9}: {median:>7.1} ms median, {min:>7.1} ms min  (single image, batch 1, 640 px, {TIMED_RUNS} runs, burn-cpu CubeCL; first run incl. JIT/autotune: {} ms)",
        first_run.as_secs_f64() * 1e3,
    );
}

#[derive(Deserialize)]
struct GoldenFixture {
    format: String,
    model: String,
    tensors: BTreeMap<String, GoldenTensor>,
}

#[derive(Deserialize)]
struct GoldenTensor {
    shape: Vec<usize>,
    mean: f64,
    rms: f64,
    min: f64,
    max: f64,
    samples: Vec<(usize, f64)>,
}

/// Same statistics-based golden comparison as the in-tree Flex tests (2e-4 relative tolerance).
fn assert_golden<B: Backend, const D: usize>(
    name: &str,
    actual: Tensor<B, D>,
    expected: &GoldenTensor,
) {
    assert_eq!(actual.dims().to_vec(), expected.shape, "{name} shape");
    let values: Vec<f64> = actual
        .into_data()
        .iter::<f32>()
        .map(|value| value.elem::<f64>())
        .collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let rms = (values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64).sqrt();
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let close =
        |actual: f64, expected: f64| (actual - expected).abs() <= 2e-4 + expected.abs() * 2e-4;

    assert!(
        close(mean, expected.mean),
        "{name} mean: {mean} != {}",
        expected.mean
    );
    assert!(
        close(rms, expected.rms),
        "{name} rms: {rms} != {}",
        expected.rms
    );
    assert!(
        close(min, expected.min),
        "{name} min: {min} != {}",
        expected.min
    );
    assert!(
        close(max, expected.max),
        "{name} max: {max} != {}",
        expected.max
    );
    for &(index, expected_value) in &expected.samples {
        let actual_value = values[index];
        assert!(
            close(actual_value, expected_value),
            "{name}[{index}]: {actual_value} != {expected_value}"
        );
    }
}

fn load_fixture(id: &str) -> GoldenFixture {
    let fixture: GoldenFixture =
        serde_json::from_slice(&std::fs::read(format!("target/{id}-golden-v1.json")).unwrap())
            .unwrap();
    assert_eq!(fixture.format, "montgomery-ultralytics-golden-v1");
    assert_eq!(fixture.model, id);
    fixture
}

fn load_reference_image<B: Backend>(id: &str, device: &Device<B>) -> Tensor<B, 4> {
    let image = image::open(format!("target/{id}-preprocessed-reference.png"))
        .unwrap()
        .into_rgb8();
    let shape = [image.height() as usize, image.width() as usize, 3];
    Tensor::<B, 3>::from_data(
        TensorData::new(image.into_raw(), shape).convert::<f32>(),
        device,
    )
    .permute([2, 0, 1])
    .unsqueeze::<4>()
        / 255.0
}

macro_rules! cubecl_latency_test {
    ($fn_name:ident, $init:expr, $id:literal) => {
        #[test]
        #[ignore]
        fn $fn_name() {
            with_big_stack(move || {
                let device = Device::<Cpu>::default();
                let mut model = $init.init::<Cpu>(&device);
                model.load_burnpack_weights(artifact($id)).unwrap();
                measure_latency($id, &model, &device);
            });
        }
    };
}

cubecl_latency_test!(cubecl_cpu_yolo11n_latency, Yolo11NConfig, "yolo11n");
cubecl_latency_test!(cubecl_cpu_yolo11s_latency, Yolo11SConfig, "yolo11s");
cubecl_latency_test!(cubecl_cpu_yolo11x_latency, Yolo11XConfig, "yolo11x");
cubecl_latency_test!(cubecl_cpu_yolov10n_latency, Yolov10NConfig, "yolov10n");
cubecl_latency_test!(cubecl_cpu_yolov10s_latency, Yolov10SConfig, "yolov10s");
cubecl_latency_test!(cubecl_cpu_yolov10x_latency, Yolov10XConfig, "yolov10x");
cubecl_latency_test!(cubecl_cpu_yolo26n_latency, Yolo26NConfig, "yolo26n");
cubecl_latency_test!(cubecl_cpu_yolo26s_latency, Yolo26SConfig, "yolo26s");
cubecl_latency_test!(cubecl_cpu_yolo26x_latency, Yolo26XConfig, "yolo26x");

/// Golden parity for one variant per family on the CubeCL CPU backend: the full `forward`
/// (body + head decode) output statistics against the same fixture JSON the Flex golden tests
/// consume, with the same 2e-4 relative tolerance. These FAIL on burn 0.21.0-pre.4 — see the
/// module docs (reshape-after-conv bug) — and are kept as the upgrade gate for `cpu-cubecl`.
#[test]
#[ignore]
fn cubecl_cpu_yolo11n_matches_golden() {
    with_big_stack(move || {
        let device = Device::<Cpu>::default();
        let fixture = load_fixture("yolo11n");
        let mut model = Yolo11NConfig::default().init::<Cpu>(&device);
        model.load_burnpack_weights(artifact("yolo11n")).unwrap();
        let input = load_reference_image("yolo11n", &device);
        let decoded = model.forward(input);
        assert_golden(
            "decoded_boxes_cxcywh",
            decoded.boxes,
            fixture.tensors.get("decoded_boxes_cxcywh").unwrap(),
        );
        assert_golden(
            "decoded_scores",
            decoded.scores,
            fixture.tensors.get("decoded_scores").unwrap(),
        );
    });
}

#[test]
#[ignore]
fn cubecl_cpu_yolov10n_matches_golden() {
    with_big_stack(move || {
        let device = Device::<Cpu>::default();
        let fixture = load_fixture("yolov10n");
        let mut model = Yolov10NConfig::default().init::<Cpu>(&device);
        model.load_burnpack_weights(artifact("yolov10n")).unwrap();
        let input = load_reference_image("yolov10n", &device);
        let decoded = model.forward(input);
        assert_golden(
            "decoded_boxes_xyxy",
            decoded.boxes,
            fixture.tensors.get("decoded_boxes_xyxy").unwrap(),
        );
        assert_golden(
            "decoded_scores",
            decoded.scores,
            fixture.tensors.get("decoded_scores").unwrap(),
        );
    });
}

#[test]
#[ignore]
fn cubecl_cpu_yolo26n_matches_golden() {
    with_big_stack(move || {
        let device = Device::<Cpu>::default();
        let fixture = load_fixture("yolo26n");
        let mut model = Yolo26NConfig::default().init::<Cpu>(&device);
        model.load_burnpack_weights(artifact("yolo26n")).unwrap();
        let input = load_reference_image("yolo26n", &device);
        let decoded = model.forward(input);
        assert_golden(
            "decoded_boxes_xyxy",
            decoded.boxes,
            fixture.tensors.get("decoded_boxes_xyxy").unwrap(),
        );
        assert_golden(
            "decoded_scores",
            decoded.scores,
            fixture.tensors.get("decoded_scores").unwrap(),
        );
    });
}

/// Minimal reproduction of the CubeCL CPU reshape-after-conv unsoundness: a biased 1x1 conv
/// over the reference-image-sized input followed by `reshape([1, 64, 5120])` diverges from
/// Flex, while the same conv without the reshape matches to f32 rounding. When upgrading
/// Burn, this test is the quick check for whether `cpu-cubecl` may be revisited.
#[test]
#[ignore]
fn cubecl_cpu_reshape_after_conv_is_unsound() {
    with_big_stack(move || {
        let flex_device = Device::<Flex>::default();
        let cpu_device = Device::<Cpu>::default();
        let input_data: Vec<f32> = (0..1 * 64 * 80 * 64)
            .map(|v| (v as f32 * 0.023).sin())
            .collect();
        let weight: Vec<f32> = (0..64 * 64)
            .map(|v| (v as f32 * 0.019).cos() * 0.05)
            .collect();
        let bias: Vec<f32> = (0..64).map(|v| (v as f32 * 0.05).sin() * 0.1).collect();

        let run = |reshape: bool| -> (Vec<f64>, Vec<f64>) {
            let flex = {
                let input = Tensor::<Flex, 4>::from_data(
                    TensorData::new(input_data.clone(), [1, 64, 80, 64]).convert::<f32>(),
                    &flex_device,
                );
                let out = module::conv2d(
                    input,
                    Tensor::<Flex, 4>::from_data(
                        TensorData::new(weight.clone(), [64, 64, 1, 1]).convert::<f32>(),
                        &flex_device,
                    ),
                    Some(Tensor::<Flex, 1>::from_data(
                        TensorData::new(bias.clone(), [64]).convert::<f32>(),
                        &flex_device,
                    )),
                    ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
                );
                let to_vec3 = |t: Tensor<Flex, 3>| {
                    t.into_data()
                        .iter::<f32>()
                        .map(f64::from)
                        .collect::<Vec<f64>>()
                };
                let to_vec4 = |t: Tensor<Flex, 4>| {
                    t.into_data()
                        .iter::<f32>()
                        .map(f64::from)
                        .collect::<Vec<f64>>()
                };
                if reshape {
                    to_vec3(out.reshape([1, 64, 5120]))
                } else {
                    to_vec4(out)
                }
            };
            let cpu = {
                let input = Tensor::<Cpu, 4>::from_data(
                    TensorData::new(input_data.clone(), [1, 64, 80, 64]).convert::<f32>(),
                    &cpu_device,
                );
                let out = module::conv2d(
                    input,
                    Tensor::<Cpu, 4>::from_data(
                        TensorData::new(weight.clone(), [64, 64, 1, 1]).convert::<f32>(),
                        &cpu_device,
                    ),
                    Some(Tensor::<Cpu, 1>::from_data(
                        TensorData::new(bias.clone(), [64]).convert::<f32>(),
                        &cpu_device,
                    )),
                    ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
                );
                let to_vec3 = |t: Tensor<Cpu, 3>| {
                    t.into_data()
                        .iter::<f32>()
                        .map(f64::from)
                        .collect::<Vec<f64>>()
                };
                let to_vec4 = |t: Tensor<Cpu, 4>| {
                    t.into_data()
                        .iter::<f32>()
                        .map(f64::from)
                        .collect::<Vec<f64>>()
                };
                if reshape {
                    to_vec3(out.reshape([1, 64, 5120]))
                } else {
                    to_vec4(out)
                }
            };
            (flex, cpu)
        };

        let (flex_plain, cpu_plain) = run(false);
        let (flex_reshaped, cpu_reshaped) = run(true);
        let diff = |a: &[f64], b: &[f64]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f64, f64::max)
        };
        println!(
            "1x1 conv [1,64,80,64]: no-reshape max|diff| = {:.6}, after reshape([1,64,5120]) max|diff| = {:.6}",
            diff(&flex_plain, &cpu_plain),
            diff(&flex_reshaped, &cpu_reshaped)
        );
        // The conv itself is fine; only the reshape diverges.
        assert!(diff(&flex_plain, &cpu_plain) < 1e-3);
        assert!(diff(&flex_reshaped, &cpu_reshaped) > 0.5);
    });
}
