//! Native Burn implementation of the Ultralytics YOLOv8-cls classification family (n/s/m/l/x).
//!
//! The classification graph differs structurally from YOLO26-cls/YOLO11-cls: the backbone is the
//! YOLOv8 C2f chain (layers 0-8) with **no** C2PSA stage and **no** SPPF, and Ultralytics'
//! `Classify` head sits at `model.9` (1x1 Conv to 1280 channels, global average pooling, one
//! linear layer to 1000 ImageNet classes). The scale rows differ too: every scale keeps
//! `max_channels` at 1024 and the n/s depth gain is 0.33 (not 0.50), so the m/l/x cap rule that
//! shapes the 26/11 classify bodies does not apply here.
//!
//! Like every Ultralytics classify-task checkpoint, the batch norms carry plain PyTorch defaults
//! (eps 1e-5, momentum 0.1) — see the BnFlavor invariant in AGENTS.md. The `Classify` head module
//! itself is shared with the YOLO26-cls family ([`crate::models::yolo26::classification`]).

use burn::{
    module::Module,
    tensor::{Device, Tensor, backend::Backend},
};

#[cfg(feature = "pretrained")]
use burn_store::{ModuleSnapshot, PytorchStore};

use super::blocks::{BnFlavor, C2f, C2fConfig, Conv, ConvConfig};
use crate::models::yolo26::classification::{
    ClassificationOutput, ClassifyHead, ClassifyHeadConfig,
};

/// Every Conv in the official YOLOv8-cls checkpoints carries plain PyTorch BatchNorm defaults
/// (eps 1e-5, momentum 0.1) instead of the Ultralytics-initialized values the detect family uses.
fn conv_cfg(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
) -> ConvConfig {
    ConvConfig::new(in_channels, out_channels, kernel_size, stride)
        .with_bn_flavor(BnFlavor::Pytorch)
}

/// YOLOv8-cls backbone (layers 0-8): a pure C2f chain with PyTorch batch-norm flavor.
#[derive(Module, Debug)]
pub struct Yolov8ClassifyBody<B: Backend> {
    model_0: Conv<B>,
    model_1: Conv<B>,
    model_2: C2f<B>,
    model_3: Conv<B>,
    model_4: C2f<B>,
    model_5: Conv<B>,
    model_6: C2f<B>,
    model_7: Conv<B>,
    model_8: C2f<B>,
}

impl<B: Backend> Yolov8ClassifyBody<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.model_0.forward(input);
        let x = self.model_1.forward(x);
        let x = self.model_2.forward(x);
        let x = self.model_3.forward(x);
        let x = self.model_4.forward(x);
        let x = self.model_5.forward(x);
        let x = self.model_6.forward(x);
        let x = self.model_7.forward(x);
        self.model_8.forward(x)
    }
}

/// Shared backbone construction table: stem/chain widths and the depth-scaled C2f repeats.
#[derive(Debug)]
pub struct Yolov8ClassifyBodyConfig {
    widths: [usize; 9],
    repeats: [usize; 4],
}

impl Yolov8ClassifyBodyConfig {
    fn init<B: Backend>(&self, device: &Device<B>) -> Yolov8ClassifyBody<B> {
        let [w0, w1, w2, w3, w4, w5, w6, w7, w8] = self.widths;
        let [r2, r4, r6, r8] = self.repeats;
        Yolov8ClassifyBody {
            model_0: conv_cfg(3, w0, 3, 2).init(device),
            model_1: conv_cfg(w0, w1, 3, 2).init(device),
            model_2: C2fConfig::new(w1, w2, r2, true)
                .with_bn_flavor(BnFlavor::Pytorch)
                .init(device),
            model_3: conv_cfg(w2, w3, 3, 2).init(device),
            model_4: C2fConfig::new(w3, w4, r4, true)
                .with_bn_flavor(BnFlavor::Pytorch)
                .init(device),
            model_5: conv_cfg(w4, w5, 3, 2).init(device),
            model_6: C2fConfig::new(w5, w6, r6, true)
                .with_bn_flavor(BnFlavor::Pytorch)
                .init(device),
            model_7: conv_cfg(w6, w7, 3, 2).init(device),
            model_8: C2fConfig::new(w7, w8, r8, true)
                .with_bn_flavor(BnFlavor::Pytorch)
                .init(device),
        }
    }
}

#[derive(Debug, Default)]
pub struct Yolov8ClassifyBodyNConfig;

impl Yolov8ClassifyBodyNConfig {
    fn init<B: Backend>(&self, device: &Device<B>) -> Yolov8ClassifyBody<B> {
        Yolov8ClassifyBodyConfig {
            widths: [16, 32, 32, 64, 64, 128, 128, 256, 256],
            repeats: [1, 2, 2, 1],
        }
        .init(device)
    }
}

#[derive(Debug, Default)]
pub struct Yolov8ClassifyBodySConfig;

impl Yolov8ClassifyBodySConfig {
    fn init<B: Backend>(&self, device: &Device<B>) -> Yolov8ClassifyBody<B> {
        Yolov8ClassifyBodyConfig {
            widths: [32, 64, 64, 128, 128, 256, 256, 512, 512],
            repeats: [1, 2, 2, 1],
        }
        .init(device)
    }
}

#[derive(Debug, Default)]
pub struct Yolov8ClassifyBodyMConfig;

impl Yolov8ClassifyBodyMConfig {
    fn init<B: Backend>(&self, device: &Device<B>) -> Yolov8ClassifyBody<B> {
        Yolov8ClassifyBodyConfig {
            widths: [48, 96, 96, 192, 192, 384, 384, 768, 768],
            repeats: [2, 4, 4, 2],
        }
        .init(device)
    }
}

#[derive(Debug, Default)]
pub struct Yolov8ClassifyBodyLConfig;

impl Yolov8ClassifyBodyLConfig {
    fn init<B: Backend>(&self, device: &Device<B>) -> Yolov8ClassifyBody<B> {
        Yolov8ClassifyBodyConfig {
            widths: [64, 128, 128, 256, 256, 512, 512, 1024, 1024],
            repeats: [3, 6, 6, 3],
        }
        .init(device)
    }
}

#[derive(Debug, Default)]
pub struct Yolov8ClassifyBodyXConfig;

impl Yolov8ClassifyBodyXConfig {
    fn init<B: Backend>(&self, device: &Device<B>) -> Yolov8ClassifyBody<B> {
        Yolov8ClassifyBodyConfig {
            widths: [80, 160, 160, 320, 320, 640, 640, 1280, 1280],
            repeats: [3, 6, 6, 3],
        }
        .init(device)
    }
}

classify_model!(
    Yolov8ClsN,
    Yolov8ClsNConfig,
    Yolov8ClassifyBody,
    Yolov8ClassifyBodyNConfig,
    256,
    "yolov8n-cls",
    "Native Burn YOLOv8n-cls model.",
    "Import tensor-only state exported from an official Ultralytics checkpoint."
);

classify_model!(
    Yolov8ClsS,
    Yolov8ClsSConfig,
    Yolov8ClassifyBody,
    Yolov8ClassifyBodySConfig,
    512,
    "yolov8s-cls",
    "Native Burn YOLOv8s-cls model.",
    "Import tensor-only state exported from an official Ultralytics checkpoint."
);

classify_model!(
    Yolov8ClsM,
    Yolov8ClsMConfig,
    Yolov8ClassifyBody,
    Yolov8ClassifyBodyMConfig,
    768,
    "yolov8m-cls",
    "Native Burn YOLOv8m-cls model.",
    "Import tensor-only state exported from an official Ultralytics checkpoint."
);

classify_model!(
    Yolov8ClsL,
    Yolov8ClsLConfig,
    Yolov8ClassifyBody,
    Yolov8ClassifyBodyLConfig,
    1024,
    "yolov8l-cls",
    "Native Burn YOLOv8l-cls model.",
    "Import tensor-only state exported from an official Ultralytics checkpoint."
);

classify_model!(
    Yolov8ClsX,
    Yolov8ClsXConfig,
    Yolov8ClassifyBody,
    Yolov8ClassifyBodyXConfig,
    1280,
    "yolov8x-cls",
    "Native Burn YOLOv8x-cls model.",
    "Import tensor-only state exported from an official Ultralytics checkpoint."
);

/// Build the PyTorch-state store shared by every YOLOv8-cls scale variant.
///
/// The backbone is layers 0-8 and the `Classify` head is model.9: one `Conv` (conv+bn) and one
/// `nn.Linear` per scale.
#[cfg(feature = "pretrained")]
fn pytorch_store(path: impl Into<std::path::PathBuf>) -> PytorchStore {
    PytorchStore::from_file(path)
        .with_top_level_key("model")
        // Backbone layers 0-8 keep their Ultralytics graph indices. The head is model.9, so the
        // single-digit rule must not match it.
        .with_key_remapping("model\\.([0-8])\\.(.+)", "body.model_$1.$2")
        // model.9.conv.{conv,bn}.* is the 1x1 classification convolution.
        .with_key_remapping("model\\.9\\.conv\\.conv\\.(.+)", "head.conv.conv.$1")
        .with_key_remapping("model\\.9\\.conv\\.bn\\.(.+)", "head.conv.bn.$1")
        // model.9.linear.* is the final classifier.
        .with_key_remapping("model\\.9\\.linear\\.(.+)", "head.linear.$1")
}

#[cfg(all(test, feature = "pretrained"))]
mod tests {
    use super::*;
    use burn::tensor::{ElementConversion, TensorData};
    use burn_flex::Flex;
    use serde::Deserialize;
    use std::collections::BTreeMap;

    #[cfg(feature = "gpu")]
    use burn::backend::Wgpu;

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

    fn assert_golden<const D: usize>(name: &str, actual: Tensor<Flex, D>, expected: &GoldenTensor) {
        assert_eq!(actual.dims().to_vec(), expected.shape, "{name} shape");
        let values: Vec<f64> = actual
            .into_data()
            .iter::<f32>()
            .map(|value| value.elem::<f64>())
            .collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let rms =
            (values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64).sqrt();
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

    fn load_reference_image(id: &str, device: &Device<Flex>) -> Tensor<Flex, 4> {
        let image = image::open(format!("target/{id}-preprocessed-reference.png"))
            .unwrap()
            .into_rgb8();
        let shape = [image.height() as usize, image.width() as usize, 3];
        Tensor::<Flex, 3>::from_data(
            TensorData::new(image.into_raw(), shape).convert::<f32>(),
            device,
        )
        .permute([2, 0, 1])
        .unsqueeze::<4>()
            / 255.0
    }

    macro_rules! checkpoint_test {
        ($fn_name:ident, $config:ty, $id:literal) => {
            /// Run manually after converting the official checkpoint. Kept ignored in CI because
            /// the source checkpoint is an external AGPL asset.
            #[test]
            #[ignore]
            fn $fn_name() {
                let checkpoint = std::path::PathBuf::from(concat!("target/", $id, "-state.pt"));
                assert!(
                    checkpoint.exists(),
                    "convert {}.pt with tools/export_ultralytics_state.py first",
                    $id
                );
                let worker = std::thread::Builder::new()
                    .stack_size(64 * 1024 * 1024)
                    .spawn(move || {
                        let device = Default::default();
                        let mut model = <$config>::default().init::<Flex>(&device);
                        model.load_pytorch_weights(checkpoint).unwrap();
                        let output = model.forward(Tensor::zeros([1, 3, 64, 64], &device));
                        assert_eq!(
                            output.probs.dims(),
                            [1, crate::models::yolo26::classification::NUM_CLASSES]
                        );
                    })
                    .unwrap();
                worker.join().unwrap();
            }
        };
    }

    macro_rules! golden_test {
        ($fn_name:ident, $config:ty, $id:literal) => {
            #[test]
            #[ignore]
            fn $fn_name() {
                let checkpoint = std::path::PathBuf::from(format!(
                    "target/{}",
                    crate::models::yolov8::weights::artifact_filename($id)
                ));
                let fixture: GoldenFixture = serde_json::from_slice(
                    &std::fs::read(format!("target/{}-golden-v1.json", $id)).unwrap_or_else(|_| {
                        panic!(
                            "generate fixtures with tools/export_yolov8_cls_fixtures.py --model {}",
                            $id
                        )
                    }),
                )
                .unwrap();
                assert_eq!(fixture.format, "montgomery-ultralytics-golden-v1");
                assert_eq!(fixture.model, $id);

                let worker = std::thread::Builder::new()
                    .stack_size(64 * 1024 * 1024)
                    .spawn(move || {
                        let device = Default::default();
                        let mut model = <$config>::default().init::<Flex>(&device);
                        model.load_burnpack_weights(checkpoint).unwrap();
                        let input = load_reference_image($id, &device);
                        let backbone = model.body.forward(input);
                        let output = model.head.forward(backbone.clone());

                        assert_golden(
                            "backbone_p5",
                            backbone,
                            fixture.tensors.get("backbone_p5").unwrap(),
                        );
                        assert_golden(
                            "logits",
                            output.logits,
                            fixture.tensors.get("logits").unwrap(),
                        );
                        assert_golden("probs", output.probs, fixture.tensors.get("probs").unwrap());
                    })
                    .unwrap();
                worker.join().unwrap();
            }
        };
    }

    macro_rules! latency_test {
        ($fn_name:ident, $config:ty, $id:literal) => {
            /// Measure single-image batch-1 inference latency with the packed native artifact on
            /// the Flex CPU backend at the family's 224 px classify input. Run with
            /// `cargo test --release <id> -- --ignored --nocapture` after the weight-prep loop.
            #[test]
            #[ignore]
            fn $fn_name() {
                let checkpoint = std::path::PathBuf::from(format!(
                    "target/{}",
                    crate::models::yolov8::weights::artifact_filename($id)
                ));
                assert!(
                    checkpoint.exists(),
                    "pack the {} artifact with pack-weights first",
                    $id
                );
                let worker = std::thread::Builder::new()
                    .stack_size(64 * 1024 * 1024)
                    .spawn(move || {
                        let device = Default::default();
                        let mut model = <$config>::default().init::<Flex>(&device);
                        model.load_burnpack_weights(checkpoint).unwrap();
                        let input = Tensor::<Flex, 4>::zeros([1, 3, 224, 224], &device);
                        const WARMUP_RUNS: usize = 3;
                        const TIMED_RUNS: usize = 10;

                        for _ in 0..WARMUP_RUNS {
                            let output = model.forward(input.clone());
                            let _ = output.probs.sum().into_data();
                        }
                        let mut samples = Vec::with_capacity(TIMED_RUNS);
                        for _ in 0..TIMED_RUNS {
                            let started = std::time::Instant::now();
                            let output = model.forward(input.clone());
                            let _ = output.probs.sum().into_data();
                            samples.push(started.elapsed().as_secs_f64() * 1e3);
                        }
                        samples.sort_by(|a, b| a.total_cmp(b));
                        let median = samples[samples.len() / 2];
                        let min = samples[0];
                        println!(
                            "{:>11}: {:>7.1} ms median, {:>7.1} ms min  (single image, batch 1, 224 px, {TIMED_RUNS} runs)",
                            $id, median, min,
                        );
                    })
                    .unwrap();
                worker.join().unwrap();
            }
        };
    }

    /// Compare the classification runtime end to end against the official Ultralytics prediction
    /// on the reference image (top-5 classes and probabilities). Run the generator first:
    /// `python tools/export_yolov8_cls_fixtures.py target/<id>.pt docs/dog_bike_man.jpg target --model <id>`
    macro_rules! cls_e2e_test {
        ($fn_name:ident, $config:ty, $id:literal) => {
            #[test]
            #[ignore]
            fn $fn_name() {
                use std::str::FromStr;
                let expected_path =
                    std::path::PathBuf::from(format!("target/{}-e2e-expected.json", $id));
                assert!(
                    expected_path.exists(),
                    "generate the official expectation with tools/export_yolov8_cls_fixtures.py first"
                );
                #[derive(Deserialize)]
                struct Expected {
                    top5: Vec<ExpectedClass>,
                }
                #[derive(Deserialize)]
                struct ExpectedClass {
                    class_id: usize,
                    name: String,
                    confidence: f32,
                }

                let expected: Expected =
                    serde_json::from_slice(&std::fs::read(&expected_path).unwrap()).unwrap();
                let checkpoint = std::path::PathBuf::from(format!(
                    "target/{}",
                    crate::models::yolov8::weights::artifact_filename($id)
                ));
                assert!(
                    checkpoint.exists(),
                    "pack the {} artifact with pack-weights first",
                    $id
                );
                let predictor = crate::Predictor::<Flex>::from_checkpoint(
                    crate::ModelId::from_str($id).unwrap(),
                    checkpoint,
                    Default::default(),
                )
                .unwrap();
                let (image, classifications) = predictor
                    .predict_classification_path("docs/dog_bike_man.jpg")
                    .unwrap();
                let _ = image;
                assert_eq!(
                    classifications.len(),
                    expected.top5.len(),
                    "top-5 count differs from Ultralytics"
                );
                // Flat distributions (near-tied probabilities) can swap adjacent ranks when the
                // anti-aliased resize rounds differently (PIL vs the Rust transform), so compare
                // the top-5 class set and per-class probabilities rather than rank order.
                let mut expected_ids: Vec<usize> =
                    expected.top5.iter().map(|entry| entry.class_id).collect();
                expected_ids.sort_unstable();
                let mut actual_ids: Vec<usize> =
                    classifications.iter().map(|entry| entry.class_id).collect();
                actual_ids.sort_unstable();
                assert_eq!(
                    actual_ids, expected_ids,
                    "top-5 class set differs from Ultralytics"
                );
                for actual in &classifications {
                    let expected = expected
                        .top5
                        .iter()
                        .find(|entry| entry.class_id == actual.class_id)
                        .unwrap();
                    assert_eq!(actual.class_name, expected.name, "class name table");
                    let delta = (actual.confidence - expected.confidence).abs();
                    // The 1000-way softmax is sensitive to the +-1 per-channel rounding of any
                    // anti-aliased bilinear resize; the golden test pins the graph at 2e-4 on the
                    // shared canvas, so this delta is preprocessing rounding, not graph drift.
                    assert!(
                        delta <= 4.5e-2,
                        "{} confidence: {} vs {}",
                        actual.class_name,
                        actual.confidence,
                        expected.confidence
                    );
                }
            }
        };
    }

    #[cfg(feature = "gpu")]
    macro_rules! gpu_latency_test {
        ($fn_name:ident, $config:ty, $id:literal) => {
            /// Measure single-image batch-1 inference latency with the packed native artifact on
            /// the Wgpu GPU backend. Requires the gpu feature and a packed native artifact.
            #[test]
            #[ignore]
            fn $fn_name() {
                let checkpoint = std::path::PathBuf::from(format!(
                    "target/{}",
                    crate::models::yolov8::weights::artifact_filename($id)
                ));
                assert!(
                    checkpoint.exists(),
                    "pack the {} artifact with pack-weights first",
                    $id
                );
                let (device, adapter) = crate::default_wgpu_device();
                println!("GPU adapter: {adapter}");
                let worker = std::thread::Builder::new()
                    .stack_size(64 * 1024 * 1024)
                    .spawn(move || {
                        let mut model = <$config>::default().init::<Wgpu>(&device);
                        model.load_burnpack_weights(checkpoint).unwrap();
                        let input = Tensor::<Wgpu, 4>::zeros([1, 3, 224, 224], &device);
                        const WARMUP_RUNS: usize = 3;
                        const TIMED_RUNS: usize = 10;

                        for _ in 0..WARMUP_RUNS {
                            let output = model.forward(input.clone());
                            let _ = output.probs.sum().into_data();
                        }
                        let mut samples = Vec::with_capacity(TIMED_RUNS);
                        for _ in 0..TIMED_RUNS {
                            let started = std::time::Instant::now();
                            let output = model.forward(input.clone());
                            let _ = output.probs.sum().into_data();
                            samples.push(started.elapsed().as_secs_f64() * 1e3);
                        }
                        samples.sort_by(|a, b| a.total_cmp(b));
                        let median = samples[samples.len() / 2];
                        let min = samples[0];
                        println!(
                            "{:>11}: {:>7.1} ms median, {:>7.1} ms min  (single image, batch 1, 224 px, {TIMED_RUNS} runs, Wgpu GPU)",
                            $id, median, min,
                        );
                    })
                    .unwrap();
                worker.join().unwrap();
            }
        };
    }

    checkpoint_test!(
        yolov8n_cls_imports_official_checkpoint_and_runs_forward,
        Yolov8ClsNConfig,
        "yolov8n-cls"
    );
    checkpoint_test!(
        yolov8s_cls_imports_official_checkpoint_and_runs_forward,
        Yolov8ClsSConfig,
        "yolov8s-cls"
    );
    checkpoint_test!(
        yolov8m_cls_imports_official_checkpoint_and_runs_forward,
        Yolov8ClsMConfig,
        "yolov8m-cls"
    );
    checkpoint_test!(
        yolov8l_cls_imports_official_checkpoint_and_runs_forward,
        Yolov8ClsLConfig,
        "yolov8l-cls"
    );
    checkpoint_test!(
        yolov8x_cls_imports_official_checkpoint_and_runs_forward,
        Yolov8ClsXConfig,
        "yolov8x-cls"
    );

    golden_test!(
        yolov8n_cls_matches_ultralytics_golden_tensors,
        Yolov8ClsNConfig,
        "yolov8n-cls"
    );
    golden_test!(
        yolov8s_cls_matches_ultralytics_golden_tensors,
        Yolov8ClsSConfig,
        "yolov8s-cls"
    );
    golden_test!(
        yolov8m_cls_matches_ultralytics_golden_tensors,
        Yolov8ClsMConfig,
        "yolov8m-cls"
    );
    golden_test!(
        yolov8l_cls_matches_ultralytics_golden_tensors,
        Yolov8ClsLConfig,
        "yolov8l-cls"
    );
    golden_test!(
        yolov8x_cls_matches_ultralytics_golden_tensors,
        Yolov8ClsXConfig,
        "yolov8x-cls"
    );

    latency_test!(
        yolov8n_cls_measures_single_inference_latency,
        Yolov8ClsNConfig,
        "yolov8n-cls"
    );
    latency_test!(
        yolov8s_cls_measures_single_inference_latency,
        Yolov8ClsSConfig,
        "yolov8s-cls"
    );
    latency_test!(
        yolov8m_cls_measures_single_inference_latency,
        Yolov8ClsMConfig,
        "yolov8m-cls"
    );
    latency_test!(
        yolov8l_cls_measures_single_inference_latency,
        Yolov8ClsLConfig,
        "yolov8l-cls"
    );
    latency_test!(
        yolov8x_cls_measures_single_inference_latency,
        Yolov8ClsXConfig,
        "yolov8x-cls"
    );

    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolov8n_cls_measures_single_inference_latency_gpu,
        Yolov8ClsNConfig,
        "yolov8n-cls"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolov8s_cls_measures_single_inference_latency_gpu,
        Yolov8ClsSConfig,
        "yolov8s-cls"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolov8m_cls_measures_single_inference_latency_gpu,
        Yolov8ClsMConfig,
        "yolov8m-cls"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolov8l_cls_measures_single_inference_latency_gpu,
        Yolov8ClsLConfig,
        "yolov8l-cls"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolov8x_cls_measures_single_inference_latency_gpu,
        Yolov8ClsXConfig,
        "yolov8x-cls"
    );

    cls_e2e_test!(
        yolov8n_cls_matches_ultralytics_end_to_end,
        Yolov8ClsNConfig,
        "yolov8n-cls"
    );
    cls_e2e_test!(
        yolov8s_cls_matches_ultralytics_end_to_end,
        Yolov8ClsSConfig,
        "yolov8s-cls"
    );
    cls_e2e_test!(
        yolov8m_cls_matches_ultralytics_end_to_end,
        Yolov8ClsMConfig,
        "yolov8m-cls"
    );
    cls_e2e_test!(
        yolov8l_cls_matches_ultralytics_end_to_end,
        Yolov8ClsLConfig,
        "yolov8l-cls"
    );
    cls_e2e_test!(
        yolov8x_cls_matches_ultralytics_end_to_end,
        Yolov8ClsXConfig,
        "yolov8x-cls"
    );
}
