//! Native Burn implementation of the Ultralytics YOLO11-cls classification family (n/s/m/l/x).
//!
//! The YOLO11-cls graph is identical to YOLO26-cls: the shared backbone truncated at the C2PSA
//! stage plus Ultralytics' `Classify` head, trained on ImageNet-1k at 224 px with plain PyTorch
//! batch-norm defaults. The backbone YAMLs, module graphs, checkpoint key layouts, and batch-norm
//! flavors match (verified against both checkpoints), so this module reuses the shared
//! classification graph from [`crate::models::yolo26::classification`] and declares only the five
//! scale models with their own artifact metadata.

use burn::{
    module::Module,
    tensor::{Device, Tensor, backend::Backend},
};

#[cfg(feature = "pretrained")]
use burn_store::{ModuleSnapshot, PytorchStore};

use crate::models::yolo26::classification::{
    ClassificationOutput, ClassifyHead, ClassifyHeadConfig, Yolo26ClassifyBodyLConfig,
    Yolo26ClassifyBodyLarge, Yolo26ClassifyBodyMConfig, Yolo26ClassifyBodyNConfig,
    Yolo26ClassifyBodySConfig, Yolo26ClassifyBodySmall, Yolo26ClassifyBodyXConfig,
};

classify_model!(
    Yolo11ClsN,
    Yolo11ClsNConfig,
    Yolo26ClassifyBodySmall,
    Yolo26ClassifyBodyNConfig,
    256,
    "yolo11n-cls",
    "Native Burn YOLO11n-cls model.",
    "Import tensor-only state exported from an official Ultralytics checkpoint."
);
classify_model!(
    Yolo11ClsS,
    Yolo11ClsSConfig,
    Yolo26ClassifyBodySmall,
    Yolo26ClassifyBodySConfig,
    512,
    "yolo11s-cls",
    "Native Burn YOLO11s-cls model.",
    "Import tensor-only state exported from an official Ultralytics checkpoint."
);
classify_model!(
    Yolo11ClsM,
    Yolo11ClsMConfig,
    Yolo26ClassifyBodyLarge,
    Yolo26ClassifyBodyMConfig,
    512,
    "yolo11m-cls",
    "Native Burn YOLO11m-cls model.",
    "Import tensor-only state exported from an official Ultralytics checkpoint."
);
classify_model!(
    Yolo11ClsL,
    Yolo11ClsLConfig,
    Yolo26ClassifyBodyLarge,
    Yolo26ClassifyBodyLConfig,
    512,
    "yolo11l-cls",
    "Native Burn YOLO11l-cls model.",
    "Import tensor-only state exported from an official Ultralytics checkpoint."
);
classify_model!(
    Yolo11ClsX,
    Yolo11ClsXConfig,
    Yolo26ClassifyBodyLarge,
    Yolo26ClassifyBodyXConfig,
    768,
    "yolo11x-cls",
    "Native Burn YOLO11x-cls model.",
    "Import tensor-only state exported from an official Ultralytics checkpoint."
);

/// Build the PyTorch-state store shared by every YOLO11-cls scale variant.
///
/// Identical key layout to the YOLO26-cls checkpoints: backbone layers 0-9 and the `Classify` head
/// at model.10.
#[cfg(feature = "pretrained")]
fn pytorch_store(path: impl Into<std::path::PathBuf>) -> PytorchStore {
    PytorchStore::from_file(path)
        .with_top_level_key("model")
        .with_key_remapping("model\\.([0-9])\\.(.+)", "body.model_$1.$2")
        .with_key_remapping("model\\.10\\.conv\\.conv\\.(.+)", "head.conv.conv.$1")
        .with_key_remapping("model\\.10\\.conv\\.bn\\.(.+)", "head.conv.bn.$1")
        .with_key_remapping("model\\.10\\.linear\\.(.+)", "head.linear.$1")
}

#[cfg(all(test, feature = "pretrained"))]
mod parity_tests {
    use super::*;
    use crate::models::yolo26::classification::NUM_CLASSES;
    use burn::tensor::{ElementConversion, TensorData};
    use burn_flex::Flex;
    use serde::Deserialize;
    use std::collections::BTreeMap;

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
            #[test]
            #[ignore]
            fn $fn_name() {
                let checkpoint = std::path::PathBuf::from(concat!("target/", $id, "-state.pt"));
                assert!(
                    checkpoint.exists(),
                    "convert {}.pt with tools/export_checkpoint_state.py first",
                    $id
                );
                let worker = std::thread::Builder::new()
                    .stack_size(64 * 1024 * 1024)
                    .spawn(move || {
                        let device = Default::default();
                        let mut model = <$config>::default().init::<Flex>(&device);
                        model.load_pytorch_weights(checkpoint).unwrap();
                        let output = model.forward(Tensor::zeros([1, 3, 64, 64], &device));
                        assert_eq!(output.probs.dims(), [1, NUM_CLASSES]);
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
                    crate::models::yolo11::weights::artifact_filename($id)
                ));
                let fixture: GoldenFixture = serde_json::from_slice(
                    &std::fs::read(format!("target/{}-golden-v1.json", $id)).unwrap_or_else(|_| {
                        panic!(
                            "generate fixtures with tools/export_classification_fixtures.py --model {}",
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
            #[test]
            #[ignore]
            fn $fn_name() {
                let checkpoint = std::path::PathBuf::from(format!(
                    "target/{}",
                    crate::models::yolo11::weights::artifact_filename($id)
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

    #[cfg(feature = "gpu")]
    macro_rules! gpu_latency_test {
        ($fn_name:ident, $config:ty, $id:literal) => {
            #[test]
            #[ignore]
            fn $fn_name() {
                let checkpoint = std::path::PathBuf::from(format!(
                    "target/{}",
                    crate::models::yolo11::weights::artifact_filename($id)
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
                        let mut model = <$config>::default().init::<burn::backend::Wgpu>(&device);
                        model.load_burnpack_weights(checkpoint).unwrap();
                        let input = Tensor::<burn::backend::Wgpu, 4>::zeros([1, 3, 224, 224], &device);
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

    #[cfg(feature = "pretrained")]
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
                    "generate the official expectation with tools/export_classification_fixtures.py first"
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
                    crate::models::yolo11::weights::artifact_filename($id)
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

    checkpoint_test!(
        yolo11n_cls_imports_official_checkpoint_and_runs_forward,
        Yolo11ClsNConfig,
        "yolo11n-cls"
    );
    checkpoint_test!(
        yolo11s_cls_imports_official_checkpoint_and_runs_forward,
        Yolo11ClsSConfig,
        "yolo11s-cls"
    );
    checkpoint_test!(
        yolo11m_cls_imports_official_checkpoint_and_runs_forward,
        Yolo11ClsMConfig,
        "yolo11m-cls"
    );
    checkpoint_test!(
        yolo11l_cls_imports_official_checkpoint_and_runs_forward,
        Yolo11ClsLConfig,
        "yolo11l-cls"
    );
    checkpoint_test!(
        yolo11x_cls_imports_official_checkpoint_and_runs_forward,
        Yolo11ClsXConfig,
        "yolo11x-cls"
    );

    golden_test!(
        yolo11n_cls_matches_ultralytics_golden_tensors,
        Yolo11ClsNConfig,
        "yolo11n-cls"
    );
    golden_test!(
        yolo11s_cls_matches_ultralytics_golden_tensors,
        Yolo11ClsSConfig,
        "yolo11s-cls"
    );
    golden_test!(
        yolo11m_cls_matches_ultralytics_golden_tensors,
        Yolo11ClsMConfig,
        "yolo11m-cls"
    );
    golden_test!(
        yolo11l_cls_matches_ultralytics_golden_tensors,
        Yolo11ClsLConfig,
        "yolo11l-cls"
    );
    golden_test!(
        yolo11x_cls_matches_ultralytics_golden_tensors,
        Yolo11ClsXConfig,
        "yolo11x-cls"
    );

    latency_test!(
        yolo11n_cls_measures_single_inference_latency,
        Yolo11ClsNConfig,
        "yolo11n-cls"
    );
    latency_test!(
        yolo11s_cls_measures_single_inference_latency,
        Yolo11ClsSConfig,
        "yolo11s-cls"
    );
    latency_test!(
        yolo11m_cls_measures_single_inference_latency,
        Yolo11ClsMConfig,
        "yolo11m-cls"
    );
    latency_test!(
        yolo11l_cls_measures_single_inference_latency,
        Yolo11ClsLConfig,
        "yolo11l-cls"
    );
    latency_test!(
        yolo11x_cls_measures_single_inference_latency,
        Yolo11ClsXConfig,
        "yolo11x-cls"
    );

    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo11n_cls_measures_single_inference_latency_gpu,
        Yolo11ClsNConfig,
        "yolo11n-cls"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo11s_cls_measures_single_inference_latency_gpu,
        Yolo11ClsSConfig,
        "yolo11s-cls"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo11m_cls_measures_single_inference_latency_gpu,
        Yolo11ClsMConfig,
        "yolo11m-cls"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo11l_cls_measures_single_inference_latency_gpu,
        Yolo11ClsLConfig,
        "yolo11l-cls"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo11x_cls_measures_single_inference_latency_gpu,
        Yolo11ClsXConfig,
        "yolo11x-cls"
    );

    cls_e2e_test!(
        yolo11n_cls_matches_ultralytics_end_to_end,
        Yolo11ClsNConfig,
        "yolo11n-cls"
    );
    cls_e2e_test!(
        yolo11s_cls_matches_ultralytics_end_to_end,
        Yolo11ClsSConfig,
        "yolo11s-cls"
    );
    cls_e2e_test!(
        yolo11m_cls_matches_ultralytics_end_to_end,
        Yolo11ClsMConfig,
        "yolo11m-cls"
    );
    cls_e2e_test!(
        yolo11l_cls_matches_ultralytics_end_to_end,
        Yolo11ClsLConfig,
        "yolo11l-cls"
    );
    cls_e2e_test!(
        yolo11x_cls_matches_ultralytics_end_to_end,
        Yolo11ClsXConfig,
        "yolo11x-cls"
    );
}
