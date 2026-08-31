use burn::{
    module::Module,
    tensor::{Device, Tensor, backend::Backend},
};

use super::{
    body::{Yolov3TinyBody, Yolov3TinyBodyConfig},
    head::{DecodedPredictions, DetectHead, DetectHeadConfig, RawPredictions},
};

#[cfg(feature = "pretrained")]
use {
    super::weights,
    burn_store::{
        BurnpackError, BurnpackStore, HalfPrecisionAdapter, ModuleSnapshot, PytorchStore,
        PytorchStoreError,
    },
    std::path::PathBuf,
};

/// Native Burn YOLOv3-Tiny-Ultralytics model.
#[derive(Module, Debug)]
pub struct Yolov3Tiny<B: Backend> {
    body: Yolov3TinyBody<B>,
    head: DetectHead<B>,
}

impl<B: Backend> Yolov3Tiny<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> DecodedPredictions<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> RawPredictions<B> {
        self.head.forward_raw(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLOv3-Tiny-U checkpoint.
    #[cfg(feature = "pretrained")]
    pub fn load_pytorch_weights(
        &mut self,
        path: impl Into<PathBuf>,
    ) -> Result<(), PytorchStoreError> {
        let mut store = PytorchStore::from_file(path)
            .with_top_level_key("model")
            // Body layers retain their Ultralytics graph indices.
            .with_key_remapping("model\\.([0-9]|1[0-9])\\.(.+)", "body.model_$1.$2")
            // model.20 is Detect: cv2 is box regression and cv3 is classification.
            .with_key_remapping("model\\.20\\.cv2\\.0\\.0\\.(.+)", "head.p4.box_0.$1")
            .with_key_remapping("model\\.20\\.cv2\\.0\\.1\\.(.+)", "head.p4.box_1.$1")
            .with_key_remapping("model\\.20\\.cv2\\.0\\.2\\.(.+)", "head.p4.box_2.$1")
            .with_key_remapping("model\\.20\\.cv2\\.1\\.0\\.(.+)", "head.p5.box_0.$1")
            .with_key_remapping("model\\.20\\.cv2\\.1\\.1\\.(.+)", "head.p5.box_1.$1")
            .with_key_remapping("model\\.20\\.cv2\\.1\\.2\\.(.+)", "head.p5.box_2.$1")
            .with_key_remapping("model\\.20\\.cv3\\.0\\.0\\.(.+)", "head.p4.cls_0.$1")
            .with_key_remapping("model\\.20\\.cv3\\.0\\.1\\.(.+)", "head.p4.cls_1.$1")
            .with_key_remapping("model\\.20\\.cv3\\.0\\.2\\.(.+)", "head.p4.cls_2.$1")
            .with_key_remapping("model\\.20\\.cv3\\.1\\.0\\.(.+)", "head.p5.cls_0.$1")
            .with_key_remapping("model\\.20\\.cv3\\.1\\.1\\.(.+)", "head.p5.cls_1.$1")
            .with_key_remapping("model\\.20\\.cv3\\.1\\.2\\.(.+)", "head.p5.cls_2.$1");
        self.load_from(&mut store).map(|_| ())
    }

    /// Load Montgomery's versioned, half-precision native Burnpack artifact.
    #[cfg(feature = "pretrained")]
    pub fn load_burnpack_weights(&mut self, path: impl Into<PathBuf>) -> Result<(), BurnpackError> {
        let mut store = BurnpackStore::from_file(path.into())
            .with_from_adapter(HalfPrecisionAdapter::new())
            .zero_copy(true);
        self.load_from(&mut store).map(|_| ())
    }

    /// Save a versioned native artifact. Existing files are deliberately not overwritten.
    #[cfg(feature = "pretrained")]
    pub fn save_burnpack_weights(&self, path: impl Into<PathBuf>) -> Result<(), BurnpackError> {
        let mut store = BurnpackStore::from_file(path.into())
            .metadata("montgomery.artifact-format", weights::ARTIFACT_FORMAT)
            .metadata("montgomery.model", "yolov3-tinyu")
            .metadata("montgomery.classes", "coco-80")
            .metadata("montgomery.precision", "f16")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolov3TinyConfig;

impl Yolov3TinyConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolov3Tiny<B> {
        self.init_with_classes(80, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolov3Tiny<B> {
        assert!(num_classes > 0, "class count must be positive");
        Yolov3Tiny {
            body: Yolov3TinyBodyConfig.init(device),
            head: DetectHeadConfig::new(num_classes).init(device),
        }
    }
}

#[cfg(all(test, feature = "pretrained"))]
mod tests {
    use super::*;
    use crate::models::yolov3_tiny::body::Yolov3TinyFeatures;
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

    /// Run manually after converting the official checkpoint. Kept ignored in CI because the
    /// source checkpoint is an external AGPL asset.
    #[test]
    #[ignore]
    fn imports_official_checkpoint_and_runs_forward() {
        let checkpoint = std::path::PathBuf::from("target/yolov3-tinyu-state.pt");
        assert!(
            checkpoint.exists(),
            "convert yolov3-tinyu.pt with tools/export_ultralytics_state.py first"
        );
        let worker = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let device = Default::default();
                let mut model = Yolov3TinyConfig.init::<Flex>(&device);
                model.load_pytorch_weights(checkpoint).unwrap();
                let output = model.forward(Tensor::zeros([1, 3, 64, 64], &device));
                assert_eq!(output.boxes.dims(), [1, 20, 4]);
                assert_eq!(output.scores.dims(), [1, 20, 80]);
            })
            .unwrap();
        worker.join().unwrap();
    }

    #[test]
    #[ignore]
    fn matches_ultralytics_golden_tensors() {
        let checkpoint =
            std::path::PathBuf::from("target/yolov3-tinyu-coco-ultralytics-v8.4-montgomery-v1.bpk");
        let fixture: GoldenFixture = serde_json::from_slice(
            &std::fs::read("target/yolov3-tinyu-golden-v1.json")
                .expect("generate fixtures with tools/export_ultralytics_fixtures.py"),
        )
        .unwrap();
        assert_eq!(fixture.format, "montgomery-ultralytics-golden-v1");
        assert_eq!(fixture.model, "yolov3-tinyu");

        let worker = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let device = Default::default();
                let mut model = Yolov3TinyConfig.init::<Flex>(&device);
                model.load_burnpack_weights(checkpoint).unwrap();
                let image = image::open("target/yolov3-tinyu-preprocessed-reference.png")
                    .unwrap()
                    .into_rgb8();
                let shape = [image.height() as usize, image.width() as usize, 3];
                let input = Tensor::<Flex, 3>::from_data(
                    TensorData::new(image.into_raw(), shape).convert::<f32>(),
                    &device,
                )
                .permute([2, 0, 1])
                .unsqueeze::<4>()
                    / 255.0;
                let features = model.body.forward(input);
                let p4 = features.p4.clone();
                let p5 = features.p5.clone();
                let raw = model.head.forward_raw(Yolov3TinyFeatures {
                    p4: features.p4.clone(),
                    p5: features.p5.clone(),
                });
                let decoded = model.head.forward(features);

                assert_golden("body_p4", p4, fixture.tensors.get("body_p4").unwrap());
                assert_golden("body_p5", p5, fixture.tensors.get("body_p5").unwrap());
                assert_golden(
                    "raw_boxes",
                    raw.boxes,
                    fixture.tensors.get("raw_boxes").unwrap(),
                );
                assert_golden(
                    "raw_scores",
                    raw.scores,
                    fixture.tensors.get("raw_scores").unwrap(),
                );
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
            })
            .unwrap();
        worker.join().unwrap();
    }
}
