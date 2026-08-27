use burn::{
    module::Module,
    tensor::{Device, Tensor, backend::Backend},
};

use super::{
    body::{Yolov10Body, Yolov10BodyConfig},
    head::{DecodedPredictions, Yolov10Head, Yolov10HeadConfig},
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

/// Native Burn YOLOv10n model.
///
/// Only the inference path is implemented: the body feeds the one2one detection head whose
/// predictions are decoded to source-space candidates. The training-only one2many branch of the
/// official checkpoint is not loaded.
#[derive(Module, Debug)]
pub struct Yolov10<B: Backend> {
    body: Yolov10Body<B>,
    head: Yolov10Head<B>,
}

impl<B: Backend> Yolov10<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> DecodedPredictions<B> {
        self.head.forward(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLOv10n checkpoint.
    #[cfg(feature = "pretrained")]
    pub fn load_pytorch_weights(
        &mut self,
        path: impl Into<PathBuf>,
    ) -> Result<(), PytorchStoreError> {
        let mut store = PytorchStore::from_file(path)
            .with_top_level_key("model")
            // Body layers retain their Ultralytics graph indices. The head is model.23, so this
            // rule must not match it.
            .with_key_remapping("model\\.([0-9]|1[0-9]|2[0-2])\\.(.+)", "body.model_$1.$2")
            // model.23.one2one_cv2.{scale}.{layer} is the box regression tower.
            .with_key_remapping(
                "model\\.23\\.one2one_cv2\\.0\\.0\\.(.+)",
                "head.p3.box_0.$1",
            )
            .with_key_remapping(
                "model\\.23\\.one2one_cv2\\.0\\.1\\.(.+)",
                "head.p3.box_1.$1",
            )
            .with_key_remapping(
                "model\\.23\\.one2one_cv2\\.0\\.2\\.(.+)",
                "head.p3.box_out.$1",
            )
            .with_key_remapping(
                "model\\.23\\.one2one_cv2\\.1\\.0\\.(.+)",
                "head.p4.box_0.$1",
            )
            .with_key_remapping(
                "model\\.23\\.one2one_cv2\\.1\\.1\\.(.+)",
                "head.p4.box_1.$1",
            )
            .with_key_remapping(
                "model\\.23\\.one2one_cv2\\.1\\.2\\.(.+)",
                "head.p4.box_out.$1",
            )
            .with_key_remapping(
                "model\\.23\\.one2one_cv2\\.2\\.0\\.(.+)",
                "head.p5.box_0.$1",
            )
            .with_key_remapping(
                "model\\.23\\.one2one_cv2\\.2\\.1\\.(.+)",
                "head.p5.box_1.$1",
            )
            .with_key_remapping(
                "model\\.23\\.one2one_cv2\\.2\\.2\\.(.+)",
                "head.p5.box_out.$1",
            )
            // model.23.one2one_cv3.{scale}.{tower}.{conv} is the light classification tower.
            .with_key_remapping(
                "model\\.23\\.one2one_cv3\\.0\\.0\\.0\\.(.+)",
                "head.p3.cls_dw_0.$1",
            )
            .with_key_remapping(
                "model\\.23\\.one2one_cv3\\.0\\.0\\.1\\.(.+)",
                "head.p3.cls_pw_0.$1",
            )
            .with_key_remapping(
                "model\\.23\\.one2one_cv3\\.0\\.1\\.0\\.(.+)",
                "head.p3.cls_dw_1.$1",
            )
            .with_key_remapping(
                "model\\.23\\.one2one_cv3\\.0\\.1\\.1\\.(.+)",
                "head.p3.cls_pw_1.$1",
            )
            .with_key_remapping(
                "model\\.23\\.one2one_cv3\\.0\\.2\\.(.+)",
                "head.p3.cls_out.$1",
            )
            .with_key_remapping(
                "model\\.23\\.one2one_cv3\\.1\\.0\\.0\\.(.+)",
                "head.p4.cls_dw_0.$1",
            )
            .with_key_remapping(
                "model\\.23\\.one2one_cv3\\.1\\.0\\.1\\.(.+)",
                "head.p4.cls_pw_0.$1",
            )
            .with_key_remapping(
                "model\\.23\\.one2one_cv3\\.1\\.1\\.0\\.(.+)",
                "head.p4.cls_dw_1.$1",
            )
            .with_key_remapping(
                "model\\.23\\.one2one_cv3\\.1\\.1\\.1\\.(.+)",
                "head.p4.cls_pw_1.$1",
            )
            .with_key_remapping(
                "model\\.23\\.one2one_cv3\\.1\\.2\\.(.+)",
                "head.p4.cls_out.$1",
            )
            .with_key_remapping(
                "model\\.23\\.one2one_cv3\\.2\\.0\\.0\\.(.+)",
                "head.p5.cls_dw_0.$1",
            )
            .with_key_remapping(
                "model\\.23\\.one2one_cv3\\.2\\.0\\.1\\.(.+)",
                "head.p5.cls_pw_0.$1",
            )
            .with_key_remapping(
                "model\\.23\\.one2one_cv3\\.2\\.1\\.0\\.(.+)",
                "head.p5.cls_dw_1.$1",
            )
            .with_key_remapping(
                "model\\.23\\.one2one_cv3\\.2\\.1\\.1\\.(.+)",
                "head.p5.cls_pw_1.$1",
            )
            .with_key_remapping(
                "model\\.23\\.one2one_cv3\\.2\\.2\\.(.+)",
                "head.p5.cls_out.$1",
            );
        self.load_from(&mut store).map(|_| ())
    }

    /// Load boquilens' versioned, half-precision native Burnpack artifact.
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
            .metadata("boquilens.artifact-format", weights::ARTIFACT_FORMAT)
            .metadata("boquilens.model", "yolov10n")
            .metadata("boquilens.classes", "coco-80")
            .metadata("boquilens.precision", "f16")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolov10Config;

impl Yolov10Config {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolov10<B> {
        Yolov10 {
            body: Yolov10BodyConfig.init(device),
            head: Yolov10HeadConfig.init(device),
        }
    }
}

#[cfg(all(test, feature = "pretrained"))]
mod tests {
    use super::*;
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
        let checkpoint = std::path::PathBuf::from("target/yolov10n-state.pt");
        assert!(
            checkpoint.exists(),
            "convert yolov10n.pt with tools/export_ultralytics_state.py first"
        );
        let worker = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let device = Default::default();
                let mut model = Yolov10Config.init::<Flex>(&device);
                model.load_pytorch_weights(checkpoint).unwrap();
                let output = model.forward(Tensor::zeros([1, 3, 64, 64], &device));
                assert_eq!(output.boxes.dims(), [1, 84, 4]);
                assert_eq!(output.scores.dims(), [1, 84, 80]);
            })
            .unwrap();
        worker.join().unwrap();
    }

    #[test]
    #[ignore]
    fn matches_ultralytics_golden_tensors() {
        let checkpoint =
            std::path::PathBuf::from("target/yolov10n-coco-ultralytics-v8.4-boquilens-v1.bpk");
        let fixture: GoldenFixture = serde_json::from_slice(
            &std::fs::read("target/yolov10n-golden-v1.json")
                .expect("generate fixtures with tools/export_yolov10_fixtures.py"),
        )
        .unwrap();
        assert_eq!(fixture.format, "boquilens-ultralytics-golden-v1");
        assert_eq!(fixture.model, "yolov10n");

        let worker = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let device = Default::default();
                let mut model = Yolov10Config.init::<Flex>(&device);
                model.load_burnpack_weights(checkpoint).unwrap();
                let image = image::open("target/yolov10n-preprocessed-reference.png")
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
                let p3 = features.p3.clone();
                let p4 = features.p4.clone();
                let p5 = features.p5.clone();
                let raw = model.head.forward_raw(super::super::body::Yolov10Features {
                    p3: features.p3.clone(),
                    p4: features.p4.clone(),
                    p5: features.p5.clone(),
                });
                let decoded = model.head.forward(features);

                assert_golden("body_p3", p3, fixture.tensors.get("body_p3").unwrap());
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
