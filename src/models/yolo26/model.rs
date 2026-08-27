use burn::{
    module::Module,
    tensor::{Device, Tensor, backend::Backend},
};

use super::{
    body::{
        Yolo26BodyLConfig, Yolo26BodyLarge, Yolo26BodyMConfig, Yolo26BodyNConfig,
        Yolo26BodySConfig, Yolo26BodySmall, Yolo26BodyXConfig,
    },
    head::{DecodedPredictions, Yolo26Head, Yolo26HeadConfig},
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

/// Build the PyTorch-state store shared by every YOLO26 scale variant.
///
/// Body layers retain their Ultralytics graph indices (the head is model.23, which the body rule
/// must not match); the head's one2one box and classification towers are remapped one rule per
/// path-segment pattern. The remaps are scale-independent because every variant shares the
/// checkpoint key layout.
#[cfg(feature = "pretrained")]
fn pytorch_store(path: impl Into<PathBuf>) -> PytorchStore {
    PytorchStore::from_file(path)
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
        )
}

/// Native Burn YOLO26n model.
///
/// Only the inference path is implemented: the body feeds the end-to-end one2one detection head
/// whose predictions are decoded to source-space candidates. The training-only one2many branch of
/// the official checkpoint is not loaded.
#[derive(Module, Debug)]
pub struct Yolo26N<B: Backend> {
    body: Yolo26BodySmall<B>,
    head: Yolo26Head<B>,
}

impl<B: Backend> Yolo26N<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> DecodedPredictions<B> {
        self.head.forward(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLO26n checkpoint.
    #[cfg(feature = "pretrained")]
    pub fn load_pytorch_weights(
        &mut self,
        path: impl Into<PathBuf>,
    ) -> Result<(), PytorchStoreError> {
        let mut store = pytorch_store(path);
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
            .metadata(
                "boquilens.artifact-format",
                weights::artifact_format("yolo26n"),
            )
            .metadata("boquilens.model", "yolo26n")
            .metadata("boquilens.classes", "coco-80")
            .metadata("boquilens.precision", "f16")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolo26NConfig;

impl Yolo26NConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo26N<B> {
        Yolo26N {
            body: Yolo26BodyNConfig.init(device),
            head: Yolo26HeadConfig::new(64, 128, 256).init(device),
        }
    }
}

/// Native Burn YOLO26s model.
#[derive(Module, Debug)]
pub struct Yolo26S<B: Backend> {
    body: Yolo26BodySmall<B>,
    head: Yolo26Head<B>,
}

impl<B: Backend> Yolo26S<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> DecodedPredictions<B> {
        self.head.forward(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLO26s checkpoint.
    #[cfg(feature = "pretrained")]
    pub fn load_pytorch_weights(
        &mut self,
        path: impl Into<PathBuf>,
    ) -> Result<(), PytorchStoreError> {
        let mut store = pytorch_store(path);
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
            .metadata(
                "boquilens.artifact-format",
                weights::artifact_format("yolo26s"),
            )
            .metadata("boquilens.model", "yolo26s")
            .metadata("boquilens.classes", "coco-80")
            .metadata("boquilens.precision", "f16")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolo26SConfig;

impl Yolo26SConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo26S<B> {
        Yolo26S {
            body: Yolo26BodySConfig.init(device),
            head: Yolo26HeadConfig::new(128, 256, 512).init(device),
        }
    }
}

/// Native Burn YOLO26m model. The m-scale body forces the C3k chain onto the early backbone
/// stages (`parse_model`'s m/l/x rule), so it shares [`Yolo26BodyLarge`] with l/x.
#[derive(Module, Debug)]
pub struct Yolo26M<B: Backend> {
    body: Yolo26BodyLarge<B>,
    head: Yolo26Head<B>,
}

impl<B: Backend> Yolo26M<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> DecodedPredictions<B> {
        self.head.forward(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLO26m checkpoint.
    #[cfg(feature = "pretrained")]
    pub fn load_pytorch_weights(
        &mut self,
        path: impl Into<PathBuf>,
    ) -> Result<(), PytorchStoreError> {
        let mut store = pytorch_store(path);
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
            .metadata(
                "boquilens.artifact-format",
                weights::artifact_format("yolo26m"),
            )
            .metadata("boquilens.model", "yolo26m")
            .metadata("boquilens.classes", "coco-80")
            .metadata("boquilens.precision", "f16")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolo26MConfig;

impl Yolo26MConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo26M<B> {
        Yolo26M {
            body: Yolo26BodyMConfig.init(device),
            head: Yolo26HeadConfig::new(256, 512, 512).init(device),
        }
    }
}

/// Native Burn YOLO26l model. Shares YOLO26m's body graph with depth-scaled repeats.
#[derive(Module, Debug)]
pub struct Yolo26L<B: Backend> {
    body: Yolo26BodyLarge<B>,
    head: Yolo26Head<B>,
}

impl<B: Backend> Yolo26L<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> DecodedPredictions<B> {
        self.head.forward(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLO26l checkpoint.
    #[cfg(feature = "pretrained")]
    pub fn load_pytorch_weights(
        &mut self,
        path: impl Into<PathBuf>,
    ) -> Result<(), PytorchStoreError> {
        let mut store = pytorch_store(path);
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
            .metadata(
                "boquilens.artifact-format",
                weights::artifact_format("yolo26l"),
            )
            .metadata("boquilens.model", "yolo26l")
            .metadata("boquilens.classes", "coco-80")
            .metadata("boquilens.precision", "f16")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolo26LConfig;

impl Yolo26LConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo26L<B> {
        Yolo26L {
            body: Yolo26BodyLConfig.init(device),
            head: Yolo26HeadConfig::new(256, 512, 512).init(device),
        }
    }
}

/// Native Burn YOLO26x model.
#[derive(Module, Debug)]
pub struct Yolo26X<B: Backend> {
    body: Yolo26BodyLarge<B>,
    head: Yolo26Head<B>,
}

impl<B: Backend> Yolo26X<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> DecodedPredictions<B> {
        self.head.forward(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLO26x checkpoint.
    #[cfg(feature = "pretrained")]
    pub fn load_pytorch_weights(
        &mut self,
        path: impl Into<PathBuf>,
    ) -> Result<(), PytorchStoreError> {
        let mut store = pytorch_store(path);
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
            .metadata(
                "boquilens.artifact-format",
                weights::artifact_format("yolo26x"),
            )
            .metadata("boquilens.model", "yolo26x")
            .metadata("boquilens.classes", "coco-80")
            .metadata("boquilens.precision", "f16")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolo26XConfig;

impl Yolo26XConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo26X<B> {
        Yolo26X {
            body: Yolo26BodyXConfig.init(device),
            head: Yolo26HeadConfig::new(384, 768, 768).init(device),
        }
    }
}

#[cfg(all(test, feature = "pretrained"))]
mod tests {
    use super::*;
    use crate::models::yolo26::body::Yolo26Features;
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

    fn assert_parity_tensors(
        features: Yolo26Features<Flex>,
        head: &Yolo26Head<Flex>,
        fixture: &GoldenFixture,
    ) {
        let p3 = features.p3.clone();
        let p4 = features.p4.clone();
        let p5 = features.p5.clone();
        let raw = head.forward_raw(Yolo26Features {
            p3: features.p3.clone(),
            p4: features.p4.clone(),
            p5: features.p5.clone(),
        });
        let decoded = head.forward(features);

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
                        assert_eq!(output.boxes.dims(), [1, 84, 4]);
                        assert_eq!(output.scores.dims(), [1, 84, 80]);
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
                    "target/{}-coco-ultralytics-v8.4-boquilens-v1.bpk",
                    $id
                ));
                let fixture: GoldenFixture = serde_json::from_slice(
                    &std::fs::read(format!("target/{}-golden-v1.json", $id)).unwrap_or_else(|_| {
                        panic!(
                            "generate fixtures with tools/export_yolo26_fixtures.py --model {}",
                            $id
                        )
                    }),
                )
                .unwrap();
                assert_eq!(fixture.format, "boquilens-ultralytics-golden-v1");
                assert_eq!(fixture.model, $id);

                let worker = std::thread::Builder::new()
                    .stack_size(64 * 1024 * 1024)
                    .spawn(move || {
                        let device = Default::default();
                        let mut model = <$config>::default().init::<Flex>(&device);
                        model.load_burnpack_weights(checkpoint).unwrap();
                        let input = load_reference_image($id, &device);
                        let features = model.body.forward(input);
                        assert_parity_tensors(features, &model.head, &fixture);
                    })
                    .unwrap();
                worker.join().unwrap();
            }
        };
    }

    checkpoint_test!(
        yolo26n_imports_official_checkpoint_and_runs_forward,
        Yolo26NConfig,
        "yolo26n"
    );
    checkpoint_test!(
        yolo26s_imports_official_checkpoint_and_runs_forward,
        Yolo26SConfig,
        "yolo26s"
    );
    checkpoint_test!(
        yolo26m_imports_official_checkpoint_and_runs_forward,
        Yolo26MConfig,
        "yolo26m"
    );
    checkpoint_test!(
        yolo26l_imports_official_checkpoint_and_runs_forward,
        Yolo26LConfig,
        "yolo26l"
    );
    checkpoint_test!(
        yolo26x_imports_official_checkpoint_and_runs_forward,
        Yolo26XConfig,
        "yolo26x"
    );

    golden_test!(
        yolo26n_matches_ultralytics_golden_tensors,
        Yolo26NConfig,
        "yolo26n"
    );
    golden_test!(
        yolo26s_matches_ultralytics_golden_tensors,
        Yolo26SConfig,
        "yolo26s"
    );
    golden_test!(
        yolo26m_matches_ultralytics_golden_tensors,
        Yolo26MConfig,
        "yolo26m"
    );
    golden_test!(
        yolo26l_matches_ultralytics_golden_tensors,
        Yolo26LConfig,
        "yolo26l"
    );
    golden_test!(
        yolo26x_matches_ultralytics_golden_tensors,
        Yolo26XConfig,
        "yolo26x"
    );
}
