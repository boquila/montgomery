use burn::{
    module::Module,
    tensor::{Device, Tensor, backend::Backend},
};

use super::{
    body::{
        Yolov10BodyB, Yolov10BodyBConfig, Yolov10BodyLConfig, Yolov10BodyM, Yolov10BodyMConfig,
        Yolov10BodyN, Yolov10BodyNConfig, Yolov10BodyS, Yolov10BodySConfig, Yolov10BodyX,
        Yolov10BodyXConfig,
    },
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

/// Build the PyTorch-state store shared by every YOLOv10 scale variant.
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

/// Native Burn YOLOv10n model.
///
/// Only the inference path is implemented: the body feeds the one2one detection head whose
/// predictions are decoded to source-space candidates. The training-only one2many branch of the
/// official checkpoint is not loaded.
#[derive(Module, Debug)]
pub struct Yolov10N<B: Backend> {
    body: Yolov10BodyN<B>,
    head: Yolov10Head<B>,
}

impl<B: Backend> Yolov10N<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> DecodedPredictions<B> {
        self.head.forward(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLOv10n checkpoint.
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
                weights::artifact_format("yolov10n"),
            )
            .metadata("boquilens.model", "yolov10n")
            .metadata("boquilens.classes", "coco-80")
            .metadata("boquilens.precision", "f16")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolov10NConfig;

impl Yolov10NConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolov10N<B> {
        Yolov10N {
            body: Yolov10BodyNConfig.init(device),
            head: Yolov10HeadConfig::new(64, 128, 256).init(device),
        }
    }
}

/// Native Burn YOLOv10s model. The s-scale body swaps layer 8 to a large-kernel C2fCIB tower.
#[derive(Module, Debug)]
pub struct Yolov10S<B: Backend> {
    body: Yolov10BodyS<B>,
    head: Yolov10Head<B>,
}

impl<B: Backend> Yolov10S<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> DecodedPredictions<B> {
        self.head.forward(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLOv10s checkpoint.
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
                weights::artifact_format("yolov10s"),
            )
            .metadata("boquilens.model", "yolov10s")
            .metadata("boquilens.classes", "coco-80")
            .metadata("boquilens.precision", "f16")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolov10SConfig;

impl Yolov10SConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolov10S<B> {
        Yolov10S {
            body: Yolov10BodySConfig.init(device),
            head: Yolov10HeadConfig::new(128, 256, 512).init(device),
        }
    }
}

/// Native Burn YOLOv10m model. The m-scale body uses the plain depth-wise C2fCIB flavor,
/// including neck layer 19.
#[derive(Module, Debug)]
pub struct Yolov10M<B: Backend> {
    body: Yolov10BodyM<B>,
    head: Yolov10Head<B>,
}

impl<B: Backend> Yolov10M<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> DecodedPredictions<B> {
        self.head.forward(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLOv10m checkpoint.
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
                weights::artifact_format("yolov10m"),
            )
            .metadata("boquilens.model", "yolov10m")
            .metadata("boquilens.classes", "coco-80")
            .metadata("boquilens.precision", "f16")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolov10MConfig;

impl Yolov10MConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolov10M<B> {
        Yolov10M {
            body: Yolov10BodyMConfig.init(device),
            head: Yolov10HeadConfig::new(192, 384, 576).init(device),
        }
    }
}

/// Native Burn YOLOv10b model. Neck layer 13 also becomes a C2fCIB stage.
#[derive(Module, Debug)]
pub struct Yolov10B<B: Backend> {
    body: Yolov10BodyB<B>,
    head: Yolov10Head<B>,
}

impl<B: Backend> Yolov10B<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> DecodedPredictions<B> {
        self.head.forward(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLOv10b checkpoint.
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
                weights::artifact_format("yolov10b"),
            )
            .metadata("boquilens.model", "yolov10b")
            .metadata("boquilens.classes", "coco-80")
            .metadata("boquilens.precision", "f16")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolov10BConfig;

impl Yolov10BConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolov10B<B> {
        Yolov10B {
            body: Yolov10BodyBConfig.init(device),
            head: Yolov10HeadConfig::new(256, 512, 512).init(device),
        }
    }
}

/// Native Burn YOLOv10l model. Shares YOLOv10b's body graph with depth-scaled repeats.
#[derive(Module, Debug)]
pub struct Yolov10L<B: Backend> {
    body: Yolov10BodyB<B>,
    head: Yolov10Head<B>,
}

impl<B: Backend> Yolov10L<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> DecodedPredictions<B> {
        self.head.forward(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLOv10l checkpoint.
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
                weights::artifact_format("yolov10l"),
            )
            .metadata("boquilens.model", "yolov10l")
            .metadata("boquilens.classes", "coco-80")
            .metadata("boquilens.precision", "f16")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolov10LConfig;

impl Yolov10LConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolov10L<B> {
        Yolov10L {
            body: Yolov10BodyLConfig.init(device),
            head: Yolov10HeadConfig::new(256, 512, 512).init(device),
        }
    }
}

/// Native Burn YOLOv10x model. Backbone layer 6 also becomes a C2fCIB stage at this scale.
#[derive(Module, Debug)]
pub struct Yolov10X<B: Backend> {
    body: Yolov10BodyX<B>,
    head: Yolov10Head<B>,
}

impl<B: Backend> Yolov10X<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> DecodedPredictions<B> {
        self.head.forward(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLOv10x checkpoint.
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
                weights::artifact_format("yolov10x"),
            )
            .metadata("boquilens.model", "yolov10x")
            .metadata("boquilens.classes", "coco-80")
            .metadata("boquilens.precision", "f16")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolov10XConfig;

impl Yolov10XConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolov10X<B> {
        Yolov10X {
            body: Yolov10BodyXConfig.init(device),
            head: Yolov10HeadConfig::new(320, 640, 640).init(device),
        }
    }
}

#[cfg(all(test, feature = "pretrained"))]
mod tests {
    use super::*;
    use crate::models::yolov10::body::Yolov10Features;
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

    fn assert_parity_tensors(
        features: Yolov10Features<Flex>,
        head: &Yolov10Head<Flex>,
        fixture: &GoldenFixture,
    ) {
        let p3 = features.p3.clone();
        let p4 = features.p4.clone();
        let p5 = features.p5.clone();
        let raw = head.forward_raw(Yolov10Features {
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
                            "generate fixtures with tools/export_yolov10_fixtures.py --model {}",
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

    macro_rules! latency_test {
        ($fn_name:ident, $config:ty, $id:literal) => {
            /// Measure single-image batch-1 inference latency (forward, decode, and result sync)
            /// with the packed native artifact on the Flex CPU backend. Run with
            /// `cargo test --release <id> -- --ignored --nocapture` after the weight-prep loop.
            #[test]
            #[ignore]
            fn $fn_name() {
                let checkpoint = std::path::PathBuf::from(format!(
                    "target/{}-coco-ultralytics-v8.4-boquilens-v1.bpk",
                    $id
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
                        let input = Tensor::<Flex, 4>::zeros([1, 3, 640, 640], &device);
                        const WARMUP_RUNS: usize = 3;
                        const TIMED_RUNS: usize = 10;

                        for _ in 0..WARMUP_RUNS {
                            let output = model.forward(input.clone());
                            let _ = output.boxes.sum().into_data();
                            let _ = output.scores.sum().into_data();
                        }
                        let mut samples = Vec::with_capacity(TIMED_RUNS);
                        for _ in 0..TIMED_RUNS {
                            let started = std::time::Instant::now();
                            let output = model.forward(input.clone());
                            // Force result completion so the sample includes the full compute, not
                            // just kernel dispatch (a no-op on CPU, load-bearing on GPU).
                            let _ = output.boxes.sum().into_data();
                            let _ = output.scores.sum().into_data();
                            samples.push(started.elapsed().as_secs_f64() * 1e3);
                        }
                        samples.sort_by(|a, b| a.total_cmp(b));
                        let median = samples[samples.len() / 2];
                        let min = samples[0];
                        println!(
                            "{:>9}: {:>7.1} ms median, {:>7.1} ms min  (single image, batch 1, 640 px, {TIMED_RUNS} runs)",
                            $id, median, min,
                        );
                    })
                    .unwrap();
                worker.join().unwrap();
            }
        };
    }

    checkpoint_test!(
        yolov10n_imports_official_checkpoint_and_runs_forward,
        Yolov10NConfig,
        "yolov10n"
    );
    checkpoint_test!(
        yolov10s_imports_official_checkpoint_and_runs_forward,
        Yolov10SConfig,
        "yolov10s"
    );
    checkpoint_test!(
        yolov10m_imports_official_checkpoint_and_runs_forward,
        Yolov10MConfig,
        "yolov10m"
    );
    checkpoint_test!(
        yolov10b_imports_official_checkpoint_and_runs_forward,
        Yolov10BConfig,
        "yolov10b"
    );
    checkpoint_test!(
        yolov10l_imports_official_checkpoint_and_runs_forward,
        Yolov10LConfig,
        "yolov10l"
    );
    checkpoint_test!(
        yolov10x_imports_official_checkpoint_and_runs_forward,
        Yolov10XConfig,
        "yolov10x"
    );

    golden_test!(
        yolov10n_matches_ultralytics_golden_tensors,
        Yolov10NConfig,
        "yolov10n"
    );
    golden_test!(
        yolov10s_matches_ultralytics_golden_tensors,
        Yolov10SConfig,
        "yolov10s"
    );
    golden_test!(
        yolov10m_matches_ultralytics_golden_tensors,
        Yolov10MConfig,
        "yolov10m"
    );
    golden_test!(
        yolov10b_matches_ultralytics_golden_tensors,
        Yolov10BConfig,
        "yolov10b"
    );
    golden_test!(
        yolov10l_matches_ultralytics_golden_tensors,
        Yolov10LConfig,
        "yolov10l"
    );
    golden_test!(
        yolov10x_matches_ultralytics_golden_tensors,
        Yolov10XConfig,
        "yolov10x"
    );

    latency_test!(
        yolov10n_measures_single_inference_latency,
        Yolov10NConfig,
        "yolov10n"
    );
    latency_test!(
        yolov10s_measures_single_inference_latency,
        Yolov10SConfig,
        "yolov10s"
    );
    latency_test!(
        yolov10m_measures_single_inference_latency,
        Yolov10MConfig,
        "yolov10m"
    );
    latency_test!(
        yolov10b_measures_single_inference_latency,
        Yolov10BConfig,
        "yolov10b"
    );
    latency_test!(
        yolov10l_measures_single_inference_latency,
        Yolov10LConfig,
        "yolov10l"
    );
    latency_test!(
        yolov10x_measures_single_inference_latency,
        Yolov10XConfig,
        "yolov10x"
    );

    #[cfg(feature = "gpu")]
    macro_rules! gpu_latency_test {
        ($fn_name:ident, $config:ty, $id:literal) => {
            /// Measure single-image batch-1 inference latency (forward, decode, and result sync)
            /// with the packed native artifact on the Wgpu GPU backend (Vulkan/DX12 on Windows and
            /// Linux, Metal on macOS). Requires the gpu feature and a packed native artifact.
            #[test]
            #[ignore]
            fn $fn_name() {
                let checkpoint = std::path::PathBuf::from(format!(
                    "target/{}-coco-ultralytics-v8.4-boquilens-v1.bpk",
                    $id
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
                        let input = Tensor::<Wgpu, 4>::zeros([1, 3, 640, 640], &device);
                        const WARMUP_RUNS: usize = 3;
                        const TIMED_RUNS: usize = 10;

                        for _ in 0..WARMUP_RUNS {
                            let output = model.forward(input.clone());
                            let _ = output.boxes.sum().into_data();
                            let _ = output.scores.sum().into_data();
                        }
                        let mut samples = Vec::with_capacity(TIMED_RUNS);
                        for _ in 0..TIMED_RUNS {
                            let started = std::time::Instant::now();
                            let output = model.forward(input.clone());
                            // Force result completion so the sample includes the full compute, not
                            // just kernel dispatch.
                            let _ = output.boxes.sum().into_data();
                            let _ = output.scores.sum().into_data();
                            samples.push(started.elapsed().as_secs_f64() * 1e3);
                        }
                        samples.sort_by(|a, b| a.total_cmp(b));
                        let median = samples[samples.len() / 2];
                        let min = samples[0];
                        println!(
                            "{:>9}: {:>7.1} ms median, {:>7.1} ms min  (single image, batch 1, 640 px, {TIMED_RUNS} runs, Wgpu GPU)",
                            $id, median, min,
                        );
                    })
                    .unwrap();
                worker.join().unwrap();
            }
        };
    }

    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolov10n_measures_single_inference_latency_gpu,
        Yolov10NConfig,
        "yolov10n"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolov10s_measures_single_inference_latency_gpu,
        Yolov10SConfig,
        "yolov10s"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolov10m_measures_single_inference_latency_gpu,
        Yolov10MConfig,
        "yolov10m"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolov10b_measures_single_inference_latency_gpu,
        Yolov10BConfig,
        "yolov10b"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolov10l_measures_single_inference_latency_gpu,
        Yolov10LConfig,
        "yolov10l"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolov10x_measures_single_inference_latency_gpu,
        Yolov10XConfig,
        "yolov10x"
    );
}
