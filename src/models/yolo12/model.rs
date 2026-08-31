use burn::{
    module::Module,
    tensor::{Device, Tensor, backend::Backend},
};

use super::body::{
    Yolo12BodyLConfig, Yolo12BodyLarge, Yolo12BodyMConfig, Yolo12BodyNConfig, Yolo12BodySConfig,
    Yolo12BodySmall, Yolo12BodyXConfig,
};
// The YOLO12 `Detect` head is byte-identical to YOLO11's (classic DFL towers with the light
// DWConv classification flavor, reg_max 16, verified from the checkpoints), so the head module
// and its decode are shared instead of duplicated.
use crate::models::yolo11::head::{
    DecodedPredictions, RawPredictions, Yolo11Head, Yolo11HeadConfig,
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

/// Build the PyTorch-state store shared by every YOLO12 scale variant.
///
/// Body layers retain their Ultralytics graph indices (the head is model.21, which the body rule
/// must not match); the head's box and classification towers are remapped one rule per
/// path-segment pattern. The fixed DFL projection (`model.21.dfl.conv.weight`) is intentionally
/// unmapped: it is baked into the head as a constant.
#[cfg(feature = "pretrained")]
fn pytorch_store(path: impl Into<PathBuf>) -> PytorchStore {
    PytorchStore::from_file(path)
        .with_top_level_key("model")
        // Body layers retain their Ultralytics graph indices. The head is model.21, so this rule
        // must not match it.
        .with_key_remapping("model\\.([0-9]|1[0-9]|20)\\.(.+)", "body.model_$1.$2")
        // model.21.cv2.{scale}.{layer} is the box regression tower.
        .with_key_remapping("model\\.21\\.cv2\\.0\\.0\\.(.+)", "head.p3.box_0.$1")
        .with_key_remapping("model\\.21\\.cv2\\.0\\.1\\.(.+)", "head.p3.box_1.$1")
        .with_key_remapping("model\\.21\\.cv2\\.0\\.2\\.(.+)", "head.p3.box_out.$1")
        .with_key_remapping("model\\.21\\.cv2\\.1\\.0\\.(.+)", "head.p4.box_0.$1")
        .with_key_remapping("model\\.21\\.cv2\\.1\\.1\\.(.+)", "head.p4.box_1.$1")
        .with_key_remapping("model\\.21\\.cv2\\.1\\.2\\.(.+)", "head.p4.box_out.$1")
        .with_key_remapping("model\\.21\\.cv2\\.2\\.0\\.(.+)", "head.p5.box_0.$1")
        .with_key_remapping("model\\.21\\.cv2\\.2\\.1\\.(.+)", "head.p5.box_1.$1")
        .with_key_remapping("model\\.21\\.cv2\\.2\\.2\\.(.+)", "head.p5.box_out.$1")
        // model.21.cv3.{scale}.{tower}.{conv} is the light classification tower.
        .with_key_remapping("model\\.21\\.cv3\\.0\\.0\\.0\\.(.+)", "head.p3.cls_dw_0.$1")
        .with_key_remapping("model\\.21\\.cv3\\.0\\.0\\.1\\.(.+)", "head.p3.cls_pw_0.$1")
        .with_key_remapping("model\\.21\\.cv3\\.0\\.1\\.0\\.(.+)", "head.p3.cls_dw_1.$1")
        .with_key_remapping("model\\.21\\.cv3\\.0\\.1\\.1\\.(.+)", "head.p3.cls_pw_1.$1")
        .with_key_remapping("model\\.21\\.cv3\\.0\\.2\\.(.+)", "head.p3.cls_out.$1")
        .with_key_remapping("model\\.21\\.cv3\\.1\\.0\\.0\\.(.+)", "head.p4.cls_dw_0.$1")
        .with_key_remapping("model\\.21\\.cv3\\.1\\.0\\.1\\.(.+)", "head.p4.cls_pw_0.$1")
        .with_key_remapping("model\\.21\\.cv3\\.1\\.1\\.0\\.(.+)", "head.p4.cls_dw_1.$1")
        .with_key_remapping("model\\.21\\.cv3\\.1\\.1\\.1\\.(.+)", "head.p4.cls_pw_1.$1")
        .with_key_remapping("model\\.21\\.cv3\\.1\\.2\\.(.+)", "head.p4.cls_out.$1")
        .with_key_remapping("model\\.21\\.cv3\\.2\\.0\\.0\\.(.+)", "head.p5.cls_dw_0.$1")
        .with_key_remapping("model\\.21\\.cv3\\.2\\.0\\.1\\.(.+)", "head.p5.cls_pw_0.$1")
        .with_key_remapping("model\\.21\\.cv3\\.2\\.1\\.0\\.(.+)", "head.p5.cls_dw_1.$1")
        .with_key_remapping("model\\.21\\.cv3\\.2\\.1\\.1\\.(.+)", "head.p5.cls_pw_1.$1")
        .with_key_remapping("model\\.21\\.cv3\\.2\\.2\\.(.+)", "head.p5.cls_out.$1")
}

/// Native Burn YOLO12n model.
///
/// The body feeds the classic DFL detection head whose predictions are decoded to center-size
/// model-input pixels; the runtime applies class-aware non-maximum suppression.
#[derive(Module, Debug)]
pub struct Yolo12N<B: Backend> {
    body: Yolo12BodySmall<B>,
    head: Yolo11Head<B>,
}

impl<B: Backend> Yolo12N<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> DecodedPredictions<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> RawPredictions<B> {
        self.head.forward_raw(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLO12n checkpoint.
    #[cfg(feature = "pretrained")]
    pub fn load_pytorch_weights(
        &mut self,
        path: impl Into<PathBuf>,
    ) -> Result<(), PytorchStoreError> {
        let mut store = pytorch_store(path);
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
            .metadata(
                "montgomery.artifact-format",
                weights::artifact_format("yolo12n"),
            )
            .metadata("montgomery.model", "yolo12n")
            .metadata("montgomery.classes", "coco-80")
            .metadata("montgomery.precision", "f16")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolo12NConfig;

impl Yolo12NConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo12N<B> {
        self.init_with_classes(80, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolo12N<B> {
        Yolo12N {
            body: Yolo12BodyNConfig.init(device),
            head: Yolo11HeadConfig::new(64, 128, 256)
                .with_num_classes(num_classes)
                .init(device),
        }
    }
}

/// Native Burn YOLO12s model.
#[derive(Module, Debug)]
pub struct Yolo12S<B: Backend> {
    body: Yolo12BodySmall<B>,
    head: Yolo11Head<B>,
}

impl<B: Backend> Yolo12S<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> DecodedPredictions<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> RawPredictions<B> {
        self.head.forward_raw(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLO12s checkpoint.
    #[cfg(feature = "pretrained")]
    pub fn load_pytorch_weights(
        &mut self,
        path: impl Into<PathBuf>,
    ) -> Result<(), PytorchStoreError> {
        let mut store = pytorch_store(path);
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
            .metadata(
                "montgomery.artifact-format",
                weights::artifact_format("yolo12s"),
            )
            .metadata("montgomery.model", "yolo12s")
            .metadata("montgomery.classes", "coco-80")
            .metadata("montgomery.precision", "f16")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolo12SConfig;

impl Yolo12SConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo12S<B> {
        self.init_with_classes(80, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolo12S<B> {
        Yolo12S {
            body: Yolo12BodySConfig.init(device),
            head: Yolo11HeadConfig::new(128, 256, 512)
                .with_num_classes(num_classes)
                .init(device),
        }
    }
}

/// Native Burn YOLO12m model. The m-scale body forces the C3k chain onto the early backbone
/// stages (`parse_model`'s m/l/x rule), so it shares [`Yolo12BodyLarge`] with l/x.
#[derive(Module, Debug)]
pub struct Yolo12M<B: Backend> {
    body: Yolo12BodyLarge<B>,
    head: Yolo11Head<B>,
}

impl<B: Backend> Yolo12M<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> DecodedPredictions<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> RawPredictions<B> {
        self.head.forward_raw(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLO12m checkpoint.
    #[cfg(feature = "pretrained")]
    pub fn load_pytorch_weights(
        &mut self,
        path: impl Into<PathBuf>,
    ) -> Result<(), PytorchStoreError> {
        let mut store = pytorch_store(path);
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
            .metadata(
                "montgomery.artifact-format",
                weights::artifact_format("yolo12m"),
            )
            .metadata("montgomery.model", "yolo12m")
            .metadata("montgomery.classes", "coco-80")
            .metadata("montgomery.precision", "f16")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolo12MConfig;

impl Yolo12MConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo12M<B> {
        self.init_with_classes(80, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolo12M<B> {
        Yolo12M {
            body: Yolo12BodyMConfig.init(device),
            head: Yolo11HeadConfig::new(256, 512, 512)
                .with_num_classes(num_classes)
                .init(device),
        }
    }
}

/// Native Burn YOLO12l model. The l scale carries the learnable gamma residual on its
/// area-attention stages.
#[derive(Module, Debug)]
pub struct Yolo12L<B: Backend> {
    body: Yolo12BodyLarge<B>,
    head: Yolo11Head<B>,
}

impl<B: Backend> Yolo12L<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> DecodedPredictions<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> RawPredictions<B> {
        self.head.forward_raw(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLO12l checkpoint.
    #[cfg(feature = "pretrained")]
    pub fn load_pytorch_weights(
        &mut self,
        path: impl Into<PathBuf>,
    ) -> Result<(), PytorchStoreError> {
        let mut store = pytorch_store(path);
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
            .metadata(
                "montgomery.artifact-format",
                weights::artifact_format("yolo12l"),
            )
            .metadata("montgomery.model", "yolo12l")
            .metadata("montgomery.classes", "coco-80")
            .metadata("montgomery.precision", "f16")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolo12LConfig;

impl Yolo12LConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo12L<B> {
        self.init_with_classes(80, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolo12L<B> {
        Yolo12L {
            body: Yolo12BodyLConfig.init(device),
            head: Yolo11HeadConfig::new(256, 512, 512)
                .with_num_classes(num_classes)
                .init(device),
        }
    }
}

/// Native Burn YOLO12x model.
#[derive(Module, Debug)]
pub struct Yolo12X<B: Backend> {
    body: Yolo12BodyLarge<B>,
    head: Yolo11Head<B>,
}

impl<B: Backend> Yolo12X<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> DecodedPredictions<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> RawPredictions<B> {
        self.head.forward_raw(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLO12x checkpoint.
    #[cfg(feature = "pretrained")]
    pub fn load_pytorch_weights(
        &mut self,
        path: impl Into<PathBuf>,
    ) -> Result<(), PytorchStoreError> {
        let mut store = pytorch_store(path);
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
            .metadata(
                "montgomery.artifact-format",
                weights::artifact_format("yolo12x"),
            )
            .metadata("montgomery.model", "yolo12x")
            .metadata("montgomery.classes", "coco-80")
            .metadata("montgomery.precision", "f16")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolo12XConfig;

impl Yolo12XConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo12X<B> {
        self.init_with_classes(80, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolo12X<B> {
        Yolo12X {
            body: Yolo12BodyXConfig.init(device),
            head: Yolo11HeadConfig::new(384, 768, 768)
                .with_num_classes(num_classes)
                .init(device),
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
        features: crate::models::yolo11::body::Yolo11Features<Flex>,
        head: &Yolo11Head<Flex>,
        fixture: &GoldenFixture,
    ) {
        let p3 = features.p3.clone();
        let p4 = features.p4.clone();
        let p5 = features.p5.clone();
        let raw = head.forward_raw(crate::models::yolo11::body::Yolo11Features {
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
            "decoded_boxes_cxcywh",
            decoded.boxes,
            fixture.tensors.get("decoded_boxes_cxcywh").unwrap(),
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
                    "target/{}-coco-ultralytics-v8.4-montgomery-v1.bpk",
                    $id
                ));
                let fixture: GoldenFixture = serde_json::from_slice(
                    &std::fs::read(format!("target/{}-golden-v1.json", $id)).unwrap_or_else(|_| {
                        panic!(
                            "generate fixtures with tools/export_yolo12_fixtures.py --model {}",
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
                    "target/{}-coco-ultralytics-v8.4-montgomery-v1.bpk",
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
        yolo12n_imports_official_checkpoint_and_runs_forward,
        Yolo12NConfig,
        "yolo12n"
    );
    checkpoint_test!(
        yolo12s_imports_official_checkpoint_and_runs_forward,
        Yolo12SConfig,
        "yolo12s"
    );
    checkpoint_test!(
        yolo12m_imports_official_checkpoint_and_runs_forward,
        Yolo12MConfig,
        "yolo12m"
    );
    checkpoint_test!(
        yolo12l_imports_official_checkpoint_and_runs_forward,
        Yolo12LConfig,
        "yolo12l"
    );
    checkpoint_test!(
        yolo12x_imports_official_checkpoint_and_runs_forward,
        Yolo12XConfig,
        "yolo12x"
    );

    golden_test!(
        yolo12n_matches_ultralytics_golden_tensors,
        Yolo12NConfig,
        "yolo12n"
    );
    golden_test!(
        yolo12s_matches_ultralytics_golden_tensors,
        Yolo12SConfig,
        "yolo12s"
    );
    golden_test!(
        yolo12m_matches_ultralytics_golden_tensors,
        Yolo12MConfig,
        "yolo12m"
    );
    golden_test!(
        yolo12l_matches_ultralytics_golden_tensors,
        Yolo12LConfig,
        "yolo12l"
    );
    golden_test!(
        yolo12x_matches_ultralytics_golden_tensors,
        Yolo12XConfig,
        "yolo12x"
    );

    latency_test!(
        yolo12n_measures_single_inference_latency,
        Yolo12NConfig,
        "yolo12n"
    );
    latency_test!(
        yolo12s_measures_single_inference_latency,
        Yolo12SConfig,
        "yolo12s"
    );
    latency_test!(
        yolo12m_measures_single_inference_latency,
        Yolo12MConfig,
        "yolo12m"
    );
    latency_test!(
        yolo12l_measures_single_inference_latency,
        Yolo12LConfig,
        "yolo12l"
    );
    latency_test!(
        yolo12x_measures_single_inference_latency,
        Yolo12XConfig,
        "yolo12x"
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
                    "target/{}-coco-ultralytics-v8.4-montgomery-v1.bpk",
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
        yolo12n_measures_single_inference_latency_gpu,
        Yolo12NConfig,
        "yolo12n"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo12s_measures_single_inference_latency_gpu,
        Yolo12SConfig,
        "yolo12s"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo12m_measures_single_inference_latency_gpu,
        Yolo12MConfig,
        "yolo12m"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo12l_measures_single_inference_latency_gpu,
        Yolo12LConfig,
        "yolo12l"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo12x_measures_single_inference_latency_gpu,
        Yolo12XConfig,
        "yolo12x"
    );
}
