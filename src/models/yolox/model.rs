use burn::{
    module::Module,
    tensor::{Device, Tensor, backend::Backend},
};

use super::{
    head::{Head, HeadConfig, RawPredictions},
    pafpn::{Pafpn, PafpnConfig},
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

/// [YOLOX](https://paperswithcode.com/method/yolox) object detection architecture.
#[derive(Module, Debug)]
pub struct Yolox<B: Backend> {
    backbone: Pafpn<B>,
    head: Head<B>,
}

impl<B: Backend> Yolox<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let features = self.backbone.forward(x);
        self.head.forward(features)
    }

    /// Raw logits and differentiably decoded boxes for native training.
    pub fn forward_train(&self, x: Tensor<B, 4>) -> RawPredictions<B> {
        let features = self.backbone.forward(x);
        self.head.forward_train(features)
    }

    /// YOLOX-Nano from [`YOLOX: Exceeding YOLO Series in 2021`](https://arxiv.org/abs/2107.08430).
    ///
    /// # Arguments
    ///
    /// * `num_classes`: Number of output classes of the model.
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A YOLOX-Nano module.
    pub fn yolox_nano(num_classes: usize, device: &Device<B>) -> Self {
        YoloxConfig::new(0.33, 0.25, num_classes, true).init(device)
    }

    /// YOLOX-Tiny from [`YOLOX: Exceeding YOLO Series in 2021`](https://arxiv.org/abs/2107.08430).
    ///
    /// # Arguments
    ///
    /// * `num_classes`: Number of output classes of the model.
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A YOLOX-Tiny module.
    pub fn yolox_tiny(num_classes: usize, device: &Device<B>) -> Self {
        YoloxConfig::new(0.33, 0.375, num_classes, false).init(device)
    }

    /// YOLOX-S from [`YOLOX: Exceeding YOLO Series in 2021`](https://arxiv.org/abs/2107.08430).
    ///
    /// # Arguments
    ///
    /// * `num_classes`: Number of output classes of the model.
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A YOLOX-S module.
    pub fn yolox_s(num_classes: usize, device: &Device<B>) -> Self {
        YoloxConfig::new(0.33, 0.50, num_classes, false).init(device)
    }

    /// YOLOX-M from [`YOLOX: Exceeding YOLO Series in 2021`](https://arxiv.org/abs/2107.08430).
    ///
    /// # Arguments
    ///
    /// * `num_classes`: Number of output classes of the model.
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A YOLOX-M module.
    pub fn yolox_m(num_classes: usize, device: &Device<B>) -> Self {
        YoloxConfig::new(0.67, 0.75, num_classes, false).init(device)
    }

    /// YOLOX-L from [`YOLOX: Exceeding YOLO Series in 2021`](https://arxiv.org/abs/2107.08430).
    ///
    /// # Arguments
    ///
    /// * `num_classes`: Number of output classes of the model.
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A YOLOX-L module.
    pub fn yolox_l(num_classes: usize, device: &Device<B>) -> Self {
        YoloxConfig::new(1., 1., num_classes, false).init(device)
    }

    /// YOLOX-X from [`YOLOX: Exceeding YOLO Series in 2021`](https://arxiv.org/abs/2107.08430).
    ///
    /// # Arguments
    ///
    /// * `num_classes`: Number of output classes of the model.
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A YOLOX-X module.
    pub fn yolox_x(num_classes: usize, device: &Device<B>) -> Self {
        YoloxConfig::new(1.33, 1.25, num_classes, false).init(device)
    }

    /// Import a YOLOX PyTorch checkpoint into this Burn model.
    ///
    /// The checkpoint must use the official YOLOX state-dict structure. The model must have been
    /// initialized with the same class count as the checkpoint head.
    #[cfg(feature = "pretrained")]
    pub fn load_pytorch_weights(
        &mut self,
        path: impl Into<PathBuf>,
    ) -> Result<(), PytorchStoreError> {
        let mut store = PytorchStore::from_file(path)
            // State dict contains "model", "amp", "optimizer", "start_epoch"
            .with_top_level_key("model")
            // Map backbone.C3_* -> backbone.c3_*
            .with_key_remapping("backbone\\.C3_(.+)", "backbone.c3_$1")
            // Map backbone.backbone.dark[i].0.* -> backbone.backbone.dark[i].conv.*
            .with_key_remapping("(backbone\\.backbone\\.dark[2-5])\\.0\\.(.+)", "$1.conv.$2")
            // Map backbone.backbone.dark[i].1.* -> backbone.backbone.dark[i].c3.*
            .with_key_remapping("(backbone\\.backbone\\.dark[2-4])\\.1\\.(.+)", "$1.c3.$2")
            // Map backbone.backbone.dark5.1.* -> backbone.backbone.dark5.spp.*
            .with_key_remapping("(backbone\\.backbone\\.dark5)\\.1\\.(.+)", "$1.spp.$2")
            // Map backbone.backbone.dark5.2.* -> backbone.backbone.dark5.c3.*
            .with_key_remapping("(backbone\\.backbone\\.dark5)\\.2\\.(.+)", "$1.c3.$2")
            // Map head.{cls | reg}_convs.x.[i].* -> head.{cls | reg}_convs.x.conv[i].*
            .with_key_remapping(
                "(head\\.(cls|reg)_convs\\.[0-9]+)\\.([0-9]+)\\.(.+)",
                "$1.conv$3.$4",
            );

        self.load_from(&mut store)?;

        Ok(())
    }

    /// Load a boquilens-native, half-precision YOLOX Burnpack artifact.
    #[cfg(feature = "pretrained")]
    pub fn load_burnpack_weights(&mut self, path: impl Into<PathBuf>) -> Result<(), BurnpackError> {
        let mut store = BurnpackStore::from_file(path.into())
            .with_from_adapter(HalfPrecisionAdapter::new())
            .zero_copy(true);
        self.load_from(&mut store).map(|_| ())
    }

    /// Save an imported official checkpoint as a versioned native artifact.
    #[cfg(feature = "pretrained")]
    pub fn save_burnpack_weights(
        &self,
        path: impl Into<PathBuf>,
        model_id: &str,
    ) -> Result<(), BurnpackError> {
        let mut store = BurnpackStore::from_file(path.into())
            .metadata(
                "boquilens.artifact-format",
                weights::artifact_format(model_id),
            )
            .metadata("boquilens.model", model_id)
            .metadata("boquilens.classes", "coco-80")
            .metadata("boquilens.precision", "f16")
            .metadata("boquilens.source", "yolox-official-v0.1.1rc0")
            .metadata("boquilens.license", "Apache-2.0")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

/// [YOLOX detector](Yolox) configuration.
pub struct YoloxConfig {
    backbone: PafpnConfig,
    head: HeadConfig,
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

    /// The fixture's preprocessed reference is consumed raw: YOLOX models intentionally run on the
    /// original [0, 255] pixel range without normalization.
    fn load_reference_input(id: &str, device: &Device<Flex>) -> Tensor<Flex, 4> {
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
    }

    fn official_checkpoint_path(id: &str) -> PathBuf {
        let checkpoint = weights::official_checkpoint(id).expect("registered YOLOX checkpoint");
        let path = PathBuf::from("target/checkpoints").join(checkpoint.filename);
        assert!(
            path.exists(),
            "download the official checkpoint from {} to {} (SHA-256 {})",
            checkpoint.url,
            path.display(),
            checkpoint.sha256
        );
        path
    }

    macro_rules! checkpoint_test {
        ($fn_name:ident, $constructor:ident, $id:literal) => {
            /// Import the official Apache-2.0 checkpoint and run a forward.
            #[test]
            #[ignore]
            fn $fn_name() {
                let checkpoint = official_checkpoint_path($id);
                let worker = std::thread::Builder::new()
                    .stack_size(64 * 1024 * 1024)
                    .spawn(move || {
                        let device = <Device<Flex>>::default();
                        let mut model: Yolox<Flex> = Yolox::$constructor(80, &device);
                        model.load_pytorch_weights(checkpoint).unwrap();
                        let output = model.forward(Tensor::zeros([1, 3, 64, 64], &device));
                        assert_eq!(output.dims(), [1, 84, 85]);
                    })
                    .unwrap();
                worker.join().unwrap();
            }
        };
    }

    macro_rules! golden_test {
        ($fn_name:ident, $constructor:ident, $id:literal) => {
            /// Compare Burn's graph numerics against the official YOLOX PyTorch forward on the
            /// same preprocessed input. Requires tools/export_yolox_fixtures.py output under
            /// target/. Kept ignored because the fixtures derive from the external checkpoint.
            #[test]
            #[ignore]
            fn $fn_name() {
                let fixture_path = format!("target/{}-golden-v1.json", $id);
                let fixture: GoldenFixture =
                    serde_json::from_slice(&std::fs::read(&fixture_path).unwrap_or_else(|err| {
                        panic!(
                            "generate fixtures with tools/export_yolox_fixtures.py (path \
                             {fixture_path}, cwd {}, {err})",
                            std::env::current_dir().unwrap().display()
                        )
                    }))
                    .unwrap();
                assert_eq!(fixture.format, "boquilens-yolox-golden-v1");
                assert_eq!(fixture.model, $id);

                let checkpoint = official_checkpoint_path($id);
                let worker = std::thread::Builder::new()
                    .stack_size(64 * 1024 * 1024)
                    .spawn(move || {
                        let device = <Device<Flex>>::default();
                        let mut model: Yolox<Flex> = Yolox::$constructor(80, &device);
                        model.load_pytorch_weights(checkpoint).unwrap();
                        let input = load_reference_input($id, &device);

                        let dark = model.backbone.backbone().forward(input.clone());
                        assert_golden("backbone_dark3", dark.0, &fixture.tensors["backbone_dark3"]);
                        assert_golden("backbone_dark4", dark.1, &fixture.tensors["backbone_dark4"]);
                        assert_golden("backbone_dark5", dark.2, &fixture.tensors["backbone_dark5"]);

                        let fpn = model.backbone.forward(input.clone());
                        assert_golden("pafpn_p3", fpn.0, &fixture.tensors["pafpn_p3"]);
                        assert_golden("pafpn_p4", fpn.1, &fixture.tensors["pafpn_p4"]);
                        assert_golden("pafpn_p5", fpn.2, &fixture.tensors["pafpn_p5"]);

                        let decoded = model.forward(input);
                        assert_golden("head_decoded", decoded, &fixture.tensors["head_decoded"]);
                    })
                    .unwrap();
                worker.join().unwrap();
            }
        };
    }

    checkpoint_test!(
        yolox_nano_imports_official_checkpoint_and_runs_forward,
        yolox_nano,
        "yolox-nano"
    );

    checkpoint_test!(
        yolox_tiny_imports_official_checkpoint_and_runs_forward,
        yolox_tiny,
        "yolox-tiny"
    );
    checkpoint_test!(
        yolox_s_imports_official_checkpoint_and_runs_forward,
        yolox_s,
        "yolox-s"
    );
    checkpoint_test!(
        yolox_m_imports_official_checkpoint_and_runs_forward,
        yolox_m,
        "yolox-m"
    );
    checkpoint_test!(
        yolox_l_imports_official_checkpoint_and_runs_forward,
        yolox_l,
        "yolox-l"
    );
    checkpoint_test!(
        yolox_x_imports_official_checkpoint_and_runs_forward,
        yolox_x,
        "yolox-x"
    );

    golden_test!(
        yolox_nano_matches_official_pytorch_outputs,
        yolox_nano,
        "yolox-nano"
    );
    golden_test!(
        yolox_tiny_matches_official_pytorch_outputs,
        yolox_tiny,
        "yolox-tiny"
    );
    golden_test!(yolox_s_matches_official_pytorch_outputs, yolox_s, "yolox-s");
    golden_test!(yolox_m_matches_official_pytorch_outputs, yolox_m, "yolox-m");
    golden_test!(yolox_l_matches_official_pytorch_outputs, yolox_l, "yolox-l");
    golden_test!(yolox_x_matches_official_pytorch_outputs, yolox_x, "yolox-x");
}

impl YoloxConfig {
    /// Create a new instance of the YOLOX detector [config](YoloxConfig).
    pub fn new(depth: f64, width: f64, num_classes: usize, depthwise: bool) -> Self {
        let backbone = PafpnConfig::new(depth, width, depthwise);
        let head = HeadConfig::new(num_classes, width, depthwise);

        Self { backbone, head }
    }

    /// Initialize a new [YOLOX detector](Yolox) module.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolox<B> {
        Yolox {
            backbone: self.backbone.init(device),
            head: self.head.init(device),
        }
    }
}
