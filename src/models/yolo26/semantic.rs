//! Native Burn implementation of the Ultralytics YOLO26-sem semantic-segmentation family
//! (n/s/m/l/x).
//!
//! Semantic segmentation is a separate task from instance segmentation (`-seg`): instead of
//! per-object boxes plus binary masks it predicts one dense class map for the whole image —
//! every pixel gets the class id with the highest logit. There is deliberately no sharing with
//! the instance-segmentation graph beyond the backbone/neck block vocabulary.
//!
//! The semantic graph reuses the detect body's layers 0-16 verbatim (backbone 0-10, neck
//! upsample/concat/C3k2 stages 11-16) and Ultralytics' `SemanticSegment` head (`model.17`),
//! which classifies the P3/8 feature map with a two-layer tower
//! (`Conv(c3, c3, 3)` + biased `1x1` projection to `nc`). The auxiliary P4 head
//! (`aux_head`, deep supervision) exists in the released checkpoints but is training-only and
//! is therefore absent from the inference graph — its keys are ignored on import, like the
//! one-to-many branches of default (non-training) detect builds.
//!
//! Inference returns raw logits at stride 8 (`[batch, nc, H/8, W/8]`); the runtime upsamples
//! them to the letterboxed canvas with `align_corners = false` bilinear interpolation, inverts
//! the letterbox geometry, and takes the per-pixel argmax, mirroring
//! `SemanticSegmentationPredictor.postprocess`.

use burn::{
    module::Module,
    nn::conv::{Conv2d, Conv2dConfig},
    tensor::{Device, Tensor},
};

#[cfg(feature = "pretrained")]
use burn_pack::Error as BurnpackError;
#[cfg(feature = "pretrained")]
use burn_store::{
    BurnpackStore, HalfPrecisionAdapter, ModuleSnapshot, PytorchStore, PytorchStoreError,
};

use super::blocks::{
    C2Psa, C2PsaConfig, C3k2, C3k2C3k, C3k2C3kConfig, C3k2Config, Conv, ConvConfig, Sppf,
    SppfConfig, upsample_nearest_2x,
};

#[cfg(feature = "pretrained")]
use super::weights;

/// Number of Cityscapes classes the official YOLO26-sem checkpoints were trained on.
///
/// `pack-weights` targets these 19-class checkpoints. Other class counts (e.g. ADE20K's 150)
/// are supported through `init_with_classes`, which trained-artifact loading uses with the
/// artifact's embedded class table.
pub const NUM_CLASSES: usize = 19;

/// Feature maps consumed by the semantic head: P3/8 for classification.
///
/// `p4_aux` is the intermediate neck P4 (layer 13). It is carried for fixture parity and future
/// auxiliary use; the inference classifier reads P3 only, matching `SemanticSegment.forward`.
pub struct Yolo26SemanticFeatures {
    /// P3/8 feature map (layer 16), the classifier input.
    pub p3: Tensor<4>,
    /// Intermediate neck P4 (layer 13).
    pub p4_aux: Tensor<4>,
}

/// YOLO26 semantic-segmentation model output: raw per-pixel class logits at stride 8.
pub struct SemanticOutput {
    /// Unnormalized logits, `[batch, num_classes, height / 8, width / 8]`.
    pub logits: Tensor<4>,
}

/// Ultralytics `SemanticSegment` inference head: a `Conv(c, c, 3)` plus a biased 1x1 projection
/// to the class count.
///
/// Field names deliberately match the official `model.17.classifier.*` checkpoint keys after
/// remapping (`classifier.0` is the `Conv`, `classifier.1` the projection).
#[derive(Module, Debug)]
pub struct SemanticHead {
    classifier_0: Conv,
    classifier_1: Conv2d,
    num_classes: usize,
}

impl SemanticHead {
    pub fn forward(&self, features: Yolo26SemanticFeatures) -> SemanticOutput {
        let [batch, _, _, _] = features.p3.dims();
        let logits = self
            .classifier_1
            .forward(self.classifier_0.forward(features.p3));
        debug_assert_eq!(logits.dims()[0], batch);
        debug_assert_eq!(logits.dims()[1], self.num_classes);
        SemanticOutput { logits }
    }
}

#[derive(Debug)]
pub struct SemanticHeadConfig {
    channels: usize,
    num_classes: usize,
}

impl SemanticHeadConfig {
    /// Declare the head from the P3 input width (`c_mid = ch[0]` in the Ultralytics source).
    pub fn new(channels: usize) -> Self {
        Self {
            channels,
            num_classes: NUM_CLASSES,
        }
    }

    pub fn with_num_classes(mut self, num_classes: usize) -> Self {
        assert!(num_classes > 0, "class count must be positive");
        self.num_classes = num_classes;
        self
    }

    pub fn init(&self, device: &Device) -> SemanticHead {
        SemanticHead {
            classifier_0: ConvConfig::new(self.channels, self.channels, 3, 1).init(device),
            classifier_1: Conv2dConfig::new([self.channels, self.num_classes], [1, 1])
                .with_bias(true)
                .init(device),
            num_classes: self.num_classes,
        }
    }
}

/// YOLO26 semantic body (layers 0-16), n and s scales: plain C3k2 bottleneck chains on the
/// early backbone stages, C3k chains on the later stages. Layer indices — and therefore the
/// checkpoint keys — are identical to the detect body; only the downsampling tail (layers
/// 17-22) is absent.
#[derive(Module, Debug)]
pub struct Yolo26SemanticBodySmall {
    model_0: Conv,
    model_1: Conv,
    model_2: C3k2,
    model_3: Conv,
    model_4: C3k2,
    model_5: Conv,
    model_6: C3k2C3k,
    model_7: Conv,
    model_8: C3k2C3k,
    model_9: Sppf,
    model_10: C2Psa,
    model_13: C3k2C3k,
    model_16: C3k2C3k,
}

impl Yolo26SemanticBodySmall {
    pub fn forward(&self, input: Tensor<4>) -> Yolo26SemanticFeatures {
        let x = self.model_0.forward(input);
        let x = self.model_1.forward(x);
        let x = self.model_2.forward(x);
        let x = self.model_3.forward(x);
        let route_p3 = self.model_4.forward(x);

        let x = self.model_5.forward(route_p3.clone());
        let route_p4 = self.model_6.forward(x);

        let x = self.model_7.forward(route_p4.clone());
        let x = self.model_8.forward(x);
        let x = self.model_9.forward(x);
        let route_p5 = self.model_10.forward(x);

        let x = upsample_nearest_2x(route_p5.clone());
        let x = Tensor::cat(vec![x, route_p4], 1);
        let p4_aux = self.model_13.forward(x);

        let x = upsample_nearest_2x(p4_aux.clone());
        let x = Tensor::cat(vec![x, route_p3], 1);
        let p3 = self.model_16.forward(x);

        Yolo26SemanticFeatures { p3, p4_aux }
    }
}

/// YOLO26 semantic body (layers 0-16), m/l/x scales: `parse_model` forces `c3k=True` on every
/// C3k2 stage, so the early backbone stages build C3k chains at the YAML's 0.25 expansion.
#[derive(Module, Debug)]
pub struct Yolo26SemanticBodyLarge {
    model_0: Conv,
    model_1: Conv,
    model_2: C3k2C3k,
    model_3: Conv,
    model_4: C3k2C3k,
    model_5: Conv,
    model_6: C3k2C3k,
    model_7: Conv,
    model_8: C3k2C3k,
    model_9: Sppf,
    model_10: C2Psa,
    model_13: C3k2C3k,
    model_16: C3k2C3k,
}

impl Yolo26SemanticBodyLarge {
    pub fn forward(&self, input: Tensor<4>) -> Yolo26SemanticFeatures {
        let x = self.model_0.forward(input);
        let x = self.model_1.forward(x);
        let x = self.model_2.forward(x);
        let x = self.model_3.forward(x);
        let route_p3 = self.model_4.forward(x);

        let x = self.model_5.forward(route_p3.clone());
        let route_p4 = self.model_6.forward(x);

        let x = self.model_7.forward(route_p4.clone());
        let x = self.model_8.forward(x);
        let x = self.model_9.forward(x);
        let route_p5 = self.model_10.forward(x);

        let x = upsample_nearest_2x(route_p5.clone());
        let x = Tensor::cat(vec![x, route_p4], 1);
        let p4_aux = self.model_13.forward(x);

        let x = upsample_nearest_2x(p4_aux.clone());
        let x = Tensor::cat(vec![x, route_p3], 1);
        let p3 = self.model_16.forward(x);

        Yolo26SemanticFeatures { p3, p4_aux }
    }
}

/// Build the PyTorch-state store shared by every YOLO26-sem scale variant.
///
/// The body is layers 0-16 (identical indices to the detect checkpoint; layers 11/12/14/15
/// carry no tensors) and the head is `model.17.classifier`. The training-only `aux_head`
/// keys are intentionally unmapped and ignored on import.
#[cfg(feature = "pretrained")]
fn pytorch_store(path: impl Into<std::path::PathBuf>) -> PytorchStore {
    PytorchStore::from_file(path)
        .with_top_level_key("model")
        // Body layers retain their Ultralytics graph indices. The head is model.17, so this
        // rule must not match it.
        .with_key_remapping("model\\.([0-9]|1[0-6])\\.(.+)", "body.model_$1.$2")
        // model.17.classifier.0 is the 3x3 Conv, model.17.classifier.1 the 1x1 projection.
        .with_key_remapping(
            "model\\.17\\.classifier\\.0\\.conv\\.(.+)",
            "head.classifier_0.conv.$1",
        )
        .with_key_remapping(
            "model\\.17\\.classifier\\.0\\.bn\\.(.+)",
            "head.classifier_0.bn.$1",
        )
        .with_key_remapping("model\\.17\\.classifier\\.1\\.(.+)", "head.classifier_1.$1")
}

macro_rules! sem_model {
    ($model:ident, $config:ident, $body_struct:ident, $body_config:ident, $p3_channels:expr, $id:literal, $doc:expr) => {
        #[doc = $doc]
        #[derive(Module, Debug)]
        pub struct $model {
            body: $body_struct,
            head: SemanticHead,
        }

        impl $model {
            pub fn forward(&self, input: Tensor<4>) -> SemanticOutput {
                self.head.forward(self.body.forward(input))
            }

            /// Import tensor-only state exported from an official Ultralytics checkpoint.
            #[cfg(feature = "pretrained")]
            pub fn load_pytorch_weights(
                &mut self,
                path: impl Into<std::path::PathBuf>,
            ) -> Result<(), PytorchStoreError> {
                let mut store = pytorch_store(path);
                self.load_from(&mut store).map(|_| ())
            }

            /// Load Montgomery's versioned, half-precision native Burnpack artifact.
            #[cfg(feature = "pretrained")]
            pub fn load_burnpack_weights(
                &mut self,
                path: impl Into<std::path::PathBuf>,
            ) -> Result<(), BurnpackError> {
                let mut store = BurnpackStore::from_file(path.into())
                    .with_from_adapter(HalfPrecisionAdapter::new());
                self.load_from(&mut store).map(|_| ())
            }

            /// Save a versioned native artifact. Existing files are deliberately not overwritten.
            #[cfg(feature = "pretrained")]
            pub fn save_burnpack_weights(
                &self,
                path: impl Into<std::path::PathBuf>,
            ) -> Result<(), BurnpackError> {
                let mut store = BurnpackStore::from_file(path.into())
                    .metadata("montgomery.artifact-format", weights::artifact_format($id))
                    .metadata("montgomery.model", $id)
                    .metadata("montgomery.classes", "cityscapes-19")
                    .metadata("montgomery.precision", "f16")
                    .metadata("montgomery.source", "ultralytics-v8.4")
                    .metadata("montgomery.license", "AGPL-3.0")
                    .with_to_adapter(HalfPrecisionAdapter::new());
                self.save_into(&mut store)
            }
        }

        #[derive(Debug, Default)]
        pub struct $config;

        impl $config {
            pub fn init(&self, device: &Device) -> $model {
                self.init_with_classes(NUM_CLASSES, device)
            }

            pub fn init_with_classes(&self, num_classes: usize, device: &Device) -> $model {
                $model {
                    body: $body_config.init(device),
                    head: SemanticHeadConfig::new($p3_channels)
                        .with_num_classes(num_classes)
                        .init(device),
                }
            }
        }
    };
}

sem_model!(
    Yolo26SemN,
    Yolo26SemNConfig,
    Yolo26SemanticBodySmall,
    Yolo26SemanticBodyNConfig,
    64,
    "yolo26n-sem",
    "Native Burn YOLO26n-sem semantic-segmentation model."
);
sem_model!(
    Yolo26SemS,
    Yolo26SemSConfig,
    Yolo26SemanticBodySmall,
    Yolo26SemanticBodySConfig,
    128,
    "yolo26s-sem",
    "Native Burn YOLO26s-sem semantic-segmentation model."
);
sem_model!(
    Yolo26SemM,
    Yolo26SemMConfig,
    Yolo26SemanticBodyLarge,
    Yolo26SemanticBodyMConfig,
    256,
    "yolo26m-sem",
    "Native Burn YOLO26m-sem semantic-segmentation model."
);
sem_model!(
    Yolo26SemL,
    Yolo26SemLConfig,
    Yolo26SemanticBodyLarge,
    Yolo26SemanticBodyLConfig,
    256,
    "yolo26l-sem",
    "Native Burn YOLO26l-sem semantic-segmentation model."
);
sem_model!(
    Yolo26SemX,
    Yolo26SemXConfig,
    Yolo26SemanticBodyLarge,
    Yolo26SemanticBodyXConfig,
    384,
    "yolo26x-sem",
    "Native Burn YOLO26x-sem semantic-segmentation model."
);

/// Configuration for the fixed YOLO26n-sem body (depth 0.50, width 0.25, max channels 1024).
///
/// Channel table is the detect body's layers 0-16 verbatim.
#[derive(Debug, Default)]
pub struct Yolo26SemanticBodyNConfig;

impl Yolo26SemanticBodyNConfig {
    pub fn init(&self, device: &Device) -> Yolo26SemanticBodySmall {
        Yolo26SemanticBodySmall {
            model_0: ConvConfig::new(3, 16, 3, 2).init(device),
            model_1: ConvConfig::new(16, 32, 3, 2).init(device),
            model_2: C3k2Config::new(32, 64, 1, 0.25, true).init(device),
            model_3: ConvConfig::new(64, 64, 3, 2).init(device),
            model_4: C3k2Config::new(64, 128, 1, 0.25, true).init(device),
            model_5: ConvConfig::new(128, 128, 3, 2).init(device),
            model_6: C3k2C3kConfig::new(128, 128, 1, true, 0.5).init(device),
            model_7: ConvConfig::new(128, 256, 3, 2).init(device),
            model_8: C3k2C3kConfig::new(256, 256, 1, true, 0.5).init(device),
            model_9: SppfConfig::new(256, 3, true).init(device),
            model_10: C2PsaConfig::new(256, 1).init(device),
            model_13: C3k2C3kConfig::new(384, 128, 1, true, 0.5).init(device),
            model_16: C3k2C3kConfig::new(256, 64, 1, true, 0.5).init(device),
        }
    }
}

/// Configuration for the fixed YOLO26s-sem body (depth 0.50, width 0.50, max channels 1024).
#[derive(Debug, Default)]
pub struct Yolo26SemanticBodySConfig;

impl Yolo26SemanticBodySConfig {
    pub fn init(&self, device: &Device) -> Yolo26SemanticBodySmall {
        Yolo26SemanticBodySmall {
            model_0: ConvConfig::new(3, 32, 3, 2).init(device),
            model_1: ConvConfig::new(32, 64, 3, 2).init(device),
            model_2: C3k2Config::new(64, 128, 1, 0.25, true).init(device),
            model_3: ConvConfig::new(128, 128, 3, 2).init(device),
            model_4: C3k2Config::new(128, 256, 1, 0.25, true).init(device),
            model_5: ConvConfig::new(256, 256, 3, 2).init(device),
            model_6: C3k2C3kConfig::new(256, 256, 1, true, 0.5).init(device),
            model_7: ConvConfig::new(256, 512, 3, 2).init(device),
            model_8: C3k2C3kConfig::new(512, 512, 1, true, 0.5).init(device),
            model_9: SppfConfig::new(512, 3, true).init(device),
            model_10: C2PsaConfig::new(512, 1).init(device),
            model_13: C3k2C3kConfig::new(768, 256, 1, true, 0.5).init(device),
            model_16: C3k2C3kConfig::new(512, 128, 1, true, 0.5).init(device),
        }
    }
}

/// Configuration for the fixed YOLO26m-sem body (depth 0.50, width 1.00, max channels 512).
#[derive(Debug, Default)]
pub struct Yolo26SemanticBodyMConfig;

impl Yolo26SemanticBodyMConfig {
    pub fn init(&self, device: &Device) -> Yolo26SemanticBodyLarge {
        Yolo26SemanticBodyLarge {
            model_0: ConvConfig::new(3, 64, 3, 2).init(device),
            model_1: ConvConfig::new(64, 128, 3, 2).init(device),
            model_2: C3k2C3kConfig::new(128, 256, 1, true, 0.25).init(device),
            model_3: ConvConfig::new(256, 256, 3, 2).init(device),
            model_4: C3k2C3kConfig::new(256, 512, 1, true, 0.25).init(device),
            model_5: ConvConfig::new(512, 512, 3, 2).init(device),
            model_6: C3k2C3kConfig::new(512, 512, 1, true, 0.5).init(device),
            model_7: ConvConfig::new(512, 512, 3, 2).init(device),
            model_8: C3k2C3kConfig::new(512, 512, 1, true, 0.5).init(device),
            model_9: SppfConfig::new(512, 3, true).init(device),
            model_10: C2PsaConfig::new(512, 1).init(device),
            model_13: C3k2C3kConfig::new(1024, 512, 1, true, 0.5).init(device),
            model_16: C3k2C3kConfig::new(1024, 256, 1, true, 0.5).init(device),
        }
    }
}

/// Configuration for the fixed YOLO26l-sem body (depth 1.00, width 1.00, max channels 512).
#[derive(Debug, Default)]
pub struct Yolo26SemanticBodyLConfig;

impl Yolo26SemanticBodyLConfig {
    pub fn init(&self, device: &Device) -> Yolo26SemanticBodyLarge {
        Yolo26SemanticBodyLarge {
            model_0: ConvConfig::new(3, 64, 3, 2).init(device),
            model_1: ConvConfig::new(64, 128, 3, 2).init(device),
            model_2: C3k2C3kConfig::new(128, 256, 2, true, 0.25).init(device),
            model_3: ConvConfig::new(256, 256, 3, 2).init(device),
            model_4: C3k2C3kConfig::new(256, 512, 2, true, 0.25).init(device),
            model_5: ConvConfig::new(512, 512, 3, 2).init(device),
            model_6: C3k2C3kConfig::new(512, 512, 2, true, 0.5).init(device),
            model_7: ConvConfig::new(512, 512, 3, 2).init(device),
            model_8: C3k2C3kConfig::new(512, 512, 2, true, 0.5).init(device),
            model_9: SppfConfig::new(512, 3, true).init(device),
            model_10: C2PsaConfig::new(512, 2).init(device),
            model_13: C3k2C3kConfig::new(1024, 512, 2, true, 0.5).init(device),
            model_16: C3k2C3kConfig::new(1024, 256, 2, true, 0.5).init(device),
        }
    }
}

/// Configuration for the fixed YOLO26x-sem body (depth 1.00, width 1.50, max channels 512).
#[derive(Debug, Default)]
pub struct Yolo26SemanticBodyXConfig;

impl Yolo26SemanticBodyXConfig {
    pub fn init(&self, device: &Device) -> Yolo26SemanticBodyLarge {
        Yolo26SemanticBodyLarge {
            model_0: ConvConfig::new(3, 96, 3, 2).init(device),
            model_1: ConvConfig::new(96, 192, 3, 2).init(device),
            model_2: C3k2C3kConfig::new(192, 384, 2, true, 0.25).init(device),
            model_3: ConvConfig::new(384, 384, 3, 2).init(device),
            model_4: C3k2C3kConfig::new(384, 768, 2, true, 0.25).init(device),
            model_5: ConvConfig::new(768, 768, 3, 2).init(device),
            model_6: C3k2C3kConfig::new(768, 768, 2, true, 0.5).init(device),
            model_7: ConvConfig::new(768, 768, 3, 2).init(device),
            model_8: C3k2C3kConfig::new(768, 768, 2, true, 0.5).init(device),
            model_9: SppfConfig::new(768, 3, true).init(device),
            model_10: C2PsaConfig::new(768, 2).init(device),
            model_13: C3k2C3kConfig::new(1536, 768, 2, true, 0.5).init(device),
            model_16: C3k2C3kConfig::new(1536, 384, 2, true, 0.5).init(device),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Random-initialising N to X takes tens of seconds at opt-level 0, and
    // tests/integration.rs already runs N. CI runs this with `-- --ignored every_scale`.
    #[test]
    #[ignore = "slow: builds every semantic scale"]
    fn semantic_models_decode_stride_8_logits_for_every_scale() {
        let worker = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let device = Device::flex();
                let input = Tensor::zeros([1, 3, 64, 64], &device);
                // A 64 px input yields 8x8 logits with the 19 Cityscapes classes by default.
                let output = Yolo26SemNConfig.init(&device).forward(input.clone());
                assert_eq!(output.logits.dims(), [1, NUM_CLASSES, 8, 8]);
                let output = Yolo26SemSConfig.init(&device).forward(input.clone());
                assert_eq!(output.logits.dims(), [1, NUM_CLASSES, 8, 8]);
                let output = Yolo26SemMConfig.init(&device).forward(input.clone());
                assert_eq!(output.logits.dims(), [1, NUM_CLASSES, 8, 8]);
                let output = Yolo26SemLConfig.init(&device).forward(input.clone());
                assert_eq!(output.logits.dims(), [1, NUM_CLASSES, 8, 8]);
                let output = Yolo26SemXConfig.init(&device).forward(input);
                assert_eq!(output.logits.dims(), [1, NUM_CLASSES, 8, 8]);
            })
            .expect("shape-test worker should start");
        worker.join().expect("shape-test worker should not panic");
    }

    #[test]
    fn semantic_class_count_only_resizes_the_projection() {
        let worker = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let device = Device::flex();
                // Custom class counts (e.g. ADE20K's 150) only resize the projection.
                let output = Yolo26SemNConfig
                    .init_with_classes(150, &device)
                    .forward(Tensor::zeros([1, 3, 64, 64], &device));
                assert_eq!(output.logits.dims(), [1, 150, 8, 8]);
            })
            .expect("shape-test worker should start");
        worker.join().expect("shape-test worker should not panic");
    }
}

#[cfg(all(test, feature = "pretrained"))]
mod parity_tests {
    use super::*;
    use burn::tensor::{ElementConversion, TensorData};

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

    fn assert_golden<const D: usize>(name: &str, actual: Tensor<D>, expected: &GoldenTensor) {
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

    fn load_reference_image(id: &str, device: &Device) -> Tensor<4> {
        let image = image::open(format!("target/{id}-preprocessed-reference.png"))
            .unwrap()
            .into_rgb8();
        let shape = [image.height() as usize, image.width() as usize, 3];
        Tensor::<3>::from_data(
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
                    "convert {}.pt with tools/export_checkpoint_state.py first",
                    $id
                );
                let worker = std::thread::Builder::new()
                    .stack_size(64 * 1024 * 1024)
                    .spawn(move || {
                        let device = Device::flex();
                        let mut model = <$config>::default().init(&device);
                        model.load_pytorch_weights(checkpoint).unwrap();
                        let output = model.forward(Tensor::zeros([1, 3, 64, 64], &device));
                        assert_eq!(output.logits.dims(), [1, NUM_CLASSES, 8, 8]);
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
                    crate::models::yolo26::weights::artifact_filename($id)
                ));
                let fixture: GoldenFixture = serde_json::from_slice(
                    &std::fs::read(format!("target/{}-golden-v1.json", $id)).unwrap_or_else(|_| {
                        panic!(
                            "generate fixtures with tools/export_yolo26_sem_fixtures.py --model {}",
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
                        let device = Device::flex();
                        let mut model = <$config>::default().init(&device);
                        model.load_burnpack_weights(checkpoint).unwrap();
                        let features = model.body.forward(load_reference_image($id, &device));
                        let p3 = features.p3.clone();
                        let p4_aux = features.p4_aux.clone();
                        let output = model.head.forward(features);

                        assert_golden("body_p3", p3, fixture.tensors.get("body_p3").unwrap());
                        assert_golden(
                            "body_p4_aux",
                            p4_aux,
                            fixture.tensors.get("body_p4_aux").unwrap(),
                        );
                        assert_golden(
                            "logits",
                            output.logits,
                            fixture.tensors.get("logits").unwrap(),
                        );
                    })
                    .unwrap();
                worker.join().unwrap();
            }
        };
    }

    macro_rules! latency_test {
        ($fn_name:ident, $config:ty, $id:literal) => {
            /// Measure single-image batch-1 inference latency with the packed native artifact on
            /// the Flex CPU backend. Run with
            /// `cargo test --release <id> -- --ignored --nocapture --test-threads 1` after the
            /// weight-prep loop.
            #[test]
            #[ignore]
            fn $fn_name() {
                let checkpoint = std::path::PathBuf::from(format!(
                    "target/{}",
                    crate::models::yolo26::weights::artifact_filename($id)
                ));
                assert!(
                    checkpoint.exists(),
                    "pack the {} artifact with pack-weights first",
                    $id
                );
                let worker = std::thread::Builder::new()
                    .stack_size(64 * 1024 * 1024)
                    .spawn(move || {
                        let device = Device::flex();
                        let mut model = <$config>::default().init(&device);
                        model.load_burnpack_weights(checkpoint).unwrap();
                        let input = Tensor::<4>::zeros([1, 3, 640, 640], &device);
                        const WARMUP_RUNS: usize = 3;
                        const TIMED_RUNS: usize = 10;

                        for _ in 0..WARMUP_RUNS {
                            let output = model.forward(input.clone());
                            let _ = output.logits.sum().into_data();
                        }
                        let mut samples = Vec::with_capacity(TIMED_RUNS);
                        for _ in 0..TIMED_RUNS {
                            let started = std::time::Instant::now();
                            let output = model.forward(input.clone());
                            let _ = output.logits.sum().into_data();
                            samples.push(started.elapsed().as_secs_f64() * 1e3);
                        }
                        samples.sort_by(|a, b| a.total_cmp(b));
                        let median = samples[samples.len() / 2];
                        let min = samples[0];
                        println!(
                            "{:>11}: {:>7.1} ms median, {:>7.1} ms min  (single image, batch 1, 640 px, {TIMED_RUNS} runs)",
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
            /// Measure single-image batch-1 inference latency with the packed native artifact on
            /// the Wgpu GPU backend. Requires the gpu feature and a packed native artifact.
            #[test]
            #[ignore]
            fn $fn_name() {
                let checkpoint = std::path::PathBuf::from(format!(
                    "target/{}",
                    crate::models::yolo26::weights::artifact_filename($id)
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
                        let mut model = <$config>::default().init(&device);
                        model.load_burnpack_weights(checkpoint).unwrap();
                        let input = Tensor::<4>::zeros([1, 3, 640, 640], &device);
                        const WARMUP_RUNS: usize = 3;
                        const TIMED_RUNS: usize = 10;

                        for _ in 0..WARMUP_RUNS {
                            let output = model.forward(input.clone());
                            let _ = output.logits.sum().into_data();
                        }
                        let mut samples = Vec::with_capacity(TIMED_RUNS);
                        for _ in 0..TIMED_RUNS {
                            let started = std::time::Instant::now();
                            let output = model.forward(input.clone());
                            let _ = output.logits.sum().into_data();
                            samples.push(started.elapsed().as_secs_f64() * 1e3);
                        }
                        samples.sort_by(|a, b| a.total_cmp(b));
                        let median = samples[samples.len() / 2];
                        let min = samples[0];
                        println!(
                            "{:>11}: {:>7.1} ms median, {:>7.1} ms min  (single image, batch 1, 640 px, {TIMED_RUNS} runs, Wgpu GPU)",
                            $id, median, min,
                        );
                    })
                    .unwrap();
                worker.join().unwrap();
            }
        };
    }

    checkpoint_test!(
        yolo26n_sem_imports_official_checkpoint_and_runs_forward,
        Yolo26SemNConfig,
        "yolo26n-sem"
    );
    checkpoint_test!(
        yolo26s_sem_imports_official_checkpoint_and_runs_forward,
        Yolo26SemSConfig,
        "yolo26s-sem"
    );
    checkpoint_test!(
        yolo26m_sem_imports_official_checkpoint_and_runs_forward,
        Yolo26SemMConfig,
        "yolo26m-sem"
    );
    checkpoint_test!(
        yolo26l_sem_imports_official_checkpoint_and_runs_forward,
        Yolo26SemLConfig,
        "yolo26l-sem"
    );
    checkpoint_test!(
        yolo26x_sem_imports_official_checkpoint_and_runs_forward,
        Yolo26SemXConfig,
        "yolo26x-sem"
    );

    golden_test!(
        yolo26n_sem_matches_ultralytics_golden_tensors,
        Yolo26SemNConfig,
        "yolo26n-sem"
    );
    golden_test!(
        yolo26s_sem_matches_ultralytics_golden_tensors,
        Yolo26SemSConfig,
        "yolo26s-sem"
    );
    golden_test!(
        yolo26m_sem_matches_ultralytics_golden_tensors,
        Yolo26SemMConfig,
        "yolo26m-sem"
    );
    golden_test!(
        yolo26l_sem_matches_ultralytics_golden_tensors,
        Yolo26SemLConfig,
        "yolo26l-sem"
    );
    golden_test!(
        yolo26x_sem_matches_ultralytics_golden_tensors,
        Yolo26SemXConfig,
        "yolo26x-sem"
    );

    latency_test!(
        yolo26n_sem_measures_single_inference_latency,
        Yolo26SemNConfig,
        "yolo26n-sem"
    );
    latency_test!(
        yolo26s_sem_measures_single_inference_latency,
        Yolo26SemSConfig,
        "yolo26s-sem"
    );
    latency_test!(
        yolo26m_sem_measures_single_inference_latency,
        Yolo26SemMConfig,
        "yolo26m-sem"
    );
    latency_test!(
        yolo26l_sem_measures_single_inference_latency,
        Yolo26SemLConfig,
        "yolo26l-sem"
    );
    latency_test!(
        yolo26x_sem_measures_single_inference_latency,
        Yolo26SemXConfig,
        "yolo26x-sem"
    );

    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo26n_sem_measures_single_inference_latency_gpu,
        Yolo26SemNConfig,
        "yolo26n-sem"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo26s_sem_measures_single_inference_latency_gpu,
        Yolo26SemSConfig,
        "yolo26s-sem"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo26m_sem_measures_single_inference_latency_gpu,
        Yolo26SemMConfig,
        "yolo26m-sem"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo26l_sem_measures_single_inference_latency_gpu,
        Yolo26SemLConfig,
        "yolo26l-sem"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo26x_sem_measures_single_inference_latency_gpu,
        Yolo26SemXConfig,
        "yolo26x-sem"
    );
}
