//! Native Burn implementation of the Ultralytics YOLO26-cls classification family (n/s/m/l/x).
//!
//! The classification graph is the YOLO26 backbone without the SPPF stage and without the
//! detection neck: layers 0-9 end in a C2PSA stage that feeds Ultralytics' `Classify` head
//! (1x1 Conv to 1280 channels, global average pooling, one linear layer to 1000 ImageNet classes).
//! Inference returns softmax probabilities; the runtime exposes the top-5 classes.
//!
//! The m/l/x scales force `c3k=True` on the early backbone stages (`parse_model`'s m/l/x rule)
//! and cap channels at 512, so those variants declare a structurally different body graph.

use burn::{
    module::Module,
    nn,
    tensor::{Device, Tensor, backend::Backend},
};

#[cfg(feature = "pretrained")]
use burn_store::ModuleSnapshot;

use super::blocks::{
    BnFlavor, C2Psa, C2PsaConfig, C3k2, C3k2C3k, C3k2C3kConfig, C3k2Config, Conv, ConvConfig,
};

/// Every Conv in the official YOLO26-cls checkpoints carries plain PyTorch BatchNorm defaults
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
/// Number of ImageNet classes the official YOLO26-cls checkpoints were trained on.
pub const NUM_CLASSES: usize = 1000;

/// Hidden width of the `Classify` head's 1x1 convolution (efficientnet_b0 size).
const HEAD_HIDDEN: usize = 1280;

/// Classification head output: pre-softmax logits and their softmax probabilities.
pub struct ClassificationOutput<B: Backend> {
    /// Raw linear logits, `[batch, NUM_CLASSES]`.
    pub logits: Tensor<B, 2>,
    /// Softmax probabilities, `[batch, NUM_CLASSES]` (Ultralytics' `Classify` output).
    pub probs: Tensor<B, 2>,
}

/// Ultralytics `Classify` head: 1x1 convolution, global average pooling, dropout (inert at
/// inference), and one linear layer. Field names match the official `model.10` checkpoint keys
/// after remapping.
#[derive(Module, Debug)]
pub struct ClassifyHead<B: Backend> {
    conv: Conv<B>,
    linear: nn::Linear<B>,
}

impl<B: Backend> ClassifyHead<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> ClassificationOutput<B> {
        let [batch, _, _, _] = input.dims();
        let pooled = burn::tensor::module::adaptive_avg_pool2d(self.conv.forward(input), [1, 1])
            .reshape([batch, HEAD_HIDDEN]);
        let logits = self.linear.forward(pooled);
        let probs = burn::tensor::activation::softmax(logits.clone(), 1);
        ClassificationOutput { logits, probs }
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch, _, _, _] = input.dims();
        let pooled = burn::tensor::module::adaptive_avg_pool2d(self.conv.forward(input), [1, 1])
            .reshape([batch, HEAD_HIDDEN]);
        self.linear.forward(pooled)
    }
}

#[derive(Debug)]
pub struct ClassifyHeadConfig {
    input_channels: usize,
    num_classes: usize,
}

impl ClassifyHeadConfig {
    pub fn new(input_channels: usize) -> Self {
        Self {
            input_channels,
            num_classes: NUM_CLASSES,
        }
    }

    pub fn with_num_classes(mut self, num_classes: usize) -> Self {
        assert!(num_classes > 0, "class count must be positive");
        self.num_classes = num_classes;
        self
    }

    pub fn init<B: Backend>(&self, device: &Device<B>) -> ClassifyHead<B> {
        ClassifyHead {
            conv: conv_cfg(self.input_channels, HEAD_HIDDEN, 1, 1).init(device),
            linear: nn::LinearConfig::new(HEAD_HIDDEN, self.num_classes).init(device),
        }
    }
}

/// YOLO26-cls backbone (layers 0-9), n and s scales: plain C3k2 bottleneck chains on the early
/// stages, C3k chains on the later stages, and the C2PSA stage where the detect body has SPPF.
#[derive(Module, Debug)]
pub struct Yolo26ClassifyBodySmall<B: Backend> {
    model_0: Conv<B>,
    model_1: Conv<B>,
    model_2: C3k2<B>,
    model_3: Conv<B>,
    model_4: C3k2<B>,
    model_5: Conv<B>,
    model_6: C3k2C3k<B>,
    model_7: Conv<B>,
    model_8: C3k2C3k<B>,
    model_9: C2Psa<B>,
}

impl<B: Backend> Yolo26ClassifyBodySmall<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.model_0.forward(input);
        let x = self.model_1.forward(x);
        let x = self.model_2.forward(x);
        let x = self.model_3.forward(x);
        let x = self.model_4.forward(x);
        let x = self.model_5.forward(x);
        let x = self.model_6.forward(x);
        let x = self.model_7.forward(x);
        let x = self.model_8.forward(x);
        self.model_9.forward(x)
    }
}

/// YOLO26-cls backbone (layers 0-9), m/l/x scales: `parse_model` forces `c3k=True` on every C3k2
/// stage and the YAML caps `max_channels` at 512.
#[derive(Module, Debug)]
pub struct Yolo26ClassifyBodyLarge<B: Backend> {
    model_0: Conv<B>,
    model_1: Conv<B>,
    model_2: C3k2C3k<B>,
    model_3: Conv<B>,
    model_4: C3k2C3k<B>,
    model_5: Conv<B>,
    model_6: C3k2C3k<B>,
    model_7: Conv<B>,
    model_8: C3k2C3k<B>,
    model_9: C2Psa<B>,
}

impl<B: Backend> Yolo26ClassifyBodyLarge<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.model_0.forward(input);
        let x = self.model_1.forward(x);
        let x = self.model_2.forward(x);
        let x = self.model_3.forward(x);
        let x = self.model_4.forward(x);
        let x = self.model_5.forward(x);
        let x = self.model_6.forward(x);
        let x = self.model_7.forward(x);
        let x = self.model_8.forward(x);
        self.model_9.forward(x)
    }
}

/// Native Burn YOLO26n-cls model.
#[derive(Module, Debug)]
pub struct Yolo26ClsN<B: Backend> {
    body: Yolo26ClassifyBodySmall<B>,
    head: ClassifyHead<B>,
}

impl<B: Backend> Yolo26ClsN<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> ClassificationOutput<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        self.head.forward_train(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLO26n-cls checkpoint.
    #[cfg(feature = "pretrained")]
    pub fn load_pytorch_weights(
        &mut self,
        path: impl Into<std::path::PathBuf>,
    ) -> Result<(), burn_store::PytorchStoreError> {
        let mut store = pytorch_store(path);
        self.load_from(&mut store).map(|_| ())
    }

    /// Load Montgomery's versioned, half-precision native Burnpack artifact.
    #[cfg(feature = "pretrained")]
    pub fn load_burnpack_weights(
        &mut self,
        path: impl Into<std::path::PathBuf>,
    ) -> Result<(), burn_store::BurnpackError> {
        let mut store = burn_store::BurnpackStore::from_file(path.into())
            .with_from_adapter(burn_store::HalfPrecisionAdapter::new())
            .zero_copy(true);
        self.load_from(&mut store).map(|_| ())
    }

    /// Save a versioned native artifact. Existing files are deliberately not overwritten.
    #[cfg(feature = "pretrained")]
    pub fn save_burnpack_weights(
        &self,
        path: impl Into<std::path::PathBuf>,
    ) -> Result<(), burn_store::BurnpackError> {
        let mut store = burn_store::BurnpackStore::from_file(path.into())
            .metadata(
                "montgomery.artifact-format",
                super::weights::artifact_format("yolo26n-cls"),
            )
            .metadata("montgomery.model", "yolo26n-cls")
            .metadata("montgomery.classes", "imagenet-1000")
            .metadata("montgomery.precision", "f16")
            .with_to_adapter(burn_store::HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolo26ClsNConfig;

impl Yolo26ClsNConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo26ClsN<B> {
        self.init_with_classes(NUM_CLASSES, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolo26ClsN<B> {
        Yolo26ClsN {
            body: Yolo26ClassifyBodyNConfig.init(device),
            head: ClassifyHeadConfig::new(256)
                .with_num_classes(num_classes)
                .init(device),
        }
    }
}

/// Native Burn YOLO26s-cls model.
#[derive(Module, Debug)]
pub struct Yolo26ClsS<B: Backend> {
    body: Yolo26ClassifyBodySmall<B>,
    head: ClassifyHead<B>,
}

impl<B: Backend> Yolo26ClsS<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> ClassificationOutput<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        self.head.forward_train(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLO26s-cls checkpoint.
    #[cfg(feature = "pretrained")]
    pub fn load_pytorch_weights(
        &mut self,
        path: impl Into<std::path::PathBuf>,
    ) -> Result<(), burn_store::PytorchStoreError> {
        let mut store = pytorch_store(path);
        self.load_from(&mut store).map(|_| ())
    }

    /// Load Montgomery's versioned, half-precision native Burnpack artifact.
    #[cfg(feature = "pretrained")]
    pub fn load_burnpack_weights(
        &mut self,
        path: impl Into<std::path::PathBuf>,
    ) -> Result<(), burn_store::BurnpackError> {
        let mut store = burn_store::BurnpackStore::from_file(path.into())
            .with_from_adapter(burn_store::HalfPrecisionAdapter::new())
            .zero_copy(true);
        self.load_from(&mut store).map(|_| ())
    }

    /// Save a versioned native artifact. Existing files are deliberately not overwritten.
    #[cfg(feature = "pretrained")]
    pub fn save_burnpack_weights(
        &self,
        path: impl Into<std::path::PathBuf>,
    ) -> Result<(), burn_store::BurnpackError> {
        let mut store = burn_store::BurnpackStore::from_file(path.into())
            .metadata(
                "montgomery.artifact-format",
                super::weights::artifact_format("yolo26s-cls"),
            )
            .metadata("montgomery.model", "yolo26s-cls")
            .metadata("montgomery.classes", "imagenet-1000")
            .metadata("montgomery.precision", "f16")
            .with_to_adapter(burn_store::HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolo26ClsSConfig;

impl Yolo26ClsSConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo26ClsS<B> {
        self.init_with_classes(NUM_CLASSES, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolo26ClsS<B> {
        Yolo26ClsS {
            body: Yolo26ClassifyBodySConfig.init(device),
            head: ClassifyHeadConfig::new(512)
                .with_num_classes(num_classes)
                .init(device),
        }
    }
}

/// Native Burn YOLO26m-cls model.
#[derive(Module, Debug)]
pub struct Yolo26ClsM<B: Backend> {
    body: Yolo26ClassifyBodyLarge<B>,
    head: ClassifyHead<B>,
}

impl<B: Backend> Yolo26ClsM<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> ClassificationOutput<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        self.head.forward_train(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLO26m-cls checkpoint.
    #[cfg(feature = "pretrained")]
    pub fn load_pytorch_weights(
        &mut self,
        path: impl Into<std::path::PathBuf>,
    ) -> Result<(), burn_store::PytorchStoreError> {
        let mut store = pytorch_store(path);
        self.load_from(&mut store).map(|_| ())
    }

    /// Load Montgomery's versioned, half-precision native Burnpack artifact.
    #[cfg(feature = "pretrained")]
    pub fn load_burnpack_weights(
        &mut self,
        path: impl Into<std::path::PathBuf>,
    ) -> Result<(), burn_store::BurnpackError> {
        let mut store = burn_store::BurnpackStore::from_file(path.into())
            .with_from_adapter(burn_store::HalfPrecisionAdapter::new())
            .zero_copy(true);
        self.load_from(&mut store).map(|_| ())
    }

    /// Save a versioned native artifact. Existing files are deliberately not overwritten.
    #[cfg(feature = "pretrained")]
    pub fn save_burnpack_weights(
        &self,
        path: impl Into<std::path::PathBuf>,
    ) -> Result<(), burn_store::BurnpackError> {
        let mut store = burn_store::BurnpackStore::from_file(path.into())
            .metadata(
                "montgomery.artifact-format",
                super::weights::artifact_format("yolo26m-cls"),
            )
            .metadata("montgomery.model", "yolo26m-cls")
            .metadata("montgomery.classes", "imagenet-1000")
            .metadata("montgomery.precision", "f16")
            .with_to_adapter(burn_store::HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolo26ClsMConfig;

impl Yolo26ClsMConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo26ClsM<B> {
        self.init_with_classes(NUM_CLASSES, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolo26ClsM<B> {
        Yolo26ClsM {
            body: Yolo26ClassifyBodyMConfig.init(device),
            head: ClassifyHeadConfig::new(512)
                .with_num_classes(num_classes)
                .init(device),
        }
    }
}

/// Native Burn YOLO26l-cls model.
#[derive(Module, Debug)]
pub struct Yolo26ClsL<B: Backend> {
    body: Yolo26ClassifyBodyLarge<B>,
    head: ClassifyHead<B>,
}

impl<B: Backend> Yolo26ClsL<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> ClassificationOutput<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        self.head.forward_train(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLO26l-cls checkpoint.
    #[cfg(feature = "pretrained")]
    pub fn load_pytorch_weights(
        &mut self,
        path: impl Into<std::path::PathBuf>,
    ) -> Result<(), burn_store::PytorchStoreError> {
        let mut store = pytorch_store(path);
        self.load_from(&mut store).map(|_| ())
    }

    /// Load Montgomery's versioned, half-precision native Burnpack artifact.
    #[cfg(feature = "pretrained")]
    pub fn load_burnpack_weights(
        &mut self,
        path: impl Into<std::path::PathBuf>,
    ) -> Result<(), burn_store::BurnpackError> {
        let mut store = burn_store::BurnpackStore::from_file(path.into())
            .with_from_adapter(burn_store::HalfPrecisionAdapter::new())
            .zero_copy(true);
        self.load_from(&mut store).map(|_| ())
    }

    /// Save a versioned native artifact. Existing files are deliberately not overwritten.
    #[cfg(feature = "pretrained")]
    pub fn save_burnpack_weights(
        &self,
        path: impl Into<std::path::PathBuf>,
    ) -> Result<(), burn_store::BurnpackError> {
        let mut store = burn_store::BurnpackStore::from_file(path.into())
            .metadata(
                "montgomery.artifact-format",
                super::weights::artifact_format("yolo26l-cls"),
            )
            .metadata("montgomery.model", "yolo26l-cls")
            .metadata("montgomery.classes", "imagenet-1000")
            .metadata("montgomery.precision", "f16")
            .with_to_adapter(burn_store::HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolo26ClsLConfig;

impl Yolo26ClsLConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo26ClsL<B> {
        self.init_with_classes(NUM_CLASSES, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolo26ClsL<B> {
        Yolo26ClsL {
            body: Yolo26ClassifyBodyLConfig.init(device),
            head: ClassifyHeadConfig::new(512)
                .with_num_classes(num_classes)
                .init(device),
        }
    }
}

/// Native Burn YOLO26x-cls model.
#[derive(Module, Debug)]
pub struct Yolo26ClsX<B: Backend> {
    body: Yolo26ClassifyBodyLarge<B>,
    head: ClassifyHead<B>,
}

impl<B: Backend> Yolo26ClsX<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> ClassificationOutput<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        self.head.forward_train(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLO26x-cls checkpoint.
    #[cfg(feature = "pretrained")]
    pub fn load_pytorch_weights(
        &mut self,
        path: impl Into<std::path::PathBuf>,
    ) -> Result<(), burn_store::PytorchStoreError> {
        let mut store = pytorch_store(path);
        self.load_from(&mut store).map(|_| ())
    }

    /// Load Montgomery's versioned, half-precision native Burnpack artifact.
    #[cfg(feature = "pretrained")]
    pub fn load_burnpack_weights(
        &mut self,
        path: impl Into<std::path::PathBuf>,
    ) -> Result<(), burn_store::BurnpackError> {
        let mut store = burn_store::BurnpackStore::from_file(path.into())
            .with_from_adapter(burn_store::HalfPrecisionAdapter::new())
            .zero_copy(true);
        self.load_from(&mut store).map(|_| ())
    }

    /// Save a versioned native artifact. Existing files are deliberately not overwritten.
    #[cfg(feature = "pretrained")]
    pub fn save_burnpack_weights(
        &self,
        path: impl Into<std::path::PathBuf>,
    ) -> Result<(), burn_store::BurnpackError> {
        let mut store = burn_store::BurnpackStore::from_file(path.into())
            .metadata(
                "montgomery.artifact-format",
                super::weights::artifact_format("yolo26x-cls"),
            )
            .metadata("montgomery.model", "yolo26x-cls")
            .metadata("montgomery.classes", "imagenet-1000")
            .metadata("montgomery.precision", "f16")
            .with_to_adapter(burn_store::HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolo26ClsXConfig;

impl Yolo26ClsXConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo26ClsX<B> {
        self.init_with_classes(NUM_CLASSES, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolo26ClsX<B> {
        Yolo26ClsX {
            body: Yolo26ClassifyBodyXConfig.init(device),
            head: ClassifyHeadConfig::new(768)
                .with_num_classes(num_classes)
                .init(device),
        }
    }
}

/// Build the PyTorch-state store shared by every YOLO26-cls scale variant.
///
/// The backbone is layers 0-9 and the `Classify` head is model.10: one `Conv` (conv+bn) and one
/// `nn.Linear` per scale.
#[cfg(feature = "pretrained")]
fn pytorch_store(path: impl Into<std::path::PathBuf>) -> burn_store::PytorchStore {
    burn_store::PytorchStore::from_file(path)
        .with_top_level_key("model")
        // Backbone layers 0-9 keep their Ultralytics graph indices. The head is model.10, so the
        // single-digit rule must not match it.
        .with_key_remapping("model\\.([0-9])\\.(.+)", "body.model_$1.$2")
        // model.10.conv.{conv,bn}.* is the 1x1 classification convolution.
        .with_key_remapping("model\\.10\\.conv\\.conv\\.(.+)", "head.conv.conv.$1")
        .with_key_remapping("model\\.10\\.conv\\.bn\\.(.+)", "head.conv.bn.$1")
        // model.10.linear.* is the final classifier.
        .with_key_remapping("model\\.10\\.linear\\.(.+)", "head.linear.$1")
}

/// Configuration for the fixed YOLO26n/s-cls backbone (depth 0.50, max channels 1024).
#[derive(Debug, Default)]
pub struct Yolo26ClassifyBodyNConfig;

impl Yolo26ClassifyBodyNConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo26ClassifyBodySmall<B> {
        Yolo26ClassifyBodySmall {
            model_0: conv_cfg(3, 16, 3, 2).init(device),
            model_1: conv_cfg(16, 32, 3, 2).init(device),
            model_2: C3k2Config::new(32, 64, 1, 0.25, true)
                .with_bn_flavor(BnFlavor::Pytorch)
                .init(device),
            model_3: conv_cfg(64, 64, 3, 2).init(device),
            model_4: C3k2Config::new(64, 128, 1, 0.25, true)
                .with_bn_flavor(BnFlavor::Pytorch)
                .init(device),
            model_5: conv_cfg(128, 128, 3, 2).init(device),
            model_6: C3k2C3kConfig::new(128, 128, 1, true, 0.5)
                .with_bn_flavor(BnFlavor::Pytorch)
                .init(device),
            model_7: conv_cfg(128, 256, 3, 2).init(device),
            model_8: C3k2C3kConfig::new(256, 256, 1, true, 0.5)
                .with_bn_flavor(BnFlavor::Pytorch)
                .init(device),
            model_9: C2PsaConfig::new(256, 1)
                .with_bn_flavor(BnFlavor::Pytorch)
                .init(device),
        }
    }
}

/// Configuration for the fixed YOLO26s-cls backbone (depth 0.50, width 0.50, max channels 1024).
#[derive(Debug, Default)]
pub struct Yolo26ClassifyBodySConfig;

impl Yolo26ClassifyBodySConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo26ClassifyBodySmall<B> {
        Yolo26ClassifyBodySmall {
            model_0: conv_cfg(3, 32, 3, 2).init(device),
            model_1: conv_cfg(32, 64, 3, 2).init(device),
            model_2: C3k2Config::new(64, 128, 1, 0.25, true)
                .with_bn_flavor(BnFlavor::Pytorch)
                .init(device),
            model_3: conv_cfg(128, 128, 3, 2).init(device),
            model_4: C3k2Config::new(128, 256, 1, 0.25, true)
                .with_bn_flavor(BnFlavor::Pytorch)
                .init(device),
            model_5: conv_cfg(256, 256, 3, 2).init(device),
            model_6: C3k2C3kConfig::new(256, 256, 1, true, 0.5)
                .with_bn_flavor(BnFlavor::Pytorch)
                .init(device),
            model_7: conv_cfg(256, 512, 3, 2).init(device),
            model_8: C3k2C3kConfig::new(512, 512, 1, true, 0.5)
                .with_bn_flavor(BnFlavor::Pytorch)
                .init(device),
            model_9: C2PsaConfig::new(512, 1)
                .with_bn_flavor(BnFlavor::Pytorch)
                .init(device),
        }
    }
}

/// Configuration for the fixed YOLO26m-cls backbone (depth 0.50, width 1.00, max channels 512).
#[derive(Debug, Default)]
pub struct Yolo26ClassifyBodyMConfig;

impl Yolo26ClassifyBodyMConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo26ClassifyBodyLarge<B> {
        Yolo26ClassifyBodyLarge {
            model_0: conv_cfg(3, 64, 3, 2).init(device),
            model_1: conv_cfg(64, 128, 3, 2).init(device),
            model_2: C3k2C3kConfig::new(128, 256, 1, true, 0.25)
                .with_bn_flavor(BnFlavor::Pytorch)
                .init(device),
            model_3: conv_cfg(256, 256, 3, 2).init(device),
            model_4: C3k2C3kConfig::new(256, 512, 1, true, 0.25)
                .with_bn_flavor(BnFlavor::Pytorch)
                .init(device),
            model_5: conv_cfg(512, 512, 3, 2).init(device),
            model_6: C3k2C3kConfig::new(512, 512, 1, true, 0.5)
                .with_bn_flavor(BnFlavor::Pytorch)
                .init(device),
            model_7: conv_cfg(512, 512, 3, 2).init(device),
            model_8: C3k2C3kConfig::new(512, 512, 1, true, 0.5)
                .with_bn_flavor(BnFlavor::Pytorch)
                .init(device),
            model_9: C2PsaConfig::new(512, 1)
                .with_bn_flavor(BnFlavor::Pytorch)
                .init(device),
        }
    }
}

/// Configuration for the fixed YOLO26l-cls backbone (depth 1.00, width 1.00, max channels 512).
#[derive(Debug, Default)]
pub struct Yolo26ClassifyBodyLConfig;

impl Yolo26ClassifyBodyLConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo26ClassifyBodyLarge<B> {
        Yolo26ClassifyBodyLarge {
            model_0: conv_cfg(3, 64, 3, 2).init(device),
            model_1: conv_cfg(64, 128, 3, 2).init(device),
            model_2: C3k2C3kConfig::new(128, 256, 2, true, 0.25)
                .with_bn_flavor(BnFlavor::Pytorch)
                .init(device),
            model_3: conv_cfg(256, 256, 3, 2).init(device),
            model_4: C3k2C3kConfig::new(256, 512, 2, true, 0.25)
                .with_bn_flavor(BnFlavor::Pytorch)
                .init(device),
            model_5: conv_cfg(512, 512, 3, 2).init(device),
            model_6: C3k2C3kConfig::new(512, 512, 2, true, 0.5)
                .with_bn_flavor(BnFlavor::Pytorch)
                .init(device),
            model_7: conv_cfg(512, 512, 3, 2).init(device),
            model_8: C3k2C3kConfig::new(512, 512, 2, true, 0.5)
                .with_bn_flavor(BnFlavor::Pytorch)
                .init(device),
            model_9: C2PsaConfig::new(512, 2)
                .with_bn_flavor(BnFlavor::Pytorch)
                .init(device),
        }
    }
}

/// Configuration for the fixed YOLO26x-cls backbone (depth 1.00, width 1.50, max channels 512).
#[derive(Debug, Default)]
pub struct Yolo26ClassifyBodyXConfig;

impl Yolo26ClassifyBodyXConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo26ClassifyBodyLarge<B> {
        Yolo26ClassifyBodyLarge {
            model_0: conv_cfg(3, 96, 3, 2).init(device),
            model_1: conv_cfg(96, 192, 3, 2).init(device),
            model_2: C3k2C3kConfig::new(192, 384, 2, true, 0.25)
                .with_bn_flavor(BnFlavor::Pytorch)
                .init(device),
            model_3: conv_cfg(384, 384, 3, 2).init(device),
            model_4: C3k2C3kConfig::new(384, 768, 2, true, 0.25)
                .with_bn_flavor(BnFlavor::Pytorch)
                .init(device),
            model_5: conv_cfg(768, 768, 3, 2).init(device),
            model_6: C3k2C3kConfig::new(768, 768, 2, true, 0.5)
                .with_bn_flavor(BnFlavor::Pytorch)
                .init(device),
            model_7: conv_cfg(768, 768, 3, 2).init(device),
            model_8: C3k2C3kConfig::new(768, 768, 2, true, 0.5)
                .with_bn_flavor(BnFlavor::Pytorch)
                .init(device),
            model_9: C2PsaConfig::new(768, 2)
                .with_bn_flavor(BnFlavor::Pytorch)
                .init(device),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{ElementConversion, TensorData};
    use burn_flex::Flex;
    use serde::Deserialize;
    use std::collections::BTreeMap;

    #[cfg(feature = "gpu")]
    use burn::backend::Wgpu;

    #[test]
    fn produces_class_logits_for_each_scale() {
        let worker = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let device = Default::default();
                let input = Tensor::zeros([1, 3, 64, 64], &device);

                let model = Yolo26ClsNConfig.init::<Flex>(&device);
                let output = model.forward(input.clone());
                assert_eq!(output.probs.dims(), [1, NUM_CLASSES]);

                let model = Yolo26ClsSConfig.init::<Flex>(&device);
                let output = model.forward(input.clone());
                assert_eq!(output.probs.dims(), [1, NUM_CLASSES]);

                let model = Yolo26ClsMConfig.init::<Flex>(&device);
                let output = model.forward(input.clone());
                assert_eq!(output.probs.dims(), [1, NUM_CLASSES]);

                let model = Yolo26ClsLConfig.init::<Flex>(&device);
                let output = model.forward(input.clone());
                assert_eq!(output.probs.dims(), [1, NUM_CLASSES]);

                let model = Yolo26ClsXConfig.init::<Flex>(&device);
                let output = model.forward(input);
                assert_eq!(output.probs.dims(), [1, NUM_CLASSES]);
            })
            .expect("shape-test worker should start");
        worker.join().expect("shape-test worker should not panic");
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
                    crate::models::yolo26::weights::artifact_filename($id)
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
            /// Measure single-image batch-1 inference latency with the packed native artifact on
            /// the Flex CPU backend at the family's 224 px classify input. Run with
            /// `cargo test --release <id> -- --ignored --nocapture` after the weight-prep loop.
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
    /// `python tools/export_classification_fixtures.py target/<id>.pt docs/dog_bike_man.jpg target --model <id>`
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
                    crate::models::yolo26::weights::artifact_filename($id)
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
                    // anti-aliased bilinear resize (even PIL vs torchvision differ on ~half the
                    // pixels), and flat distributions amplify the shift (observed worst case
                    // 0.037 on yolo11s-cls). Verified: Ultralytics fed Montgomery's canvas
                    // reproduces Montgomery's probabilities exactly, so this delta is preprocessing
                    // rounding, not graph drift; the golden test pins the graph at 2e-4 on the
                    // shared canvas.
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
        yolo26n_cls_imports_official_checkpoint_and_runs_forward,
        Yolo26ClsNConfig,
        "yolo26n-cls"
    );
    checkpoint_test!(
        yolo26s_cls_imports_official_checkpoint_and_runs_forward,
        Yolo26ClsSConfig,
        "yolo26s-cls"
    );
    checkpoint_test!(
        yolo26m_cls_imports_official_checkpoint_and_runs_forward,
        Yolo26ClsMConfig,
        "yolo26m-cls"
    );
    checkpoint_test!(
        yolo26l_cls_imports_official_checkpoint_and_runs_forward,
        Yolo26ClsLConfig,
        "yolo26l-cls"
    );
    checkpoint_test!(
        yolo26x_cls_imports_official_checkpoint_and_runs_forward,
        Yolo26ClsXConfig,
        "yolo26x-cls"
    );

    golden_test!(
        yolo26n_cls_matches_ultralytics_golden_tensors,
        Yolo26ClsNConfig,
        "yolo26n-cls"
    );
    golden_test!(
        yolo26s_cls_matches_ultralytics_golden_tensors,
        Yolo26ClsSConfig,
        "yolo26s-cls"
    );
    golden_test!(
        yolo26m_cls_matches_ultralytics_golden_tensors,
        Yolo26ClsMConfig,
        "yolo26m-cls"
    );
    golden_test!(
        yolo26l_cls_matches_ultralytics_golden_tensors,
        Yolo26ClsLConfig,
        "yolo26l-cls"
    );
    golden_test!(
        yolo26x_cls_matches_ultralytics_golden_tensors,
        Yolo26ClsXConfig,
        "yolo26x-cls"
    );

    latency_test!(
        yolo26n_cls_measures_single_inference_latency,
        Yolo26ClsNConfig,
        "yolo26n-cls"
    );
    latency_test!(
        yolo26s_cls_measures_single_inference_latency,
        Yolo26ClsSConfig,
        "yolo26s-cls"
    );
    latency_test!(
        yolo26m_cls_measures_single_inference_latency,
        Yolo26ClsMConfig,
        "yolo26m-cls"
    );
    latency_test!(
        yolo26l_cls_measures_single_inference_latency,
        Yolo26ClsLConfig,
        "yolo26l-cls"
    );
    latency_test!(
        yolo26x_cls_measures_single_inference_latency,
        Yolo26ClsXConfig,
        "yolo26x-cls"
    );

    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo26n_cls_measures_single_inference_latency_gpu,
        Yolo26ClsNConfig,
        "yolo26n-cls"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo26s_cls_measures_single_inference_latency_gpu,
        Yolo26ClsSConfig,
        "yolo26s-cls"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo26m_cls_measures_single_inference_latency_gpu,
        Yolo26ClsMConfig,
        "yolo26m-cls"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo26l_cls_measures_single_inference_latency_gpu,
        Yolo26ClsLConfig,
        "yolo26l-cls"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo26x_cls_measures_single_inference_latency_gpu,
        Yolo26ClsXConfig,
        "yolo26x-cls"
    );

    cls_e2e_test!(
        yolo26n_cls_matches_ultralytics_end_to_end,
        Yolo26ClsNConfig,
        "yolo26n-cls"
    );
    cls_e2e_test!(
        yolo26s_cls_matches_ultralytics_end_to_end,
        Yolo26ClsSConfig,
        "yolo26s-cls"
    );
    cls_e2e_test!(
        yolo26m_cls_matches_ultralytics_end_to_end,
        Yolo26ClsMConfig,
        "yolo26m-cls"
    );
    cls_e2e_test!(
        yolo26l_cls_matches_ultralytics_end_to_end,
        Yolo26ClsLConfig,
        "yolo26l-cls"
    );
    cls_e2e_test!(
        yolo26x_cls_matches_ultralytics_end_to_end,
        Yolo26ClsXConfig,
        "yolo26x-cls"
    );
}
