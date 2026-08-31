use burn::{
    module::Module,
    tensor::{Device, Tensor, backend::Backend},
};

use super::{
    body::{
        Yolo11BodyLConfig, Yolo11BodyLarge, Yolo11BodyMConfig, Yolo11BodyNConfig,
        Yolo11BodySConfig, Yolo11BodySmall, Yolo11BodyXConfig,
    },
    head::{DecodedPredictions, RawPredictions, Yolo11Head, Yolo11HeadConfig},
    segment_head::{SegmentOutput, SegmentTrainOutput, Yolo11SegHead, Yolo11SegHeadConfig},
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

/// Build the PyTorch-state store shared by every YOLO11 scale variant.
///
/// Body layers retain their Ultralytics graph indices (the head is model.23, which the body rule
/// must not match); the head's box and classification towers are remapped one rule per
/// path-segment pattern. The fixed DFL projection (`model.23.dfl.conv.weight`) is intentionally
/// unmapped: it is baked into the head as a constant. The remaps are scale-independent because
/// every variant shares the checkpoint key layout.
#[cfg(feature = "pretrained")]
fn pytorch_store(path: impl Into<PathBuf>) -> PytorchStore {
    PytorchStore::from_file(path)
        .with_top_level_key("model")
        // Body layers retain their Ultralytics graph indices. The head is model.23, so this
        // rule must not match it.
        .with_key_remapping("model\\.([0-9]|1[0-9]|2[0-2])\\.(.+)", "body.model_$1.$2")
        // model.23.cv2.{scale}.{layer} is the box regression tower.
        .with_key_remapping("model\\.23\\.cv2\\.0\\.0\\.(.+)", "head.p3.box_0.$1")
        .with_key_remapping("model\\.23\\.cv2\\.0\\.1\\.(.+)", "head.p3.box_1.$1")
        .with_key_remapping("model\\.23\\.cv2\\.0\\.2\\.(.+)", "head.p3.box_out.$1")
        .with_key_remapping("model\\.23\\.cv2\\.1\\.0\\.(.+)", "head.p4.box_0.$1")
        .with_key_remapping("model\\.23\\.cv2\\.1\\.1\\.(.+)", "head.p4.box_1.$1")
        .with_key_remapping("model\\.23\\.cv2\\.1\\.2\\.(.+)", "head.p4.box_out.$1")
        .with_key_remapping("model\\.23\\.cv2\\.2\\.0\\.(.+)", "head.p5.box_0.$1")
        .with_key_remapping("model\\.23\\.cv2\\.2\\.1\\.(.+)", "head.p5.box_1.$1")
        .with_key_remapping("model\\.23\\.cv2\\.2\\.2\\.(.+)", "head.p5.box_out.$1")
        // model.23.cv3.{scale}.{tower}.{conv} is the light classification tower.
        .with_key_remapping("model\\.23\\.cv3\\.0\\.0\\.0\\.(.+)", "head.p3.cls_dw_0.$1")
        .with_key_remapping("model\\.23\\.cv3\\.0\\.0\\.1\\.(.+)", "head.p3.cls_pw_0.$1")
        .with_key_remapping("model\\.23\\.cv3\\.0\\.1\\.0\\.(.+)", "head.p3.cls_dw_1.$1")
        .with_key_remapping("model\\.23\\.cv3\\.0\\.1\\.1\\.(.+)", "head.p3.cls_pw_1.$1")
        .with_key_remapping("model\\.23\\.cv3\\.0\\.2\\.(.+)", "head.p3.cls_out.$1")
        .with_key_remapping("model\\.23\\.cv3\\.1\\.0\\.0\\.(.+)", "head.p4.cls_dw_0.$1")
        .with_key_remapping("model\\.23\\.cv3\\.1\\.0\\.1\\.(.+)", "head.p4.cls_pw_0.$1")
        .with_key_remapping("model\\.23\\.cv3\\.1\\.1\\.0\\.(.+)", "head.p4.cls_dw_1.$1")
        .with_key_remapping("model\\.23\\.cv3\\.1\\.1\\.1\\.(.+)", "head.p4.cls_pw_1.$1")
        .with_key_remapping("model\\.23\\.cv3\\.1\\.2\\.(.+)", "head.p4.cls_out.$1")
        .with_key_remapping("model\\.23\\.cv3\\.2\\.0\\.0\\.(.+)", "head.p5.cls_dw_0.$1")
        .with_key_remapping("model\\.23\\.cv3\\.2\\.0\\.1\\.(.+)", "head.p5.cls_pw_0.$1")
        .with_key_remapping("model\\.23\\.cv3\\.2\\.1\\.0\\.(.+)", "head.p5.cls_dw_1.$1")
        .with_key_remapping("model\\.23\\.cv3\\.2\\.1\\.1\\.(.+)", "head.p5.cls_pw_1.$1")
        .with_key_remapping("model\\.23\\.cv3\\.2\\.2\\.(.+)", "head.p5.cls_out.$1")
}

/// Native Burn YOLO11n model.
///
/// The body feeds the classic DFL detection head whose predictions are decoded to center-size
/// model-input pixels; the runtime applies class-aware non-maximum suppression.
#[derive(Module, Debug)]
pub struct Yolo11N<B: Backend> {
    body: Yolo11BodySmall<B>,
    head: Yolo11Head<B>,
}

impl<B: Backend> Yolo11N<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> DecodedPredictions<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> RawPredictions<B> {
        self.head.forward_raw(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLO11n checkpoint.
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
                weights::artifact_format("yolo11n"),
            )
            .metadata("montgomery.model", "yolo11n")
            .metadata("montgomery.classes", "coco-80")
            .metadata("montgomery.precision", "f16")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolo11NConfig;

impl Yolo11NConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo11N<B> {
        self.init_with_classes(80, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolo11N<B> {
        Yolo11N {
            body: Yolo11BodyNConfig.init(device),
            head: Yolo11HeadConfig::new(64, 128, 256)
                .with_num_classes(num_classes)
                .init(device),
        }
    }
}

/// Native Burn YOLO11s model.
#[derive(Module, Debug)]
pub struct Yolo11S<B: Backend> {
    body: Yolo11BodySmall<B>,
    head: Yolo11Head<B>,
}

impl<B: Backend> Yolo11S<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> DecodedPredictions<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> RawPredictions<B> {
        self.head.forward_raw(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLO11s checkpoint.
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
                weights::artifact_format("yolo11s"),
            )
            .metadata("montgomery.model", "yolo11s")
            .metadata("montgomery.classes", "coco-80")
            .metadata("montgomery.precision", "f16")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolo11SConfig;

impl Yolo11SConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo11S<B> {
        self.init_with_classes(80, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolo11S<B> {
        Yolo11S {
            body: Yolo11BodySConfig.init(device),
            head: Yolo11HeadConfig::new(128, 256, 512)
                .with_num_classes(num_classes)
                .init(device),
        }
    }
}

/// Native Burn YOLO11m model. The m-scale body forces the C3k chain onto the early backbone
/// stages (`parse_model`'s m/l/x rule), so it shares [`Yolo11BodyLarge`] with l/x.
#[derive(Module, Debug)]
pub struct Yolo11M<B: Backend> {
    body: Yolo11BodyLarge<B>,
    head: Yolo11Head<B>,
}

impl<B: Backend> Yolo11M<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> DecodedPredictions<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> RawPredictions<B> {
        self.head.forward_raw(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLO11m checkpoint.
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
                weights::artifact_format("yolo11m"),
            )
            .metadata("montgomery.model", "yolo11m")
            .metadata("montgomery.classes", "coco-80")
            .metadata("montgomery.precision", "f16")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolo11MConfig;

impl Yolo11MConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo11M<B> {
        self.init_with_classes(80, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolo11M<B> {
        Yolo11M {
            body: Yolo11BodyMConfig.init(device),
            head: Yolo11HeadConfig::new(256, 512, 512)
                .with_num_classes(num_classes)
                .init(device),
        }
    }
}

/// Native Burn YOLO11l model. Shares YOLO11m's body graph with depth-scaled repeats.
#[derive(Module, Debug)]
pub struct Yolo11L<B: Backend> {
    body: Yolo11BodyLarge<B>,
    head: Yolo11Head<B>,
}

impl<B: Backend> Yolo11L<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> DecodedPredictions<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> RawPredictions<B> {
        self.head.forward_raw(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLO11l checkpoint.
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
                weights::artifact_format("yolo11l"),
            )
            .metadata("montgomery.model", "yolo11l")
            .metadata("montgomery.classes", "coco-80")
            .metadata("montgomery.precision", "f16")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolo11LConfig;

impl Yolo11LConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo11L<B> {
        self.init_with_classes(80, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolo11L<B> {
        Yolo11L {
            body: Yolo11BodyLConfig.init(device),
            head: Yolo11HeadConfig::new(256, 512, 512)
                .with_num_classes(num_classes)
                .init(device),
        }
    }
}

/// Native Burn YOLO11x model.
#[derive(Module, Debug)]
pub struct Yolo11X<B: Backend> {
    body: Yolo11BodyLarge<B>,
    head: Yolo11Head<B>,
}

impl<B: Backend> Yolo11X<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> DecodedPredictions<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> RawPredictions<B> {
        self.head.forward_raw(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLO11x checkpoint.
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
                weights::artifact_format("yolo11x"),
            )
            .metadata("montgomery.model", "yolo11x")
            .metadata("montgomery.classes", "coco-80")
            .metadata("montgomery.precision", "f16")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolo11XConfig;

impl Yolo11XConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo11X<B> {
        self.init_with_classes(80, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolo11X<B> {
        Yolo11X {
            body: Yolo11BodyXConfig.init(device),
            head: Yolo11HeadConfig::new(384, 768, 768)
                .with_num_classes(num_classes)
                .init(device),
        }
    }
}

/// Build the PyTorch-state store for the YOLO11-seg scale variants.
///
/// The seg YAML keeps the detect model's body (layers 0-22) and puts the `Segment` head at
/// `model.23`, so every rule here mirrors the detect remap with the head paths prefixed by
/// `detect.`; the additions are the Proto module and the `cv4` mask-coefficient towers. The fixed
/// DFL projection (`model.23.dfl.conv.weight`) is intentionally unmapped, exactly like the detect
/// variants.
#[cfg(feature = "pretrained")]
fn pytorch_seg_store(path: impl Into<PathBuf>) -> PytorchStore {
    PytorchStore::from_file(path)
        .with_top_level_key("model")
        .with_key_remapping("model\\.([0-9]|1[0-9]|2[0-2])\\.(.+)", "body.model_$1.$2")
        .with_key_remapping("model\\.23\\.cv2\\.0\\.0\\.(.+)", "head.detect.p3.box_0.$1")
        .with_key_remapping("model\\.23\\.cv2\\.0\\.1\\.(.+)", "head.detect.p3.box_1.$1")
        .with_key_remapping(
            "model\\.23\\.cv2\\.0\\.2\\.(.+)",
            "head.detect.p3.box_out.$1",
        )
        .with_key_remapping("model\\.23\\.cv2\\.1\\.0\\.(.+)", "head.detect.p4.box_0.$1")
        .with_key_remapping("model\\.23\\.cv2\\.1\\.1\\.(.+)", "head.detect.p4.box_1.$1")
        .with_key_remapping(
            "model\\.23\\.cv2\\.1\\.2\\.(.+)",
            "head.detect.p4.box_out.$1",
        )
        .with_key_remapping("model\\.23\\.cv2\\.2\\.0\\.(.+)", "head.detect.p5.box_0.$1")
        .with_key_remapping("model\\.23\\.cv2\\.2\\.1\\.(.+)", "head.detect.p5.box_1.$1")
        .with_key_remapping(
            "model\\.23\\.cv2\\.2\\.2\\.(.+)",
            "head.detect.p5.box_out.$1",
        )
        .with_key_remapping(
            "model\\.23\\.cv3\\.0\\.0\\.0\\.(.+)",
            "head.detect.p3.cls_dw_0.$1",
        )
        .with_key_remapping(
            "model\\.23\\.cv3\\.0\\.0\\.1\\.(.+)",
            "head.detect.p3.cls_pw_0.$1",
        )
        .with_key_remapping(
            "model\\.23\\.cv3\\.0\\.1\\.0\\.(.+)",
            "head.detect.p3.cls_dw_1.$1",
        )
        .with_key_remapping(
            "model\\.23\\.cv3\\.0\\.1\\.1\\.(.+)",
            "head.detect.p3.cls_pw_1.$1",
        )
        .with_key_remapping(
            "model\\.23\\.cv3\\.0\\.2\\.(.+)",
            "head.detect.p3.cls_out.$1",
        )
        .with_key_remapping(
            "model\\.23\\.cv3\\.1\\.0\\.0\\.(.+)",
            "head.detect.p4.cls_dw_0.$1",
        )
        .with_key_remapping(
            "model\\.23\\.cv3\\.1\\.0\\.1\\.(.+)",
            "head.detect.p4.cls_pw_0.$1",
        )
        .with_key_remapping(
            "model\\.23\\.cv3\\.1\\.1\\.0\\.(.+)",
            "head.detect.p4.cls_dw_1.$1",
        )
        .with_key_remapping(
            "model\\.23\\.cv3\\.1\\.1\\.1\\.(.+)",
            "head.detect.p4.cls_pw_1.$1",
        )
        .with_key_remapping(
            "model\\.23\\.cv3\\.1\\.2\\.(.+)",
            "head.detect.p4.cls_out.$1",
        )
        .with_key_remapping(
            "model\\.23\\.cv3\\.2\\.0\\.0\\.(.+)",
            "head.detect.p5.cls_dw_0.$1",
        )
        .with_key_remapping(
            "model\\.23\\.cv3\\.2\\.0\\.1\\.(.+)",
            "head.detect.p5.cls_pw_0.$1",
        )
        .with_key_remapping(
            "model\\.23\\.cv3\\.2\\.1\\.0\\.(.+)",
            "head.detect.p5.cls_dw_1.$1",
        )
        .with_key_remapping(
            "model\\.23\\.cv3\\.2\\.1\\.1\\.(.+)",
            "head.detect.p5.cls_pw_1.$1",
        )
        .with_key_remapping(
            "model\\.23\\.cv3\\.2\\.2\\.(.+)",
            "head.detect.p5.cls_out.$1",
        )
        .with_key_remapping("model\\.23\\.proto\\.cv1\\.(.+)", "head.proto.cv1.$1")
        .with_key_remapping(
            "model\\.23\\.proto\\.upsample\\.(.+)",
            "head.proto.upsample.$1",
        )
        .with_key_remapping("model\\.23\\.proto\\.cv2\\.(.+)", "head.proto.cv2.$1")
        .with_key_remapping("model\\.23\\.proto\\.cv3\\.(.+)", "head.proto.cv3.$1")
        .with_key_remapping("model\\.23\\.cv4\\.0\\.0\\.(.+)", "head.p3_mask.mask_0.$1")
        .with_key_remapping("model\\.23\\.cv4\\.0\\.1\\.(.+)", "head.p3_mask.mask_1.$1")
        .with_key_remapping(
            "model\\.23\\.cv4\\.0\\.2\\.(.+)",
            "head.p3_mask.mask_out.$1",
        )
        .with_key_remapping("model\\.23\\.cv4\\.1\\.0\\.(.+)", "head.p4_mask.mask_0.$1")
        .with_key_remapping("model\\.23\\.cv4\\.1\\.1\\.(.+)", "head.p4_mask.mask_1.$1")
        .with_key_remapping(
            "model\\.23\\.cv4\\.1\\.2\\.(.+)",
            "head.p4_mask.mask_out.$1",
        )
        .with_key_remapping("model\\.23\\.cv4\\.2\\.0\\.(.+)", "head.p5_mask.mask_0.$1")
        .with_key_remapping("model\\.23\\.cv4\\.2\\.1\\.(.+)", "head.p5_mask.mask_1.$1")
        .with_key_remapping(
            "model\\.23\\.cv4\\.2\\.2\\.(.+)",
            "head.p5_mask.mask_out.$1",
        )
}

/// Native Burn YOLO11n-seg model.
///
/// Shares the YOLO11n body; the Segment head adds the stride-4 Proto module and 32 raw mask
/// coefficients per anchor to the classic DFL decode. The runtime applies class-aware NMS with
/// the coefficients carried along.
#[derive(Module, Debug)]
pub struct Yolo11SegN<B: Backend> {
    body: Yolo11BodySmall<B>,
    head: Yolo11SegHead<B>,
}

impl<B: Backend> Yolo11SegN<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> SegmentOutput<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> SegmentTrainOutput<B> {
        self.head.forward_train(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLO11n-seg checkpoint.
    #[cfg(feature = "pretrained")]
    pub fn load_pytorch_weights(
        &mut self,
        path: impl Into<PathBuf>,
    ) -> Result<(), PytorchStoreError> {
        let mut store = pytorch_seg_store(path);
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
                weights::artifact_format("yolo11n-seg"),
            )
            .metadata("montgomery.model", "yolo11n-seg")
            .metadata("montgomery.classes", "coco-80")
            .metadata("montgomery.precision", "f16")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolo11SegNConfig;

impl Yolo11SegNConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo11SegN<B> {
        self.init_with_classes(80, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolo11SegN<B> {
        Yolo11SegN {
            body: Yolo11BodyNConfig.init(device),
            head: Yolo11SegHeadConfig::new(64, 128, 256, 64)
                .with_num_classes(num_classes)
                .init(device),
        }
    }
}

/// Native Burn YOLO11s-seg model.
#[derive(Module, Debug)]
pub struct Yolo11SegS<B: Backend> {
    body: Yolo11BodySmall<B>,
    head: Yolo11SegHead<B>,
}

impl<B: Backend> Yolo11SegS<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> SegmentOutput<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> SegmentTrainOutput<B> {
        self.head.forward_train(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLO11s-seg checkpoint.
    #[cfg(feature = "pretrained")]
    pub fn load_pytorch_weights(
        &mut self,
        path: impl Into<PathBuf>,
    ) -> Result<(), PytorchStoreError> {
        let mut store = pytorch_seg_store(path);
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
                weights::artifact_format("yolo11s-seg"),
            )
            .metadata("montgomery.model", "yolo11s-seg")
            .metadata("montgomery.classes", "coco-80")
            .metadata("montgomery.precision", "f16")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolo11SegSConfig;

impl Yolo11SegSConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo11SegS<B> {
        self.init_with_classes(80, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolo11SegS<B> {
        Yolo11SegS {
            body: Yolo11BodySConfig.init(device),
            head: Yolo11SegHeadConfig::new(128, 256, 512, 128)
                .with_num_classes(num_classes)
                .init(device),
        }
    }
}

/// Native Burn YOLO11m-seg model.
///
/// The m-scale body forces the C3k chain onto the early backbone stages (`parse_model`'s m/l/x
/// rule), so it shares [`Yolo11BodyLarge`] with l/x; the Segment head adds the stride-4 Proto
/// module and 32 raw mask coefficients per anchor to the classic DFL decode.
#[derive(Module, Debug)]
pub struct Yolo11SegM<B: Backend> {
    body: Yolo11BodyLarge<B>,
    head: Yolo11SegHead<B>,
}

impl<B: Backend> Yolo11SegM<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> SegmentOutput<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> SegmentTrainOutput<B> {
        self.head.forward_train(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLO11m-seg checkpoint.
    #[cfg(feature = "pretrained")]
    pub fn load_pytorch_weights(
        &mut self,
        path: impl Into<PathBuf>,
    ) -> Result<(), PytorchStoreError> {
        let mut store = pytorch_seg_store(path);
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
                weights::artifact_format("yolo11m-seg"),
            )
            .metadata("montgomery.model", "yolo11m-seg")
            .metadata("montgomery.classes", "coco-80")
            .metadata("montgomery.precision", "f16")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolo11SegMConfig;

impl Yolo11SegMConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo11SegM<B> {
        self.init_with_classes(80, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolo11SegM<B> {
        Yolo11SegM {
            body: Yolo11BodyMConfig.init(device),
            head: Yolo11SegHeadConfig::new(256, 512, 512, 256)
                .with_num_classes(num_classes)
                .init(device),
        }
    }
}

/// Native Burn YOLO11l-seg model.
#[derive(Module, Debug)]
pub struct Yolo11SegL<B: Backend> {
    body: Yolo11BodyLarge<B>,
    head: Yolo11SegHead<B>,
}

impl<B: Backend> Yolo11SegL<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> SegmentOutput<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> SegmentTrainOutput<B> {
        self.head.forward_train(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLO11l-seg checkpoint.
    #[cfg(feature = "pretrained")]
    pub fn load_pytorch_weights(
        &mut self,
        path: impl Into<PathBuf>,
    ) -> Result<(), PytorchStoreError> {
        let mut store = pytorch_seg_store(path);
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
                weights::artifact_format("yolo11l-seg"),
            )
            .metadata("montgomery.model", "yolo11l-seg")
            .metadata("montgomery.classes", "coco-80")
            .metadata("montgomery.precision", "f16")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolo11SegLConfig;

impl Yolo11SegLConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo11SegL<B> {
        self.init_with_classes(80, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolo11SegL<B> {
        Yolo11SegL {
            body: Yolo11BodyLConfig.init(device),
            head: Yolo11SegHeadConfig::new(256, 512, 512, 256)
                .with_num_classes(num_classes)
                .init(device),
        }
    }
}

/// Native Burn YOLO11x-seg model.
#[derive(Module, Debug)]
pub struct Yolo11SegX<B: Backend> {
    body: Yolo11BodyLarge<B>,
    head: Yolo11SegHead<B>,
}

impl<B: Backend> Yolo11SegX<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> SegmentOutput<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> SegmentTrainOutput<B> {
        self.head.forward_train(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLO11x-seg checkpoint.
    #[cfg(feature = "pretrained")]
    pub fn load_pytorch_weights(
        &mut self,
        path: impl Into<PathBuf>,
    ) -> Result<(), PytorchStoreError> {
        let mut store = pytorch_seg_store(path);
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
                weights::artifact_format("yolo11x-seg"),
            )
            .metadata("montgomery.model", "yolo11x-seg")
            .metadata("montgomery.classes", "coco-80")
            .metadata("montgomery.precision", "f16")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolo11SegXConfig;

impl Yolo11SegXConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo11SegX<B> {
        self.init_with_classes(80, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolo11SegX<B> {
        Yolo11SegX {
            body: Yolo11BodyXConfig.init(device),
            head: Yolo11SegHeadConfig::new(384, 768, 768, 384)
                .with_num_classes(num_classes)
                .init(device),
        }
    }
}

#[cfg(all(test, feature = "pretrained"))]
mod tests {
    use super::*;
    use crate::models::yolo11::body::Yolo11Features;
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
        features: Yolo11Features<Flex>,
        head: &Yolo11Head<Flex>,
        fixture: &GoldenFixture,
    ) {
        let p3 = features.p3.clone();
        let p4 = features.p4.clone();
        let p5 = features.p5.clone();
        let raw = head.forward_raw(Yolo11Features {
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
                            "generate fixtures with tools/export_yolo11_fixtures.py --model {}",
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
        yolo11n_imports_official_checkpoint_and_runs_forward,
        Yolo11NConfig,
        "yolo11n"
    );
    checkpoint_test!(
        yolo11s_imports_official_checkpoint_and_runs_forward,
        Yolo11SConfig,
        "yolo11s"
    );
    checkpoint_test!(
        yolo11m_imports_official_checkpoint_and_runs_forward,
        Yolo11MConfig,
        "yolo11m"
    );
    checkpoint_test!(
        yolo11l_imports_official_checkpoint_and_runs_forward,
        Yolo11LConfig,
        "yolo11l"
    );
    checkpoint_test!(
        yolo11x_imports_official_checkpoint_and_runs_forward,
        Yolo11XConfig,
        "yolo11x"
    );

    golden_test!(
        yolo11n_matches_ultralytics_golden_tensors,
        Yolo11NConfig,
        "yolo11n"
    );
    golden_test!(
        yolo11s_matches_ultralytics_golden_tensors,
        Yolo11SConfig,
        "yolo11s"
    );
    golden_test!(
        yolo11m_matches_ultralytics_golden_tensors,
        Yolo11MConfig,
        "yolo11m"
    );
    golden_test!(
        yolo11l_matches_ultralytics_golden_tensors,
        Yolo11LConfig,
        "yolo11l"
    );
    golden_test!(
        yolo11x_matches_ultralytics_golden_tensors,
        Yolo11XConfig,
        "yolo11x"
    );

    latency_test!(
        yolo11n_measures_single_inference_latency,
        Yolo11NConfig,
        "yolo11n"
    );
    latency_test!(
        yolo11s_measures_single_inference_latency,
        Yolo11SConfig,
        "yolo11s"
    );
    latency_test!(
        yolo11m_measures_single_inference_latency,
        Yolo11MConfig,
        "yolo11m"
    );
    latency_test!(
        yolo11l_measures_single_inference_latency,
        Yolo11LConfig,
        "yolo11l"
    );
    latency_test!(
        yolo11x_measures_single_inference_latency,
        Yolo11XConfig,
        "yolo11x"
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
        yolo11n_measures_single_inference_latency_gpu,
        Yolo11NConfig,
        "yolo11n"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo11s_measures_single_inference_latency_gpu,
        Yolo11SConfig,
        "yolo11s"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo11m_measures_single_inference_latency_gpu,
        Yolo11MConfig,
        "yolo11m"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo11l_measures_single_inference_latency_gpu,
        Yolo11LConfig,
        "yolo11l"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo11x_measures_single_inference_latency_gpu,
        Yolo11XConfig,
        "yolo11x"
    );

    /// Assert one tensor against the fixture at the shared 2e-4 tolerance (segmentation variant
    /// of `assert_parity_tensors`, adding the Proto and mask-coefficient tensors).
    fn assert_seg_parity_tensors(
        features: Yolo11Features<Flex>,
        head: &Yolo11SegHead<Flex>,
        fixture: &GoldenFixture,
    ) {
        let p3 = features.p3.clone();
        let p4 = features.p4.clone();
        let p5 = features.p5.clone();
        let raw = head.detect.forward_raw(Yolo11Features {
            p3: features.p3.clone(),
            p4: features.p4.clone(),
            p5: features.p5.clone(),
        });
        let output = head.forward(features);

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
            output.boxes,
            fixture.tensors.get("decoded_boxes_cxcywh").unwrap(),
        );
        assert_golden(
            "decoded_scores",
            output.scores,
            fixture.tensors.get("decoded_scores").unwrap(),
        );
        assert_golden(
            "protos",
            output.prototypes,
            fixture.tensors.get("protos").unwrap(),
        );
        assert_golden(
            "mask_coeffs",
            output.coefficients,
            fixture.tensors.get("mask_coeffs").unwrap(),
        );
    }

    macro_rules! seg_checkpoint_test {
        ($fn_name:ident, $config:ty, $id:literal) => {
            /// Run manually after converting the official seg checkpoint. Kept ignored in CI
            /// because the source checkpoint is an external AGPL asset.
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
                        assert_eq!(output.coefficients.dims(), [1, 32, 84]);
                        assert_eq!(output.prototypes.dims(), [1, 32, 16, 16]);
                    })
                    .unwrap();
                worker.join().unwrap();
            }
        };
    }

    macro_rules! seg_golden_test {
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
                            "generate fixtures with tools/export_yolo11_fixtures.py --model {}",
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
                        assert_seg_parity_tensors(features, &model.head, &fixture);
                    })
                    .unwrap();
                worker.join().unwrap();
            }
        };
    }

    macro_rules! seg_latency_test {
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
                            let _ = output.coefficients.sum().into_data();
                            let _ = output.prototypes.sum().into_data();
                        }
                        let mut samples = Vec::with_capacity(TIMED_RUNS);
                        for _ in 0..TIMED_RUNS {
                            let started = std::time::Instant::now();
                            let output = model.forward(input.clone());
                            // Force result completion so the sample includes the full compute, not
                            // just kernel dispatch (a no-op on CPU, load-bearing on GPU).
                            let _ = output.boxes.sum().into_data();
                            let _ = output.scores.sum().into_data();
                            let _ = output.coefficients.sum().into_data();
                            let _ = output.prototypes.sum().into_data();
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

    seg_checkpoint_test!(
        yolo11n_seg_imports_official_checkpoint_and_runs_forward,
        Yolo11SegNConfig,
        "yolo11n-seg"
    );
    seg_checkpoint_test!(
        yolo11s_seg_imports_official_checkpoint_and_runs_forward,
        Yolo11SegSConfig,
        "yolo11s-seg"
    );
    seg_checkpoint_test!(
        yolo11m_seg_imports_official_checkpoint_and_runs_forward,
        Yolo11SegMConfig,
        "yolo11m-seg"
    );
    seg_checkpoint_test!(
        yolo11l_seg_imports_official_checkpoint_and_runs_forward,
        Yolo11SegLConfig,
        "yolo11l-seg"
    );
    seg_checkpoint_test!(
        yolo11x_seg_imports_official_checkpoint_and_runs_forward,
        Yolo11SegXConfig,
        "yolo11x-seg"
    );

    seg_golden_test!(
        yolo11n_seg_matches_ultralytics_golden_tensors,
        Yolo11SegNConfig,
        "yolo11n-seg"
    );
    seg_golden_test!(
        yolo11s_seg_matches_ultralytics_golden_tensors,
        Yolo11SegSConfig,
        "yolo11s-seg"
    );
    seg_golden_test!(
        yolo11m_seg_matches_ultralytics_golden_tensors,
        Yolo11SegMConfig,
        "yolo11m-seg"
    );
    seg_golden_test!(
        yolo11l_seg_matches_ultralytics_golden_tensors,
        Yolo11SegLConfig,
        "yolo11l-seg"
    );
    seg_golden_test!(
        yolo11x_seg_matches_ultralytics_golden_tensors,
        Yolo11SegXConfig,
        "yolo11x-seg"
    );

    seg_latency_test!(
        yolo11n_seg_measures_single_inference_latency,
        Yolo11SegNConfig,
        "yolo11n-seg"
    );
    seg_latency_test!(
        yolo11s_seg_measures_single_inference_latency,
        Yolo11SegSConfig,
        "yolo11s-seg"
    );
    seg_latency_test!(
        yolo11m_seg_measures_single_inference_latency,
        Yolo11SegMConfig,
        "yolo11m-seg"
    );
    seg_latency_test!(
        yolo11l_seg_measures_single_inference_latency,
        Yolo11SegLConfig,
        "yolo11l-seg"
    );
    seg_latency_test!(
        yolo11x_seg_measures_single_inference_latency,
        Yolo11SegXConfig,
        "yolo11x-seg"
    );

    #[cfg(feature = "gpu")]
    macro_rules! seg_gpu_latency_test {
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
                            let _ = output.coefficients.sum().into_data();
                            let _ = output.prototypes.sum().into_data();
                        }
                        let mut samples = Vec::with_capacity(TIMED_RUNS);
                        for _ in 0..TIMED_RUNS {
                            let started = std::time::Instant::now();
                            let output = model.forward(input.clone());
                            // Force result completion so the sample includes the full compute, not
                            // just kernel dispatch.
                            let _ = output.boxes.sum().into_data();
                            let _ = output.scores.sum().into_data();
                            let _ = output.coefficients.sum().into_data();
                            let _ = output.prototypes.sum().into_data();
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
    seg_gpu_latency_test!(
        yolo11n_seg_measures_single_inference_latency_gpu,
        Yolo11SegNConfig,
        "yolo11n-seg"
    );
    #[cfg(feature = "gpu")]
    seg_gpu_latency_test!(
        yolo11s_seg_measures_single_inference_latency_gpu,
        Yolo11SegSConfig,
        "yolo11s-seg"
    );
    #[cfg(feature = "gpu")]
    seg_gpu_latency_test!(
        yolo11m_seg_measures_single_inference_latency_gpu,
        Yolo11SegMConfig,
        "yolo11m-seg"
    );
    #[cfg(feature = "gpu")]
    seg_gpu_latency_test!(
        yolo11l_seg_measures_single_inference_latency_gpu,
        Yolo11SegLConfig,
        "yolo11l-seg"
    );
    #[cfg(feature = "gpu")]
    seg_gpu_latency_test!(
        yolo11x_seg_measures_single_inference_latency_gpu,
        Yolo11SegXConfig,
        "yolo11x-seg"
    );
}
