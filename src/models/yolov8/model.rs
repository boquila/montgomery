use burn::{
    module::Module,
    tensor::{Device, Tensor, backend::Backend},
};

use super::{
    body::{
        Yolov8Body, Yolov8BodyLConfig, Yolov8BodyMConfig, Yolov8BodyNConfig, Yolov8BodySConfig,
        Yolov8BodyXConfig,
    },
    head::{DecodedPredictions, Yolov8Head, Yolov8HeadConfig},
    segmentation::{Yolov8SegHead, Yolov8SegHeadConfig},
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

/// Build the PyTorch-state store shared by every YOLOv8 detect scale variant.
///
/// Body layers retain their Ultralytics graph indices (the head is model.22, which the body rule
/// must not match); the head's box and classification towers are remapped one rule per
/// path-segment pattern. The fixed DFL projection (`model.22.dfl.conv.weight`) is intentionally
/// unmapped: it is baked into the head as a constant. The remaps are scale-independent because
/// every variant shares the checkpoint key layout.
#[cfg(feature = "pretrained")]
fn pytorch_store(path: impl Into<PathBuf>) -> PytorchStore {
    PytorchStore::from_file(path)
        .with_top_level_key("model")
        // Body layers retain their Ultralytics graph indices. The head is model.22, so this rule
        // must not match it.
        .with_key_remapping("model\\.([0-9]|1[0-9]|2[0-1])\\.(.+)", "body.model_$1.$2")
        // model.22.cv2.{scale}.{layer} is the box regression tower.
        .with_key_remapping("model\\.22\\.cv2\\.0\\.0\\.(.+)", "head.p3.box_0.$1")
        .with_key_remapping("model\\.22\\.cv2\\.0\\.1\\.(.+)", "head.p3.box_1.$1")
        .with_key_remapping("model\\.22\\.cv2\\.0\\.2\\.(.+)", "head.p3.box_out.$1")
        .with_key_remapping("model\\.22\\.cv2\\.1\\.0\\.(.+)", "head.p4.box_0.$1")
        .with_key_remapping("model\\.22\\.cv2\\.1\\.1\\.(.+)", "head.p4.box_1.$1")
        .with_key_remapping("model\\.22\\.cv2\\.1\\.2\\.(.+)", "head.p4.box_out.$1")
        .with_key_remapping("model\\.22\\.cv2\\.2\\.0\\.(.+)", "head.p5.box_0.$1")
        .with_key_remapping("model\\.22\\.cv2\\.2\\.1\\.(.+)", "head.p5.box_1.$1")
        .with_key_remapping("model\\.22\\.cv2\\.2\\.2\\.(.+)", "head.p5.box_out.$1")
        // model.22.cv3.{scale}.{tower} is the legacy full-3x3-conv classification tower.
        .with_key_remapping("model\\.22\\.cv3\\.0\\.0\\.(.+)", "head.p3.cls_0.$1")
        .with_key_remapping("model\\.22\\.cv3\\.0\\.1\\.(.+)", "head.p3.cls_1.$1")
        .with_key_remapping("model\\.22\\.cv3\\.0\\.2\\.(.+)", "head.p3.cls_out.$1")
        .with_key_remapping("model\\.22\\.cv3\\.1\\.0\\.(.+)", "head.p4.cls_0.$1")
        .with_key_remapping("model\\.22\\.cv3\\.1\\.1\\.(.+)", "head.p4.cls_1.$1")
        .with_key_remapping("model\\.22\\.cv3\\.1\\.2\\.(.+)", "head.p4.cls_out.$1")
        .with_key_remapping("model\\.22\\.cv3\\.2\\.0\\.(.+)", "head.p5.cls_0.$1")
        .with_key_remapping("model\\.22\\.cv3\\.2\\.1\\.(.+)", "head.p5.cls_1.$1")
        .with_key_remapping("model\\.22\\.cv3\\.2\\.2\\.(.+)", "head.p5.cls_out.$1")
}

/// Native Burn YOLOv8n model.
///
/// The body feeds the classic DFL detection head whose predictions are decoded to center-size
/// model-input pixels; the runtime applies class-aware non-maximum suppression.
#[derive(Module, Debug)]
pub struct Yolov8N<B: Backend> {
    body: Yolov8Body<B>,
    head: Yolov8Head<B>,
}

impl<B: Backend> Yolov8N<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> DecodedPredictions<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> super::head::RawPredictions<B> {
        self.head.forward_raw(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLOv8n checkpoint.
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
                weights::artifact_format("yolov8n"),
            )
            .metadata("montgomery.model", "yolov8n")
            .metadata("montgomery.classes", "coco-80")
            .metadata("montgomery.precision", "f16")
            .metadata("montgomery.source", "ultralytics-v8.4")
            .metadata("montgomery.license", "AGPL-3.0")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolov8NConfig;

impl Yolov8NConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolov8N<B> {
        self.init_with_classes(80, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolov8N<B> {
        Yolov8N {
            body: Yolov8BodyNConfig.init(device),
            head: Yolov8HeadConfig::new(64, 128, 256)
                .with_num_classes(num_classes)
                .init(device),
        }
    }
}

/// Native Burn YOLOv8s model.
#[derive(Module, Debug)]
pub struct Yolov8S<B: Backend> {
    body: Yolov8Body<B>,
    head: Yolov8Head<B>,
}

impl<B: Backend> Yolov8S<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> DecodedPredictions<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> super::head::RawPredictions<B> {
        self.head.forward_raw(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLOv8s checkpoint.
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
                weights::artifact_format("yolov8s"),
            )
            .metadata("montgomery.model", "yolov8s")
            .metadata("montgomery.classes", "coco-80")
            .metadata("montgomery.precision", "f16")
            .metadata("montgomery.source", "ultralytics-v8.4")
            .metadata("montgomery.license", "AGPL-3.0")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolov8SConfig;

impl Yolov8SConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolov8S<B> {
        self.init_with_classes(80, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolov8S<B> {
        Yolov8S {
            body: Yolov8BodySConfig.init(device),
            head: Yolov8HeadConfig::new(128, 256, 512)
                .with_num_classes(num_classes)
                .init(device),
        }
    }
}

/// Native Burn YOLOv8m model.
#[derive(Module, Debug)]
pub struct Yolov8M<B: Backend> {
    body: Yolov8Body<B>,
    head: Yolov8Head<B>,
}

impl<B: Backend> Yolov8M<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> DecodedPredictions<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> super::head::RawPredictions<B> {
        self.head.forward_raw(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLOv8m checkpoint.
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
                weights::artifact_format("yolov8m"),
            )
            .metadata("montgomery.model", "yolov8m")
            .metadata("montgomery.classes", "coco-80")
            .metadata("montgomery.precision", "f16")
            .metadata("montgomery.source", "ultralytics-v8.4")
            .metadata("montgomery.license", "AGPL-3.0")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolov8MConfig;

impl Yolov8MConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolov8M<B> {
        self.init_with_classes(80, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolov8M<B> {
        Yolov8M {
            body: Yolov8BodyMConfig.init(device),
            head: Yolov8HeadConfig::new(192, 384, 576)
                .with_num_classes(num_classes)
                .init(device),
        }
    }
}

/// Native Burn YOLOv8l model.
#[derive(Module, Debug)]
pub struct Yolov8L<B: Backend> {
    body: Yolov8Body<B>,
    head: Yolov8Head<B>,
}

impl<B: Backend> Yolov8L<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> DecodedPredictions<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> super::head::RawPredictions<B> {
        self.head.forward_raw(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLOv8l checkpoint.
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
                weights::artifact_format("yolov8l"),
            )
            .metadata("montgomery.model", "yolov8l")
            .metadata("montgomery.classes", "coco-80")
            .metadata("montgomery.precision", "f16")
            .metadata("montgomery.source", "ultralytics-v8.4")
            .metadata("montgomery.license", "AGPL-3.0")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolov8LConfig;

impl Yolov8LConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolov8L<B> {
        self.init_with_classes(80, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolov8L<B> {
        Yolov8L {
            body: Yolov8BodyLConfig.init(device),
            head: Yolov8HeadConfig::new(256, 512, 512)
                .with_num_classes(num_classes)
                .init(device),
        }
    }
}

/// Native Burn YOLOv8x model.
#[derive(Module, Debug)]
pub struct Yolov8X<B: Backend> {
    body: Yolov8Body<B>,
    head: Yolov8Head<B>,
}

impl<B: Backend> Yolov8X<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> DecodedPredictions<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> super::head::RawPredictions<B> {
        self.head.forward_raw(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLOv8x checkpoint.
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
                weights::artifact_format("yolov8x"),
            )
            .metadata("montgomery.model", "yolov8x")
            .metadata("montgomery.classes", "coco-80")
            .metadata("montgomery.precision", "f16")
            .metadata("montgomery.source", "ultralytics-v8.4")
            .metadata("montgomery.license", "AGPL-3.0")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolov8XConfig;

impl Yolov8XConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolov8X<B> {
        self.init_with_classes(80, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolov8X<B> {
        Yolov8X {
            body: Yolov8BodyXConfig.init(device),
            head: Yolov8HeadConfig::new(320, 640, 640)
                .with_num_classes(num_classes)
                .init(device),
        }
    }
}

/// Build the PyTorch-state store for the YOLOv8-seg scale variants.
///
/// The seg YAML keeps the detect model's body (layers 0-21) and puts the `Segment` head at
/// `model.22`, so every rule here mirrors the detect remap with the head paths prefixed by
/// `detect.`; the additions are the Proto module and the `cv4` mask-coefficient towers. The fixed
/// DFL projection (`model.22.dfl.conv.weight`) is intentionally unmapped, exactly like the detect
/// variants.
#[cfg(feature = "pretrained")]
fn pytorch_seg_store(path: impl Into<PathBuf>) -> PytorchStore {
    PytorchStore::from_file(path)
        .with_top_level_key("model")
        .with_key_remapping("model\\.([0-9]|1[0-9]|2[0-1])\\.(.+)", "body.model_$1.$2")
        .with_key_remapping("model\\.22\\.cv2\\.0\\.0\\.(.+)", "head.detect.p3.box_0.$1")
        .with_key_remapping("model\\.22\\.cv2\\.0\\.1\\.(.+)", "head.detect.p3.box_1.$1")
        .with_key_remapping(
            "model\\.22\\.cv2\\.0\\.2\\.(.+)",
            "head.detect.p3.box_out.$1",
        )
        .with_key_remapping("model\\.22\\.cv2\\.1\\.0\\.(.+)", "head.detect.p4.box_0.$1")
        .with_key_remapping("model\\.22\\.cv2\\.1\\.1\\.(.+)", "head.detect.p4.box_1.$1")
        .with_key_remapping(
            "model\\.22\\.cv2\\.1\\.2\\.(.+)",
            "head.detect.p4.box_out.$1",
        )
        .with_key_remapping("model\\.22\\.cv2\\.2\\.0\\.(.+)", "head.detect.p5.box_0.$1")
        .with_key_remapping("model\\.22\\.cv2\\.2\\.1\\.(.+)", "head.detect.p5.box_1.$1")
        .with_key_remapping(
            "model\\.22\\.cv2\\.2\\.2\\.(.+)",
            "head.detect.p5.box_out.$1",
        )
        .with_key_remapping("model\\.22\\.cv3\\.0\\.0\\.(.+)", "head.detect.p3.cls_0.$1")
        .with_key_remapping("model\\.22\\.cv3\\.0\\.1\\.(.+)", "head.detect.p3.cls_1.$1")
        .with_key_remapping(
            "model\\.22\\.cv3\\.0\\.2\\.(.+)",
            "head.detect.p3.cls_out.$1",
        )
        .with_key_remapping("model\\.22\\.cv3\\.1\\.0\\.(.+)", "head.detect.p4.cls_0.$1")
        .with_key_remapping("model\\.22\\.cv3\\.1\\.1\\.(.+)", "head.detect.p4.cls_1.$1")
        .with_key_remapping(
            "model\\.22\\.cv3\\.1\\.2\\.(.+)",
            "head.detect.p4.cls_out.$1",
        )
        .with_key_remapping("model\\.22\\.cv3\\.2\\.0\\.(.+)", "head.detect.p5.cls_0.$1")
        .with_key_remapping("model\\.22\\.cv3\\.2\\.1\\.(.+)", "head.detect.p5.cls_1.$1")
        .with_key_remapping(
            "model\\.22\\.cv3\\.2\\.2\\.(.+)",
            "head.detect.p5.cls_out.$1",
        )
        .with_key_remapping("model\\.22\\.proto\\.cv1\\.(.+)", "head.proto.cv1.$1")
        .with_key_remapping(
            "model\\.22\\.proto\\.upsample\\.(.+)",
            "head.proto.upsample.$1",
        )
        .with_key_remapping("model\\.22\\.proto\\.cv2\\.(.+)", "head.proto.cv2.$1")
        .with_key_remapping("model\\.22\\.proto\\.cv3\\.(.+)", "head.proto.cv3.$1")
        .with_key_remapping("model\\.22\\.cv4\\.0\\.0\\.(.+)", "head.p3_mask.mask_0.$1")
        .with_key_remapping("model\\.22\\.cv4\\.0\\.1\\.(.+)", "head.p3_mask.mask_1.$1")
        .with_key_remapping(
            "model\\.22\\.cv4\\.0\\.2\\.(.+)",
            "head.p3_mask.mask_out.$1",
        )
        .with_key_remapping("model\\.22\\.cv4\\.1\\.0\\.(.+)", "head.p4_mask.mask_0.$1")
        .with_key_remapping("model\\.22\\.cv4\\.1\\.1\\.(.+)", "head.p4_mask.mask_1.$1")
        .with_key_remapping(
            "model\\.22\\.cv4\\.1\\.2\\.(.+)",
            "head.p4_mask.mask_out.$1",
        )
        .with_key_remapping("model\\.22\\.cv4\\.2\\.0\\.(.+)", "head.p5_mask.mask_0.$1")
        .with_key_remapping("model\\.22\\.cv4\\.2\\.1\\.(.+)", "head.p5_mask.mask_1.$1")
        .with_key_remapping(
            "model\\.22\\.cv4\\.2\\.2\\.(.+)",
            "head.p5_mask.mask_out.$1",
        )
}

/// Native Burn YOLOv8n-seg model.
///
/// Shares the YOLOv8n body; the Segment head adds the stride-4 Proto module and 32 raw mask
/// coefficients per anchor to the classic DFL decode. The runtime applies class-aware NMS with
/// the coefficients carried along.
#[derive(Module, Debug)]
pub struct Yolov8SegN<B: Backend> {
    body: Yolov8Body<B>,
    head: Yolov8SegHead<B>,
}

impl<B: Backend> Yolov8SegN<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> crate::models::yolo11::SegmentOutput<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> super::segmentation::SegmentTrainOutput<B> {
        self.head.forward_train(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLOv8n-seg checkpoint.
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
                weights::artifact_format("yolov8n-seg"),
            )
            .metadata("montgomery.model", "yolov8n-seg")
            .metadata("montgomery.classes", "coco-80")
            .metadata("montgomery.precision", "f16")
            .metadata("montgomery.source", "ultralytics-v8.4")
            .metadata("montgomery.license", "AGPL-3.0")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolov8SegNConfig;

impl Yolov8SegNConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolov8SegN<B> {
        self.init_with_classes(80, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolov8SegN<B> {
        Yolov8SegN {
            body: Yolov8BodyNConfig.init(device),
            head: Yolov8SegHeadConfig::new(64, 128, 256, 64)
                .with_num_classes(num_classes)
                .init(device),
        }
    }
}

/// Native Burn YOLOv8s-seg model.
#[derive(Module, Debug)]
pub struct Yolov8SegS<B: Backend> {
    body: Yolov8Body<B>,
    head: Yolov8SegHead<B>,
}

impl<B: Backend> Yolov8SegS<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> crate::models::yolo11::SegmentOutput<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> super::segmentation::SegmentTrainOutput<B> {
        self.head.forward_train(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLOv8s-seg checkpoint.
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
                weights::artifact_format("yolov8s-seg"),
            )
            .metadata("montgomery.model", "yolov8s-seg")
            .metadata("montgomery.classes", "coco-80")
            .metadata("montgomery.precision", "f16")
            .metadata("montgomery.source", "ultralytics-v8.4")
            .metadata("montgomery.license", "AGPL-3.0")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolov8SegSConfig;

impl Yolov8SegSConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolov8SegS<B> {
        self.init_with_classes(80, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolov8SegS<B> {
        Yolov8SegS {
            body: Yolov8BodySConfig.init(device),
            head: Yolov8SegHeadConfig::new(128, 256, 512, 128)
                .with_num_classes(num_classes)
                .init(device),
        }
    }
}

/// Native Burn YOLOv8m-seg model.
#[derive(Module, Debug)]
pub struct Yolov8SegM<B: Backend> {
    body: Yolov8Body<B>,
    head: Yolov8SegHead<B>,
}

impl<B: Backend> Yolov8SegM<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> crate::models::yolo11::SegmentOutput<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> super::segmentation::SegmentTrainOutput<B> {
        self.head.forward_train(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLOv8m-seg checkpoint.
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
                weights::artifact_format("yolov8m-seg"),
            )
            .metadata("montgomery.model", "yolov8m-seg")
            .metadata("montgomery.classes", "coco-80")
            .metadata("montgomery.precision", "f16")
            .metadata("montgomery.source", "ultralytics-v8.4")
            .metadata("montgomery.license", "AGPL-3.0")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolov8SegMConfig;

impl Yolov8SegMConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolov8SegM<B> {
        self.init_with_classes(80, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolov8SegM<B> {
        Yolov8SegM {
            body: Yolov8BodyMConfig.init(device),
            head: Yolov8SegHeadConfig::new(192, 384, 576, 192)
                .with_num_classes(num_classes)
                .init(device),
        }
    }
}

/// Native Burn YOLOv8l-seg model.
#[derive(Module, Debug)]
pub struct Yolov8SegL<B: Backend> {
    body: Yolov8Body<B>,
    head: Yolov8SegHead<B>,
}

impl<B: Backend> Yolov8SegL<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> crate::models::yolo11::SegmentOutput<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> super::segmentation::SegmentTrainOutput<B> {
        self.head.forward_train(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLOv8l-seg checkpoint.
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
                weights::artifact_format("yolov8l-seg"),
            )
            .metadata("montgomery.model", "yolov8l-seg")
            .metadata("montgomery.classes", "coco-80")
            .metadata("montgomery.precision", "f16")
            .metadata("montgomery.source", "ultralytics-v8.4")
            .metadata("montgomery.license", "AGPL-3.0")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolov8SegLConfig;

impl Yolov8SegLConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolov8SegL<B> {
        self.init_with_classes(80, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolov8SegL<B> {
        Yolov8SegL {
            body: Yolov8BodyLConfig.init(device),
            head: Yolov8SegHeadConfig::new(256, 512, 512, 256)
                .with_num_classes(num_classes)
                .init(device),
        }
    }
}

/// Native Burn YOLOv8x-seg model.
#[derive(Module, Debug)]
pub struct Yolov8SegX<B: Backend> {
    body: Yolov8Body<B>,
    head: Yolov8SegHead<B>,
}

impl<B: Backend> Yolov8SegX<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> crate::models::yolo11::SegmentOutput<B> {
        self.head.forward(self.body.forward(input))
    }

    pub fn forward_train(&self, input: Tensor<B, 4>) -> super::segmentation::SegmentTrainOutput<B> {
        self.head.forward_train(self.body.forward(input))
    }

    /// Import tensor-only state exported from an official Ultralytics YOLOv8x-seg checkpoint.
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
                weights::artifact_format("yolov8x-seg"),
            )
            .metadata("montgomery.model", "yolov8x-seg")
            .metadata("montgomery.classes", "coco-80")
            .metadata("montgomery.precision", "f16")
            .metadata("montgomery.source", "ultralytics-v8.4")
            .metadata("montgomery.license", "AGPL-3.0")
            .with_to_adapter(HalfPrecisionAdapter::new());
        self.save_into(&mut store)
    }
}

#[derive(Debug, Default)]
pub struct Yolov8SegXConfig;

impl Yolov8SegXConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolov8SegX<B> {
        self.init_with_classes(80, device)
    }

    pub fn init_with_classes<B: Backend>(
        &self,
        num_classes: usize,
        device: &Device<B>,
    ) -> Yolov8SegX<B> {
        Yolov8SegX {
            body: Yolov8BodyXConfig.init(device),
            head: Yolov8SegHeadConfig::new(320, 640, 640, 320)
                .with_num_classes(num_classes)
                .init(device),
        }
    }
}

#[cfg(all(test, feature = "pretrained"))]
mod tests {
    use super::*;
    use crate::models::yolov8::body::Yolov8Features;
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
        features: Yolov8Features<Flex>,
        head: &Yolov8Head<Flex>,
        fixture: &GoldenFixture,
    ) {
        let p3 = features.p3.clone();
        let p4 = features.p4.clone();
        let p5 = features.p5.clone();
        let raw = head.forward_raw(Yolov8Features {
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
                let checkpoint = std::path::PathBuf::from(format!("target/{}.bpk", $id));
                let fixture: GoldenFixture = serde_json::from_slice(
                    &std::fs::read(format!("target/{}-golden-v1.json", $id)).unwrap_or_else(|_| {
                        panic!(
                            "generate fixtures with tools/export_yolov8_fixtures.py --model {}",
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
                    "target/{}.bpk",
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
        yolov8n_imports_official_checkpoint_and_runs_forward,
        Yolov8NConfig,
        "yolov8n"
    );
    checkpoint_test!(
        yolov8s_imports_official_checkpoint_and_runs_forward,
        Yolov8SConfig,
        "yolov8s"
    );
    checkpoint_test!(
        yolov8m_imports_official_checkpoint_and_runs_forward,
        Yolov8MConfig,
        "yolov8m"
    );
    checkpoint_test!(
        yolov8l_imports_official_checkpoint_and_runs_forward,
        Yolov8LConfig,
        "yolov8l"
    );
    checkpoint_test!(
        yolov8x_imports_official_checkpoint_and_runs_forward,
        Yolov8XConfig,
        "yolov8x"
    );

    golden_test!(
        yolov8n_matches_ultralytics_golden_tensors,
        Yolov8NConfig,
        "yolov8n"
    );
    golden_test!(
        yolov8s_matches_ultralytics_golden_tensors,
        Yolov8SConfig,
        "yolov8s"
    );
    golden_test!(
        yolov8m_matches_ultralytics_golden_tensors,
        Yolov8MConfig,
        "yolov8m"
    );
    golden_test!(
        yolov8l_matches_ultralytics_golden_tensors,
        Yolov8LConfig,
        "yolov8l"
    );
    golden_test!(
        yolov8x_matches_ultralytics_golden_tensors,
        Yolov8XConfig,
        "yolov8x"
    );

    latency_test!(
        yolov8n_measures_single_inference_latency,
        Yolov8NConfig,
        "yolov8n"
    );
    latency_test!(
        yolov8s_measures_single_inference_latency,
        Yolov8SConfig,
        "yolov8s"
    );
    latency_test!(
        yolov8m_measures_single_inference_latency,
        Yolov8MConfig,
        "yolov8m"
    );
    latency_test!(
        yolov8l_measures_single_inference_latency,
        Yolov8LConfig,
        "yolov8l"
    );
    latency_test!(
        yolov8x_measures_single_inference_latency,
        Yolov8XConfig,
        "yolov8x"
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
                    "target/{}.bpk",
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
        yolov8n_measures_single_inference_latency_gpu,
        Yolov8NConfig,
        "yolov8n"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolov8s_measures_single_inference_latency_gpu,
        Yolov8SConfig,
        "yolov8s"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolov8m_measures_single_inference_latency_gpu,
        Yolov8MConfig,
        "yolov8m"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolov8l_measures_single_inference_latency_gpu,
        Yolov8LConfig,
        "yolov8l"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolov8x_measures_single_inference_latency_gpu,
        Yolov8XConfig,
        "yolov8x"
    );

    /// Assert one tensor against the fixture at the shared 2e-4 tolerance (segmentation variant
    /// of `assert_parity_tensors`, adding the Proto and mask-coefficient tensors).
    fn assert_seg_parity_tensors(
        features: Yolov8Features<Flex>,
        head: &Yolov8SegHead<Flex>,
        fixture: &GoldenFixture,
    ) {
        let p3 = features.p3.clone();
        let p4 = features.p4.clone();
        let p5 = features.p5.clone();
        let raw = head.detect.forward_raw(Yolov8Features {
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
                let checkpoint = std::path::PathBuf::from(format!("target/{}.bpk", $id));
                let fixture: GoldenFixture = serde_json::from_slice(
                    &std::fs::read(format!("target/{}-golden-v1.json", $id)).unwrap_or_else(|_| {
                        panic!(
                            "generate fixtures with tools/export_yolov8_fixtures.py --model {}",
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
                    "target/{}.bpk",
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
        yolov8n_seg_imports_official_checkpoint_and_runs_forward,
        Yolov8SegNConfig,
        "yolov8n-seg"
    );
    seg_checkpoint_test!(
        yolov8s_seg_imports_official_checkpoint_and_runs_forward,
        Yolov8SegSConfig,
        "yolov8s-seg"
    );
    seg_checkpoint_test!(
        yolov8m_seg_imports_official_checkpoint_and_runs_forward,
        Yolov8SegMConfig,
        "yolov8m-seg"
    );
    seg_checkpoint_test!(
        yolov8l_seg_imports_official_checkpoint_and_runs_forward,
        Yolov8SegLConfig,
        "yolov8l-seg"
    );
    seg_checkpoint_test!(
        yolov8x_seg_imports_official_checkpoint_and_runs_forward,
        Yolov8SegXConfig,
        "yolov8x-seg"
    );

    seg_golden_test!(
        yolov8n_seg_matches_ultralytics_golden_tensors,
        Yolov8SegNConfig,
        "yolov8n-seg"
    );
    seg_golden_test!(
        yolov8s_seg_matches_ultralytics_golden_tensors,
        Yolov8SegSConfig,
        "yolov8s-seg"
    );
    seg_golden_test!(
        yolov8m_seg_matches_ultralytics_golden_tensors,
        Yolov8SegMConfig,
        "yolov8m-seg"
    );
    seg_golden_test!(
        yolov8l_seg_matches_ultralytics_golden_tensors,
        Yolov8SegLConfig,
        "yolov8l-seg"
    );
    seg_golden_test!(
        yolov8x_seg_matches_ultralytics_golden_tensors,
        Yolov8SegXConfig,
        "yolov8x-seg"
    );

    seg_latency_test!(
        yolov8n_seg_measures_single_inference_latency,
        Yolov8SegNConfig,
        "yolov8n-seg"
    );
    seg_latency_test!(
        yolov8s_seg_measures_single_inference_latency,
        Yolov8SegSConfig,
        "yolov8s-seg"
    );
    seg_latency_test!(
        yolov8m_seg_measures_single_inference_latency,
        Yolov8SegMConfig,
        "yolov8m-seg"
    );
    seg_latency_test!(
        yolov8l_seg_measures_single_inference_latency,
        Yolov8SegLConfig,
        "yolov8l-seg"
    );
    seg_latency_test!(
        yolov8x_seg_measures_single_inference_latency,
        Yolov8SegXConfig,
        "yolov8x-seg"
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
                    "target/{}.bpk",
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
        yolov8n_seg_measures_single_inference_latency_gpu,
        Yolov8SegNConfig,
        "yolov8n-seg"
    );
    #[cfg(feature = "gpu")]
    seg_gpu_latency_test!(
        yolov8s_seg_measures_single_inference_latency_gpu,
        Yolov8SegSConfig,
        "yolov8s-seg"
    );
    #[cfg(feature = "gpu")]
    seg_gpu_latency_test!(
        yolov8m_seg_measures_single_inference_latency_gpu,
        Yolov8SegMConfig,
        "yolov8m-seg"
    );
    #[cfg(feature = "gpu")]
    seg_gpu_latency_test!(
        yolov8l_seg_measures_single_inference_latency_gpu,
        Yolov8SegLConfig,
        "yolov8l-seg"
    );
    #[cfg(feature = "gpu")]
    seg_gpu_latency_test!(
        yolov8x_seg_measures_single_inference_latency_gpu,
        Yolov8SegXConfig,
        "yolov8x-seg"
    );
}
