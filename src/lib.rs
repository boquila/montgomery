//! A small, end-to-end object detection API built on [Burn](https://burn.dev).
//!
//! The stable path supports YOLOX (nano/tiny/s/m/l/x) trained on COCO, with experimental native
//! YOLOv3-Tiny-Ultralytics, YOLOv10 (n/s/m/b/l/x), YOLO26 (n/s/m/l/x), YOLO11 (n/s/m/l/x),
//! YOLOv8 (n/s/m/l/x), and YOLO12 (n/s/m/l/x) inference paths, plus YOLO11-seg (n/s/m/l/x),
//! YOLO26-seg (n/s/m/l/x), and YOLOv8-seg (n/s/m/l/x) instance segmentation and the
//! YOLO26-cls/YOLO11-cls/YOLOv8-cls (n/s/m/l/x) ImageNet-1k classification. Model inference and
//! post-processing run from Rust — on the Flex CPU backend by default, or on the Wgpu GPU backend
//! (Vulkan/DX12/Metal) when built with the `gpu` feature. No Python runtime or ONNX runtime is
//! involved.

extern crate alloc;

mod data;
#[cfg(feature = "onnx")]
pub mod export;
pub mod models;
mod postprocess;
#[cfg(feature = "training")]
pub mod training;

use std::{
    borrow::Cow,
    error::Error,
    fmt,
    path::{Path, PathBuf},
    str::FromStr,
};

#[cfg(feature = "pretrained")]
use crate::data::IMAGENET_CLASSES;
use crate::data::{LetterboxedImage, classify_transform};
#[cfg(feature = "pretrained")]
use crate::models::yolo11::{
    Yolo11ClsLConfig, Yolo11ClsMConfig, Yolo11ClsNConfig, Yolo11ClsSConfig, Yolo11ClsXConfig,
    Yolo11LConfig, Yolo11MConfig, Yolo11NConfig, Yolo11SConfig, Yolo11SegLConfig, Yolo11SegMConfig,
    Yolo11SegNConfig, Yolo11SegSConfig, Yolo11SegXConfig, Yolo11XConfig,
};
#[cfg(feature = "pretrained")]
use crate::models::yolo12::{
    Yolo12LConfig, Yolo12MConfig, Yolo12NConfig, Yolo12SConfig, Yolo12XConfig,
};
use crate::models::yolo26::head::MAX_DETECTIONS as YOLO26_MAX_DETECTIONS;
#[cfg(feature = "pretrained")]
use crate::models::yolo26::{
    Yolo26ClsLConfig, Yolo26ClsMConfig, Yolo26ClsNConfig, Yolo26ClsSConfig, Yolo26ClsXConfig,
    Yolo26LConfig, Yolo26MConfig, Yolo26NConfig, Yolo26SConfig, Yolo26SegLConfig, Yolo26SegMConfig,
    Yolo26SegNConfig, Yolo26SegSConfig, Yolo26SegXConfig, Yolo26XConfig,
};
use crate::models::yolov3_tiny::Yolov3Tiny;
#[cfg(feature = "pretrained")]
use crate::models::yolov3_tiny::Yolov3TinyConfig;
#[cfg(feature = "pretrained")]
use crate::models::yolov8::{
    Yolov8ClsLConfig, Yolov8ClsMConfig, Yolov8ClsNConfig, Yolov8ClsSConfig, Yolov8ClsXConfig,
    Yolov8LConfig, Yolov8MConfig, Yolov8NConfig, Yolov8SConfig, Yolov8SegLConfig, Yolov8SegMConfig,
    Yolov8SegNConfig, Yolov8SegSConfig, Yolov8SegXConfig, Yolov8XConfig,
};
use crate::models::yolov10::head::MAX_DETECTIONS as YOLOV10_MAX_DETECTIONS;
#[cfg(feature = "pretrained")]
use crate::models::yolov10::{
    Yolov10BConfig, Yolov10LConfig, Yolov10MConfig, Yolov10NConfig, Yolov10SConfig, Yolov10XConfig,
};
use crate::models::yolox::Yolox;
use crate::postprocess::{BoundingBox, nms};
use burn::tensor::{Device, ElementConversion, Tensor, TensorData, backend::Backend};
#[cfg(feature = "pretrained")]
use burn_flex::Flex;
use image::{DynamicImage, GrayImage, ImageBuffer, Rgb, RgbImage, RgbaImage};
use serde::{Deserialize, Serialize};
#[cfg(feature = "pretrained")]
use sha2::{Digest, Sha256};

/// The square input size used by the currently supported pretrained models.
pub const INPUT_SIZE: usize = 640;

/// The square input size used by the YOLO26-cls classification models (Ultralytics' classify
/// default; the official checkpoints were trained at 224 px).
pub const CLASSIFY_INPUT_SIZE: usize = 224;

/// Number of ranked classes returned by [`Predictor::predict_classification`] (Ultralytics'
/// `probs.top5` convention).
pub const CLASSIFICATION_TOP_K: usize = 5;

/// Stable identifier for a model architecture/scale in the Montgomery catalog.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelId {
    #[default]
    YoloxNano,
    YoloxTiny,
    YoloxS,
    YoloxM,
    YoloxL,
    YoloxX,
    Yolov3TinyU,
    Yolov10N,
    Yolov10S,
    Yolov10M,
    Yolov10B,
    Yolov10L,
    Yolov10X,
    Yolo11N,
    Yolo11S,
    Yolo11M,
    Yolo11L,
    Yolo11X,
    Yolo11NSeg,
    Yolo11SSeg,
    Yolo11MSeg,
    Yolo11LSeg,
    Yolo11XSeg,
    Yolo11NCls,
    Yolo11SCls,
    Yolo11MCls,
    Yolo11LCls,
    Yolo11XCls,
    Yolov8N,
    Yolov8S,
    Yolov8M,
    Yolov8L,
    Yolov8X,
    Yolov8NSeg,
    Yolov8SSeg,
    Yolov8MSeg,
    Yolov8LSeg,
    Yolov8XSeg,
    Yolov8NCls,
    Yolov8SCls,
    Yolov8MCls,
    Yolov8LCls,
    Yolov8XCls,
    Yolo12N,
    Yolo12S,
    Yolo12M,
    Yolo12L,
    Yolo12X,
    Yolo26N,
    Yolo26S,
    Yolo26M,
    Yolo26L,
    Yolo26X,
    Yolo26NSeg,
    Yolo26SSeg,
    Yolo26MSeg,
    Yolo26LSeg,
    Yolo26XSeg,
    Yolo26NCls,
    Yolo26SCls,
    Yolo26MCls,
    Yolo26LCls,
    Yolo26XCls,
}

impl ModelId {
    /// Exhaustive catalog used by registries whose coverage must track every public model.
    pub const ALL: [Self; 63] = [
        Self::YoloxNano,
        Self::YoloxTiny,
        Self::YoloxS,
        Self::YoloxM,
        Self::YoloxL,
        Self::YoloxX,
        Self::Yolov3TinyU,
        Self::Yolov10N,
        Self::Yolov10S,
        Self::Yolov10M,
        Self::Yolov10B,
        Self::Yolov10L,
        Self::Yolov10X,
        Self::Yolo11N,
        Self::Yolo11S,
        Self::Yolo11M,
        Self::Yolo11L,
        Self::Yolo11X,
        Self::Yolo11NSeg,
        Self::Yolo11SSeg,
        Self::Yolo11MSeg,
        Self::Yolo11LSeg,
        Self::Yolo11XSeg,
        Self::Yolo11NCls,
        Self::Yolo11SCls,
        Self::Yolo11MCls,
        Self::Yolo11LCls,
        Self::Yolo11XCls,
        Self::Yolov8N,
        Self::Yolov8S,
        Self::Yolov8M,
        Self::Yolov8L,
        Self::Yolov8X,
        Self::Yolov8NSeg,
        Self::Yolov8SSeg,
        Self::Yolov8MSeg,
        Self::Yolov8LSeg,
        Self::Yolov8XSeg,
        Self::Yolov8NCls,
        Self::Yolov8SCls,
        Self::Yolov8MCls,
        Self::Yolov8LCls,
        Self::Yolov8XCls,
        Self::Yolo12N,
        Self::Yolo12S,
        Self::Yolo12M,
        Self::Yolo12L,
        Self::Yolo12X,
        Self::Yolo26N,
        Self::Yolo26S,
        Self::Yolo26M,
        Self::Yolo26L,
        Self::Yolo26X,
        Self::Yolo26NSeg,
        Self::Yolo26SSeg,
        Self::Yolo26MSeg,
        Self::Yolo26LSeg,
        Self::Yolo26XSeg,
        Self::Yolo26NCls,
        Self::Yolo26SCls,
        Self::Yolo26MCls,
        Self::Yolo26LCls,
        Self::Yolo26XCls,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::YoloxNano => "yolox-nano",
            Self::YoloxTiny => "yolox-tiny",
            Self::YoloxS => "yolox-s",
            Self::YoloxM => "yolox-m",
            Self::YoloxL => "yolox-l",
            Self::YoloxX => "yolox-x",
            Self::Yolov3TinyU => "yolov3-tinyu",
            Self::Yolov10N => "yolov10n",
            Self::Yolov10S => "yolov10s",
            Self::Yolov10M => "yolov10m",
            Self::Yolov10B => "yolov10b",
            Self::Yolov10L => "yolov10l",
            Self::Yolov10X => "yolov10x",
            Self::Yolo11N => "yolo11n",
            Self::Yolo11S => "yolo11s",
            Self::Yolo11M => "yolo11m",
            Self::Yolo11L => "yolo11l",
            Self::Yolo11X => "yolo11x",
            Self::Yolo11NSeg => "yolo11n-seg",
            Self::Yolo11SSeg => "yolo11s-seg",
            Self::Yolo11MSeg => "yolo11m-seg",
            Self::Yolo11LSeg => "yolo11l-seg",
            Self::Yolo11XSeg => "yolo11x-seg",
            Self::Yolo11NCls => "yolo11n-cls",
            Self::Yolo11SCls => "yolo11s-cls",
            Self::Yolo11MCls => "yolo11m-cls",
            Self::Yolo11LCls => "yolo11l-cls",
            Self::Yolo11XCls => "yolo11x-cls",
            Self::Yolov8N => "yolov8n",
            Self::Yolov8S => "yolov8s",
            Self::Yolov8M => "yolov8m",
            Self::Yolov8L => "yolov8l",
            Self::Yolov8X => "yolov8x",
            Self::Yolov8NSeg => "yolov8n-seg",
            Self::Yolov8SSeg => "yolov8s-seg",
            Self::Yolov8MSeg => "yolov8m-seg",
            Self::Yolov8LSeg => "yolov8l-seg",
            Self::Yolov8XSeg => "yolov8x-seg",
            Self::Yolov8NCls => "yolov8n-cls",
            Self::Yolov8SCls => "yolov8s-cls",
            Self::Yolov8MCls => "yolov8m-cls",
            Self::Yolov8LCls => "yolov8l-cls",
            Self::Yolov8XCls => "yolov8x-cls",
            Self::Yolo12N => "yolo12n",
            Self::Yolo12S => "yolo12s",
            Self::Yolo12M => "yolo12m",
            Self::Yolo12L => "yolo12l",
            Self::Yolo12X => "yolo12x",
            Self::Yolo26N => "yolo26n",
            Self::Yolo26S => "yolo26s",
            Self::Yolo26M => "yolo26m",
            Self::Yolo26L => "yolo26l",
            Self::Yolo26X => "yolo26x",
            Self::Yolo26NSeg => "yolo26n-seg",
            Self::Yolo26SSeg => "yolo26s-seg",
            Self::Yolo26MSeg => "yolo26m-seg",
            Self::Yolo26LSeg => "yolo26l-seg",
            Self::Yolo26XSeg => "yolo26x-seg",
            Self::Yolo26NCls => "yolo26n-cls",
            Self::Yolo26SCls => "yolo26s-cls",
            Self::Yolo26MCls => "yolo26m-cls",
            Self::Yolo26LCls => "yolo26l-cls",
            Self::Yolo26XCls => "yolo26x-cls",
        }
    }

    /// Conventional filename for this model's self-describing native artifact.
    pub fn artifact_filename(self) -> String {
        format!("{}.bpk", self.as_str())
    }

    /// Read the architecture identifier embedded in a Montgomery Burnpack artifact.
    #[cfg(feature = "pretrained")]
    pub fn from_burnpack(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        require_burnpack_path(path)?;
        artifact_metadata(path)?
            .model_id
            .ok_or_else(|| "Burnpack artifact is missing montgomery.model metadata".into())
    }

    /// Default square input side for this catalog model.
    pub const fn default_input_size(self) -> usize {
        match self {
            Self::YoloxNano | Self::YoloxTiny => 416,
            Self::Yolo11NCls
            | Self::Yolo11SCls
            | Self::Yolo11MCls
            | Self::Yolo11LCls
            | Self::Yolo11XCls
            | Self::Yolov8NCls
            | Self::Yolov8SCls
            | Self::Yolov8MCls
            | Self::Yolov8LCls
            | Self::Yolov8XCls
            | Self::Yolo26NCls
            | Self::Yolo26SCls
            | Self::Yolo26MCls
            | Self::Yolo26LCls
            | Self::Yolo26XCls => CLASSIFY_INPUT_SIZE,
            _ => INPUT_SIZE,
        }
    }

    /// The prediction task encoded by this architecture.
    pub const fn task(self) -> ModelTask {
        match self {
            Self::Yolo11NSeg
            | Self::Yolo11SSeg
            | Self::Yolo11MSeg
            | Self::Yolo11LSeg
            | Self::Yolo11XSeg
            | Self::Yolov8NSeg
            | Self::Yolov8SSeg
            | Self::Yolov8MSeg
            | Self::Yolov8LSeg
            | Self::Yolov8XSeg
            | Self::Yolo26NSeg
            | Self::Yolo26SSeg
            | Self::Yolo26MSeg
            | Self::Yolo26LSeg
            | Self::Yolo26XSeg => ModelTask::Segmentation,
            Self::Yolo11NCls
            | Self::Yolo11SCls
            | Self::Yolo11MCls
            | Self::Yolo11LCls
            | Self::Yolo11XCls
            | Self::Yolov8NCls
            | Self::Yolov8SCls
            | Self::Yolov8MCls
            | Self::Yolov8LCls
            | Self::Yolov8XCls
            | Self::Yolo26NCls
            | Self::Yolo26SCls
            | Self::Yolo26MCls
            | Self::Yolo26LCls
            | Self::Yolo26XCls => ModelTask::Classification,
            _ => ModelTask::Detection,
        }
    }
}

/// The kind of prediction produced by a model artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelTask {
    Detection,
    Segmentation,
    Classification,
}

impl fmt::Display for ModelId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for ModelId {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "yolox-nano" | "yolox_nano" | "nano" => Ok(Self::YoloxNano),
            "yolox-tiny" | "yolox_tiny" | "tiny" => Ok(Self::YoloxTiny),
            "yolox-s" | "yolox_s" | "s" => Ok(Self::YoloxS),
            "yolox-m" | "yolox_m" | "m" => Ok(Self::YoloxM),
            "yolox-l" | "yolox_l" | "l" => Ok(Self::YoloxL),
            "yolox-x" | "yolox_x" | "x" => Ok(Self::YoloxX),
            "yolov3-tinyu" | "yolov3_tinyu" => Ok(Self::Yolov3TinyU),
            "yolov10n" | "yolov10-nano" => Ok(Self::Yolov10N),
            "yolov10s" | "yolov10-small" => Ok(Self::Yolov10S),
            "yolov10m" | "yolov10-medium" => Ok(Self::Yolov10M),
            "yolov10b" | "yolov10-balanced" => Ok(Self::Yolov10B),
            "yolov10l" | "yolov10-large" => Ok(Self::Yolov10L),
            "yolov10x" | "yolov10-xlarge" => Ok(Self::Yolov10X),
            "yolo11n" | "yolo11-nano" => Ok(Self::Yolo11N),
            "yolo11s" | "yolo11-small" => Ok(Self::Yolo11S),
            "yolo11m" | "yolo11-medium" => Ok(Self::Yolo11M),
            "yolo11l" | "yolo11-large" => Ok(Self::Yolo11L),
            "yolo11x" | "yolo11-xlarge" => Ok(Self::Yolo11X),
            "yolo11n-seg" | "yolo11n_seg" => Ok(Self::Yolo11NSeg),
            "yolo11s-seg" | "yolo11s_seg" => Ok(Self::Yolo11SSeg),
            "yolo11m-seg" | "yolo11m_seg" => Ok(Self::Yolo11MSeg),
            "yolo11l-seg" | "yolo11l_seg" => Ok(Self::Yolo11LSeg),
            "yolo11x-seg" | "yolo11x_seg" => Ok(Self::Yolo11XSeg),
            "yolo11n-cls" | "yolo11n_cls" => Ok(Self::Yolo11NCls),
            "yolo11s-cls" | "yolo11s_cls" => Ok(Self::Yolo11SCls),
            "yolo11m-cls" | "yolo11m_cls" => Ok(Self::Yolo11MCls),
            "yolo11l-cls" | "yolo11l_cls" => Ok(Self::Yolo11LCls),
            "yolo11x-cls" | "yolo11x_cls" => Ok(Self::Yolo11XCls),
            "yolov8n" | "yolov8-nano" => Ok(Self::Yolov8N),
            "yolov8s" | "yolov8-small" => Ok(Self::Yolov8S),
            "yolov8m" | "yolov8-medium" => Ok(Self::Yolov8M),
            "yolov8l" | "yolov8-large" => Ok(Self::Yolov8L),
            "yolov8x" | "yolov8-xlarge" => Ok(Self::Yolov8X),
            "yolov8n-seg" | "yolov8n_seg" => Ok(Self::Yolov8NSeg),
            "yolov8s-seg" | "yolov8s_seg" => Ok(Self::Yolov8SSeg),
            "yolov8m-seg" | "yolov8m_seg" => Ok(Self::Yolov8MSeg),
            "yolov8l-seg" | "yolov8l_seg" => Ok(Self::Yolov8LSeg),
            "yolov8x-seg" | "yolov8x_seg" => Ok(Self::Yolov8XSeg),
            "yolov8n-cls" | "yolov8n_cls" => Ok(Self::Yolov8NCls),
            "yolov8s-cls" | "yolov8s_cls" => Ok(Self::Yolov8SCls),
            "yolov8m-cls" | "yolov8m_cls" => Ok(Self::Yolov8MCls),
            "yolov8l-cls" | "yolov8l_cls" => Ok(Self::Yolov8LCls),
            "yolov8x-cls" | "yolov8x_cls" => Ok(Self::Yolov8XCls),
            "yolo12n" | "yolo12-nano" => Ok(Self::Yolo12N),
            "yolo12s" | "yolo12-small" => Ok(Self::Yolo12S),
            "yolo12m" | "yolo12-medium" => Ok(Self::Yolo12M),
            "yolo12l" | "yolo12-large" => Ok(Self::Yolo12L),
            "yolo12x" | "yolo12-xlarge" => Ok(Self::Yolo12X),
            "yolo26n" | "yolo26-nano" => Ok(Self::Yolo26N),
            "yolo26s" | "yolo26-small" => Ok(Self::Yolo26S),
            "yolo26m" | "yolo26-medium" => Ok(Self::Yolo26M),
            "yolo26l" | "yolo26-large" => Ok(Self::Yolo26L),
            "yolo26x" | "yolo26-xlarge" => Ok(Self::Yolo26X),
            "yolo26n-seg" | "yolo26n_seg" => Ok(Self::Yolo26NSeg),
            "yolo26s-seg" | "yolo26s_seg" => Ok(Self::Yolo26SSeg),
            "yolo26m-seg" | "yolo26m_seg" => Ok(Self::Yolo26MSeg),
            "yolo26l-seg" | "yolo26l_seg" => Ok(Self::Yolo26LSeg),
            "yolo26x-seg" | "yolo26x_seg" => Ok(Self::Yolo26XSeg),
            "yolo26n-cls" | "yolo26n_cls" => Ok(Self::Yolo26NCls),
            "yolo26s-cls" | "yolo26s_cls" => Ok(Self::Yolo26SCls),
            "yolo26m-cls" | "yolo26m_cls" => Ok(Self::Yolo26MCls),
            "yolo26l-cls" | "yolo26l_cls" => Ok(Self::Yolo26LCls),
            "yolo26x-cls" | "yolo26x_cls" => Ok(Self::Yolo26XCls),
            _ => Err(format!(
                "unknown model '{value}'; available models: yolox-nano/tiny/s/m/l/x, \
                 yolov3-tinyu, yolov10n/s/m/b/l/x, yolo11n/s/m/l/x, yolo11n/s/m/l/x-seg, \
                 yolo11n/s/m/l/x-cls, yolov8n/s/m/l/x, yolov8n/s/m/l/x-seg, yolov8n/s/m/l/x-cls, \
                 yolo12n/s/m/l/x, yolo26n/s/m/l/x, yolo26n/s/m/l/x-seg, yolo26n/s/m/l/x-cls"
            )),
        }
    }
}

/// A detected object in the original source-image coordinate space.
///
/// Bounding boxes use unnormalized floating-point `XYXY` pixel coordinates: `(xmin, ymin)` is the
/// top-left corner and `(xmax, ymax)` is the bottom-right corner. The origin is the source image's
/// top-left image corner. Values are continuous box-edge positions clipped to `[0, width]` and
/// `[0, height]`; consequently `xmax == width` or `ymax == height` is valid.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Detection {
    pub class_id: usize,
    pub class_name: String,
    pub confidence: f32,
    /// Left edge in source-image pixels.
    pub xmin: f32,
    /// Top edge in source-image pixels.
    pub ymin: f32,
    /// Right edge in source-image pixels.
    pub xmax: f32,
    /// Bottom edge in source-image pixels.
    pub ymax: f32,
}

/// An instance mask for one detected object, in the original source-image coordinate space.
///
/// `data` is a boolean coverage bitmask over the full source image, row-major with `width`
/// elements per row: `data[y * width + x]` is `true` when source-image pixel `(x, y)` is covered
/// by the object's instance mask. A `Vec<bool>` stores one byte per element, so one mask costs
/// `width * height` bytes (about 0.75 MB per instance for a 1024x768 image); masks exist only for
/// the models and predictor methods that produce them.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct InstanceMask {
    /// Mask (and source image) width in pixels.
    pub width: u32,
    /// Mask (and source image) height in pixels.
    pub height: u32,
    /// Row-major boolean coverage over the source image.
    pub data: Vec<bool>,
}

/// A detected object with its instance segmentation mask, in the original source-image
/// coordinate space.
///
/// The bounding box follows the same contract as [`Detection`]: unnormalized floating-point
/// `XYXY` pixel edges clipped to the source image. The `mask` is boolean coverage over the full
/// source image (see [`InstanceMask`]); pixels are the model's instance-mask prediction binarized
/// with Ultralytics' semantics and mapped through the letterbox geometry exactly like the box
/// edges.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SegmentationDetection {
    pub class_id: usize,
    pub class_name: String,
    pub confidence: f32,
    /// Left edge in source-image pixels.
    pub xmin: f32,
    /// Top edge in source-image pixels.
    pub ymin: f32,
    /// Right edge in source-image pixels.
    pub xmax: f32,
    /// Bottom edge in source-image pixels.
    pub ymax: f32,
    /// Boolean instance coverage over the source image.
    pub mask: InstanceMask,
}

/// One image-classification result for the classification models.
///
/// `confidence` is the softmax probability of `class_id` in the model's class table (ImageNet-1k
/// for the YOLO26-cls family). The predictor returns the strongest classes in descending order.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Classification {
    pub class_id: usize,
    pub class_name: String,
    pub confidence: f32,
}

/// Results from [`Predictor::inference`], selected automatically from the loaded architecture.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Prediction {
    Detections(Vec<Detection>),
    Segmentations(Vec<SegmentationDetection>),
    Classifications(Vec<Classification>),
}

impl Prediction {
    pub fn detections(&self) -> Option<&[Detection]> {
        match self {
            Self::Detections(items) => Some(items),
            _ => None,
        }
    }

    pub fn segmentations(&self) -> Option<&[SegmentationDetection]> {
        match self {
            Self::Segmentations(items) => Some(items),
            _ => None,
        }
    }

    pub fn classifications(&self) -> Option<&[Classification]> {
        match self {
            Self::Classifications(items) => Some(items),
            _ => None,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Detections(items) => items.len(),
            Self::Segmentations(items) => items.len(),
            Self::Classifications(items) => items.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// An image accepted by [`Predictor::inference`].
///
/// Paths and encoded bytes are decoded once. Borrowing a [`DynamicImage`] avoids copying the
/// source image; passing an owned image buffer transfers its allocation without copying.
pub enum ImageSource<'a> {
    Path(Cow<'a, Path>),
    Image(Cow<'a, DynamicImage>),
    Encoded(Cow<'a, [u8]>),
}

impl<'a> ImageSource<'a> {
    fn decode(self) -> Result<Cow<'a, DynamicImage>> {
        match self {
            Self::Path(path) => Ok(Cow::Owned(image::open(path.as_ref())?)),
            Self::Image(image) => Ok(image),
            Self::Encoded(bytes) => Ok(Cow::Owned(image::load_from_memory(bytes.as_ref())?)),
        }
    }
}

impl<'a> From<&'a DynamicImage> for ImageSource<'a> {
    fn from(image: &'a DynamicImage) -> Self {
        Self::Image(Cow::Borrowed(image))
    }
}

impl From<DynamicImage> for ImageSource<'static> {
    fn from(image: DynamicImage) -> Self {
        Self::Image(Cow::Owned(image))
    }
}

macro_rules! owned_image_source {
    ($image:ty, $variant:ident) => {
        impl From<$image> for ImageSource<'static> {
            fn from(image: $image) -> Self {
                Self::Image(Cow::Owned(DynamicImage::$variant(image)))
            }
        }

        impl<'a> From<&'a $image> for ImageSource<'static> {
            fn from(image: &'a $image) -> Self {
                Self::Image(Cow::Owned(DynamicImage::$variant(image.clone())))
            }
        }
    };
}

owned_image_source!(RgbImage, ImageRgb8);
owned_image_source!(RgbaImage, ImageRgba8);
owned_image_source!(GrayImage, ImageLuma8);

impl<'a> From<&'a Path> for ImageSource<'a> {
    fn from(path: &'a Path) -> Self {
        Self::Path(Cow::Borrowed(path))
    }
}

impl From<PathBuf> for ImageSource<'static> {
    fn from(path: PathBuf) -> Self {
        Self::Path(Cow::Owned(path))
    }
}

impl<'a> From<&'a PathBuf> for ImageSource<'a> {
    fn from(path: &'a PathBuf) -> Self {
        Self::Path(Cow::Borrowed(path.as_path()))
    }
}

impl<'a> From<&'a str> for ImageSource<'a> {
    fn from(path: &'a str) -> Self {
        Self::Path(Cow::Borrowed(Path::new(path)))
    }
}

impl From<String> for ImageSource<'static> {
    fn from(path: String) -> Self {
        Self::Path(Cow::Owned(PathBuf::from(path)))
    }
}

impl<'a> From<&'a [u8]> for ImageSource<'a> {
    fn from(bytes: &'a [u8]) -> Self {
        Self::Encoded(Cow::Borrowed(bytes))
    }
}

impl From<Vec<u8>> for ImageSource<'static> {
    fn from(bytes: Vec<u8>) -> Self {
        Self::Encoded(Cow::Owned(bytes))
    }
}

impl<'a> From<&'a Vec<u8>> for ImageSource<'a> {
    fn from(bytes: &'a Vec<u8>) -> Self {
        Self::Encoded(Cow::Borrowed(bytes.as_slice()))
    }
}

/// Thresholds used during class-aware non-maximum suppression.
#[derive(Debug, Clone, Copy)]
pub struct PredictOptions {
    pub confidence: f32,
    pub iou: f32,
}

impl Default for PredictOptions {
    fn default() -> Self {
        Self {
            confidence: 0.25,
            iou: 0.45,
        }
    }
}

impl PredictOptions {
    pub fn validate(self) -> Result<Self> {
        if !(0.0..=1.0).contains(&self.confidence) {
            return Err("confidence threshold must be between 0 and 1".into());
        }
        if !(0.0..=1.0).contains(&self.iou) {
            return Err("IoU threshold must be between 0 and 1".into());
        }
        Ok(self)
    }
}

#[cfg_attr(not(feature = "pretrained"), allow(dead_code))]
enum RuntimeModel<B: Backend> {
    Yolox(Box<Yolox<B>>),
    Yolov3Tiny(Box<Yolov3Tiny<B>>),
    Yolov10N(Box<crate::models::yolov10::Yolov10N<B>>),
    Yolov10S(Box<crate::models::yolov10::Yolov10S<B>>),
    Yolov10M(Box<crate::models::yolov10::Yolov10M<B>>),
    Yolov10B(Box<crate::models::yolov10::Yolov10B<B>>),
    Yolov10L(Box<crate::models::yolov10::Yolov10L<B>>),
    Yolov10X(Box<crate::models::yolov10::Yolov10X<B>>),
    Yolo11N(Box<crate::models::yolo11::Yolo11N<B>>),
    Yolo11S(Box<crate::models::yolo11::Yolo11S<B>>),
    Yolo11M(Box<crate::models::yolo11::Yolo11M<B>>),
    Yolo11L(Box<crate::models::yolo11::Yolo11L<B>>),
    Yolo11X(Box<crate::models::yolo11::Yolo11X<B>>),
    Yolo11SegN(Box<crate::models::yolo11::Yolo11SegN<B>>),
    Yolo11SegS(Box<crate::models::yolo11::Yolo11SegS<B>>),
    Yolo11SegM(Box<crate::models::yolo11::Yolo11SegM<B>>),
    Yolo11SegL(Box<crate::models::yolo11::Yolo11SegL<B>>),
    Yolo11SegX(Box<crate::models::yolo11::Yolo11SegX<B>>),
    Yolo11ClsN(Box<crate::models::yolo11::Yolo11ClsN<B>>),
    Yolo11ClsS(Box<crate::models::yolo11::Yolo11ClsS<B>>),
    Yolo11ClsM(Box<crate::models::yolo11::Yolo11ClsM<B>>),
    Yolo11ClsL(Box<crate::models::yolo11::Yolo11ClsL<B>>),
    Yolo11ClsX(Box<crate::models::yolo11::Yolo11ClsX<B>>),
    Yolov8N(Box<crate::models::yolov8::Yolov8N<B>>),
    Yolov8S(Box<crate::models::yolov8::Yolov8S<B>>),
    Yolov8M(Box<crate::models::yolov8::Yolov8M<B>>),
    Yolov8L(Box<crate::models::yolov8::Yolov8L<B>>),
    Yolov8X(Box<crate::models::yolov8::Yolov8X<B>>),
    Yolov8SegN(Box<crate::models::yolov8::Yolov8SegN<B>>),
    Yolov8SegS(Box<crate::models::yolov8::Yolov8SegS<B>>),
    Yolov8SegM(Box<crate::models::yolov8::Yolov8SegM<B>>),
    Yolov8SegL(Box<crate::models::yolov8::Yolov8SegL<B>>),
    Yolov8SegX(Box<crate::models::yolov8::Yolov8SegX<B>>),
    Yolov8ClsN(Box<crate::models::yolov8::Yolov8ClsN<B>>),
    Yolov8ClsS(Box<crate::models::yolov8::Yolov8ClsS<B>>),
    Yolov8ClsM(Box<crate::models::yolov8::Yolov8ClsM<B>>),
    Yolov8ClsL(Box<crate::models::yolov8::Yolov8ClsL<B>>),
    Yolov8ClsX(Box<crate::models::yolov8::Yolov8ClsX<B>>),
    Yolo12N(Box<crate::models::yolo12::Yolo12N<B>>),
    Yolo12S(Box<crate::models::yolo12::Yolo12S<B>>),
    Yolo12M(Box<crate::models::yolo12::Yolo12M<B>>),
    Yolo12L(Box<crate::models::yolo12::Yolo12L<B>>),
    Yolo12X(Box<crate::models::yolo12::Yolo12X<B>>),
    Yolo26N(Box<crate::models::yolo26::Yolo26N<B>>),
    Yolo26S(Box<crate::models::yolo26::Yolo26S<B>>),
    Yolo26M(Box<crate::models::yolo26::Yolo26M<B>>),
    Yolo26L(Box<crate::models::yolo26::Yolo26L<B>>),
    Yolo26X(Box<crate::models::yolo26::Yolo26X<B>>),
    Yolo26SegN(Box<crate::models::yolo26::Yolo26SegN<B>>),
    Yolo26SegS(Box<crate::models::yolo26::Yolo26SegS<B>>),
    Yolo26SegM(Box<crate::models::yolo26::Yolo26SegM<B>>),
    Yolo26SegL(Box<crate::models::yolo26::Yolo26SegL<B>>),
    Yolo26SegX(Box<crate::models::yolo26::Yolo26SegX<B>>),
    Yolo26ClsN(Box<crate::models::yolo26::Yolo26ClsN<B>>),
    Yolo26ClsS(Box<crate::models::yolo26::Yolo26ClsS<B>>),
    Yolo26ClsM(Box<crate::models::yolo26::Yolo26ClsM<B>>),
    Yolo26ClsL(Box<crate::models::yolo26::Yolo26ClsL<B>>),
    Yolo26ClsX(Box<crate::models::yolo26::Yolo26ClsX<B>>),
}

/// A ready-to-run object detector over a Burn backend.
///
/// The backend is a type parameter: [`Flex`] on CPU, or `Wgpu` for GPU inference (Vulkan/DX12 on
/// Windows and Linux, Metal on macOS) when built with the `gpu` feature. Constructors resolve the
/// backend's default device; the `_on_device` variants accept an explicit device.
pub struct Predictor<B: Backend> {
    model_id: ModelId,
    model: RuntimeModel<B>,
    device: Device<B>,
    options: PredictOptions,
    class_names: Vec<String>,
    input_size: usize,
}

/// A model loaded on Montgomery's default CPU backend.
///
/// This is the ergonomic entry point for normal inference. It dereferences to [`Predictor<Flex>`],
/// so the task-specific methods remain available without an extra accessor. Use [`Predictor`]
/// directly when selecting a different backend or device.
#[cfg(feature = "pretrained")]
pub struct Model(Predictor<Flex>);

#[cfg(feature = "pretrained")]
impl Model {
    /// Load a native Burnpack artifact, inferring the architecture from its metadata.
    pub fn new(checkpoint: impl Into<PathBuf>) -> Result<Self> {
        Predictor::new(checkpoint).map(Self)
    }

    /// Load an artifact with explicit confidence and IoU thresholds.
    pub fn with_options(checkpoint: impl Into<PathBuf>, options: PredictOptions) -> Result<Self> {
        Predictor::with_options(checkpoint, options).map(Self)
    }

    /// Consume the convenience wrapper and return the underlying CPU predictor.
    pub fn into_predictor(self) -> Predictor<Flex> {
        self.0
    }
}

#[cfg(feature = "pretrained")]
impl std::ops::Deref for Model {
    type Target = Predictor<Flex>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DetectionPreprocess {
    Yolox,
    Ultralytics,
}

impl ModelId {
    const fn detection_preprocess(self) -> DetectionPreprocess {
        match self {
            Self::YoloxNano
            | Self::YoloxTiny
            | Self::YoloxS
            | Self::YoloxM
            | Self::YoloxL
            | Self::YoloxX => DetectionPreprocess::Yolox,
            _ => DetectionPreprocess::Ultralytics,
        }
    }
}

#[cfg(feature = "pretrained")]
fn catalog_class_names(model_id: ModelId) -> Vec<String> {
    if model_id.as_str().ends_with("-cls") {
        IMAGENET_CLASSES
            .iter()
            .map(|name| (*name).to_owned())
            .collect()
    } else {
        COCO_CLASSES.iter().map(|name| (*name).to_owned()).collect()
    }
}

#[cfg(feature = "pretrained")]
fn catalog_input_size(model_id: ModelId) -> usize {
    model_id.default_input_size()
}

#[cfg(feature = "pretrained")]
struct ArtifactMetadata {
    model_id: Option<ModelId>,
    class_names: Option<Vec<String>>,
    input_size: Option<usize>,
}

#[cfg(feature = "pretrained")]
fn artifact_metadata(path: &Path) -> Result<ArtifactMetadata> {
    #[derive(Deserialize)]
    struct MetadataEnvelope {
        metadata: std::collections::BTreeMap<String, String>,
    }
    use std::io::Read as _;

    let mut file = std::fs::File::open(path)?;
    let mut header = [0_u8; 10];
    file.read_exact(&mut header)?;
    if &header[0..4] != b"NRUB" {
        return Err("trained artifact is not a Burnpack file".into());
    }
    let metadata_size = u32::from_le_bytes(header[6..10].try_into().unwrap()) as usize;
    if metadata_size > 100 * 1024 * 1024 {
        return Err("trained artifact metadata exceeds the Burnpack safety limit".into());
    }
    let mut metadata_bytes = vec![0_u8; metadata_size];
    file.read_exact(&mut metadata_bytes)?;
    let envelope: MetadataEnvelope = ciborium::de::from_reader(metadata_bytes.as_slice())?;
    let metadata = &envelope.metadata;
    let model_id = metadata
        .get("montgomery.model")
        .map(|encoded| {
            encoded
                .parse::<ModelId>()
                .map_err(|error| format!("invalid montgomery.model metadata {encoded:?}: {error}"))
        })
        .transpose()?;

    let class_names = metadata
        .get("montgomery.class-names-json")
        .map(|encoded| -> Result<Vec<String>> {
            let names: Vec<String> = serde_json::from_str(encoded)?;
            if names.is_empty()
                || names.iter().any(|name| name.trim().is_empty())
                || names
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    != names.len()
            {
                return Err("artifact class names must be non-empty and unique".into());
            }
            Ok(names)
        })
        .transpose()?;
    let input_size = metadata
        .get("montgomery.input-size-json")
        .map(|encoded| -> Result<usize> {
            let size: [usize; 2] = serde_json::from_str(encoded)?;
            if size[0] == 0 || size[0] != size[1] {
                return Err("artifact currently requires a positive square input size".into());
            }
            Ok(size[0])
        })
        .transpose()?;

    Ok(ArtifactMetadata {
        model_id,
        class_names,
        input_size,
    })
}

#[cfg(feature = "pretrained")]
fn require_burnpack_path(path: &Path) -> Result<()> {
    if path.extension().and_then(|value| value.to_str()) != Some("bpk") {
        return Err(
            "Model requires a native .bpk artifact; convert upstream checkpoints with \
             the pack-weights CLI"
                .into(),
        );
    }
    Ok(())
}

pub type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

/// Construct the requested model inside a large-stack worker and load either its native Burnpack
/// artifact or an upstream tensor state accepted by that family.
///
/// Deep module construction overflows the small default main-thread stack on Windows in debug
/// builds, so the graph is built in the worker thread.
/// A model construction-and-load closure handed to the large-stack loader worker.
#[cfg(feature = "pretrained")]
type ModelLoader<B> = Box<dyn FnOnce(&Device<B>) -> Result<RuntimeModel<B>> + Send>;

#[cfg(feature = "pretrained")]
fn load_model_checkpoint<B: Backend>(
    model_id: ModelId,
    checkpoint: PathBuf,
    device: Device<B>,
    num_classes: usize,
) -> Result<RuntimeModel<B>> {
    use burn_store::ModuleSnapshot as _;

    macro_rules! load_variant {
        ($config:ty, $variant:path) => {
            move |device: &Device<B>| -> Result<RuntimeModel<B>> {
                let mut model = <$config>::default().init_with_classes::<B>(num_classes, device);
                let mut store = burn_store::BurnpackStore::from_file(&checkpoint)
                    .with_from_adapter(burn_store::HalfPrecisionAdapter::new())
                    .allow_partial(cfg!(feature = "training"))
                    .zero_copy(true);
                let result = model.load_from(&mut store)?;
                if result.missing.iter().any(|(path, _)| {
                    !path.contains(".o2m_") && !path.starts_with("head.proto.sem_")
                }) {
                    return Err(format!(
                        "inference artifact is missing non-training tensors:\n{result}"
                    )
                    .into());
                }
                Ok($variant(Box::new(model)))
            }
        };
    }
    let loader: ModelLoader<B> = match model_id {
        ModelId::YoloxNano
        | ModelId::YoloxTiny
        | ModelId::YoloxS
        | ModelId::YoloxM
        | ModelId::YoloxL
        | ModelId::YoloxX => Box::new(move |device: &Device<B>| {
            let constructor = yolox_constructor::<B>(model_id);
            let mut model = constructor(num_classes, device);
            model.load_burnpack_weights(checkpoint)?;
            Ok(RuntimeModel::Yolox(Box::new(model)))
        }),
        ModelId::Yolov3TinyU => Box::new(load_variant!(Yolov3TinyConfig, RuntimeModel::Yolov3Tiny)),
        ModelId::Yolov10N => Box::new(load_variant!(Yolov10NConfig, RuntimeModel::Yolov10N)),
        ModelId::Yolov10S => Box::new(load_variant!(Yolov10SConfig, RuntimeModel::Yolov10S)),
        ModelId::Yolov10M => Box::new(load_variant!(Yolov10MConfig, RuntimeModel::Yolov10M)),
        ModelId::Yolov10B => Box::new(load_variant!(Yolov10BConfig, RuntimeModel::Yolov10B)),
        ModelId::Yolov10L => Box::new(load_variant!(Yolov10LConfig, RuntimeModel::Yolov10L)),
        ModelId::Yolov10X => Box::new(load_variant!(Yolov10XConfig, RuntimeModel::Yolov10X)),
        ModelId::Yolo11N => Box::new(load_variant!(Yolo11NConfig, RuntimeModel::Yolo11N)),
        ModelId::Yolo11S => Box::new(load_variant!(Yolo11SConfig, RuntimeModel::Yolo11S)),
        ModelId::Yolo11M => Box::new(load_variant!(Yolo11MConfig, RuntimeModel::Yolo11M)),
        ModelId::Yolo11L => Box::new(load_variant!(Yolo11LConfig, RuntimeModel::Yolo11L)),
        ModelId::Yolo11X => Box::new(load_variant!(Yolo11XConfig, RuntimeModel::Yolo11X)),
        ModelId::Yolo11NSeg => Box::new(load_variant!(Yolo11SegNConfig, RuntimeModel::Yolo11SegN)),
        ModelId::Yolo11SSeg => Box::new(load_variant!(Yolo11SegSConfig, RuntimeModel::Yolo11SegS)),
        ModelId::Yolo11MSeg => Box::new(load_variant!(Yolo11SegMConfig, RuntimeModel::Yolo11SegM)),
        ModelId::Yolo11LSeg => Box::new(load_variant!(Yolo11SegLConfig, RuntimeModel::Yolo11SegL)),
        ModelId::Yolo11XSeg => Box::new(load_variant!(Yolo11SegXConfig, RuntimeModel::Yolo11SegX)),
        ModelId::Yolo11NCls => Box::new(load_variant!(Yolo11ClsNConfig, RuntimeModel::Yolo11ClsN)),
        ModelId::Yolo11SCls => Box::new(load_variant!(Yolo11ClsSConfig, RuntimeModel::Yolo11ClsS)),
        ModelId::Yolo11MCls => Box::new(load_variant!(Yolo11ClsMConfig, RuntimeModel::Yolo11ClsM)),
        ModelId::Yolo11LCls => Box::new(load_variant!(Yolo11ClsLConfig, RuntimeModel::Yolo11ClsL)),
        ModelId::Yolo11XCls => Box::new(load_variant!(Yolo11ClsXConfig, RuntimeModel::Yolo11ClsX)),
        ModelId::Yolov8N => Box::new(load_variant!(Yolov8NConfig, RuntimeModel::Yolov8N)),
        ModelId::Yolov8S => Box::new(load_variant!(Yolov8SConfig, RuntimeModel::Yolov8S)),
        ModelId::Yolov8M => Box::new(load_variant!(Yolov8MConfig, RuntimeModel::Yolov8M)),
        ModelId::Yolov8L => Box::new(load_variant!(Yolov8LConfig, RuntimeModel::Yolov8L)),
        ModelId::Yolov8X => Box::new(load_variant!(Yolov8XConfig, RuntimeModel::Yolov8X)),
        ModelId::Yolov8NSeg => Box::new(load_variant!(Yolov8SegNConfig, RuntimeModel::Yolov8SegN)),
        ModelId::Yolov8SSeg => Box::new(load_variant!(Yolov8SegSConfig, RuntimeModel::Yolov8SegS)),
        ModelId::Yolov8MSeg => Box::new(load_variant!(Yolov8SegMConfig, RuntimeModel::Yolov8SegM)),
        ModelId::Yolov8LSeg => Box::new(load_variant!(Yolov8SegLConfig, RuntimeModel::Yolov8SegL)),
        ModelId::Yolov8XSeg => Box::new(load_variant!(Yolov8SegXConfig, RuntimeModel::Yolov8SegX)),
        ModelId::Yolov8NCls => Box::new(load_variant!(Yolov8ClsNConfig, RuntimeModel::Yolov8ClsN)),
        ModelId::Yolov8SCls => Box::new(load_variant!(Yolov8ClsSConfig, RuntimeModel::Yolov8ClsS)),
        ModelId::Yolov8MCls => Box::new(load_variant!(Yolov8ClsMConfig, RuntimeModel::Yolov8ClsM)),
        ModelId::Yolov8LCls => Box::new(load_variant!(Yolov8ClsLConfig, RuntimeModel::Yolov8ClsL)),
        ModelId::Yolov8XCls => Box::new(load_variant!(Yolov8ClsXConfig, RuntimeModel::Yolov8ClsX)),
        ModelId::Yolo12N => Box::new(load_variant!(Yolo12NConfig, RuntimeModel::Yolo12N)),
        ModelId::Yolo12S => Box::new(load_variant!(Yolo12SConfig, RuntimeModel::Yolo12S)),
        ModelId::Yolo12M => Box::new(load_variant!(Yolo12MConfig, RuntimeModel::Yolo12M)),
        ModelId::Yolo12L => Box::new(load_variant!(Yolo12LConfig, RuntimeModel::Yolo12L)),
        ModelId::Yolo12X => Box::new(load_variant!(Yolo12XConfig, RuntimeModel::Yolo12X)),
        ModelId::Yolo26N => Box::new(load_variant!(Yolo26NConfig, RuntimeModel::Yolo26N)),
        ModelId::Yolo26S => Box::new(load_variant!(Yolo26SConfig, RuntimeModel::Yolo26S)),
        ModelId::Yolo26M => Box::new(load_variant!(Yolo26MConfig, RuntimeModel::Yolo26M)),
        ModelId::Yolo26L => Box::new(load_variant!(Yolo26LConfig, RuntimeModel::Yolo26L)),
        ModelId::Yolo26X => Box::new(load_variant!(Yolo26XConfig, RuntimeModel::Yolo26X)),
        ModelId::Yolo26NSeg => Box::new(load_variant!(Yolo26SegNConfig, RuntimeModel::Yolo26SegN)),
        ModelId::Yolo26SSeg => Box::new(load_variant!(Yolo26SegSConfig, RuntimeModel::Yolo26SegS)),
        ModelId::Yolo26MSeg => Box::new(load_variant!(Yolo26SegMConfig, RuntimeModel::Yolo26SegM)),
        ModelId::Yolo26LSeg => Box::new(load_variant!(Yolo26SegLConfig, RuntimeModel::Yolo26SegL)),
        ModelId::Yolo26XSeg => Box::new(load_variant!(Yolo26SegXConfig, RuntimeModel::Yolo26SegX)),
        ModelId::Yolo26NCls => Box::new(load_variant!(Yolo26ClsNConfig, RuntimeModel::Yolo26ClsN)),
        ModelId::Yolo26SCls => Box::new(load_variant!(Yolo26ClsSConfig, RuntimeModel::Yolo26ClsS)),
        ModelId::Yolo26MCls => Box::new(load_variant!(Yolo26ClsMConfig, RuntimeModel::Yolo26ClsM)),
        ModelId::Yolo26LCls => Box::new(load_variant!(Yolo26ClsLConfig, RuntimeModel::Yolo26ClsL)),
        ModelId::Yolo26XCls => Box::new(load_variant!(Yolo26ClsXConfig, RuntimeModel::Yolo26ClsX)),
    };
    let worker = std::thread::Builder::new()
        .name("montgomery-model-loader".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || loader(&device))?;
    worker
        .join()
        .map_err(|_| format!("{model_id} model loader thread panicked"))?
}

/// Resolve the YOLOX graph constructor for a YOLOX scale identifier. Every YOLOX scale shares the
/// same [`Yolox`] graph; the scale is baked into the depth/width parameters passed here.
#[cfg(feature = "pretrained")]
fn yolox_constructor<B: Backend>(model_id: ModelId) -> fn(usize, &Device<B>) -> Yolox<B> {
    match model_id {
        ModelId::YoloxNano => Yolox::<B>::yolox_nano,
        ModelId::YoloxTiny => Yolox::<B>::yolox_tiny,
        ModelId::YoloxS => Yolox::<B>::yolox_s,
        ModelId::YoloxM => Yolox::<B>::yolox_m,
        ModelId::YoloxL => Yolox::<B>::yolox_l,
        ModelId::YoloxX => Yolox::<B>::yolox_x,
        _ => unreachable!("non-YOLOX models do not use the YOLOX constructor"),
    }
}

/// Uniform end-to-end detection entry point shared by every YOLOv10/YOLO26 scale variant, so the
/// runtime can dispatch to any of them without naming the concrete scale type.
trait EndToEndDetector<B: Backend> {
    fn detect(&self, input: Tensor<B, 4>) -> (Tensor<B, 3>, Tensor<B, 3>);
}

macro_rules! impl_end_to_end_detector {
    ($family:ident: [$($model:ident),+ $(,)?]) => {
        $(
            impl<B: Backend> EndToEndDetector<B> for crate::models::$family::$model<B> {
                fn detect(&self, input: Tensor<B, 4>) -> (Tensor<B, 3>, Tensor<B, 3>) {
                    let output = self.forward(input);
                    (output.boxes, output.scores)
                }
            }
        )+
    };
}

impl_end_to_end_detector!(yolov10: [Yolov10N, Yolov10S, Yolov10M, Yolov10B, Yolov10L, Yolov10X]);
impl_end_to_end_detector!(yolo26: [Yolo26N, Yolo26S, Yolo26M, Yolo26L, Yolo26X]);

/// Uniform classification entry point shared by every YOLO26-cls and YOLO11-cls scale variant, so
/// the runtime can dispatch to any of them without naming the concrete scale type.
trait EndToEndClassifier<B: Backend> {
    fn classify(
        &self,
        input: Tensor<B, 4>,
    ) -> crate::models::yolo26::classification::ClassificationOutput<B>;
}

impl<B: Backend, M: EndToEndClassifier<B>> EndToEndClassifier<B> for Box<M> {
    fn classify(
        &self,
        input: Tensor<B, 4>,
    ) -> crate::models::yolo26::classification::ClassificationOutput<B> {
        (**self).classify(input)
    }
}

macro_rules! impl_end_to_end_classifier {
    ($family:ident: [$($model:ident),+ $(,)?]) => {
        $(
            impl<B: Backend> EndToEndClassifier<B> for crate::models::$family::$model<B> {
                fn classify(
                    &self,
                    input: Tensor<B, 4>,
                ) -> crate::models::yolo26::classification::ClassificationOutput<B> {
                    self.forward(input)
                }
            }
        )+
    };
}

impl_end_to_end_classifier!(yolo26: [Yolo26ClsN, Yolo26ClsS, Yolo26ClsM, Yolo26ClsL, Yolo26ClsX]);
impl_end_to_end_classifier!(yolo11: [Yolo11ClsN, Yolo11ClsS, Yolo11ClsM, Yolo11ClsL, Yolo11ClsX]);
impl_end_to_end_classifier!(yolov8: [Yolov8ClsN, Yolov8ClsS, Yolov8ClsM, Yolov8ClsL, Yolov8ClsX]);

/// Uniform classic-detection entry point shared by every NMS-based detector family.
trait ClassicDetector<B: Backend> {
    fn detect(&self, input: Tensor<B, 4>) -> (Tensor<B, 3>, Tensor<B, 3>);
}

impl<B: Backend> ClassicDetector<B> for Yolox<B> {
    fn detect(&self, input: Tensor<B, 4>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let output = self.forward(input);
        let [batch, anchors, channels] = output.dims();
        let boxes = output.clone().slice([0..batch, 0..anchors, 0..4]);
        let objectness = output.clone().slice([0..batch, 0..anchors, 4..5]);
        let class_scores = output.slice([0..batch, 0..anchors, 5..channels]);
        (boxes, class_scores * objectness)
    }
}

impl<B: Backend> ClassicDetector<B> for Yolov3Tiny<B> {
    fn detect(&self, input: Tensor<B, 4>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let output = self.forward(input);
        let [batch, anchors, _] = output.boxes.dims();
        let left_top = output.boxes.clone().slice([0..batch, 0..anchors, 0..2]);
        let right_bottom = output.boxes.slice([0..batch, 0..anchors, 2..4]);
        let center = (left_top.clone() + right_bottom.clone()) / 2.0;
        let size = right_bottom - left_top;
        (Tensor::cat(vec![center, size], 2), output.scores)
    }
}

macro_rules! impl_classic_detector {
    ($family:ident: [$($model:ident),+ $(,)?]) => {
        $(
            impl<B: Backend> ClassicDetector<B> for crate::models::$family::$model<B> {
                fn detect(&self, input: Tensor<B, 4>) -> (Tensor<B, 3>, Tensor<B, 3>) {
                    let output = self.forward(input);
                    (output.boxes, output.scores)
                }
            }
        )+
    };
}

impl_classic_detector!(yolo11: [Yolo11N, Yolo11S, Yolo11M, Yolo11L, Yolo11X]);
impl_classic_detector!(yolov8: [Yolov8N, Yolov8S, Yolov8M, Yolov8L, Yolov8X]);
impl_classic_detector!(yolo12: [Yolo12N, Yolo12S, Yolo12M, Yolo12L, Yolo12X]);

impl<B: Backend, M: ClassicDetector<B>> ClassicDetector<B> for Box<M> {
    fn detect(&self, input: Tensor<B, 4>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        (**self).detect(input)
    }
}

/// Decode and suppress classic (NMS-based) predictions for any scale variant.
///
/// Implementations normalize family-specific head layouts to center-size boxes and per-class
/// detection scores before feeding the generic class-aware NMS helper.
fn run_classic_detections<B: Backend>(
    model: &impl ClassicDetector<B>,
    input: Tensor<B, 4>,
    iou_threshold: f32,
    confidence_threshold: f32,
) -> Vec<Vec<Vec<BoundingBox>>> {
    let (boxes, scores) = model.detect(input);
    nms(boxes, scores, iou_threshold, confidence_threshold)
}

/// Uniform classic instance-segmentation entry point shared by the YOLO11-seg scale variants, so
/// the runtime can dispatch to any of them without naming the concrete scale type.
pub(crate) trait ClassicSegmenter<B: Backend> {
    fn segment(&self, input: Tensor<B, 4>) -> crate::models::yolo11::SegmentOutput<B>;
}

macro_rules! impl_classic_segmenter {
    ($family:ident: [$($model:ident),+ $(,)?]) => {
        $(
            impl<B: Backend> ClassicSegmenter<B> for crate::models::$family::$model<B> {
                fn segment(&self, input: Tensor<B, 4>) -> crate::models::yolo11::SegmentOutput<B> {
                    self.forward(input)
                }
            }
        )+
    };
}

impl_classic_segmenter!(yolo11: [Yolo11SegN, Yolo11SegS, Yolo11SegM, Yolo11SegL, Yolo11SegX]);
impl_classic_segmenter!(yolov8: [Yolov8SegN, Yolov8SegS, Yolov8SegM, Yolov8SegL, Yolov8SegX]);

impl<B: Backend, M: ClassicSegmenter<B>> ClassicSegmenter<B> for Box<M> {
    fn segment(&self, input: Tensor<B, 4>) -> crate::models::yolo11::SegmentOutput<B> {
        (**self).segment(input)
    }
}

/// Uniform end-to-end instance-segmentation entry point shared by the YOLO26-seg scale variants,
/// so the runtime can dispatch to any of them without naming the concrete scale type.
pub(crate) trait EndToEndSegmenter<B: Backend> {
    fn segment(&self, input: Tensor<B, 4>)
    -> crate::models::yolo26::segmentation::SegmentOutput<B>;
}

macro_rules! impl_end_to_end_segmenter {
    ($family:ident: [$($model:ident),+ $(,)?]) => {
        $(
            impl<B: Backend> EndToEndSegmenter<B> for crate::models::$family::$model<B> {
                fn segment(
                    &self,
                    input: Tensor<B, 4>,
                ) -> crate::models::yolo26::segmentation::SegmentOutput<B> {
                    self.forward(input)
                }
            }
        )+
    };
}

impl_end_to_end_segmenter!(yolo26: [Yolo26SegN, Yolo26SegS, Yolo26SegM, Yolo26SegL, Yolo26SegX]);

impl<B: Backend, M: EndToEndSegmenter<B>> EndToEndSegmenter<B> for Box<M> {
    fn segment(
        &self,
        input: Tensor<B, 4>,
    ) -> crate::models::yolo26::segmentation::SegmentOutput<B> {
        (**self).segment(input)
    }
}

/// Decode and select end-to-end (NMS-free) segmentation predictions for any scale variant.
///
/// The YOLO26-seg head output mirrors Ultralytics' end2end postprocess: the top
/// `max_detections` anchors by best-class score are kept, then the top `max_detections`
/// (anchor, class) pairs among them, and finally the confidence filter is applied — no
/// non-maximum suppression. The surviving anchors' raw mask coefficients ride along in
/// [`SegmentationOutputCpu`] for the shared mask assembly.
pub(crate) fn run_end_to_end_segmentations<B: Backend>(
    model: &impl EndToEndSegmenter<B>,
    input: Tensor<B, 4>,
    max_detections: usize,
    confidence_threshold: f32,
) -> SegmentationOutputCpu {
    let output = model.segment(input);
    let [_, proto_channels, proto_height, proto_width] = output.prototypes.dims();
    let [_, _, anchors] = output.coefficients.dims();
    let [batch, anchors_scores, num_classes] = output.decoded.scores.dims();
    assert_eq!(anchors, anchors_scores, "head anchor mismatch");
    assert_eq!(batch, 1, "batch-1 inference only");

    let boxes: Vec<f32> = output
        .decoded
        .boxes
        .into_data()
        .iter::<B::FloatElem>()
        .map(|value| value.elem::<f32>())
        .collect();
    let scores: Vec<f32> = output
        .decoded
        .scores
        .into_data()
        .iter::<B::FloatElem>()
        .map(|value| value.elem::<f32>())
        .collect();
    let coefficients: Vec<f32> = output
        .coefficients
        .into_data()
        .iter::<B::FloatElem>()
        .map(|value| value.elem::<f32>())
        .collect();
    let prototypes: Vec<f32> = output
        .prototypes
        .into_data()
        .iter::<B::FloatElem>()
        .map(|value| value.elem::<f32>())
        .collect();

    // Two-stage top-k exactly like `end2end_topk_detections`, with anchor indices kept so the
    // mask coefficients of every survivor can be gathered.
    let keep = max_detections.min(anchors);
    let mut best_scores = Vec::with_capacity(anchors);
    for anchor in 0..anchors {
        let row = &scores[anchor * num_classes..(anchor + 1) * num_classes];
        let mut best = f32::NEG_INFINITY;
        for &score in row {
            if score > best {
                best = score;
            }
        }
        best_scores.push(best);
    }
    let mut anchor_order: Vec<usize> = (0..anchors)
        .filter(|&anchor| best_scores[anchor] >= confidence_threshold)
        .collect();
    if anchor_order.len() > keep {
        anchor_order
            .select_nth_unstable_by(keep, |&a, &b| best_scores[b].total_cmp(&best_scores[a]));
        anchor_order.truncate(keep);
    }
    anchor_order.sort_unstable_by(|&a, &b| best_scores[b].total_cmp(&best_scores[a]));

    let mut candidates = Vec::with_capacity(anchor_order.len() * num_classes.min(8));
    for (selected_index, &anchor) in anchor_order.iter().enumerate() {
        let base = anchor * num_classes;
        for class in 0..num_classes {
            let score = scores[base + class];
            if score >= confidence_threshold {
                candidates.push((score, selected_index, class));
            }
        }
    }
    if candidates.len() > keep {
        candidates.select_nth_unstable_by(keep, |a, b| b.0.total_cmp(&a.0));
        candidates.truncate(keep);
    }
    candidates.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

    let mut survivors = Vec::with_capacity(candidates.len());
    for (score, selected_index, class) in candidates {
        let anchor = anchor_order[selected_index];
        let bbox = &boxes[anchor * 4..anchor * 4 + 4];
        survivors.push(SegmentationCandidate {
            bbox: BoundingBox {
                xmin: bbox[0],
                ymin: bbox[1],
                xmax: bbox[2],
                ymax: bbox[3],
                confidence: score,
            },
            class_id: class,
            anchor,
        });
    }

    SegmentationOutputCpu {
        candidates: survivors,
        prototypes,
        proto_channels,
        proto_width,
        proto_height,
        coefficients,
        anchors,
    }
}

/// One NMS-surviving segmentation candidate: a box plus the anchor index its 32 mask
/// coefficients live at.
pub(crate) struct SegmentationCandidate {
    pub(crate) bbox: BoundingBox,
    pub(crate) class_id: usize,
    pub(crate) anchor: usize,
}

/// CPU-side result of the segmentation decode: NMS survivors plus everything the mask assembly
/// needs (prototypes and per-anchor coefficients), already synced from the backend.
pub(crate) struct SegmentationOutputCpu {
    pub(crate) candidates: Vec<SegmentationCandidate>,
    /// Prototype masks, row-major `[channels, proto_height, proto_width]`.
    pub(crate) prototypes: Vec<f32>,
    pub(crate) proto_channels: usize,
    pub(crate) proto_width: usize,
    pub(crate) proto_height: usize,
    /// Raw mask coefficients, row-major `[channels, anchors]`.
    pub(crate) coefficients: Vec<f32>,
    pub(crate) anchors: usize,
}

/// Decode and suppress classic (NMS-based) segmentation predictions for any scale variant.
///
/// Mirrors Ultralytics' segmentation inference path: the head's `[boxes, scores, coefficients]`
/// rows are filtered by the best class score, suppressed with class-aware NMS (per-class greedy
/// suppression on center-size boxes converted to XYXY), and the mask coefficients of every
/// surviving anchor are carried along for the mask assembly.
pub(crate) fn run_classic_segmentations<B: Backend>(
    model: &impl ClassicSegmenter<B>,
    input: Tensor<B, 4>,
    iou_threshold: f32,
    confidence_threshold: f32,
) -> SegmentationOutputCpu {
    let output = model.segment(input);
    let [_, proto_channels, proto_height, proto_width] = output.prototypes.dims();
    let [_, _, anchors] = output.coefficients.dims();
    let [_, anchors_scores, num_classes] = output.scores.dims();
    assert_eq!(anchors, anchors_scores, "head anchor mismatch");

    let boxes: Vec<f32> = output
        .boxes
        .into_data()
        .iter::<B::FloatElem>()
        .map(|value| value.elem::<f32>())
        .collect();
    let scores: Vec<f32> = output
        .scores
        .into_data()
        .iter::<B::FloatElem>()
        .map(|value| value.elem::<f32>())
        .collect();
    let coefficients: Vec<f32> = output
        .coefficients
        .into_data()
        .iter::<B::FloatElem>()
        .map(|value| value.elem::<f32>())
        .collect();
    let prototypes: Vec<f32> = output
        .prototypes
        .into_data()
        .iter::<B::FloatElem>()
        .map(|value| value.elem::<f32>())
        .collect();

    // Best class per anchor, thresholded (the `nms` helper's filter semantics).
    let mut by_class: Vec<Vec<(BoundingBox, usize)>> =
        (0..num_classes).map(|_| Vec::new()).collect();
    // Survivors are sparse after thresholding; avoid regrowth without over-allocating.
    let per_class_cap = (anchors / num_classes.max(1)).clamp(4, 256);
    for vec in by_class.iter_mut() {
        vec.reserve(per_class_cap);
    }
    for anchor in 0..anchors {
        let row = &scores[anchor * num_classes..(anchor + 1) * num_classes];
        let (best_class, best_score) = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(class, score)| (class, *score))
            .unwrap_or((0, f32::NEG_INFINITY));
        if best_score < confidence_threshold {
            continue;
        }
        let bbox = &boxes[anchor * 4..anchor * 4 + 4];
        by_class[best_class].push((
            BoundingBox {
                xmin: bbox[0] - bbox[2] / 2.0,
                ymin: bbox[1] - bbox[3] / 2.0,
                xmax: bbox[0] + bbox[2] / 2.0,
                ymax: bbox[1] + bbox[3] / 2.0,
                confidence: best_score,
            },
            anchor,
        ));
    }

    // Per-class greedy suppression with the shared IoU definition (a box survives when its IoU
    // with every previously kept box of the same class is at most the threshold).
    let mut candidates = Vec::new();
    for (class_id, mut class_candidates) in by_class.into_iter().enumerate() {
        if class_candidates.is_empty() {
            continue;
        }
        class_candidates.sort_unstable_by(|a, b| b.0.confidence.total_cmp(&a.0.confidence));
        let len = class_candidates.len();
        let mut areas = Vec::with_capacity(len);
        for (bbox, _) in class_candidates.iter() {
            areas.push((bbox.xmax - bbox.xmin).max(0.) * (bbox.ymax - bbox.ymin).max(0.));
        }
        let mut kept: Vec<(BoundingBox, usize)> = Vec::with_capacity(len);
        let mut kept_areas: Vec<f32> = Vec::with_capacity(len);
        for (index, (bbox, anchor)) in class_candidates.into_iter().enumerate() {
            let area = areas[index];
            let mut drop = false;
            for (kept_idx, (kept_box, _)) in kept.iter().enumerate() {
                if kept_box.xmax <= bbox.xmin
                    || kept_box.xmin >= bbox.xmax
                    || kept_box.ymax <= bbox.ymin
                    || kept_box.ymin >= bbox.ymax
                {
                    continue;
                }
                let i_xmin = kept_box.xmin.max(bbox.xmin);
                let i_xmax = kept_box.xmax.min(bbox.xmax);
                let i_ymin = kept_box.ymin.max(bbox.ymin);
                let i_ymax = kept_box.ymax.min(bbox.ymax);
                let i_area = (i_xmax - i_xmin).max(0.) * (i_ymax - i_ymin).max(0.);
                if i_area == 0.0 {
                    continue;
                }
                let union = kept_areas[kept_idx] + area - i_area;
                if union > 0.0 && i_area / union > iou_threshold {
                    drop = true;
                    break;
                }
            }
            if !drop {
                kept.push((bbox, anchor));
                kept_areas.push(area);
            }
        }
        candidates.extend(
            kept.into_iter()
                .map(|(bbox, anchor)| SegmentationCandidate {
                    bbox,
                    class_id,
                    anchor,
                }),
        );
    }

    SegmentationOutputCpu {
        candidates,
        prototypes,
        proto_channels,
        proto_width,
        proto_height,
        coefficients,
        anchors,
    }
}

/// Regroup segmentation candidates into the per-class box layout shared by the detection paths.
fn classic_candidates_to_boxes(output: SegmentationOutputCpu) -> Vec<Vec<Vec<BoundingBox>>> {
    let mut per_class: Vec<Vec<BoundingBox>> =
        (0..COCO_CLASSES.len()).map(|_| Vec::new()).collect();
    for candidate in output.candidates {
        per_class[candidate.class_id].push(candidate.bbox);
    }
    vec![per_class]
}

/// Assemble one instance mask in the letterboxed canvas frame.
///
/// Mirrors Ultralytics' `process_mask(..., upsample=True)` exactly: the mask is the raw linear
/// combination `coefficients @ prototypes` (no sigmoid), bilinearly upsampled to the letterboxed
/// canvas at `align_corners = False` semantics, binarized at `> 0`, and cropped to the box.
pub(crate) fn canvas_instance_mask(
    output: &SegmentationOutputCpu,
    anchor: usize,
    canvas_width: usize,
    canvas_height: usize,
    box_canvas: [f32; 4],
) -> Vec<bool> {
    let channels = output.proto_channels;
    let proto_width = output.proto_width;
    let proto_height = output.proto_height;
    let plane = proto_width * proto_height;

    // Mask logits at prototype resolution.
    let mut logits = vec![0_f64; plane];
    for channel in 0..channels {
        let coefficient = output.coefficients[channel * output.anchors + anchor] as f64;
        let proto = &output.prototypes[channel * plane..(channel + 1) * plane];
        for (index, value) in proto.iter().enumerate() {
            logits[index] += coefficient * *value as f64;
        }
    }

    // Bilinear upsample to the canvas, threshold at > 0, and crop to the box, per canvas pixel.
    // Integerize the `crop_mask` window once (`x >= box[0] && x < box[2]` <=> `x in
    // [ceil(box[0]), ceil(box[2]))`) so pixels outside the box are never visited instead of
    // being branched away per pixel.
    let scale_x = proto_width as f64 / canvas_width.max(1) as f64;
    let scale_y = proto_height as f64 / canvas_height.max(1) as f64;
    let mut mask = vec![false; canvas_width * canvas_height];
    let x_start = (box_canvas[0].ceil().clamp(0.0, canvas_width as f32) as usize).min(canvas_width);
    let x_end = (box_canvas[2].ceil().clamp(0.0, canvas_width as f32) as usize).min(canvas_width);
    let y_start =
        (box_canvas[1].ceil().clamp(0.0, canvas_height as f32) as usize).min(canvas_height);
    let y_end = (box_canvas[3].ceil().clamp(0.0, canvas_height as f32) as usize).min(canvas_height);
    if x_start >= x_end || y_start >= y_end {
        return mask;
    }
    // Precompute the horizontal sampling tables once per mask; the same `source_x / x0 / x1 /
    // lambda_x` was recomputed for every row before.
    let width = x_end - x_start;
    let mut x0_table = Vec::with_capacity(width);
    let mut x1_table = Vec::with_capacity(width);
    let mut lambda_x_table = Vec::with_capacity(width);
    for x in x_start..x_end {
        let source_x = ((x as f64 + 0.5) * scale_x - 0.5).max(0.0);
        let x0 = source_x.floor();
        let x1 = (x0 + 1.0).min((proto_width - 1) as f64);
        x0_table.push(x0 as usize);
        x1_table.push(x1 as usize);
        lambda_x_table.push(source_x - x0);
    }
    for y in y_start..y_end {
        let source_y = ((y as f64 + 0.5) * scale_y - 0.5).max(0.0);
        let y0 = source_y.floor();
        let y1 = (y0 + 1.0).min((proto_height - 1) as f64);
        let y0 = y0 as usize;
        let y1 = y1 as usize;
        let lambda_y = source_y - y0 as f64;
        let inv_lambda_y = 1.0 - lambda_y;
        let row = y * canvas_width;
        let row0 = y0 * proto_width;
        let row1 = y1 * proto_width;
        for (i, x) in (x_start..x_end).enumerate() {
            let x0 = x0_table[i];
            let x1 = x1_table[i];
            let lambda_x = lambda_x_table[i];
            let inv_lambda_x = 1.0 - lambda_x;
            let top = logits[row0 + x0] * inv_lambda_x + logits[row0 + x1] * lambda_x;
            let bottom = logits[row1 + x0] * inv_lambda_x + logits[row1 + x1] * lambda_x;
            mask[row + x] = top * inv_lambda_y + bottom * lambda_y > 0.0;
        }
    }
    mask
}

/// Sample a canvas-frame boolean mask onto the full source-image grid.
///
/// Every source pixel `(x, y)` samples the canvas mask at the nearest canvas pixel to
/// `(x * scale + pad_x, y * scale + pad_y)` — the exact inverse of the letterbox geometry that
/// [`LetterboxedImage::to_source_box`] applies to box edges. Pixels outside the canvas (possible
/// only through rounding at the borders) stay uncovered.
fn source_instance_mask(
    canvas_mask: &[bool],
    canvas_width: usize,
    canvas_height: usize,
    prepared: &LetterboxedImage,
) -> InstanceMask {
    let (scale, pad_x, pad_y) = prepared.letterbox_geometry();
    let (source_width, source_height) = prepared.source_dimensions();
    let mut data = vec![false; source_width as usize * source_height as usize];
    let mut column_samples = Vec::with_capacity(source_width as usize);
    for x in 0..source_width {
        let canvas_x = (x as f32 * scale + pad_x + 0.5).floor();
        column_samples.push(canvas_x.clamp(0.0, canvas_width as f32 - 1.0) as usize);
    }
    for y in 0..source_height {
        let canvas_y = (y as f32 * scale + pad_y + 0.5).floor();
        let canvas_y = canvas_y.clamp(0.0, canvas_height as f32 - 1.0) as usize;
        for x in 0..source_width {
            data[y as usize * source_width as usize + x as usize] =
                canvas_mask[canvas_y * canvas_width + column_samples[x as usize]];
        }
    }
    InstanceMask {
        width: source_width,
        height: source_height,
        data,
    }
}

impl<B: Backend, M: EndToEndDetector<B>> EndToEndDetector<B> for Box<M> {
    fn detect(&self, input: Tensor<B, 4>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        (**self).detect(input)
    }
}

/// Decode normalized end-to-end one2one predictions for any scale variant.
fn run_end_to_end<B: Backend>(
    model: &impl EndToEndDetector<B>,
    input: Tensor<B, 4>,
    max_detections: usize,
    confidence_threshold: f32,
) -> Vec<Vec<Vec<BoundingBox>>> {
    let (boxes, scores) = model.detect(input);
    end2end_topk_detections(boxes, scores, max_detections, confidence_threshold)
}

impl<B: Backend> Predictor<B> {
    /// Load a native Burnpack artifact and infer its architecture from embedded metadata.
    ///
    /// This is the preferred constructor. Model construction and weight loading happen once;
    /// subsequent [`inference`](Self::inference) calls reuse the loaded graph.
    #[cfg(feature = "pretrained")]
    pub fn new(checkpoint: impl Into<PathBuf>) -> Result<Self> {
        Self::with_options(checkpoint, PredictOptions::default())
    }

    /// Load an artifact with explicit prediction thresholds.
    #[cfg(feature = "pretrained")]
    pub fn with_options(checkpoint: impl Into<PathBuf>, options: PredictOptions) -> Result<Self> {
        Self::with_options_on_device(checkpoint, Device::<B>::default(), options)
    }

    /// Load an artifact on an explicit device using default prediction thresholds.
    #[cfg(feature = "pretrained")]
    pub fn new_on_device(checkpoint: impl Into<PathBuf>, device: Device<B>) -> Result<Self> {
        Self::with_options_on_device(checkpoint, device, PredictOptions::default())
    }

    /// Load an artifact on an explicit device with explicit prediction thresholds.
    #[cfg(feature = "pretrained")]
    pub fn with_options_on_device(
        checkpoint: impl Into<PathBuf>,
        device: Device<B>,
        options: PredictOptions,
    ) -> Result<Self> {
        let options = options.validate()?;
        let checkpoint = checkpoint.into();
        require_burnpack_path(&checkpoint)?;
        let metadata = artifact_metadata(&checkpoint)?;
        if metadata.model_id.is_none() {
            return Err(
                "Burnpack artifact is missing montgomery.model metadata; repack it or use \
                 Predictor::from_checkpoint with an explicit ModelId"
                    .into(),
            );
        }
        Self::load_from_metadata(checkpoint, device, options, metadata)
    }

    #[cfg(feature = "pretrained")]
    fn load_from_metadata(
        checkpoint: PathBuf,
        device: Device<B>,
        options: PredictOptions,
        metadata: ArtifactMetadata,
    ) -> Result<Self> {
        let options = options.validate()?;
        let model_id = metadata
            .model_id
            .ok_or("internal error: artifact architecture was not resolved")?;
        let class_names = metadata
            .class_names
            .unwrap_or_else(|| catalog_class_names(model_id));
        let input_size = metadata
            .input_size
            .unwrap_or_else(|| catalog_input_size(model_id));
        let model = load_model_checkpoint(model_id, checkpoint, device.clone(), class_names.len())?;
        Ok(Self {
            model_id,
            model,
            device,
            options,
            class_names,
            input_size,
        })
    }

    /// Load a model from a native Burnpack artifact on the backend's default device.
    #[cfg(feature = "pretrained")]
    pub fn from_checkpoint(
        model_id: ModelId,
        checkpoint: impl Into<PathBuf>,
        options: PredictOptions,
    ) -> Result<Self> {
        Self::from_checkpoint_on_device(model_id, checkpoint, Device::<B>::default(), options)
    }

    /// Load a native Burnpack artifact on an explicit device.
    #[cfg(feature = "pretrained")]
    pub fn from_checkpoint_on_device(
        model_id: ModelId,
        checkpoint: impl Into<PathBuf>,
        device: Device<B>,
        options: PredictOptions,
    ) -> Result<Self> {
        let options = options.validate()?;
        let checkpoint = checkpoint.into();
        require_burnpack_path(&checkpoint)?;
        let mut metadata = artifact_metadata(&checkpoint)?;
        if metadata
            .model_id
            .is_some_and(|embedded| embedded != model_id)
        {
            return Err(format!(
                "Burnpack architecture is {}, but {model_id} was requested",
                metadata.model_id.unwrap()
            )
            .into());
        }
        metadata.model_id = Some(model_id);
        Self::load_from_metadata(checkpoint, device, options, metadata)
    }

    /// Load a native artifact exported from a training checkpoint, using its embedded ordered
    /// class table to construct the graph and label predictions.
    #[cfg(feature = "pretrained")]
    pub fn from_trained_artifact(
        model_id: ModelId,
        checkpoint: impl Into<PathBuf>,
        options: PredictOptions,
    ) -> Result<Self> {
        Self::from_trained_artifact_on_device(model_id, checkpoint, Device::<B>::default(), options)
    }

    /// Explicit-device variant of [`Predictor::from_trained_artifact`].
    #[cfg(feature = "pretrained")]
    pub fn from_trained_artifact_on_device(
        model_id: ModelId,
        checkpoint: impl Into<PathBuf>,
        device: Device<B>,
        options: PredictOptions,
    ) -> Result<Self> {
        let options = options.validate()?;
        let checkpoint = checkpoint.into();
        require_burnpack_path(&checkpoint)?;
        let mut metadata = artifact_metadata(&checkpoint)?;
        if metadata
            .model_id
            .is_some_and(|embedded| embedded != model_id)
        {
            return Err(format!(
                "Burnpack architecture is {}, but {model_id} was requested",
                metadata.model_id.unwrap()
            )
            .into());
        }
        if metadata.class_names.is_none() || metadata.input_size.is_none() {
            return Err("trained artifact is missing class-name or input-size metadata".into());
        }
        metadata.model_id = Some(model_id);
        Self::load_from_metadata(checkpoint, device, options, metadata)
    }

    /// The stable catalog identifier for the loaded model.
    pub const fn model_id(&self) -> ModelId {
        self.model_id
    }

    /// The task inferred from the artifact's architecture.
    pub const fn task(&self) -> ModelTask {
        self.model_id.task()
    }

    /// Prediction thresholds used by this model.
    pub const fn options(&self) -> PredictOptions {
        self.options
    }

    /// Ordered class table used by this predictor.
    pub fn class_names(&self) -> &[String] {
        &self.class_names
    }

    /// Square input side used by this artifact's preprocessing contract.
    pub const fn input_size(&self) -> usize {
        self.input_size
    }

    /// Decode or borrow an image and run the task represented by the loaded artifact.
    ///
    /// Accepts paths (`&str`, [`Path`], [`PathBuf`]), [`DynamicImage`], RGB/RGBA/grayscale
    /// [`ImageBuffer`] aliases, and encoded image bytes (`&[u8]` or `Vec<u8>`).
    pub fn inference<'a>(&self, source: impl Into<ImageSource<'a>>) -> Result<Prediction> {
        let image = source.into().decode()?;
        match self.task() {
            ModelTask::Detection => Ok(Prediction::Detections(self.predict(image.as_ref()))),
            ModelTask::Segmentation => Ok(Prediction::Segmentations(
                self.predict_segmentation(image.as_ref())?,
            )),
            ModelTask::Classification => Ok(Prediction::Classifications(
                self.predict_classification(image.as_ref())?,
            )),
        }
    }

    /// Run object detection on an already-decoded image.
    pub fn predict(&self, image: &DynamicImage) -> Vec<Detection> {
        let mut prepared = match self.model_id.detection_preprocess() {
            DetectionPreprocess::Yolox => LetterboxedImage::yolox(image, self.input_size),
            DetectionPreprocess::Ultralytics => {
                LetterboxedImage::ultralytics(image, self.input_size, 32)
            }
        };
        let input = image_to_tensor(prepared.take_image(), &self.device).unsqueeze::<4>();
        let input = match self.model_id.detection_preprocess() {
            DetectionPreprocess::Yolox => input,
            DetectionPreprocess::Ultralytics => input / 255.0,
        };
        let boxes_by_class = match &self.model {
            RuntimeModel::Yolox(model) => {
                run_classic_detections(model, input, self.options.iou, self.options.confidence)
            }
            RuntimeModel::Yolov3Tiny(model) => {
                run_classic_detections(model, input, self.options.iou, self.options.confidence)
            }
            RuntimeModel::Yolov10N(model) => run_end_to_end(
                model,
                input,
                YOLOV10_MAX_DETECTIONS,
                self.options.confidence,
            ),
            RuntimeModel::Yolov10S(model) => run_end_to_end(
                model,
                input,
                YOLOV10_MAX_DETECTIONS,
                self.options.confidence,
            ),
            RuntimeModel::Yolov10M(model) => run_end_to_end(
                model,
                input,
                YOLOV10_MAX_DETECTIONS,
                self.options.confidence,
            ),
            RuntimeModel::Yolov10B(model) => run_end_to_end(
                model,
                input,
                YOLOV10_MAX_DETECTIONS,
                self.options.confidence,
            ),
            RuntimeModel::Yolov10L(model) => run_end_to_end(
                model,
                input,
                YOLOV10_MAX_DETECTIONS,
                self.options.confidence,
            ),
            RuntimeModel::Yolov10X(model) => run_end_to_end(
                model,
                input,
                YOLOV10_MAX_DETECTIONS,
                self.options.confidence,
            ),
            RuntimeModel::Yolo11N(model) => {
                // YOLO11 is the first NMS-based Ultralytics detect family here: its classic DFL
                // head output (center-size boxes plus per-class sigmoid scores) is decoded by the
                // head and suppressed by the generic class-aware NMS helper, mirroring
                // Ultralytics' non_max_suppression at conf 0.25 / IoU 0.45.
                run_classic_detections(model, input, self.options.iou, self.options.confidence)
            }
            RuntimeModel::Yolo11S(model) => {
                run_classic_detections(model, input, self.options.iou, self.options.confidence)
            }
            RuntimeModel::Yolo11M(model) => {
                run_classic_detections(model, input, self.options.iou, self.options.confidence)
            }
            RuntimeModel::Yolo11L(model) => {
                run_classic_detections(model, input, self.options.iou, self.options.confidence)
            }
            RuntimeModel::Yolo11X(model) => {
                run_classic_detections(model, input, self.options.iou, self.options.confidence)
            }
            RuntimeModel::Yolo11SegM(model) => classic_candidates_to_boxes(
                run_classic_segmentations(model, input, self.options.iou, self.options.confidence),
            ),
            RuntimeModel::Yolo11SegL(model) => classic_candidates_to_boxes(
                run_classic_segmentations(model, input, self.options.iou, self.options.confidence),
            ),
            RuntimeModel::Yolo11SegX(model) => classic_candidates_to_boxes(
                run_classic_segmentations(model, input, self.options.iou, self.options.confidence),
            ),
            RuntimeModel::Yolov8N(model) => {
                // YOLOv8 shares YOLO11's classic decode path: DFL head output decoded to
                // center-size boxes plus per-class sigmoid scores, suppressed by the generic
                // class-aware NMS helper.
                run_classic_detections(model, input, self.options.iou, self.options.confidence)
            }
            RuntimeModel::Yolov8S(model) => {
                run_classic_detections(model, input, self.options.iou, self.options.confidence)
            }
            RuntimeModel::Yolov8M(model) => {
                run_classic_detections(model, input, self.options.iou, self.options.confidence)
            }
            RuntimeModel::Yolov8L(model) => {
                run_classic_detections(model, input, self.options.iou, self.options.confidence)
            }
            RuntimeModel::Yolov8X(model) => {
                run_classic_detections(model, input, self.options.iou, self.options.confidence)
            }
            RuntimeModel::Yolov8SegN(model) => classic_candidates_to_boxes(
                run_classic_segmentations(model, input, self.options.iou, self.options.confidence),
            ),
            RuntimeModel::Yolov8SegS(model) => classic_candidates_to_boxes(
                run_classic_segmentations(model, input, self.options.iou, self.options.confidence),
            ),
            RuntimeModel::Yolov8SegM(model) => classic_candidates_to_boxes(
                run_classic_segmentations(model, input, self.options.iou, self.options.confidence),
            ),
            RuntimeModel::Yolov8SegL(model) => classic_candidates_to_boxes(
                run_classic_segmentations(model, input, self.options.iou, self.options.confidence),
            ),
            RuntimeModel::Yolov8SegX(model) => classic_candidates_to_boxes(
                run_classic_segmentations(model, input, self.options.iou, self.options.confidence),
            ),
            RuntimeModel::Yolo12N(model) => {
                // YOLO12 rides the same classic decode as YOLO11 (its head is byte-identical).
                run_classic_detections(model, input, self.options.iou, self.options.confidence)
            }
            RuntimeModel::Yolo12S(model) => {
                run_classic_detections(model, input, self.options.iou, self.options.confidence)
            }
            RuntimeModel::Yolo12M(model) => {
                run_classic_detections(model, input, self.options.iou, self.options.confidence)
            }
            RuntimeModel::Yolo12L(model) => {
                run_classic_detections(model, input, self.options.iou, self.options.confidence)
            }
            RuntimeModel::Yolo12X(model) => {
                run_classic_detections(model, input, self.options.iou, self.options.confidence)
            }
            RuntimeModel::Yolo11SegN(model) => {
                // Segmentation models share the classic detection decode; predict() exposes the
                // box branch only (masks require predict_segmentation).
                classic_candidates_to_boxes(run_classic_segmentations(
                    model,
                    input,
                    self.options.iou,
                    self.options.confidence,
                ))
            }
            RuntimeModel::Yolo11SegS(model) => classic_candidates_to_boxes(
                run_classic_segmentations(model, input, self.options.iou, self.options.confidence),
            ),
            RuntimeModel::Yolo26N(model) => {
                // YOLO26 is end-to-end like YOLOv10 and consumes the same RGB normalization. Its
                // DFL-free one2one head is likewise decoded by top-score selection and a
                // confidence filter with no non-maximum suppression.
                run_end_to_end(model, input, YOLO26_MAX_DETECTIONS, self.options.confidence)
            }
            RuntimeModel::Yolo26S(model) => {
                run_end_to_end(model, input, YOLO26_MAX_DETECTIONS, self.options.confidence)
            }
            RuntimeModel::Yolo26M(model) => {
                run_end_to_end(model, input, YOLO26_MAX_DETECTIONS, self.options.confidence)
            }
            RuntimeModel::Yolo26L(model) => {
                run_end_to_end(model, input, YOLO26_MAX_DETECTIONS, self.options.confidence)
            }
            RuntimeModel::Yolo26X(model) => {
                run_end_to_end(model, input, YOLO26_MAX_DETECTIONS, self.options.confidence)
            }
            RuntimeModel::Yolo26SegN(model) => {
                // YOLO26-seg shares the end-to-end decode (top-k, no NMS); predict() exposes the
                // box branch only (masks require predict_segmentation).
                classic_candidates_to_boxes(run_end_to_end_segmentations(
                    model,
                    input,
                    YOLO26_MAX_DETECTIONS,
                    self.options.confidence,
                ))
            }
            RuntimeModel::Yolo26SegS(model) => {
                classic_candidates_to_boxes(run_end_to_end_segmentations(
                    model,
                    input,
                    YOLO26_MAX_DETECTIONS,
                    self.options.confidence,
                ))
            }
            RuntimeModel::Yolo26SegM(model) => {
                classic_candidates_to_boxes(run_end_to_end_segmentations(
                    model,
                    input,
                    YOLO26_MAX_DETECTIONS,
                    self.options.confidence,
                ))
            }
            RuntimeModel::Yolo26SegL(model) => {
                classic_candidates_to_boxes(run_end_to_end_segmentations(
                    model,
                    input,
                    YOLO26_MAX_DETECTIONS,
                    self.options.confidence,
                ))
            }
            RuntimeModel::Yolo26SegX(model) => {
                classic_candidates_to_boxes(run_end_to_end_segmentations(
                    model,
                    input,
                    YOLO26_MAX_DETECTIONS,
                    self.options.confidence,
                ))
            }
            // Classification models carry no spatial detections; the class probabilities are
            // exposed through predict_classification.
            RuntimeModel::Yolo26ClsN(_)
            | RuntimeModel::Yolo26ClsS(_)
            | RuntimeModel::Yolo26ClsM(_)
            | RuntimeModel::Yolo26ClsL(_)
            | RuntimeModel::Yolo26ClsX(_)
            | RuntimeModel::Yolo11ClsN(_)
            | RuntimeModel::Yolo11ClsS(_)
            | RuntimeModel::Yolo11ClsM(_)
            | RuntimeModel::Yolo11ClsL(_)
            | RuntimeModel::Yolo11ClsX(_)
            | RuntimeModel::Yolov8ClsN(_)
            | RuntimeModel::Yolov8ClsS(_)
            | RuntimeModel::Yolov8ClsM(_)
            | RuntimeModel::Yolov8ClsL(_)
            | RuntimeModel::Yolov8ClsX(_) => vec![Vec::new()],
        };

        let mut detections = Vec::new();

        for (class_id, class_boxes) in boxes_by_class[0].iter().enumerate() {
            for bbox in class_boxes {
                let [xmin, ymin, xmax, ymax] =
                    prepared.to_source_box([bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]);
                detections.push(Detection {
                    class_id,
                    class_name: self.class_names[class_id].clone(),
                    confidence: bbox.confidence,
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                });
            }
        }

        detections.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
        detections
    }

    /// Decode an image from disk and run object detection.
    pub fn predict_path(&self, path: impl AsRef<Path>) -> Result<(DynamicImage, Vec<Detection>)> {
        let image = image::open(path)?;
        let detections = self.predict(&image);
        Ok((image, detections))
    }

    /// Run instance segmentation on an already-decoded image.
    ///
    /// Returns one [`SegmentationDetection`] per surviving instance: the same box contract as
    /// [`Predictor::predict`] plus a boolean source-image coverage mask. Detections whose cropped
    /// mask is empty are dropped, mirroring Ultralytics' segmentation postprocess. Requires a
    /// segmentation model (`yolo11n/s/m/l/x-seg`, `yolov8n/s/m/l/x-seg`, or
    /// `yolo26n/s/m/l/x-seg`); detect models should use [`Predictor::predict`].
    pub fn predict_segmentation(&self, image: &DynamicImage) -> Result<Vec<SegmentationDetection>> {
        let mut prepared = self.prepare_segmentation(image)?;
        let (canvas_width, canvas_height) = (
            prepared.image().width() as usize,
            prepared.image().height() as usize,
        );
        let input = image_to_tensor(prepared.take_image(), &self.device).unsqueeze::<4>() / 255.0;
        let output = match &self.model {
            RuntimeModel::Yolo11SegN(model) => {
                run_classic_segmentations(model, input, self.options.iou, self.options.confidence)
            }
            RuntimeModel::Yolo11SegS(model) => {
                run_classic_segmentations(model, input, self.options.iou, self.options.confidence)
            }
            RuntimeModel::Yolo11SegM(model) => {
                run_classic_segmentations(model, input, self.options.iou, self.options.confidence)
            }
            RuntimeModel::Yolo11SegL(model) => {
                run_classic_segmentations(model, input, self.options.iou, self.options.confidence)
            }
            RuntimeModel::Yolo11SegX(model) => {
                run_classic_segmentations(model, input, self.options.iou, self.options.confidence)
            }
            RuntimeModel::Yolov8SegN(model) => {
                run_classic_segmentations(model, input, self.options.iou, self.options.confidence)
            }
            RuntimeModel::Yolov8SegS(model) => {
                run_classic_segmentations(model, input, self.options.iou, self.options.confidence)
            }
            RuntimeModel::Yolov8SegM(model) => {
                run_classic_segmentations(model, input, self.options.iou, self.options.confidence)
            }
            RuntimeModel::Yolov8SegL(model) => {
                run_classic_segmentations(model, input, self.options.iou, self.options.confidence)
            }
            RuntimeModel::Yolov8SegX(model) => {
                run_classic_segmentations(model, input, self.options.iou, self.options.confidence)
            }
            RuntimeModel::Yolo26SegN(model) => run_end_to_end_segmentations(
                model,
                input,
                YOLO26_MAX_DETECTIONS,
                self.options.confidence,
            ),
            RuntimeModel::Yolo26SegS(model) => run_end_to_end_segmentations(
                model,
                input,
                YOLO26_MAX_DETECTIONS,
                self.options.confidence,
            ),
            RuntimeModel::Yolo26SegM(model) => run_end_to_end_segmentations(
                model,
                input,
                YOLO26_MAX_DETECTIONS,
                self.options.confidence,
            ),
            RuntimeModel::Yolo26SegL(model) => run_end_to_end_segmentations(
                model,
                input,
                YOLO26_MAX_DETECTIONS,
                self.options.confidence,
            ),
            RuntimeModel::Yolo26SegX(model) => run_end_to_end_segmentations(
                model,
                input,
                YOLO26_MAX_DETECTIONS,
                self.options.confidence,
            ),
            _ => {
                return Err(format!(
                    "{} is not a segmentation model; instance masks are available for \
                     yolo11n/s/m/l/x-seg, yolov8n/s/m/l/x-seg, and yolo26n/s/m/l/x-seg",
                    self.model_id
                )
                .into());
            }
        };

        let mut detections = Vec::new();
        for candidate in &output.candidates {
            let canvas_mask = canvas_instance_mask(
                &output,
                candidate.anchor,
                canvas_width,
                canvas_height,
                [
                    candidate.bbox.xmin,
                    candidate.bbox.ymin,
                    candidate.bbox.xmax,
                    candidate.bbox.ymax,
                ],
            );
            // Ultralytics drops post-NMS detections whose cropped mask is fully empty.
            if !canvas_mask.iter().any(|covered| *covered) {
                continue;
            }
            let mask = source_instance_mask(&canvas_mask, canvas_width, canvas_height, &prepared);
            let [xmin, ymin, xmax, ymax] = prepared.to_source_box([
                candidate.bbox.xmin,
                candidate.bbox.ymin,
                candidate.bbox.xmax,
                candidate.bbox.ymax,
            ]);
            detections.push(SegmentationDetection {
                class_id: candidate.class_id,
                class_name: self.class_names[candidate.class_id].clone(),
                confidence: candidate.bbox.confidence,
                xmin,
                ymin,
                xmax,
                ymax,
                mask,
            });
        }

        detections.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
        Ok(detections)
    }

    /// Decode an image from disk and run instance segmentation.
    pub fn predict_segmentation_path(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<(DynamicImage, Vec<SegmentationDetection>)> {
        let image = image::open(path)?;
        let detections = self.predict_segmentation(&image)?;
        Ok((image, detections))
    }

    /// Run image classification on an already-decoded image.
    ///
    /// Returns the top-5 classes by probability (Ultralytics' `probs.top5` convention), in
    /// descending order. Requires a YOLOv8, YOLO11, or YOLO26 `-cls` model; detect models should
    /// use [`Predictor::predict`]. The input mirrors Ultralytics' classify inference transform:
    /// bilinear resize of the shortest edge to 224 px (anti-aliased), a centered 224x224 crop, and
    /// RGB values scaled to `[0, 1]`.
    pub fn predict_classification(&self, image: &DynamicImage) -> Result<Vec<Classification>> {
        self.prepare_classification(image)?;
        let input = image_to_tensor(classify_transform(image, self.input_size), &self.device)
            .unsqueeze::<4>();
        let output = match &self.model {
            RuntimeModel::Yolo26ClsN(model) => model.classify(input / 255.0),
            RuntimeModel::Yolo26ClsS(model) => model.classify(input / 255.0),
            RuntimeModel::Yolo26ClsM(model) => model.classify(input / 255.0),
            RuntimeModel::Yolo26ClsL(model) => model.classify(input / 255.0),
            RuntimeModel::Yolo26ClsX(model) => model.classify(input / 255.0),
            RuntimeModel::Yolo11ClsN(model) => model.classify(input / 255.0),
            RuntimeModel::Yolo11ClsS(model) => model.classify(input / 255.0),
            RuntimeModel::Yolo11ClsM(model) => model.classify(input / 255.0),
            RuntimeModel::Yolo11ClsL(model) => model.classify(input / 255.0),
            RuntimeModel::Yolo11ClsX(model) => model.classify(input / 255.0),
            RuntimeModel::Yolov8ClsN(model) => model.classify(input / 255.0),
            RuntimeModel::Yolov8ClsS(model) => model.classify(input / 255.0),
            RuntimeModel::Yolov8ClsM(model) => model.classify(input / 255.0),
            RuntimeModel::Yolov8ClsL(model) => model.classify(input / 255.0),
            RuntimeModel::Yolov8ClsX(model) => model.classify(input / 255.0),
            _ => unreachable!("prepare_classification rejects non-classification models"),
        };
        let probs: Vec<f32> = output
            .probs
            .into_data()
            .iter::<B::FloatElem>()
            .map(|value| value.elem::<f32>())
            .collect();
        let mut order: Vec<usize> = (0..probs.len()).collect();
        order.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));
        order.truncate(CLASSIFICATION_TOP_K);
        Ok(order
            .into_iter()
            .map(|class_id| Classification {
                class_id,
                class_name: self.class_names[class_id].clone(),
                confidence: probs[class_id],
            })
            .collect())
    }

    /// Decode an image from disk and run image classification.
    pub fn predict_classification_path(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<(DynamicImage, Vec<Classification>)> {
        let image = image::open(path)?;
        let classifications = self.predict_classification(&image)?;
        Ok((image, classifications))
    }

    /// Verify the loaded model is a classification model.
    fn prepare_classification(&self, _image: &DynamicImage) -> Result<()> {
        match &self.model {
            RuntimeModel::Yolo26ClsN(_)
            | RuntimeModel::Yolo26ClsS(_)
            | RuntimeModel::Yolo26ClsM(_)
            | RuntimeModel::Yolo26ClsL(_)
            | RuntimeModel::Yolo26ClsX(_)
            | RuntimeModel::Yolo11ClsN(_)
            | RuntimeModel::Yolo11ClsS(_)
            | RuntimeModel::Yolo11ClsM(_)
            | RuntimeModel::Yolo11ClsL(_)
            | RuntimeModel::Yolo11ClsX(_)
            | RuntimeModel::Yolov8ClsN(_)
            | RuntimeModel::Yolov8ClsS(_)
            | RuntimeModel::Yolov8ClsM(_)
            | RuntimeModel::Yolov8ClsL(_)
            | RuntimeModel::Yolov8ClsX(_) => Ok(()),
            _ => Err(format!(
                "{} is not a classification model; class probabilities are available for the \
                 yolo26n/s/m/l/x-cls, yolo11n/s/m/l/x-cls, and yolov8n/s/m/l/x-cls variants",
                self.model_id
            )
            .into()),
        }
    }

    /// Letterbox an image the way the loaded segmentation model expects.
    fn prepare_segmentation(&self, image: &DynamicImage) -> Result<LetterboxedImage> {
        match &self.model {
            RuntimeModel::Yolo11SegN(_)
            | RuntimeModel::Yolo11SegS(_)
            | RuntimeModel::Yolo11SegM(_)
            | RuntimeModel::Yolo11SegL(_)
            | RuntimeModel::Yolo11SegX(_)
            | RuntimeModel::Yolov8SegN(_)
            | RuntimeModel::Yolov8SegS(_)
            | RuntimeModel::Yolov8SegM(_)
            | RuntimeModel::Yolov8SegL(_)
            | RuntimeModel::Yolov8SegX(_)
            | RuntimeModel::Yolo26SegN(_)
            | RuntimeModel::Yolo26SegS(_)
            | RuntimeModel::Yolo26SegM(_)
            | RuntimeModel::Yolo26SegL(_)
            | RuntimeModel::Yolo26SegX(_) => {
                Ok(LetterboxedImage::ultralytics(image, self.input_size, 32))
            }
            _ => Err(format!(
                "{} is not a segmentation model; instance masks are available for \
                 yolo11n/s/m/l/x-seg, yolov8n/s/m/l/x-seg, and yolo26n/s/m/l/x-seg",
                self.model_id
            )
            .into()),
        }
    }
}

/// Convert an imported tensor-only checkpoint state into Montgomery's versioned native Burnpack
/// format. All families use states generated by `tools/export_checkpoint_state.py`. The output is
/// `<model>.bpk` in the current directory and stores half-precision tensors. Existing files are
/// never overwritten.
#[cfg(feature = "pretrained")]
#[derive(Debug)]
pub struct PackedWeights {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
}

#[cfg(feature = "pretrained")]
pub fn pack_weights(model_id: ModelId, input: impl Into<PathBuf>) -> Result<PackedWeights> {
    let output = model_id.artifact_filename();
    pack_weights_to(model_id, input, output)
}

/// Pack weights to an explicit path instead of the conventional `<model>.bpk` filename.
#[cfg(feature = "pretrained")]
pub fn pack_weights_to(
    model_id: ModelId,
    input: impl Into<PathBuf>,
    output: impl Into<PathBuf>,
) -> Result<PackedWeights> {
    let input = input.into();
    let output = output.into();
    if input.extension().and_then(|value| value.to_str()) != Some("pt") {
        return Err(
            "pack-weights input must be a tensor-only .pt state produced by tools/export_checkpoint_state.py"
                .into(),
        );
    }
    if output.extension().and_then(|value| value.to_str()) != Some("bpk") {
        return Err("native weight artifact output must use the .bpk extension".into());
    }
    if output.exists() {
        return Err(format!(
            "refusing to overwrite existing artifact: {}",
            output.display()
        )
        .into());
    }
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }

    let packed_path = output.clone();
    let worker = std::thread::Builder::new()
        .name("montgomery-weight-packer".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            let device = Default::default();
            macro_rules! pack_variant {
                ($config:ty) => {{
                    let mut model = <$config>::default().init::<Flex>(&device);
                    model.load_pytorch_weights(&input)?;
                    model.save_burnpack_weights(&output)?;
                }};
            }
            macro_rules! pack_yolox {
                ($constructor:path) => {{
                    let mut model = $constructor(COCO_CLASSES.len(), &device);
                    model.load_pytorch_weights(&input)?;
                    model.save_burnpack_weights(&output, model_id.as_str())?;
                }};
            }
            match model_id {
                ModelId::YoloxNano => pack_yolox!(Yolox::<Flex>::yolox_nano),
                ModelId::YoloxTiny => pack_yolox!(Yolox::<Flex>::yolox_tiny),
                ModelId::YoloxS => pack_yolox!(Yolox::<Flex>::yolox_s),
                ModelId::YoloxM => pack_yolox!(Yolox::<Flex>::yolox_m),
                ModelId::YoloxL => pack_yolox!(Yolox::<Flex>::yolox_l),
                ModelId::YoloxX => pack_yolox!(Yolox::<Flex>::yolox_x),
                ModelId::Yolov3TinyU => pack_variant!(Yolov3TinyConfig),
                ModelId::Yolov10N => pack_variant!(Yolov10NConfig),
                ModelId::Yolov10S => pack_variant!(Yolov10SConfig),
                ModelId::Yolov10M => pack_variant!(Yolov10MConfig),
                ModelId::Yolov10B => pack_variant!(Yolov10BConfig),
                ModelId::Yolov10L => pack_variant!(Yolov10LConfig),
                ModelId::Yolov10X => pack_variant!(Yolov10XConfig),
                ModelId::Yolo11N => pack_variant!(Yolo11NConfig),
                ModelId::Yolo11S => pack_variant!(Yolo11SConfig),
                ModelId::Yolo11M => pack_variant!(Yolo11MConfig),
                ModelId::Yolo11L => pack_variant!(Yolo11LConfig),
                ModelId::Yolo11X => pack_variant!(Yolo11XConfig),
                ModelId::Yolo11NSeg => pack_variant!(Yolo11SegNConfig),
                ModelId::Yolo11SSeg => pack_variant!(Yolo11SegSConfig),
                ModelId::Yolo11MSeg => pack_variant!(Yolo11SegMConfig),
                ModelId::Yolo11LSeg => pack_variant!(Yolo11SegLConfig),
                ModelId::Yolo11XSeg => pack_variant!(Yolo11SegXConfig),
                ModelId::Yolo11NCls => pack_variant!(Yolo11ClsNConfig),
                ModelId::Yolo11SCls => pack_variant!(Yolo11ClsSConfig),
                ModelId::Yolo11MCls => pack_variant!(Yolo11ClsMConfig),
                ModelId::Yolo11LCls => pack_variant!(Yolo11ClsLConfig),
                ModelId::Yolo11XCls => pack_variant!(Yolo11ClsXConfig),
                ModelId::Yolov8N => pack_variant!(Yolov8NConfig),
                ModelId::Yolov8S => pack_variant!(Yolov8SConfig),
                ModelId::Yolov8M => pack_variant!(Yolov8MConfig),
                ModelId::Yolov8L => pack_variant!(Yolov8LConfig),
                ModelId::Yolov8X => pack_variant!(Yolov8XConfig),
                ModelId::Yolov8NSeg => pack_variant!(Yolov8SegNConfig),
                ModelId::Yolov8SSeg => pack_variant!(Yolov8SegSConfig),
                ModelId::Yolov8MSeg => pack_variant!(Yolov8SegMConfig),
                ModelId::Yolov8LSeg => pack_variant!(Yolov8SegLConfig),
                ModelId::Yolov8XSeg => pack_variant!(Yolov8SegXConfig),
                ModelId::Yolov8NCls => pack_variant!(Yolov8ClsNConfig),
                ModelId::Yolov8SCls => pack_variant!(Yolov8ClsSConfig),
                ModelId::Yolov8MCls => pack_variant!(Yolov8ClsMConfig),
                ModelId::Yolov8LCls => pack_variant!(Yolov8ClsLConfig),
                ModelId::Yolov8XCls => pack_variant!(Yolov8ClsXConfig),
                ModelId::Yolo12N => pack_variant!(Yolo12NConfig),
                ModelId::Yolo12S => pack_variant!(Yolo12SConfig),
                ModelId::Yolo12M => pack_variant!(Yolo12MConfig),
                ModelId::Yolo12L => pack_variant!(Yolo12LConfig),
                ModelId::Yolo12X => pack_variant!(Yolo12XConfig),
                ModelId::Yolo26N => pack_variant!(Yolo26NConfig),
                ModelId::Yolo26S => pack_variant!(Yolo26SConfig),
                ModelId::Yolo26M => pack_variant!(Yolo26MConfig),
                ModelId::Yolo26L => pack_variant!(Yolo26LConfig),
                ModelId::Yolo26X => pack_variant!(Yolo26XConfig),
                ModelId::Yolo26NSeg => pack_variant!(Yolo26SegNConfig),
                ModelId::Yolo26SSeg => pack_variant!(Yolo26SegSConfig),
                ModelId::Yolo26MSeg => pack_variant!(Yolo26SegMConfig),
                ModelId::Yolo26LSeg => pack_variant!(Yolo26SegLConfig),
                ModelId::Yolo26XSeg => pack_variant!(Yolo26SegXConfig),
                ModelId::Yolo26NCls => pack_variant!(Yolo26ClsNConfig),
                ModelId::Yolo26SCls => pack_variant!(Yolo26ClsSConfig),
                ModelId::Yolo26MCls => pack_variant!(Yolo26ClsMConfig),
                ModelId::Yolo26LCls => pack_variant!(Yolo26ClsLConfig),
                ModelId::Yolo26XCls => pack_variant!(Yolo26ClsXConfig),
            }
            Ok::<_, Box<dyn Error + Send + Sync>>(())
        })?;
    worker
        .join()
        .map_err(|_| "weight packer thread panicked")??;

    let mut file = std::fs::File::open(&packed_path)?;
    let mut digest = Sha256::new();
    // Heap-allocated on purpose: a stack buffer this large would overflow the small default
    // main-thread stack on Windows as soon as this function is entered.
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        use std::io::Read;
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    let sha256 = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02X}"))
        .collect();
    Ok(PackedWeights {
        bytes: std::fs::metadata(&packed_path)?.len(),
        path: packed_path,
        sha256,
    })
}

fn image_to_tensor<B: Backend>(image: DynamicImage, device: &Device<B>) -> Tensor<B, 3> {
    let rgb = image.into_rgb8();
    let width = rgb.width() as usize;
    let height = rgb.height() as usize;
    let raw = rgb.into_raw();
    // Fuse the HWC->CHW transpose with the u8->float cast on the host so the backend
    // receives contiguous CHW directly (previously this uploaded HWC then ran a separate
    // `permute` kernel over ~1.2M floats).
    let mut chw = vec![0.0f32; width * height * 3];
    let plane = width * height;
    let (pixels, remainder) = raw.as_chunks::<3>();
    debug_assert!(remainder.is_empty());
    for (index, pixel) in pixels.iter().enumerate() {
        chw[index] = pixel[0] as f32;
        chw[plane + index] = pixel[1] as f32;
        chw[2 * plane + index] = pixel[2] as f32;
    }
    Tensor::<B, 3>::from_data(
        TensorData::new(chw, [3, height, width]).convert::<B::FloatElem>(),
        device,
    )
}

/// Select the strongest detections from decoded end-to-end one2one predictions.
///
/// This mirrors Ultralytics' end-to-end head postprocess, shared by YOLOv10 and YOLO26: keep the
/// `max_detections` anchors with the highest best-class score, then keep the `max_detections`
/// strongest (anchor, class) pairs among them, and finally apply the confidence threshold. No
/// non-maximum suppression is applied because the one2one head is trained to emit one prediction
/// per object.
pub(crate) fn end2end_topk_detections<B: Backend>(
    boxes: Tensor<B, 3>,
    scores: Tensor<B, 3>,
    max_detections: usize,
    confidence_threshold: f32,
) -> Vec<Vec<Vec<BoundingBox>>> {
    let [batch, anchors, classes] = scores.dims();
    let keep = max_detections.min(anchors);
    let boxes: Vec<f32> = boxes.into_data().iter::<f32>().collect();
    let scores: Vec<f32> = scores.into_data().iter::<f32>().collect();

    let mut batches = Vec::with_capacity(batch);
    for image in 0..batch {
        let image_scores = &scores[image * anchors * classes..(image + 1) * anchors * classes];
        let image_boxes = &boxes[image * anchors * 4..(image + 1) * anchors * 4];

        let mut best_scores = Vec::with_capacity(anchors);
        for anchor in 0..anchors {
            let row = &image_scores[anchor * classes..(anchor + 1) * classes];
            let mut best = f32::NEG_INFINITY;
            for &score in row {
                if score > best {
                    best = score;
                }
            }
            best_scores.push(best);
        }
        // Anchors whose best score is below threshold cannot survive the confidence
        // filter below, so drop them before the (anchor, class) expansion. When more
        // than `keep` anchors remain above threshold this selects the same top-`keep`
        // set as a full sort; otherwise the survivors are identical after filtering.
        let mut anchor_order: Vec<usize> = (0..anchors)
            .filter(|&anchor| best_scores[anchor] >= confidence_threshold)
            .collect();
        if anchor_order.len() > keep {
            anchor_order
                .select_nth_unstable_by(keep, |&a, &b| best_scores[b].total_cmp(&best_scores[a]));
            anchor_order.truncate(keep);
        }
        anchor_order.sort_unstable_by(|&a, &b| best_scores[b].total_cmp(&best_scores[a]));

        let mut candidates = Vec::with_capacity(anchor_order.len() * classes.min(8));
        for (selected_index, &anchor) in anchor_order.iter().enumerate() {
            let base = anchor * classes;
            for class in 0..classes {
                let score = image_scores[base + class];
                if score >= confidence_threshold {
                    candidates.push((score, selected_index, class));
                }
            }
        }
        if candidates.len() > keep {
            candidates.select_nth_unstable_by(keep, |a, b| b.0.total_cmp(&a.0));
            candidates.truncate(keep);
        }
        candidates.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

        let mut per_class = (0..classes).map(|_| Vec::new()).collect::<Vec<_>>();
        for (score, selected_index, class) in candidates {
            let anchor = anchor_order[selected_index];
            let bbox = &image_boxes[anchor * 4..anchor * 4 + 4];
            per_class[class].push(BoundingBox {
                xmin: bbox[0],
                ymin: bbox[1],
                xmax: bbox[2],
                ymax: bbox[3],
                confidence: score,
            });
        }
        batches.push(per_class);
    }
    batches
}

/// Resolve the default wgpu device, reporting the graphics adapter and API actually chosen.
///
/// The first call initializes the wgpu runtime (adapter selection, shader compiler setup) and the
/// returned description names the physical adapter, the graphics backend (Vulkan, DX12, Metal,
/// OpenGL, or WebGPU), and the driver. Every later operation on the same device reuses that
/// runtime.
#[cfg(feature = "gpu")]
pub fn default_wgpu_device() -> (burn::backend::wgpu::WgpuDevice, String) {
    use burn::backend::wgpu::{RuntimeOptions, WgpuDevice, graphics::AutoGraphicsApi, init_setup};
    use std::sync::OnceLock;

    // The wgpu runtime registers one compute client per device per process; re-initializing an
    // already-registered device panics, so resolve the default device exactly once and share it.
    static DEVICE: OnceLock<(WgpuDevice, String)> = OnceLock::new();
    DEVICE
        .get_or_init(|| {
            let device = WgpuDevice::default();
            let setup = init_setup::<AutoGraphicsApi>(&device, RuntimeOptions::default());
            (device, format!("{:?}", setup.adapter.get_info()))
        })
        .clone()
}

/// Draw labeled detection rectangles on a copy of the image.
pub fn annotate(image: &DynamicImage, detections: &[Detection]) -> DynamicImage {
    let mut output = image.to_rgb8();
    for detection in detections {
        let color = class_color(detection.class_id);
        draw_labeled_rect(
            &mut output,
            [
                detection.xmin as u32,
                detection.ymin as u32,
                detection.xmax as u32,
                detection.ymax as u32,
            ],
            color,
            &detection.class_name,
            detection.confidence,
        );
    }
    DynamicImage::ImageRgb8(output)
}

/// Draw instance-mask outlines and labeled detection rectangles on a copy of the image.
///
/// The outline is the mask's boolean boundary: every covered pixel with at least one uncovered
/// 4-neighbor is drawn in the class color, then the boxes are stroked on top. Masks live in
/// source-image pixels, so no coordinate mapping happens here.
pub fn annotate_segmentation(
    image: &DynamicImage,
    detections: &[SegmentationDetection],
) -> DynamicImage {
    let mut output = image.to_rgb8();
    for detection in detections {
        draw_mask_outline(
            &mut output,
            &detection.mask,
            class_color(detection.class_id),
        );
    }
    for detection in detections {
        let color = class_color(detection.class_id);
        draw_labeled_rect(
            &mut output,
            [
                detection.xmin as u32,
                detection.ymin as u32,
                detection.xmax as u32,
                detection.ymax as u32,
            ],
            color,
            &detection.class_name,
            detection.confidence,
        );
    }
    DynamicImage::ImageRgb8(output)
}

/// Stroke the boundary pixels of a source-space boolean mask.
fn draw_mask_outline(
    image: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
    mask: &InstanceMask,
    color: Rgb<u8>,
) {
    let width = mask.width as usize;
    let height = mask.height as usize;
    if width == 0 || height == 0 {
        return;
    }
    let covered = |x: usize, y: usize| mask.data[y * width + x];
    for y in 0..height {
        for x in 0..width {
            if !covered(x, y) {
                continue;
            }
            let boundary = (x == 0 || !covered(x - 1, y))
                || (x + 1 == width || !covered(x + 1, y))
                || (y == 0 || !covered(x, y - 1))
                || (y + 1 == height || !covered(x, y + 1));
            if boundary {
                image.put_pixel(x as u32, y as u32, color);
            }
        }
    }
}

fn class_color(class_id: usize) -> Rgb<u8> {
    const PALETTE: [[u8; 3]; 6] = [
        [239, 62, 5],
        [34, 197, 94],
        [59, 130, 246],
        [234, 179, 8],
        [168, 85, 247],
        [236, 72, 153],
    ];
    Rgb(PALETTE[class_id % PALETTE.len()])
}

fn draw_labeled_rect(
    image: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
    bounds: [u32; 4],
    color: Rgb<u8>,
    class_name: &str,
    confidence: f32,
) {
    let [x1, y1, x2, y2] = bounds;
    draw_rect(image, x1, y1, x2, y2, color);
    draw_label(image, x1, y1, color, class_name, confidence);
}

/// Draw a compact class-colored chip above the box, matching the visual convention used by
/// BoquilaHub. The chip falls inside the image at the top edge and shifts left at the right edge.
fn draw_label(
    image: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
    x: u32,
    y: u32,
    color: Rgb<u8>,
    class_name: &str,
    confidence: f32,
) {
    use font8x8::{BASIC_FONTS, LATIN_FONTS, UnicodeFonts};

    if image.width() < 16 || image.height() < 12 {
        return;
    }

    let scale = match image.width().min(image.height()) {
        0..=399 => 1,
        400..=899 => 2,
        _ => 3,
    };
    let padding = 2 * scale;
    let advance = 9 * scale;
    let max_chars = ((image.width() - 2 * padding) / advance).max(1) as usize;
    let text = label_text(class_name, confidence, max_chars);
    let text_chars = text.chars().count() as u32;
    let chip_width = (text_chars * advance + 2 * padding).min(image.width());
    let chip_height = 8 * scale + 2 * padding;
    if chip_height > image.height() {
        return;
    }

    let chip_x = x.min(image.width() - chip_width);
    let chip_y = if y >= chip_height {
        y - chip_height
    } else {
        y.min(image.height() - chip_height)
    };
    fill_rect(image, chip_x, chip_y, chip_width, chip_height, color);

    let white = Rgb([255, 255, 255]);
    let mut cursor_x = chip_x + padding;
    for character in text.chars() {
        let glyph = BASIC_FONTS
            .get(character)
            .or_else(|| LATIN_FONTS.get(character))
            .or_else(|| BASIC_FONTS.get('?'))
            .expect("the built-in bitmap font contains a question mark");
        for (row, bits) in glyph.iter().copied().enumerate() {
            for column in 0..8 {
                if bits & (1 << column) == 0 {
                    continue;
                }
                let glyph_x = cursor_x + column * scale;
                let glyph_y = chip_y + padding + row as u32 * scale;
                fill_rect(image, glyph_x, glyph_y, scale, scale, white);
            }
        }
        cursor_x += advance;
    }
}

fn label_text(class_name: &str, confidence: f32, max_chars: usize) -> String {
    let confidence = format!("{confidence:.2}");
    if max_chars <= confidence.len() {
        return confidence.chars().take(max_chars).collect();
    }

    let name_limit = max_chars - confidence.len() - 1;
    let mut name: String = class_name
        .chars()
        .map(|character| {
            if character.is_control() {
                '?'
            } else if character.is_whitespace() {
                ' '
            } else {
                character
            }
        })
        .take(name_limit)
        .collect();
    if class_name.chars().count() > name_limit && name_limit > 0 {
        name.pop();
        name.push('~');
    }
    format!("{name} {confidence}")
}

fn fill_rect(
    image: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    color: Rgb<u8>,
) {
    let right = x.saturating_add(width).min(image.width());
    let bottom = y.saturating_add(height).min(image.height());
    for pixel_y in y..bottom {
        for pixel_x in x..right {
            image.put_pixel(pixel_x, pixel_y, color);
        }
    }
}

fn draw_rect(
    image: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
    x1: u32,
    y1: u32,
    x2: u32,
    y2: u32,
    color: Rgb<u8>,
) {
    if image.width() == 0 || image.height() == 0 {
        return;
    }
    let (x1, x2) = (x1.min(image.width() - 1), x2.min(image.width() - 1));
    let (y1, y2) = (y1.min(image.height() - 1), y2.min(image.height() - 1));
    if x1 > x2 || y1 > y2 {
        return;
    }
    for x in x1..=x2 {
        image.put_pixel(x, y1, color);
        image.put_pixel(x, y2, color);
    }
    for y in y1..=y2 {
        image.put_pixel(x1, y, color);
        image.put_pixel(x2, y, color);
    }
}

/// Class names in the standard COCO-80 order used by the pretrained weights.
pub const COCO_CLASSES: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn annotation_labels_include_name_and_confidence_and_fit_the_canvas() {
        assert_eq!(label_text("person", 0.876, usize::MAX), "person 0.88");
        assert_eq!(label_text("motorcycle", 0.5, 8), "mo~ 0.50");

        let color = class_color(0);
        let mut image = ImageBuffer::from_pixel(200, 100, Rgb([0, 0, 0]));
        draw_label(&mut image, 190, 20, color, "person", 0.9);

        // The 103 px chip is shifted left so it remains fully visible at the right edge.
        assert_eq!(*image.get_pixel(97, 8), color);
        assert!(
            image
                .enumerate_pixels()
                .any(|(_, _, pixel)| *pixel == Rgb([255, 255, 255])),
            "label text should be drawn in white"
        );
    }

    /// Check that the production YOLOX artifact preserves the direct official-checkpoint result.
    /// Requires the external checkpoint and generated artifact under `target/`.
    #[cfg(feature = "pretrained")]
    #[test]
    #[ignore]
    fn yolox_nano_burnpack_matches_official_checkpoint_end_to_end() {
        let worker = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let device = Device::<Flex>::default();
                let mut official = Yolox::<Flex>::yolox_nano(COCO_CLASSES.len(), &device);
                official
                    .load_pytorch_weights("target/checkpoints/yolox_nano.pth")
                    .unwrap();
                let mut artifact = Yolox::<Flex>::yolox_nano(COCO_CLASSES.len(), &device);
                artifact
                    .load_burnpack_weights("target/yolox-nano.bpk")
                    .unwrap();

                let image = image::open("docs/dog_bike_man.jpg").unwrap();
                let prepared = LetterboxedImage::yolox(&image, 416);
                let input = image_to_tensor(prepared.image().clone(), &device).unsqueeze::<4>();
                let flatten = |batches: Vec<Vec<Vec<BoundingBox>>>| {
                    batches
                        .into_iter()
                        .next()
                        .unwrap()
                        .into_iter()
                        .enumerate()
                        .flat_map(|(class_id, boxes)| {
                            boxes.into_iter().map(move |bbox| (class_id, bbox))
                        })
                        .collect::<Vec<_>>()
                };
                let expected =
                    flatten(run_classic_detections(&official, input.clone(), 0.45, 0.25));
                let actual = flatten(run_classic_detections(&artifact, input, 0.45, 0.25));

                let classes = actual
                    .iter()
                    .map(|(class_id, _)| COCO_CLASSES[*class_id])
                    .collect::<std::collections::HashSet<_>>();
                assert!(
                    actual.len() <= 10,
                    "unexpectedly dense YOLOX output: {}",
                    actual.len()
                );
                for expected in ["person", "bicycle", "dog"] {
                    assert!(
                        classes.contains(expected),
                        "YOLOX missed {expected}: {classes:?}"
                    );
                }

                // The common f16 artifact policy can move candidates across the confidence and
                // NMS cutoffs. Require bidirectional agreement for the stable subset instead.
                for (name, references, candidates) in [
                    ("official", &expected[..], &actual[..]),
                    ("artifact", &actual[..], &expected[..]),
                ] {
                    for (class_id, reference) in references
                        .iter()
                        .filter(|(_, item)| item.confidence >= 0.65)
                    {
                        let matched = candidates.iter().any(|(candidate_class, candidate)| {
                            candidate_class == class_id
                                && (candidate.confidence - reference.confidence).abs() <= 0.03
                                && test_box_iou(
                                    (
                                        reference.xmin,
                                        reference.ymin,
                                        reference.xmax,
                                        reference.ymax,
                                    ),
                                    (
                                        candidate.xmin,
                                        candidate.ymin,
                                        candidate.xmax,
                                        candidate.ymax,
                                    ),
                                ) >= 0.90
                        });
                        assert!(
                            matched,
                            "no {name} match for class {class_id}, confidence {}",
                            reference.confidence
                        );
                    }
                }
            })
            .unwrap();
        worker.join().unwrap();
    }

    fn test_box_iou(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
        let width = (a.2.min(b.2) - a.0.max(b.0)).max(0.0);
        let height = (a.3.min(b.3) - a.1.max(b.1)).max(0.0);
        let intersection = width * height;
        let area_a = (a.2 - a.0) * (a.3 - a.1);
        let area_b = (b.2 - b.0) * (b.3 - b.1);
        intersection / (area_a + area_b - intersection)
    }

    /// Compare the seg runtime end to end against the official Ultralytics prediction on the
    /// reference image, including per-detection mask IoU in source-image space. Run the
    /// generator first:
    /// `python tools/export_yolo11_seg_e2e.py target/<id>.pt docs/dog_bike_man.jpg target --model <id>`
    #[cfg(feature = "pretrained")]
    macro_rules! seg_e2e_test {
        ($fn_name:ident, $model_id:expr, $id:literal) => {
            #[test]
            #[ignore]
            fn $fn_name() {
                let expected_path =
                    std::path::PathBuf::from(format!("target/{}-e2e-expected.json", $id));
                assert!(
                    expected_path.exists(),
                    "generate the official expectation with tools/export_yolo11_seg_e2e.py first"
                );
                #[derive(serde::Deserialize)]
                struct Expected {
                    detections: Vec<ExpectedDetection>,
                }
                #[derive(serde::Deserialize)]
                struct ExpectedDetection {
                    class_id: usize,
                    class_name: String,
                    confidence: f32,
                    box_xyxy_px: [f32; 4],
                    mask_file: String,
                }

                let expected: Expected =
                    serde_json::from_slice(&std::fs::read(&expected_path).unwrap()).unwrap();
                let checkpoint = std::path::PathBuf::from(format!("target/{}.bpk", $id));
                assert!(
                    checkpoint.exists(),
                    "pack the {} artifact with pack-weights first",
                    $id
                );
                let predictor =
                    Predictor::<Flex>::from_checkpoint($model_id, checkpoint, Default::default())
                        .unwrap();
                let (image, detections) = predictor
                    .predict_segmentation_path("docs/dog_bike_man.jpg")
                    .unwrap();
                let _ = image;
                assert_eq!(
                    detections.len(),
                    expected.detections.len(),
                    "detection count differs from Ultralytics"
                );

                let mut used = vec![false; expected.detections.len()];
                for detection in &detections {
                    // Match by class and best box IoU (both sides use the same decode, so the
                    // boxes are near-identical; the matching tolerates ordering).
                    let (index, match_iou) = expected
                        .detections
                        .iter()
                        .enumerate()
                        .filter(|(index, candidate)| {
                            !used[*index] && candidate.class_id == detection.class_id
                        })
                        .map(|(index, candidate)| {
                            (
                                index,
                                test_box_iou(
                                    (
                                        detection.xmin,
                                        detection.ymin,
                                        detection.xmax,
                                        detection.ymax,
                                    ),
                                    (
                                        candidate.box_xyxy_px[0],
                                        candidate.box_xyxy_px[1],
                                        candidate.box_xyxy_px[2],
                                        candidate.box_xyxy_px[3],
                                    ),
                                ),
                            )
                        })
                        .max_by(|a, b| a.1.total_cmp(&b.1))
                        .unwrap_or_else(|| {
                            panic!(
                                "no unmatched Ultralytics detection for {}",
                                detection.class_name
                            )
                        });
                    assert!(match_iou > 0.9, "box match IoU too low: {match_iou}");
                    used[index] = true;
                    let candidate = &expected.detections[index];

                    // Ultralytics' end2end (one2one) heads keep near-duplicate detections that
                    // classic NMS would suppress. The scores of those weak duplicates sit deep in
                    // the top-k near-tie region where f16 rounding reorders membership, so their
                    // confidence can move by far more than rounding on a stable detection (0.09
                    // observed on yolo26l-seg); duplicates are exempt from the confidence gate but
                    // still pass through the IoU and mask gates below.
                    let is_near_duplicate =
                        expected
                            .detections
                            .iter()
                            .enumerate()
                            .any(|(other, strong)| {
                                other != index
                                    && strong.class_id == candidate.class_id
                                    && strong.confidence > candidate.confidence
                                    && test_box_iou(
                                        (
                                            candidate.box_xyxy_px[0],
                                            candidate.box_xyxy_px[1],
                                            candidate.box_xyxy_px[2],
                                            candidate.box_xyxy_px[3],
                                        ),
                                        (
                                            strong.box_xyxy_px[0],
                                            strong.box_xyxy_px[1],
                                            strong.box_xyxy_px[2],
                                            strong.box_xyxy_px[3],
                                        ),
                                    ) >= 0.9
                            });
                    let confidence_delta = (detection.confidence - candidate.confidence).abs();
                    // f16 weight rounding shifts sigmoid scores by up to ~1% relative on the
                    // end-to-end heads (observed worst case 0.009 absolute on yolo26n-seg's dog);
                    // box IoU, the per-edge deltas, and the mask IoU below are the parity gates.
                    if !is_near_duplicate {
                        assert!(
                            confidence_delta <= 1.5e-2,
                            "{} confidence: {} vs {}",
                            detection.class_name,
                            detection.confidence,
                            candidate.confidence
                        );
                    }
                    // f16 weight rounding can flip a sharp (multi-peak) DFL side distribution and
                    // move a single box edge by a couple of pixels (observed worst case ~2.8 px);
                    // box IoU stays high and the mask IoU below is the real parity gate.
                    let box_iou = test_box_iou(
                        (
                            detection.xmin,
                            detection.ymin,
                            detection.xmax,
                            detection.ymax,
                        ),
                        (
                            candidate.box_xyxy_px[0],
                            candidate.box_xyxy_px[1],
                            candidate.box_xyxy_px[2],
                            candidate.box_xyxy_px[3],
                        ),
                    );
                    assert!(
                        box_iou >= 0.98,
                        "{} box IoU {box_iou} below 0.98",
                        detection.class_name
                    );
                    for (actual, expected_edge) in [
                        detection.xmin,
                        detection.ymin,
                        detection.xmax,
                        detection.ymax,
                    ]
                    .iter()
                    .zip(candidate.box_xyxy_px.iter())
                    {
                        assert!(
                            (actual - expected_edge).abs() <= 3.5,
                            "{} box edge {actual} vs {expected_edge}",
                            detection.class_name
                        );
                    }

                    let official = image::open(format!("target/{}", candidate.mask_file))
                        .unwrap()
                        .into_luma8();
                    assert_eq!(official.width(), detection.mask.width);
                    assert_eq!(official.height(), detection.mask.height);
                    let mut intersection = 0_u64;
                    let mut union = 0_u64;
                    for (actual, expected_pixel) in
                        detection.mask.data.iter().zip(official.pixels())
                    {
                        let expected_pixel = expected_pixel[0] > 0;
                        match (*actual, expected_pixel) {
                            (true, true) => intersection += 1,
                            (true, false) | (false, true) => union += 1,
                            (false, false) => {}
                        }
                    }
                    let mask_iou = intersection as f64 / (intersection + union) as f64;
                    println!(
                        "{:<14} conf={:.3} mask_px={} mask_IoU={:.4}",
                        detection.class_name,
                        detection.confidence,
                        detection.mask.data.iter().filter(|pixel| **pixel).count(),
                        mask_iou
                    );
                    // Small masks (hundreds of pixels) are dominated by their boundary: a couple
                    // of f16-rounded logit flips move their IoU far more than on an
                    // object-sized mask (0.92 observed on yolo26x-seg's 313-px "tie"). Tiny masks
                    // get a relaxed 0.85 gate; the 0.95 gate applies to object-sized masks.
                    let mask_iou_gate =
                        if detection.mask.data.iter().filter(|pixel| **pixel).count() < 2000 {
                            0.85
                        } else {
                            0.95
                        };
                    assert!(
                        mask_iou >= mask_iou_gate,
                        "{} mask IoU {mask_iou} below {mask_iou_gate}",
                        detection.class_name
                    );
                }

                // Every non-duplicate official detection at conf >= 0.55 must still be matched by
                // the runtime (see the duplicate exemption inside the loop above).
                let is_unmatched_duplicate = |index: usize| {
                    expected
                        .detections
                        .iter()
                        .enumerate()
                        .any(|(other, strong)| {
                            other != index
                                && strong.class_id == expected.detections[index].class_id
                                && strong.confidence > expected.detections[index].confidence
                                && test_box_iou(
                                    (
                                        expected.detections[index].box_xyxy_px[0],
                                        expected.detections[index].box_xyxy_px[1],
                                        expected.detections[index].box_xyxy_px[2],
                                        expected.detections[index].box_xyxy_px[3],
                                    ),
                                    (
                                        strong.box_xyxy_px[0],
                                        strong.box_xyxy_px[1],
                                        strong.box_xyxy_px[2],
                                        strong.box_xyxy_px[3],
                                    ),
                                ) >= 0.9
                        })
                };
                for (index, candidate) in expected.detections.iter().enumerate() {
                    if !used[index]
                        && candidate.confidence >= 0.55
                        && !is_unmatched_duplicate(index)
                    {
                        panic!(
                            "strong Ultralytics detection {} at conf {:.3} was not matched",
                            candidate.class_name, candidate.confidence
                        );
                    }
                }
            }
        };
    }

    seg_e2e_test!(
        yolo11n_seg_matches_ultralytics_end_to_end,
        ModelId::Yolo11NSeg,
        "yolo11n-seg"
    );
    seg_e2e_test!(
        yolo11s_seg_matches_ultralytics_end_to_end,
        ModelId::Yolo11SSeg,
        "yolo11s-seg"
    );
    seg_e2e_test!(
        yolo11m_seg_matches_ultralytics_end_to_end,
        ModelId::Yolo11MSeg,
        "yolo11m-seg"
    );
    seg_e2e_test!(
        yolo11l_seg_matches_ultralytics_end_to_end,
        ModelId::Yolo11LSeg,
        "yolo11l-seg"
    );
    seg_e2e_test!(
        yolo11x_seg_matches_ultralytics_end_to_end,
        ModelId::Yolo11XSeg,
        "yolo11x-seg"
    );
    seg_e2e_test!(
        yolov8n_seg_matches_ultralytics_end_to_end,
        ModelId::Yolov8NSeg,
        "yolov8n-seg"
    );
    seg_e2e_test!(
        yolov8s_seg_matches_ultralytics_end_to_end,
        ModelId::Yolov8SSeg,
        "yolov8s-seg"
    );
    seg_e2e_test!(
        yolov8m_seg_matches_ultralytics_end_to_end,
        ModelId::Yolov8MSeg,
        "yolov8m-seg"
    );
    seg_e2e_test!(
        yolov8l_seg_matches_ultralytics_end_to_end,
        ModelId::Yolov8LSeg,
        "yolov8l-seg"
    );
    seg_e2e_test!(
        yolov8x_seg_matches_ultralytics_end_to_end,
        ModelId::Yolov8XSeg,
        "yolov8x-seg"
    );
    seg_e2e_test!(
        yolo26n_seg_matches_ultralytics_end_to_end,
        ModelId::Yolo26NSeg,
        "yolo26n-seg"
    );
    seg_e2e_test!(
        yolo26s_seg_matches_ultralytics_end_to_end,
        ModelId::Yolo26SSeg,
        "yolo26s-seg"
    );
    seg_e2e_test!(
        yolo26m_seg_matches_ultralytics_end_to_end,
        ModelId::Yolo26MSeg,
        "yolo26m-seg"
    );
    seg_e2e_test!(
        yolo26l_seg_matches_ultralytics_end_to_end,
        ModelId::Yolo26LSeg,
        "yolo26l-seg"
    );
    seg_e2e_test!(
        yolo26x_seg_matches_ultralytics_end_to_end,
        ModelId::Yolo26XSeg,
        "yolo26x-seg"
    );
}
