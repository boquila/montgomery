//! A small, end-to-end object detection API built on [Burn](https://burn.dev).
//!
//! The stable MVP supports YOLOX-Nano trained on COCO, with experimental native
//! YOLOv3-Tiny-Ultralytics, YOLOv10 (n/s/m/b/l/x), and YOLO26 (n/s/m/l/x) inference paths. Model
//! inference and post-processing run from Rust — on the Flex CPU backend by default, or on the
//! Wgpu GPU backend (Vulkan/DX12/Metal) when built with the `gpu` feature. No Python runtime or
//! ONNX runtime is involved.

extern crate alloc;

mod data;
pub mod models;

#[cfg(feature = "pretrained")]
use std::path::PathBuf;
use std::{error::Error, fmt, path::Path, str::FromStr};

use crate::data::LetterboxedImage;
use crate::models::yolo26::head::MAX_DETECTIONS as YOLO26_MAX_DETECTIONS;
#[cfg(feature = "pretrained")]
use crate::models::yolo26::{
    Yolo26LConfig, Yolo26MConfig, Yolo26NConfig, Yolo26SConfig, Yolo26XConfig,
};
use crate::models::yolov3_tiny::Yolov3Tiny;
#[cfg(feature = "pretrained")]
use crate::models::yolov3_tiny::Yolov3TinyConfig;
use crate::models::yolov10::head::MAX_DETECTIONS as YOLOV10_MAX_DETECTIONS;
#[cfg(feature = "pretrained")]
use crate::models::yolov10::{
    Yolov10BConfig, Yolov10LConfig, Yolov10MConfig, Yolov10NConfig, Yolov10SConfig, Yolov10XConfig,
};
#[cfg(feature = "pretrained")]
use crate::models::yolox::weights;
use crate::models::yolox::{Yolox, boxes::BoundingBox, boxes::nms};
use burn::tensor::{Device, Tensor, TensorData, backend::Backend};
use burn_flex::Flex;
use image::{DynamicImage, ImageBuffer, Rgb};
use serde::Serialize;
#[cfg(feature = "pretrained")]
use sha2::{Digest, Sha256};

/// The square input size used by the currently supported pretrained models.
pub const INPUT_SIZE: usize = 640;

/// Stable identifier for a model architecture/scale in the boquilens catalog.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelId {
    #[default]
    YoloxNano,
    Yolov3TinyU,
    Yolov10N,
    Yolov10S,
    Yolov10M,
    Yolov10B,
    Yolov10L,
    Yolov10X,
    Yolo26N,
    Yolo26S,
    Yolo26M,
    Yolo26L,
    Yolo26X,
}

impl ModelId {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::YoloxNano => "yolox-nano",
            Self::Yolov3TinyU => "yolov3-tinyu",
            Self::Yolov10N => "yolov10n",
            Self::Yolov10S => "yolov10s",
            Self::Yolov10M => "yolov10m",
            Self::Yolov10B => "yolov10b",
            Self::Yolov10L => "yolov10l",
            Self::Yolov10X => "yolov10x",
            Self::Yolo26N => "yolo26n",
            Self::Yolo26S => "yolo26s",
            Self::Yolo26M => "yolo26m",
            Self::Yolo26L => "yolo26l",
            Self::Yolo26X => "yolo26x",
        }
    }
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
            "yolov3-tinyu" | "yolov3_tinyu" => Ok(Self::Yolov3TinyU),
            "yolov10n" | "yolov10-nano" => Ok(Self::Yolov10N),
            "yolov10s" | "yolov10-small" => Ok(Self::Yolov10S),
            "yolov10m" | "yolov10-medium" => Ok(Self::Yolov10M),
            "yolov10b" | "yolov10-balanced" => Ok(Self::Yolov10B),
            "yolov10l" | "yolov10-large" => Ok(Self::Yolov10L),
            "yolov10x" | "yolov10-xlarge" => Ok(Self::Yolov10X),
            "yolo26n" | "yolo26-nano" => Ok(Self::Yolo26N),
            "yolo26s" | "yolo26-small" => Ok(Self::Yolo26S),
            "yolo26m" | "yolo26-medium" => Ok(Self::Yolo26M),
            "yolo26l" | "yolo26-large" => Ok(Self::Yolo26L),
            "yolo26x" | "yolo26-xlarge" => Ok(Self::Yolo26X),
            _ => Err(format!(
                "unknown model '{value}'; available models: yolox-nano, yolov3-tinyu, \
                 yolov10n/s/m/b/l/x, yolo26n/s/m/l/x"
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
    pub class_name: &'static str,
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
    Yolo26N(Box<crate::models::yolo26::Yolo26N<B>>),
    Yolo26S(Box<crate::models::yolo26::Yolo26S<B>>),
    Yolo26M(Box<crate::models::yolo26::Yolo26M<B>>),
    Yolo26L(Box<crate::models::yolo26::Yolo26L<B>>),
    Yolo26X(Box<crate::models::yolo26::Yolo26X<B>>),
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
}

pub type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

/// Construct the requested Ultralytics-family model inside a large-stack worker and load either
/// its native Burnpack artifact or its tensor-only PyTorch state.
///
/// Deep module construction overflows the small default main-thread stack on Windows in debug
/// builds, so the graph is built in the worker thread.
/// A model construction-and-load closure handed to the large-stack loader worker.
#[cfg(feature = "pretrained")]
type ModelLoader<B> = Box<dyn FnOnce(&Device<B>) -> Result<RuntimeModel<B>> + Send>;

#[cfg(feature = "pretrained")]
fn load_ultralytics_checkpoint<B: Backend>(
    model_id: ModelId,
    checkpoint: PathBuf,
    device: Device<B>,
) -> Result<RuntimeModel<B>> {
    macro_rules! load_variant {
        ($config:ty, $variant:path) => {
            move |device: &Device<B>| -> Result<RuntimeModel<B>> {
                let mut model = <$config>::default().init::<B>(device);
                if checkpoint.extension().and_then(|value| value.to_str()) == Some("bpk") {
                    model.load_burnpack_weights(&checkpoint)?;
                } else {
                    model.load_pytorch_weights(&checkpoint)?;
                }
                Ok($variant(Box::new(model)))
            }
        };
    }
    let loader: ModelLoader<B> = match model_id {
        ModelId::Yolov3TinyU => Box::new(load_variant!(Yolov3TinyConfig, RuntimeModel::Yolov3Tiny)),
        ModelId::Yolov10N => Box::new(load_variant!(Yolov10NConfig, RuntimeModel::Yolov10N)),
        ModelId::Yolov10S => Box::new(load_variant!(Yolov10SConfig, RuntimeModel::Yolov10S)),
        ModelId::Yolov10M => Box::new(load_variant!(Yolov10MConfig, RuntimeModel::Yolov10M)),
        ModelId::Yolov10B => Box::new(load_variant!(Yolov10BConfig, RuntimeModel::Yolov10B)),
        ModelId::Yolov10L => Box::new(load_variant!(Yolov10LConfig, RuntimeModel::Yolov10L)),
        ModelId::Yolov10X => Box::new(load_variant!(Yolov10XConfig, RuntimeModel::Yolov10X)),
        ModelId::Yolo26N => Box::new(load_variant!(Yolo26NConfig, RuntimeModel::Yolo26N)),
        ModelId::Yolo26S => Box::new(load_variant!(Yolo26SConfig, RuntimeModel::Yolo26S)),
        ModelId::Yolo26M => Box::new(load_variant!(Yolo26MConfig, RuntimeModel::Yolo26M)),
        ModelId::Yolo26L => Box::new(load_variant!(Yolo26LConfig, RuntimeModel::Yolo26L)),
        ModelId::Yolo26X => Box::new(load_variant!(Yolo26XConfig, RuntimeModel::Yolo26X)),
        ModelId::YoloxNano => {
            return Err("YOLOX accepts its official .pth checkpoint directly and is loaded without the Ultralytics state bridge".into());
        }
    };
    let worker = std::thread::Builder::new()
        .name("boquilens-model-loader".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || loader(&device))?;
    worker
        .join()
        .map_err(|_| format!("{model_id} model loader thread panicked"))?
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
    let (boxes, scores) = model.detect(input / 255.0);
    end2end_topk_detections(boxes, scores, max_detections, confidence_threshold)
}

impl<B: Backend> Predictor<B> {
    /// Load a catalog model with its official pretrained weights on the backend's default device.
    #[cfg(feature = "pretrained")]
    pub fn new(model_id: ModelId, options: PredictOptions) -> Result<Self> {
        Self::new_on_device(model_id, options, Device::<B>::default())
    }

    /// Load a catalog model with its official pretrained weights on an explicit device.
    #[cfg(feature = "pretrained")]
    pub fn new_on_device(
        model_id: ModelId,
        options: PredictOptions,
        device: Device<B>,
    ) -> Result<Self> {
        let options = options.validate()?;
        match model_id {
            ModelId::YoloxNano => Self::load_yolox_nano_on_device(model_id, options, device),
            ModelId::Yolov3TinyU
            | ModelId::Yolov10N
            | ModelId::Yolov10S
            | ModelId::Yolov10M
            | ModelId::Yolov10B
            | ModelId::Yolov10L
            | ModelId::Yolov10X
            | ModelId::Yolo26N
            | ModelId::Yolo26S
            | ModelId::Yolo26M
            | ModelId::Yolo26L
            | ModelId::Yolo26X => Err(format!(
                "{} currently requires --weights with a boquilens .bpk artifact; see the README's one-time weight preparation",
                model_id
            )
            .into()),
        }
    }

    /// Load a model from a supported PyTorch state checkpoint on the backend's default device.
    ///
    /// YOLOX accepts its official checkpoint directly. The Ultralytics-family models require the
    /// tensor-only state produced by `tools/export_ultralytics_state.py` because Burn's pickle
    /// reader cannot deserialize the Python model objects stored in a full Ultralytics checkpoint.
    #[cfg(feature = "pretrained")]
    pub fn from_checkpoint(
        model_id: ModelId,
        checkpoint: impl Into<PathBuf>,
        options: PredictOptions,
    ) -> Result<Self> {
        Self::from_checkpoint_on_device(model_id, checkpoint, Device::<B>::default(), options)
    }

    /// Load a model from a supported PyTorch state checkpoint on an explicit device.
    #[cfg(feature = "pretrained")]
    pub fn from_checkpoint_on_device(
        model_id: ModelId,
        checkpoint: impl Into<PathBuf>,
        device: Device<B>,
        options: PredictOptions,
    ) -> Result<Self> {
        let options = options.validate()?;
        let checkpoint = checkpoint.into();
        match model_id {
            ModelId::YoloxNano => {
                let worker = std::thread::Builder::new()
                    .name("boquilens-model-loader".into())
                    .stack_size(64 * 1024 * 1024)
                    .spawn({
                        let device = device.clone();
                        move || {
                            let mut model: Yolox<B> =
                                Yolox::yolox_nano(COCO_CLASSES.len(), &device);
                            model.load_pytorch_weights(checkpoint)?;
                            Ok::<_, Box<dyn Error + Send + Sync>>(RuntimeModel::Yolox(Box::new(
                                model,
                            )))
                        }
                    })?;
                let model = worker
                    .join()
                    .map_err(|_| "YOLOX model loader thread panicked")??;
                Ok(Self {
                    model_id,
                    model,
                    device,
                    options,
                })
            }
            _ => {
                let model = load_ultralytics_checkpoint(model_id, checkpoint, device.clone())?;
                Ok(Self {
                    model_id,
                    model,
                    device,
                    options,
                })
            }
        }
    }

    /// Load pretrained COCO weights, downloading them to the model cache on first use.
    #[cfg(feature = "pretrained")]
    pub fn yolox_nano(options: PredictOptions) -> Result<Self> {
        Self::new(ModelId::YoloxNano, options)
    }

    #[cfg(feature = "pretrained")]
    fn load_yolox_nano_on_device(
        model_id: ModelId,
        options: PredictOptions,
        device: Device<B>,
    ) -> Result<Self> {
        // Constructing this deeply nested module can exceed the small default main-thread stack
        // on Windows in debug builds. Keep that platform detail out of the public API.
        let worker = std::thread::Builder::new()
            .name("boquilens-model-loader".into())
            .stack_size(64 * 1024 * 1024)
            .spawn({
                let device = device.clone();
                move || {
                    let model: Yolox<B> =
                        Yolox::yolox_nano_pretrained(weights::YoloxNano::Coco, &device)?;
                    Ok::<_, Box<dyn Error + Send + Sync>>(RuntimeModel::Yolox(Box::new(model)))
                }
            })?;
        let model = worker
            .join()
            .map_err(|_| "YOLOX model loader thread panicked")??;
        Ok(Self {
            model_id,
            model,
            device,
            options,
        })
    }

    /// The stable catalog identifier for the loaded model.
    pub const fn model_id(&self) -> ModelId {
        self.model_id
    }

    /// Run object detection on an already-decoded image.
    pub fn predict(&self, image: &DynamicImage) -> Vec<Detection> {
        let prepared = match &self.model {
            RuntimeModel::Yolox(_) => LetterboxedImage::yolox(image, INPUT_SIZE),
            RuntimeModel::Yolov3Tiny(_)
            | RuntimeModel::Yolov10N(_)
            | RuntimeModel::Yolov10S(_)
            | RuntimeModel::Yolov10M(_)
            | RuntimeModel::Yolov10B(_)
            | RuntimeModel::Yolov10L(_)
            | RuntimeModel::Yolov10X(_)
            | RuntimeModel::Yolo26N(_)
            | RuntimeModel::Yolo26S(_)
            | RuntimeModel::Yolo26M(_)
            | RuntimeModel::Yolo26L(_)
            | RuntimeModel::Yolo26X(_) => LetterboxedImage::ultralytics(image, INPUT_SIZE, 32),
        };
        let input = image_to_tensor(prepared.image().clone(), &self.device).unsqueeze::<4>();
        let boxes_by_class = match &self.model {
            RuntimeModel::Yolox(model) => {
                let output = model.forward(input);
                let [_, num_boxes, num_outputs] = output.dims();
                let boxes = output.clone().slice([0..1, 0..num_boxes, 0..4]);
                let objectness = output.clone().slice([0..1, 0..num_boxes, 4..5]);
                let class_scores = output.slice([0..1, 0..num_boxes, 5..num_outputs]);
                nms(
                    boxes,
                    class_scores * objectness,
                    self.options.iou,
                    self.options.confidence,
                )
            }
            RuntimeModel::Yolov3Tiny(model) => {
                // Ultralytics inference consumes RGB values in [0, 1]. YOLOX's transform above
                // intentionally consumes the original [0, 255] range, so normalization belongs
                // in this model-specific branch.
                let output = model.forward(input / 255.0);
                let [batch, anchors, _] = output.boxes.dims();
                let left_top = output.boxes.clone().slice([0..batch, 0..anchors, 0..2]);
                let right_bottom = output.boxes.slice([0..batch, 0..anchors, 2..4]);
                let center = (left_top.clone() + right_bottom.clone()) / 2.0;
                let size = right_bottom - left_top;
                nms(
                    Tensor::cat(vec![center, size], 2),
                    output.scores,
                    self.options.iou,
                    self.options.confidence,
                )
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
        };

        let mut detections = Vec::new();

        for (class_id, class_boxes) in boxes_by_class[0].iter().enumerate() {
            for bbox in class_boxes {
                let [xmin, ymin, xmax, ymax] =
                    prepared.to_source_box([bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]);
                detections.push(Detection {
                    class_id,
                    class_name: COCO_CLASSES[class_id],
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
}

/// Convert imported Ultralytics-family tensor state into boquilens' versioned native Burnpack
/// format.
///
/// The input is the tensor-only state generated by `tools/export_ultralytics_state.py`. The output
/// must end in `.bpk` and is stored with half-precision tensors, matching the precision of the
/// official checkpoint. Existing output files are never overwritten.
#[cfg(feature = "pretrained")]
#[derive(Debug)]
pub struct PackedWeights {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
}

#[cfg(feature = "pretrained")]
pub fn pack_weights(
    model_id: ModelId,
    input: impl Into<PathBuf>,
    output: impl Into<PathBuf>,
) -> Result<PackedWeights> {
    if matches!(model_id, ModelId::YoloxNano) {
        return Err(
            "native weight packing is currently implemented only for the Ultralytics-family models \
             (yolov3-tinyu, yolov10n/s/m/b/l/x, and yolo26n/s/m/l/x)"
                .into(),
        );
    }
    let input = input.into();
    let output = output.into();
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
        .name("boquilens-weight-packer".into())
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
            match model_id {
                ModelId::Yolov3TinyU => pack_variant!(Yolov3TinyConfig),
                ModelId::Yolov10N => pack_variant!(Yolov10NConfig),
                ModelId::Yolov10S => pack_variant!(Yolov10SConfig),
                ModelId::Yolov10M => pack_variant!(Yolov10MConfig),
                ModelId::Yolov10B => pack_variant!(Yolov10BConfig),
                ModelId::Yolov10L => pack_variant!(Yolov10LConfig),
                ModelId::Yolov10X => pack_variant!(Yolov10XConfig),
                ModelId::Yolo26N => pack_variant!(Yolo26NConfig),
                ModelId::Yolo26S => pack_variant!(Yolo26SConfig),
                ModelId::Yolo26M => pack_variant!(Yolo26MConfig),
                ModelId::Yolo26L => pack_variant!(Yolo26LConfig),
                ModelId::Yolo26X => pack_variant!(Yolo26XConfig),
                ModelId::YoloxNano => {
                    return Err(
                        "native weight packing is currently implemented only for the Ultralytics-family models \
                         (yolov3-tinyu, yolov10n/s/m/b/l/x, and yolo26n/s/m/l/x)"
                            .into(),
                    );
                }
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
    let shape = [rgb.height() as usize, rgb.width() as usize, 3];
    Tensor::<B, 3>::from_data(
        TensorData::new(rgb.into_raw(), shape).convert::<B::FloatElem>(),
        device,
    )
    .permute([2, 0, 1])
}

/// Select the strongest detections from decoded end-to-end one2one predictions.
///
/// This mirrors Ultralytics' end-to-end head postprocess, shared by YOLOv10 and YOLO26: keep the
/// `max_detections` anchors with the highest best-class score, then keep the `max_detections`
/// strongest (anchor, class) pairs among them, and finally apply the confidence threshold. No
/// non-maximum suppression is applied because the one2one head is trained to emit one prediction
/// per object.
fn end2end_topk_detections<B: Backend>(
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

        let best_scores = (0..anchors)
            .map(|anchor| {
                let row = &image_scores[anchor * classes..(anchor + 1) * classes];
                row.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            })
            .collect::<Vec<_>>();
        let mut anchor_order = (0..anchors).collect::<Vec<_>>();
        anchor_order.sort_unstable_by(|&a, &b| best_scores[b].total_cmp(&best_scores[a]));
        anchor_order.truncate(keep);

        let mut candidates = Vec::with_capacity(keep * classes);
        for (selected_index, &anchor) in anchor_order.iter().enumerate() {
            for class in 0..classes {
                candidates.push((
                    image_scores[anchor * classes + class],
                    selected_index,
                    class,
                ));
            }
        }
        candidates.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        candidates.truncate(keep);

        let mut per_class = (0..classes).map(|_| Vec::new()).collect::<Vec<_>>();
        for (score, selected_index, class) in candidates {
            if score < confidence_threshold {
                continue;
            }
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

/// Draw detection rectangles on a copy of the image.
pub fn annotate(image: &DynamicImage, detections: &[Detection]) -> DynamicImage {
    let mut output = image.to_rgb8();
    for detection in detections {
        let color = class_color(detection.class_id);
        draw_rect(
            &mut output,
            detection.xmin as u32,
            detection.ymin as u32,
            detection.xmax as u32,
            detection.ymax as u32,
            color,
        );
    }
    DynamicImage::ImageRgb8(output)
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
    fn rejects_invalid_thresholds() {
        assert!(
            PredictOptions {
                confidence: -0.1,
                iou: 0.5
            }
            .validate()
            .is_err()
        );
        assert!(
            PredictOptions {
                confidence: 0.5,
                iou: 1.1
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn parses_stable_model_names() {
        assert_eq!("yolox-nano".parse(), Ok(ModelId::YoloxNano));
        assert_eq!("nano".parse(), Ok(ModelId::YoloxNano));
        assert_eq!("yolov3-tinyu".parse(), Ok(ModelId::Yolov3TinyU));
        assert_eq!("yolov10n".parse(), Ok(ModelId::Yolov10N));
        assert_eq!("yolov10s".parse(), Ok(ModelId::Yolov10S));
        assert_eq!("yolov10m".parse(), Ok(ModelId::Yolov10M));
        assert_eq!("yolov10b".parse(), Ok(ModelId::Yolov10B));
        assert_eq!("yolov10l".parse(), Ok(ModelId::Yolov10L));
        assert_eq!("yolov10x".parse(), Ok(ModelId::Yolov10X));
        assert_eq!("yolo26n".parse(), Ok(ModelId::Yolo26N));
        assert_eq!("yolo26s".parse(), Ok(ModelId::Yolo26S));
        assert_eq!("yolo26m".parse(), Ok(ModelId::Yolo26M));
        assert_eq!("yolo26l".parse(), Ok(ModelId::Yolo26L));
        assert_eq!("yolo26x".parse(), Ok(ModelId::Yolo26X));
        assert!("yolo26".parse::<ModelId>().is_err());
    }

    #[cfg(feature = "pretrained")]
    #[test]
    fn native_weight_packer_rejects_unsupported_model_and_extension() {
        assert!(
            pack_weights(ModelId::YoloxNano, "unused.pt", "unused.bpk")
                .unwrap_err()
                .to_string()
                .contains("Ultralytics-family models")
        );
        for model_id in [
            ModelId::Yolov3TinyU,
            ModelId::Yolov10N,
            ModelId::Yolov10S,
            ModelId::Yolov10M,
            ModelId::Yolov10B,
            ModelId::Yolov10L,
            ModelId::Yolov10X,
            ModelId::Yolo26N,
            ModelId::Yolo26S,
            ModelId::Yolo26M,
            ModelId::Yolo26L,
            ModelId::Yolo26X,
        ] {
            assert!(
                pack_weights(model_id, "unused.pt", "unused.bin")
                    .unwrap_err()
                    .to_string()
                    .contains(".bpk extension")
            );
        }
    }

    #[test]
    fn annotation_draws_expected_border() {
        let input = DynamicImage::new_rgb8(10, 10);
        let detection = Detection {
            class_id: 0,
            class_name: "person",
            confidence: 0.9,
            xmin: 2.0,
            ymin: 3.0,
            xmax: 7.0,
            ymax: 8.0,
        };
        let output = annotate(&input, &[detection]).to_rgb8();
        assert_eq!(*output.get_pixel(2, 3), class_color(0));
        assert_eq!(*output.get_pixel(7, 8), class_color(0));
        assert_eq!(*output.get_pixel(4, 5), Rgb([0, 0, 0]));
    }
}
