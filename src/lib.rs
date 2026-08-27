//! A small, end-to-end object detection API built on [Burn](https://burn.dev).
//!
//! The stable MVP supports YOLOX-Nano trained on COCO, with experimental native
//! YOLOv3-Tiny-Ultralytics and YOLOv10n inference paths. Model inference and post-processing run
//! from Rust; no Python runtime or ONNX runtime is involved.

extern crate alloc;

mod data;
pub mod models;

#[cfg(feature = "pretrained")]
use std::path::PathBuf;
use std::{error::Error, fmt, path::Path, str::FromStr};

use crate::data::LetterboxedImage;
use crate::models::yolov3_tiny::Yolov3Tiny;
#[cfg(feature = "pretrained")]
use crate::models::yolov3_tiny::Yolov3TinyConfig;
#[cfg(feature = "pretrained")]
use crate::models::yolov10::Yolov10Config;
use crate::models::yolov10::{Yolov10, head::MAX_DETECTIONS as YOLOV10_MAX_DETECTIONS};
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
}

impl ModelId {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::YoloxNano => "yolox-nano",
            Self::Yolov3TinyU => "yolov3-tinyu",
            Self::Yolov10N => "yolov10n",
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
            _ => Err(format!(
                "unknown model '{value}'; available models: yolox-nano, yolov3-tinyu, yolov10n"
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
enum RuntimeModel {
    Yolox(Box<Yolox<Flex>>),
    Yolov3Tiny(Box<Yolov3Tiny<Flex>>),
    Yolov10(Box<Yolov10<Flex>>),
}

/// A ready-to-run object detector using Burn's flexible backend.
pub struct Predictor {
    model_id: ModelId,
    model: RuntimeModel,
    device: Device<Flex>,
    options: PredictOptions,
}

pub type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

impl Predictor {
    /// Load a catalog model with its official pretrained weights.
    #[cfg(feature = "pretrained")]
    pub fn new(model_id: ModelId, options: PredictOptions) -> Result<Self> {
        match model_id {
            ModelId::YoloxNano => Self::load_yolox_nano(model_id, options),
            ModelId::Yolov3TinyU | ModelId::Yolov10N => Err(format!(
                "{} currently requires --weights with a boquilens .bpk artifact; see the README's one-time weight preparation",
                model_id
            )
            .into()),
        }
    }

    /// Load a model from a supported PyTorch state checkpoint on disk.
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
        let options = options.validate()?;
        let checkpoint = checkpoint.into();
        match model_id {
            ModelId::YoloxNano => {
                let worker = std::thread::Builder::new()
                    .name("boquilens-model-loader".into())
                    .stack_size(64 * 1024 * 1024)
                    .spawn(move || {
                        let device = Default::default();
                        let mut model: Yolox<Flex> = Yolox::yolox_nano(COCO_CLASSES.len(), &device);
                        model.load_pytorch_weights(checkpoint)?;
                        Ok::<_, Box<dyn Error + Send + Sync>>((
                            RuntimeModel::Yolox(Box::new(model)),
                            device,
                        ))
                    })?;
                let (model, device) = worker
                    .join()
                    .map_err(|_| "YOLOX model loader thread panicked")??;
                Ok(Self {
                    model_id,
                    model,
                    device,
                    options,
                })
            }
            ModelId::Yolov3TinyU => {
                let worker = std::thread::Builder::new()
                    .name("boquilens-model-loader".into())
                    .stack_size(64 * 1024 * 1024)
                    .spawn(move || {
                        let device = Default::default();
                        let mut model = Yolov3TinyConfig.init::<Flex>(&device);
                        if checkpoint.extension().and_then(|value| value.to_str()) == Some("bpk") {
                            model.load_burnpack_weights(&checkpoint)?;
                        } else {
                            model.load_pytorch_weights(&checkpoint)?;
                        }
                        Ok::<_, Box<dyn Error + Send + Sync>>((
                            RuntimeModel::Yolov3Tiny(Box::new(model)),
                            device,
                        ))
                    })?;
                let (model, device) = worker
                    .join()
                    .map_err(|_| "YOLOv3-Tiny-U model loader thread panicked")??;
                Ok(Self {
                    model_id,
                    model,
                    device,
                    options,
                })
            }
            ModelId::Yolov10N => {
                let worker = std::thread::Builder::new()
                    .name("boquilens-model-loader".into())
                    .stack_size(64 * 1024 * 1024)
                    .spawn(move || {
                        let device = Default::default();
                        let mut model = Yolov10Config.init::<Flex>(&device);
                        if checkpoint.extension().and_then(|value| value.to_str()) == Some("bpk") {
                            model.load_burnpack_weights(&checkpoint)?;
                        } else {
                            model.load_pytorch_weights(&checkpoint)?;
                        }
                        Ok::<_, Box<dyn Error + Send + Sync>>((
                            RuntimeModel::Yolov10(Box::new(model)),
                            device,
                        ))
                    })?;
                let (model, device) = worker
                    .join()
                    .map_err(|_| "YOLOv10n model loader thread panicked")??;
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
    fn load_yolox_nano(model_id: ModelId, options: PredictOptions) -> Result<Self> {
        let options = options.validate()?;
        // Constructing this deeply nested module can exceed the small default main-thread stack
        // on Windows in debug builds. Keep that platform detail out of the public API.
        let worker = std::thread::Builder::new()
            .name("boquilens-model-loader".into())
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let device = Default::default();
                let model: Yolox<Flex> =
                    Yolox::yolox_nano_pretrained(weights::YoloxNano::Coco, &device)?;
                Ok::<_, Box<dyn Error + Send + Sync>>((
                    RuntimeModel::Yolox(Box::new(model)),
                    device,
                ))
            })?;
        let (model, device) = worker
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
            RuntimeModel::Yolov3Tiny(_) | RuntimeModel::Yolov10(_) => {
                LetterboxedImage::ultralytics(image, INPUT_SIZE, 32)
            }
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
            RuntimeModel::Yolov10(model) => {
                // Same RGB normalization as YOLOv3-Tiny-U. The one2one head is trained NMS-free;
                // official inference selects the top-scoring detections and filters by
                // confidence without non-maximum suppression.
                let output = model.forward(input / 255.0);
                yolov10_topk_detections(output, YOLOV10_MAX_DETECTIONS, self.options.confidence)
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
            "native weight packing is currently implemented only for yolov3-tinyu and yolov10n"
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
            match model_id {
                ModelId::Yolov3TinyU => {
                    let mut model = Yolov3TinyConfig.init::<Flex>(&device);
                    model.load_pytorch_weights(&input)?;
                    model.save_burnpack_weights(&output)?;
                }
                ModelId::Yolov10N => {
                    let mut model = Yolov10Config.init::<Flex>(&device);
                    model.load_pytorch_weights(&input)?;
                    model.save_burnpack_weights(&output)?;
                }
                ModelId::YoloxNano => {
                    return Err(
                        "native weight packing is currently implemented only for yolov3-tinyu and yolov10n"
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

/// Select the strongest detections from decoded YOLOv10 one2one predictions.
///
/// This mirrors Ultralytics' end-to-end head postprocess: keep the `max_detections` anchors with
/// the highest best-class score, then keep the `max_detections` strongest (anchor, class) pairs
/// among them, and finally apply the confidence threshold. No non-maximum suppression is applied
/// because the one2one head is trained to emit one prediction per object.
fn yolov10_topk_detections(
    predictions: crate::models::yolov10::head::DecodedPredictions<Flex>,
    max_detections: usize,
    confidence_threshold: f32,
) -> Vec<Vec<Vec<BoundingBox>>> {
    let [batch, anchors, classes] = predictions.scores.dims();
    let keep = max_detections.min(anchors);
    let boxes: Vec<f32> = predictions.boxes.into_data().iter::<f32>().collect();
    let scores: Vec<f32> = predictions.scores.into_data().iter::<f32>().collect();

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
        assert!("yolo26".parse::<ModelId>().is_err());
    }

    #[cfg(feature = "pretrained")]
    #[test]
    fn native_weight_packer_rejects_unsupported_model_and_extension() {
        assert!(
            pack_weights(ModelId::YoloxNano, "unused.pt", "unused.bpk")
                .unwrap_err()
                .to_string()
                .contains("only for yolov3-tinyu and yolov10n")
        );
        assert!(
            pack_weights(ModelId::Yolov3TinyU, "unused.pt", "unused.bin")
                .unwrap_err()
                .to_string()
                .contains(".bpk extension")
        );
        assert!(
            pack_weights(ModelId::Yolov10N, "unused.pt", "unused.bin")
                .unwrap_err()
                .to_string()
                .contains(".bpk extension")
        );
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
