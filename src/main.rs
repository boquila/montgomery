use std::path::PathBuf;

use boquilens::{
    ModelId, PredictOptions, Predictor, annotate, annotate_segmentation, pack_weights,
};
#[cfg(feature = "gpu")]
use burn::backend::Wgpu;
use burn::tensor::{Device, backend::Backend};
use burn_flex::Flex;
use clap::{Args as ClapArgs, Parser, Subcommand, ValueEnum};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum DeviceSelection {
    /// Burn Flex backend on the CPU (default).
    Cpu,
    /// Burn Wgpu backend on the GPU: Vulkan/DX12 on Windows and Linux, Metal on macOS. Requires
    /// building with `--features gpu`.
    Gpu,
}

#[derive(Debug, Parser)]
#[command(
    name = "boquilens",
    version,
    about = "Object detection in Rust with Burn"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run object detection on an image.
    Predict(PredictArgs),
    /// Pack imported tensor state into a versioned native Burnpack artifact.
    PackWeights(PackWeightsArgs),
}

#[derive(Debug, ClapArgs)]
struct PackWeightsArgs {
    /// Model architecture represented by the checkpoint.
    #[arg(long)]
    model: ModelId,

    /// Tensor-only imported checkpoint produced by the model-specific development bridge.
    #[arg(long)]
    input: PathBuf,

    /// Native output artifact; must end in .bpk and must not already exist.
    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, ClapArgs)]
#[command(
    after_long_help = "BOUNDING BOXES:\n  Output coordinates are unnormalized, continuous XYXY pixel edges in the source image.\n  (xmin, ymin) is the top-left edge; (xmax, ymax) is the bottom-right edge.\n  Values are clipped to [0, width] x [0, height]."
)]
struct PredictArgs {
    /// Input JPEG, PNG, or WebP image.
    #[arg(long)]
    source: PathBuf,

    /// Model architecture and scale to run: yolox-nano/tiny/s/m/l/x, yolov3-tinyu,
    /// yolov10n/s/m/b/l/x, yolo11n/s/m/l/x, yolo11n/s-seg, yolo11n/s/m/l/x-cls, yolo26n/s/m/l/x,
    /// yolo26n/s/m/l/x-seg, or yolo26n/s/m/l/x-cls.
    #[arg(long)]
    model: ModelId,

    /// Local checkpoint. YOLOX accepts official .pth; Ultralytics-family models prefer native .bpk.
    #[arg(long)]
    weights: Option<PathBuf>,

    /// Compute device for inference.
    #[arg(long, value_enum, default_value = "cpu")]
    device: DeviceSelection,

    /// Annotated output image (defaults to <input-stem>-detections.png).
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Minimum object confidence.
    #[arg(long, default_value_t = 0.25)]
    confidence: f32,

    /// Intersection-over-union threshold used by NMS.
    #[arg(long, default_value_t = 0.45)]
    iou: f32,

    /// Render instance-mask outlines over the annotated image and report per-detection mask
    /// coverage. Requires a segmentation model (yolo11n/s-seg or yolo26n/s/m/l/x-seg).
    #[arg(long)]
    masks: bool,

    /// Print detections as JSON instead of a compact table.
    #[arg(long)]
    json: bool,
}

fn default_output(input: &std::path::Path) -> PathBuf {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("prediction");
    input.with_file_name(format!("{stem}-detections.png"))
}

fn main() -> boquilens::Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Predict(args) => predict(args),
        Command::PackWeights(args) => {
            let packed = pack_weights(args.model, &args.input, &args.output)?;
            eprintln!(
                "Packed {} weights into {} ({} bytes, SHA-256 {})",
                args.model,
                packed.path.display(),
                packed.bytes,
                packed.sha256,
            );
            Ok(())
        }
    }
}

fn predict(args: PredictArgs) -> boquilens::Result<()> {
    let options = PredictOptions {
        confidence: args.confidence,
        iou: args.iou,
    };
    match args.device {
        DeviceSelection::Cpu => run_predict::<Flex>(&args, options, Device::<Flex>::default()),
        #[cfg(feature = "gpu")]
        DeviceSelection::Gpu => {
            let (device, adapter) = boquilens::default_wgpu_device();
            eprintln!("GPU adapter: {adapter}");
            run_predict::<Wgpu>(&args, options, device)
        }
        #[cfg(not(feature = "gpu"))]
        DeviceSelection::Gpu => Err(
            "GPU inference requires building boquilens with the gpu feature: \
             cargo build --release --features gpu"
                .into(),
        ),
    }
}

fn run_predict<B: Backend>(
    args: &PredictArgs,
    options: PredictOptions,
    device: Device<B>,
) -> boquilens::Result<()> {
    let output = args
        .output
        .clone()
        .unwrap_or_else(|| default_output(&args.source));

    eprintln!("Loading {} weights with Burn...", args.model);
    let predictor: Predictor<B> = match &args.weights {
        Some(checkpoint) => {
            Predictor::from_checkpoint_on_device(args.model, checkpoint.clone(), device, options)?
        }
        None => Predictor::new_on_device(args.model, options, device)?,
    };

    if matches!(
        args.model,
        ModelId::Yolo26NCls
            | ModelId::Yolo26SCls
            | ModelId::Yolo26MCls
            | ModelId::Yolo26LCls
            | ModelId::Yolo26XCls
            | ModelId::Yolo11NCls
            | ModelId::Yolo11SCls
            | ModelId::Yolo11MCls
            | ModelId::Yolo11LCls
            | ModelId::Yolo11XCls
    ) {
        let (image, classifications) = predictor.predict_classification_path(&args.source)?;
        report_classifications(args, &image, &output, &classifications)?;
        return Ok(());
    }
    if matches!(
        args.model,
        ModelId::Yolo11NSeg
            | ModelId::Yolo11SSeg
            | ModelId::Yolo26NSeg
            | ModelId::Yolo26SSeg
            | ModelId::Yolo26MSeg
            | ModelId::Yolo26LSeg
            | ModelId::Yolo26XSeg
    ) {
        let (image, detections) = predictor.predict_segmentation_path(&args.source)?;
        report_segmentations(args, &image, &output, &detections)?;
        return Ok(());
    }
    if args.masks {
        return Err(
            "--masks requires a segmentation model (yolo11n/s-seg or yolo26n/s/m/l/x-seg)".into(),
        );
    }
    let (image, detections) = predictor.predict_path(&args.source)?;

    if args.json {
        #[derive(Serialize)]
        struct JsonOutput<'a> {
            coordinate_format: &'static str,
            coordinate_units: &'static str,
            coordinate_space: &'static str,
            detections: &'a [boquilens::Detection],
        }

        println!(
            "{}",
            serde_json::to_string_pretty(&JsonOutput {
                coordinate_format: "xyxy",
                coordinate_units: "pixels",
                coordinate_space: "source_image",
                detections: &detections,
            })?
        );
    } else if detections.is_empty() {
        println!("No objects detected.");
    } else {
        for detection in &detections {
            println!(
                "{:<16} {:>5.1}%  xyxy_px=[{:>6.1}, {:>6.1}, {:>6.1}, {:>6.1}]",
                detection.class_name,
                detection.confidence * 100.0,
                detection.xmin,
                detection.ymin,
                detection.xmax,
                detection.ymax,
            );
        }
    }

    annotate(&image, &detections).save(&output)?;
    eprintln!(
        "Saved {} detections to {}",
        detections.len(),
        output.display()
    );
    Ok(())
}

/// Print classification results (top-5 table or JSON).
///
/// Classification models return class probabilities instead of spatial detections, so no annotated
/// image is produced; the strongest class is reported on stderr for parity with the other tasks'
/// output line.
fn report_classifications(
    args: &PredictArgs,
    image: &image::DynamicImage,
    output: &std::path::Path,
    classifications: &[boquilens::Classification],
) -> boquilens::Result<()> {
    let _ = image;
    let _ = output;
    if args.json {
        #[derive(Serialize)]
        struct JsonOutput<'a> {
            task: &'static str,
            input_size_px: usize,
            classes: &'a [boquilens::Classification],
        }

        println!(
            "{}",
            serde_json::to_string_pretty(&JsonOutput {
                task: "classification",
                input_size_px: boquilens::CLASSIFY_INPUT_SIZE,
                classes: classifications,
            })?
        );
    } else if classifications.is_empty() {
        println!("No classes predicted.");
    } else {
        println!("Top-{} classes:", classifications.len());
        for (rank, classification) in classifications.iter().enumerate() {
            println!(
                "{:>2}. {:<24} {:>6.2}%",
                rank + 1,
                classification.class_name,
                classification.confidence * 100.0
            );
        }
    }
    eprintln!(
        "Top-1: {} ({:.2}%)",
        classifications[0].class_name,
        classifications[0].confidence * 100.0
    );
    Ok(())
}

/// Print segmentation results (table or JSON) and save the annotated image.
///
/// With `--masks`, the table gains a per-detection covered-pixel count, the JSON gains a mask
/// summary per detection (the full bitmask is too large to print), and the annotated image gets
/// the mask outlines stroked under the boxes. Without it, output matches the detect task except
/// that the annotated image is produced from the same boxes.
fn report_segmentations(
    args: &PredictArgs,
    image: &image::DynamicImage,
    output: &std::path::Path,
    detections: &[boquilens::SegmentationDetection],
) -> boquilens::Result<()> {
    let masks = args.masks;
    if args.json {
        #[derive(Serialize)]
        struct JsonOutput {
            coordinate_format: &'static str,
            coordinate_units: &'static str,
            coordinate_space: &'static str,
            detections: Vec<JsonSegmentationDetection>,
        }

        #[derive(Serialize)]
        struct JsonSegmentationDetection {
            class_id: usize,
            class_name: &'static str,
            confidence: f32,
            box_xyxy_px: [f32; 4],
            mask: Option<JsonMaskSummary>,
        }

        #[derive(Serialize)]
        struct JsonMaskSummary {
            width: u32,
            height: u32,
            coordinate_space: &'static str,
            filled_pixels: u64,
        }

        println!(
            "{}",
            serde_json::to_string_pretty(&JsonOutput {
                coordinate_format: "xyxy",
                coordinate_units: "pixels",
                coordinate_space: "source_image",
                detections: detections
                    .iter()
                    .map(|detection| JsonSegmentationDetection {
                        class_id: detection.class_id,
                        class_name: detection.class_name,
                        confidence: detection.confidence,
                        box_xyxy_px: [
                            detection.xmin,
                            detection.ymin,
                            detection.xmax,
                            detection.ymax
                        ],
                        mask: masks.then(|| JsonMaskSummary {
                            width: detection.mask.width,
                            height: detection.mask.height,
                            coordinate_space: "source_image",
                            filled_pixels: detection
                                .mask
                                .data
                                .iter()
                                .filter(|pixel| **pixel)
                                .count() as u64,
                        }),
                    })
                    .collect(),
            })?
        );
    } else if detections.is_empty() {
        println!("No objects detected.");
    } else {
        for detection in detections {
            let mut line = format!(
                "{:<16} {:>5.1}%  xyxy_px=[{:>6.1}, {:>6.1}, {:>6.1}, {:>6.1}]",
                detection.class_name,
                detection.confidence * 100.0,
                detection.xmin,
                detection.ymin,
                detection.xmax,
                detection.ymax,
            );
            if masks {
                line.push_str(&format!(
                    "  mask_px={}",
                    detection.mask.data.iter().filter(|pixel| **pixel).count()
                ));
            }
            println!("{line}");
        }
    }

    let annotated = if masks {
        annotate_segmentation(image, detections)
    } else {
        let boxes: Vec<boquilens::Detection> = detections
            .iter()
            .map(|detection| boquilens::Detection {
                class_id: detection.class_id,
                class_name: detection.class_name,
                confidence: detection.confidence,
                xmin: detection.xmin,
                ymin: detection.ymin,
                xmax: detection.xmax,
                ymax: detection.ymax,
            })
            .collect();
        annotate(image, &boxes)
    };
    annotated.save(output)?;
    eprintln!(
        "Saved {} detections to {}",
        detections.len(),
        output.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_default_output_path() {
        assert_eq!(
            default_output(std::path::Path::new("photos/dog.jpg")),
            PathBuf::from("photos/dog-detections.png")
        );
    }
}
