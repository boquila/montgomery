use std::path::PathBuf;

#[cfg(feature = "onnx")]
use boquilens::export::{
    CheckpointState, ExternalDataPolicy, OnnxExportOptions, OnnxPrecision, OnnxProfile, export_onnx,
};
#[cfg(feature = "training")]
use boquilens::training::runtime::{
    TrainingRequest, export as export_training, train as train_native, validate as validate_native,
};
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
    /// Pack an imported upstream checkpoint into a versioned native Burnpack artifact.
    PackWeights(PackWeightsArgs),
    /// Export the exact loaded Burn model weights to a validated portable ONNX artifact.
    #[cfg(feature = "onnx")]
    ExportOnnx(ExportOnnxArgs),
    /// Train a model with the native Burn/WGPU trainer.
    #[cfg(feature = "training")]
    Train(TrainArgs),
    /// Validate and inspect a resumable native training checkpoint.
    #[cfg(feature = "training")]
    Val(ValArgs),
    /// Export a native training checkpoint to the existing inference Burnpack format.
    #[cfg(feature = "training")]
    Export(ExportTrainingArgs),
}

#[cfg(feature = "training")]
#[derive(Debug, ClapArgs)]
struct TrainArgs {
    #[arg(long)]
    model: ModelId,
    #[arg(long)]
    data: PathBuf,
    #[arg(long, default_value_t = 100)]
    epochs: usize,
    #[arg(long, default_value_t = 8)]
    batch: usize,
    #[arg(long, default_value_t = 1)]
    accumulation: usize,
    #[arg(long)]
    imgsz: Option<usize>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value = "runs")]
    project: PathBuf,
    #[arg(long, default_value = "train")]
    name: String,
    /// Run data -> forward -> loss -> backward without mutating model or optimizer state.
    #[arg(long)]
    dry_run: bool,
    /// Resume a full native checkpoint. Model, task, classes, and dataset metadata are immutable.
    #[arg(long)]
    resume: Option<PathBuf>,
    /// Initialize from an official tensor-only checkpoint. Mutually exclusive with --resume.
    #[arg(long)]
    weights: Option<PathBuf>,
    /// Confidence floor used for AP validation (low by default to preserve the PR curve).
    #[arg(long)]
    val_confidence: Option<f32>,
    /// Class-aware NMS IoU used by classic detector/segment validation.
    #[arg(long)]
    val_iou: Option<f32>,
    /// Maximum predictions retained per validation image.
    #[arg(long)]
    max_detections: Option<usize>,
}

#[cfg(feature = "training")]
#[derive(Debug, ClapArgs)]
struct ValArgs {
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    json: bool,
}

#[cfg(feature = "training")]
#[derive(Debug, ClapArgs)]
struct ExportTrainingArgs {
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

#[cfg(feature = "onnx")]
#[derive(Debug, ClapArgs)]
struct ExportOnnxArgs {
    /// Model architecture represented by the checkpoint.
    #[arg(long)]
    model: ModelId,
    /// Local boquilens .bpk artifact.
    #[arg(long)]
    weights: PathBuf,
    /// Final ONNX path. A missing .onnx suffix is added explicitly.
    #[arg(long)]
    output: PathBuf,
    /// Square size or H,W. Detect/segment dimensions must be divisible by 32.
    #[arg(long)]
    imgsz: Option<String>,
    /// Fixed batch size (dynamic batch is gated separately).
    #[arg(long, default_value_t = 1)]
    batch: usize,
    #[arg(long)]
    dynamic_batch: bool,
    #[arg(long)]
    dynamic_spatial: bool,
    #[arg(long, default_value_t = 17)]
    opset: u32,
    #[arg(long, value_enum, default_value = "portable")]
    profile: OnnxProfile,
    #[arg(long, value_enum, default_value = "fp32")]
    precision: OnnxPrecision,
    #[arg(long, value_enum, default_value = "auto")]
    external_data: ExternalDataPolicy,
    /// Exact Python executable from the locked export environment.
    #[arg(long)]
    python: Option<PathBuf>,
    /// Official YOLOX 0.1.1rc0 checkout (YOLOX only).
    #[arg(long)]
    yolox_repo: Option<PathBuf>,
    /// Reserved state preference for future multi-state training checkpoints (EMA only today).
    #[arg(long, value_enum, default_value = "ema")]
    checkpoint_state: CheckpointState,
    #[arg(long)]
    simplify: bool,
    #[arg(long)]
    force: bool,
    #[arg(long)]
    keep_intermediate: bool,
    #[arg(long)]
    reproducible: bool,
    /// Skip the exact Burn-vs-PyTorch comparison. ONNX Runtime validation remains mandatory.
    #[arg(long)]
    no_verify: bool,
    /// Print the resulting artifact record as JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, ClapArgs)]
struct PackWeightsArgs {
    /// Model architecture represented by the checkpoint.
    #[arg(long)]
    model: ModelId,

    /// Official YOLOX .pth or tensor-only state produced by the Ultralytics development bridge.
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
    /// yolov10n/s/m/b/l/x, yolo11n/s/m/l/x, yolo11n/s/m/l/x-seg, yolo11n/s/m/l/x-cls,
    /// yolov8n/s/m/l/x, yolov8n/s/m/l/x-seg, yolov8n/s/m/l/x-cls, yolo12n/s/m/l/x,
    /// yolo26n/s/m/l/x, yolo26n/s/m/l/x-seg, or yolo26n/s/m/l/x-cls.
    #[arg(long)]
    model: ModelId,

    /// Local boquilens .bpk artifact.
    #[arg(long)]
    weights: PathBuf,

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
    /// coverage. Requires a segmentation model (yolo11n/s/m/l/x-seg, yolov8n/s/m/l/x-seg, or
    /// yolo26n/s/m/l/x-seg).
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
    #[cfg(all(windows, feature = "training", debug_assertions))]
    if std::env::var_os("RUST_MIN_STACK").is_none() {
        // Burn/CubeCL creates deep named worker threads after CLI startup. Windows' default Rust
        // worker stack is too small for debug training graphs. Optimized training does not need
        // this process-wide override. This point is still single-threaded and therefore satisfies
        // `set_var`'s process-environment safety rule.
        unsafe { std::env::set_var("RUST_MIN_STACK", "67108864") };
    }
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
        #[cfg(feature = "onnx")]
        Command::ExportOnnx(args) => export_onnx_command(args),
        #[cfg(feature = "training")]
        Command::Train(args) => {
            let run = train_native(TrainingRequest {
                model: args.model,
                data: args.data,
                epochs: args.epochs,
                batch_size: args.batch,
                accumulation: args.accumulation,
                image_size: args.imgsz,
                seed: args.seed,
                run_root: args.project,
                name: args.name,
                dry_run: args.dry_run,
                resume: args.resume,
                weights: args.weights,
                val_confidence: args.val_confidence,
                val_iou: args.val_iou,
                max_detections: args.max_detections,
            })?;
            eprintln!("Training run: {}", run.display());
            Ok(())
        }
        #[cfg(feature = "training")]
        Command::Val(args) => {
            let summary = validate_native(args.checkpoint)?;
            if args.json {
                println!("{}", serde_json::to_string_pretty(&summary)?);
            } else if let (Some(box_metrics), Some(mask_metrics)) =
                (&summary.box_metrics, &summary.mask_metrics)
            {
                println!(
                    "{} {} images: box mAP50-95 {:.2}%, mAP50 {:.2}%; mask mAP50-95 {:.2}%, mAP50 {:.2}%",
                    summary.model,
                    summary.images,
                    box_metrics.map_50_95 * 100.0,
                    box_metrics.map_50 * 100.0,
                    mask_metrics.map_50_95 * 100.0,
                    mask_metrics.map_50 * 100.0,
                );
            } else if let Some(metrics) = &summary.box_metrics {
                println!(
                    "{} {} images: box mAP50-95 {:.2}%, mAP50 {:.2}%",
                    summary.model,
                    summary.images,
                    metrics.map_50_95 * 100.0,
                    metrics.map_50 * 100.0,
                );
            } else {
                println!(
                    "{} {} images: loss {:.6}, top-1 {:.2}%, top-5 {:.2}%",
                    summary.model,
                    summary.images,
                    summary.mean_loss.unwrap_or_default(),
                    summary.top1_accuracy.unwrap_or_default() * 100.0,
                    summary.top5_accuracy.unwrap_or_default() * 100.0,
                );
            }
            Ok(())
        }
        #[cfg(feature = "training")]
        Command::Export(args) => {
            let output = export_training(args.checkpoint, args.output)?;
            eprintln!("Exported inference artifact to {}", output.display());
            Ok(())
        }
    }
}

#[cfg(feature = "onnx")]
fn export_onnx_command(args: ExportOnnxArgs) -> boquilens::Result<()> {
    let mut options = OnnxExportOptions::for_model(args.model, args.output);
    if let Some(imgsz) = &args.imgsz {
        let (height, width) = parse_imgsz(imgsz)?;
        options.input_shape = [args.batch, 3, height, width];
    } else {
        options.input_shape[0] = args.batch;
    }
    options.profile = args.profile;
    options.opset = args.opset;
    options.precision = args.precision;
    options.dynamic_batch = args.dynamic_batch;
    options.dynamic_spatial = args.dynamic_spatial;
    options.external_data = args.external_data;
    options.verify = !args.no_verify;
    options.python = args.python;
    options.yolox_repo = args.yolox_repo;
    options.checkpoint_state = args.checkpoint_state;
    options.simplify = args.simplify;
    options.force = args.force;
    options.keep_intermediate = args.keep_intermediate;
    options.reproducible = args.reproducible;
    let artifact = export_onnx(args.model, &args.weights, options)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&artifact)?);
    } else {
        eprintln!(
            "Exported {} to {} ({} bytes, SHA-256 {}); sidecar {}",
            args.model,
            artifact.path.display(),
            artifact.bytes,
            artifact.sha256,
            artifact.sidecar.display()
        );
    }
    Ok(())
}

#[cfg(feature = "onnx")]
fn parse_imgsz(value: &str) -> boquilens::Result<(usize, usize)> {
    let parts = value
        .split([',', 'x', 'X'])
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::parse::<usize>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    match parts.as_slice() {
        [size] => Ok((*size, *size)),
        [height, width] => Ok((*height, *width)),
        _ => Err("--imgsz must be one integer or H,W".into()),
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
    if args.weights.extension().and_then(|value| value.to_str()) != Some("bpk") {
        return Err(
            "predict --weights requires a native .bpk artifact; convert upstream checkpoints with pack-weights"
                .into(),
        );
    }
    let output = args
        .output
        .clone()
        .unwrap_or_else(|| default_output(&args.source));

    eprintln!("Loading {} weights with Burn...", args.model);
    let predictor: Predictor<B> =
        Predictor::from_checkpoint_on_device(args.model, args.weights.clone(), device, options)?;

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
            | ModelId::Yolov8NCls
            | ModelId::Yolov8SCls
            | ModelId::Yolov8MCls
            | ModelId::Yolov8LCls
            | ModelId::Yolov8XCls
    ) {
        let (image, classifications) = predictor.predict_classification_path(&args.source)?;
        report_classifications(
            args,
            &image,
            &output,
            predictor.input_size(),
            &classifications,
        )?;
        return Ok(());
    }
    if matches!(
        args.model,
        ModelId::Yolo11NSeg
            | ModelId::Yolo11SSeg
            | ModelId::Yolo11MSeg
            | ModelId::Yolo11LSeg
            | ModelId::Yolo11XSeg
            | ModelId::Yolov8NSeg
            | ModelId::Yolov8SSeg
            | ModelId::Yolov8MSeg
            | ModelId::Yolov8LSeg
            | ModelId::Yolov8XSeg
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
            "--masks requires a segmentation model (yolo11n/s/m/l/x-seg, yolov8n/s/m/l/x-seg, or \
             yolo26n/s/m/l/x-seg)"
                .into(),
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
    input_size: usize,
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
                input_size_px: input_size,
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
            class_name: String,
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
                        class_name: detection.class_name.clone(),
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
                class_name: detection.class_name.clone(),
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
