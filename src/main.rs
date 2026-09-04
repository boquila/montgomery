use std::path::PathBuf;

#[cfg(feature = "gpu")]
use burn::backend::Wgpu;
use burn::tensor::{Device, backend::Backend};
use burn_flex::Flex;
#[cfg(feature = "training")]
use clap::ArgGroup;
use clap::{Args as ClapArgs, Parser, Subcommand, ValueEnum};
#[cfg(feature = "onnx")]
use montgomery::export::{
    CheckpointState, ExternalDataPolicy, OnnxExportOptions, OnnxPrecision, OnnxProfile, export_onnx,
};
#[cfg(feature = "training")]
use montgomery::training::automatic_worker_count;
#[cfg(feature = "training")]
use montgomery::training::runtime::{
    TrainingInitialization, TrainingRequest, export as export_training, train as train_native,
    validate as validate_native,
};
use montgomery::{
    ModelId, ModelTask, PredictOptions, Predictor, annotate, annotate_segmentation, pack_weights,
    pack_weights_to,
};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum DeviceSelection {
    /// Burn Flex backend on the CPU (default).
    Cpu,
    /// Burn Wgpu backend on the GPU: Vulkan/DX12 on Windows and Linux, Metal on macOS.
    Gpu,
}

#[derive(Debug, Parser)]
#[command(
    name = "montgomery",
    version,
    about = "YOLO inference and training in Rust with Burn"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run detection, instance segmentation, or classification on an image.
    Predict(PredictArgs),
    /// Pack an imported tensor-only state into a versioned native Burnpack artifact.
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
#[command(group(
    ArgGroup::new("initialization")
        .required(true)
        .multiple(false)
        .args(["architecture", "model", "resume"])
))]
struct TrainArgs {
    /// Initialize a new model architecture from scratch.
    #[arg(long)]
    architecture: Option<ModelId>,
    /// Initialize from a pretrained Montgomery .bpk model; its architecture is read from metadata.
    #[arg(long, value_name = "MODEL.bpk")]
    model: Option<PathBuf>,
    /// Resume a full native training checkpoint; its model and dataset configuration are retained.
    #[arg(long, value_name = "CHECKPOINT")]
    resume: Option<PathBuf>,
    /// Dataset manifest for a scratch or pretrained run. Resume uses the checkpoint's dataset.
    #[arg(long, required_unless_present = "resume", conflicts_with = "resume")]
    data: Option<PathBuf>,
    #[arg(long, default_value_t = 100)]
    epochs: usize,
    #[arg(long, default_value_t = 8)]
    batch: usize,
    #[arg(long, default_value_t = 1)]
    accumulation: usize,
    /// CPU preprocessing workers.
    #[arg(long, default_value_t = automatic_worker_count())]
    workers: usize,
    /// Number of prepared CPU batches retained ahead of the active batch.
    #[arg(long, default_value_t = 2)]
    prefetch: usize,
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
    /// Confidence floor used for AP validation (low by default to preserve the PR curve).
    #[arg(long)]
    val_confidence: Option<f32>,
    /// Class-aware NMS IoU used by classic detector/segment validation.
    #[arg(long)]
    val_iou: Option<f32>,
    /// Maximum predictions retained per validation image.
    #[arg(long)]
    max_detections: Option<usize>,
    /// Skip validation during training. Useful for throughput benchmarks.
    #[arg(long)]
    no_val: bool,
    /// Skip final best.bpk and last.bpk inference exports.
    #[arg(long)]
    no_export: bool,
    /// Save a resumable `last` checkpoint every N epochs; improvements and the final epoch always save.
    #[arg(long, default_value_t = 10)]
    save_period: usize,
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
    /// Montgomery .bpk model to export; architecture is read from artifact metadata.
    #[arg(long, value_name = "MODEL.bpk")]
    model: PathBuf,
    /// Final ONNX path (defaults to <model>.onnx). A missing suffix is added explicitly.
    #[arg(long)]
    output: Option<PathBuf>,
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
    /// Training checkpoint state to export. Only EMA is currently available.
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
    architecture: ModelId,

    /// Tensor-only state produced by tools/export_checkpoint_state.py.
    #[arg(long, value_name = "STATE.pt")]
    state: PathBuf,

    /// Native output artifact (defaults to <model>.bpk); must not already exist.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, ClapArgs)]
#[command(
    after_long_help = "BOUNDING BOXES:\n  Output coordinates are unnormalized, continuous XYXY pixel edges in the source image.\n  (xmin, ymin) is the top-left edge; (xmax, ymax) is the bottom-right edge.\n  Values are clipped to [0, width] x [0, height]."
)]
struct PredictArgs {
    /// Input JPEG, PNG, or WebP image.
    #[arg(long)]
    source: PathBuf,

    /// Montgomery .bpk model to run; architecture and task are read from artifact metadata.
    #[arg(long, value_name = "MODEL.bpk")]
    model: PathBuf,

    /// Compute device for inference.
    #[arg(long, value_enum, default_value = "cpu")]
    device: DeviceSelection,

    /// Annotated output image (defaults to <input-stem>-detections.png, or
    /// <input-stem>-segmentation.png with --masks).
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

fn default_output(input: &std::path::Path, masks: bool) -> PathBuf {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("prediction");
    // The suffix names the rendered content: mask outlines only appear with --masks, so a
    // segmentation rendering must not default to a *-detections.png path.
    let suffix = if masks { "segmentation" } else { "detections" };
    input.with_file_name(format!("{stem}-{suffix}.png"))
}

fn main() -> montgomery::Result<()> {
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
            let packed = match args.output {
                Some(output) => pack_weights_to(args.architecture, &args.state, output)?,
                None => pack_weights(args.architecture, &args.state)?,
            };
            eprintln!(
                "Packed {} weights into {} ({} bytes, SHA-256 {})",
                args.architecture,
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
            let initialization = match (args.architecture, args.model, args.resume) {
                (Some(architecture), None, None) => TrainingInitialization::Scratch(architecture),
                (None, Some(model), None) => TrainingInitialization::Pretrained(model),
                (None, None, Some(checkpoint)) => TrainingInitialization::Resume(checkpoint),
                _ => unreachable!("clap enforces exactly one training initialization mode"),
            };
            let run = train_native(TrainingRequest {
                initialization,
                data: args.data,
                epochs: args.epochs,
                batch_size: args.batch,
                accumulation: args.accumulation,
                workers: args.workers,
                prefetch: args.prefetch,
                image_size: args.imgsz,
                seed: args.seed,
                run_root: args.project,
                name: args.name,
                dry_run: args.dry_run,
                val_confidence: args.val_confidence,
                val_iou: args.val_iou,
                max_detections: args.max_detections,
                validation_enabled: !args.no_val,
                export_artifacts: !args.no_export,
                checkpoint_interval: args.save_period,
            })?;
            eprintln!("Training run: {}", run.display());
            let best = run.join("exports/best.bpk");
            if best.exists() {
                eprintln!("Best model: {}", best.display());
                eprintln!("Last model: {}", run.join("exports/last.bpk").display());
            } else if !args.dry_run {
                eprintln!(
                    "Best checkpoint: {}",
                    run.join("checkpoints/best").display()
                );
                eprintln!(
                    "Last checkpoint: {}",
                    run.join("checkpoints/last").display()
                );
            }
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
                    "{} {} images: box P {:.2}%, R {:.2}%, mAP50 {:.2}%, mAP50-95 {:.2}%; mask P {:.2}%, R {:.2}%, mAP50 {:.2}%, mAP50-95 {:.2}%",
                    summary.model,
                    summary.images,
                    box_metrics.precision * 100.0,
                    box_metrics.recall * 100.0,
                    box_metrics.map_50 * 100.0,
                    box_metrics.map_50_95 * 100.0,
                    mask_metrics.precision * 100.0,
                    mask_metrics.recall * 100.0,
                    mask_metrics.map_50 * 100.0,
                    mask_metrics.map_50_95 * 100.0,
                );
            } else if let Some(metrics) = &summary.box_metrics {
                println!(
                    "{} {} images: box P {:.2}%, R {:.2}%, mAP50 {:.2}%, mAP50-95 {:.2}%",
                    summary.model,
                    summary.images,
                    metrics.precision * 100.0,
                    metrics.recall * 100.0,
                    metrics.map_50 * 100.0,
                    metrics.map_50_95 * 100.0,
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
fn export_onnx_command(args: ExportOnnxArgs) -> montgomery::Result<()> {
    let weights = args.model;
    let model = ModelId::from_burnpack(&weights)?;
    let output = args
        .output
        .unwrap_or_else(|| weights.with_extension("onnx"));
    let mut options = OnnxExportOptions::for_model(model, output);
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
    let artifact = export_onnx(model, &weights, options)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&artifact)?);
    } else {
        eprintln!(
            "Exported {} to {} ({} bytes, SHA-256 {}); sidecar {}",
            model,
            artifact.path.display(),
            artifact.bytes,
            artifact.sha256,
            artifact.sidecar.display()
        );
    }
    Ok(())
}

#[cfg(feature = "onnx")]
fn parse_imgsz(value: &str) -> montgomery::Result<(usize, usize)> {
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

fn predict(args: PredictArgs) -> montgomery::Result<()> {
    let options = PredictOptions {
        confidence: args.confidence,
        iou: args.iou,
    };
    match args.device {
        DeviceSelection::Cpu => run_predict::<Flex>(&args, options, Device::<Flex>::default()),
        #[cfg(feature = "gpu")]
        DeviceSelection::Gpu => {
            let (device, adapter) = montgomery::default_wgpu_device();
            eprintln!("GPU adapter: {adapter}");
            run_predict::<Wgpu>(&args, options, device)
        }
        #[cfg(not(feature = "gpu"))]
        DeviceSelection::Gpu => Err(
            "GPU inference requires building Montgomery with the gpu feature: \
             cargo build --release --features gpu"
                .into(),
        ),
    }
}

fn run_predict<B: Backend>(
    args: &PredictArgs,
    options: PredictOptions,
    device: Device<B>,
) -> montgomery::Result<()> {
    let model = args.model.clone();
    let output = args
        .output
        .clone()
        .unwrap_or_else(|| default_output(&args.source, args.masks));

    let predictor: Predictor<B> = Predictor::with_options_on_device(model, device, options)?;
    eprintln!(
        "Loaded {} ({}) with Burn.",
        args.model.display(),
        predictor.model_id()
    );

    match predictor.task() {
        ModelTask::Classification => {
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
        ModelTask::Segmentation => {
            let (image, detections) = predictor.predict_segmentation_path(&args.source)?;
            report_segmentations(args, &image, &output, &detections)?;
            return Ok(());
        }
        ModelTask::Detection => {}
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
            detections: &'a [montgomery::Detection],
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
    classifications: &[montgomery::Classification],
) -> montgomery::Result<()> {
    let _ = image;
    let _ = output;
    if args.json {
        #[derive(Serialize)]
        struct JsonOutput<'a> {
            task: &'static str,
            input_size_px: usize,
            classes: &'a [montgomery::Classification],
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
    detections: &[montgomery::SegmentationDetection],
) -> montgomery::Result<()> {
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
        let boxes: Vec<montgomery::Detection> = detections
            .iter()
            .map(|detection| montgomery::Detection {
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
            default_output(std::path::Path::new("photos/dog.jpg"), false),
            PathBuf::from("photos/dog-detections.png")
        );
        assert_eq!(
            default_output(std::path::Path::new("photos/dog.jpg"), true),
            PathBuf::from("photos/dog-segmentation.png")
        );
    }
}
