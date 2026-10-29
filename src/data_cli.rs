//! Dataset-only commands; no model or GPU initialization.
use clap::{ArgGroup, Args, Subcommand};
use montgomery::dataset::{self, Conversion, ImageSize, ResizeMode, TransformOptions};
use std::path::PathBuf;

#[derive(Debug, Args)]
#[command(
    after_help = "Examples:\n  montgomery data --path dataset.yaml\n  montgomery data inspect --path dataset.yaml --json\n  montgomery data convert --path dataset/data.yaml --to detect --out-dir boxes\n  montgomery data convert --path boxes/data.yaml --to classify --out-dir crops\n  montgomery data resize --path dataset/data.yaml --imgsz 224 --override\n\nTransforms require exactly one of --out-dir DIRECTORY or --override."
)]
pub(crate) struct DataArgs {
    /// Dataset YAML. Relative dataset paths resolve against the YAML, not the current directory.
    #[arg(
        long,
        visible_alias = "data",
        global = true,
        value_name = "DATASET.yaml"
    )]
    path: Option<PathBuf>,
    /// Emit a structured report as JSON.
    #[arg(long, global = true)]
    json: bool,
    /// Concurrent image workers (default: available CPU cores, capped at four).
    #[arg(long, global = true, default_value_t = dataset::default_workers(), value_parser = positive_workers)]
    workers: usize,
    #[command(subcommand)]
    command: Option<DataCommand>,
}

fn positive_workers(value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|n| (1..=256).contains(n))
        .ok_or_else(|| "workers must be between 1 and 256".into())
}

#[derive(Debug, Subcommand)]
enum DataCommand {
    /// Count images, labels, classes, dimensions, backgrounds, and report data issues.
    #[command(visible_alias = "info")]
    Inspect(InspectArgs),
    /// Convert instance labels to boxes, or crop boxes/polygons into class folders.
    Convert(ConvertArgs),
    /// Resize every image and its annotations.
    Resize(ResizeArgs),
}

#[derive(Debug, Default, Args)]
struct InspectArgs {
    /// Decode every image to detect damaged pixel data; otherwise read headers only.
    #[arg(long)]
    verify_images: bool,
    /// Return a failing exit status if invalid images/labels or cross-split leakage are found.
    #[arg(long)]
    check: bool,
}

#[derive(Debug, Args)]
#[command(group(ArgGroup::new("destination").required(true).args(["out_dir", "override_source"])))]
struct OutputArgs {
    /// Write a new dataset to this directory, preserving the source dataset.
    #[arg(long, value_name = "DIRECTORY")]
    out_dir: Option<PathBuf>,
    /// Replace the source dataset directory after successful processing; retain a recovery backup.
    #[arg(long = "override")]
    override_source: bool,
    /// Validate and preview counts without creating any files.
    #[arg(long)]
    dry_run: bool,
}

#[derive(Debug, Args)]
struct ImageArgs {
    /// Fit preserves aspect ratio; letterbox pads to exact size; stretch changes aspect ratio.
    #[arg(long, value_enum, default_value = "fit")]
    mode: ResizeMode,
    /// Allow fit/letterbox to enlarge small images (stretch always uses exact dimensions).
    #[arg(long)]
    upscale: bool,
    /// JPEG quality for resized images or crops; copied detection images are unchanged.
    #[arg(long, default_value_t = 90, value_parser = clap::value_parser!(u8).range(1..=100))]
    quality: u8,
}

#[derive(Debug, Args)]
struct ConvertArgs {
    #[arg(long, value_enum)]
    to: Conversion,
    /// Optionally resize classification crops: N for a square bound, or WIDTHxHEIGHT.
    #[arg(long, value_name = "N|WIDTHxHEIGHT")]
    imgsz: Option<ImageSize>,
    /// Expand classification crops on each side by this fraction of box width/height (0..1).
    #[arg(long, default_value_t = 0.0)]
    padding: f64,
    /// Include COCO crowd annotations in classification crops (excluded by default).
    #[arg(long)]
    include_crowd: bool,
    #[command(flatten)]
    output: OutputArgs,
    #[command(flatten)]
    image: ImageArgs,
}

#[derive(Debug, Args)]
struct ResizeArgs {
    /// N for a square bound, or WIDTHxHEIGHT. Default fit never upscales small images.
    #[arg(long, value_name = "N|WIDTHxHEIGHT")]
    imgsz: ImageSize,
    #[command(flatten)]
    output: OutputArgs,
    #[command(flatten)]
    image: ImageArgs,
}

pub(crate) fn run(args: DataArgs) -> montgomery::Result<()> {
    let Some(path) = args.path else {
        clap::Error::raw(
            clap::error::ErrorKind::MissingRequiredArgument,
            "--path DATASET.yaml is required\n\nExample: montgomery data --path dataset.yaml\n",
        )
        .exit();
    };
    let command = args
        .command
        .unwrap_or(DataCommand::Inspect(InspectArgs::default()));
    let (operation, output, image, size, padding, include_crowd) = match command {
        DataCommand::Inspect(inspect) => {
            let report = dataset::inspect(&path, inspect.verify_images, args.workers)?;
            if args.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print!("{}", report.text());
            }
            if inspect.check && (report.invalid_images > 0 || report.cross_split_images > 0) {
                return Err("dataset check failed; see the report above".into());
            }
            return Ok(());
        }
        DataCommand::Convert(convert) => (
            Some(convert.to),
            convert.output,
            convert.image,
            convert.imgsz,
            convert.padding,
            convert.include_crowd,
        ),
        DataCommand::Resize(resize) => (
            None,
            resize.output,
            resize.image,
            Some(resize.imgsz),
            0.,
            false,
        ),
    };
    if operation == Some(Conversion::Detect) && (padding != 0. || include_crowd) {
        return Err("--padding and --include-crowd apply only to --to classify".into());
    }
    if size.is_none() && (image.mode != ResizeMode::Fit || image.upscale) {
        return Err("--mode and --upscale require --imgsz".into());
    }
    let options = TransformOptions {
        out_dir: output.out_dir,
        override_source: output.override_source,
        size,
        mode: image.mode,
        upscale: image.upscale,
        jpeg_quality: image.quality,
        workers: args.workers,
        dry_run: output.dry_run,
        padding,
        include_crowd,
    };
    let path = if options.override_source {
        let path = std::fs::canonicalize(path)?;
        // Windows holds the process's current directory open. Leave the source
        // before replacing it; this changes only this CLI process, not the shell.
        if let Some(source) = path.parent()
            && let Some(parent) = source.parent()
            && std::fs::canonicalize(std::env::current_dir()?)?.starts_with(source)
        {
            std::env::set_current_dir(parent)?;
        }
        path
    } else {
        path
    };
    let report = if let Some(to) = operation {
        dataset::convert(&path, to, &options)?
    } else {
        dataset::resize(&path, &options)?
    };
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!(
            "{}: {} input images -> {} output images, {} labels",
            if report.dry_run { "Preview" } else { "Done" },
            report.input_images,
            report.output_images,
            report.output_labels
        );
        if report.skipped_background_images > 0 || report.skipped_crowd_objects > 0 {
            println!(
                "Skipped {} background images and {} crowd objects.",
                report.skipped_background_images, report.skipped_crowd_objects
            );
        }
        if report.output_bytes > 0 {
            println!(
                "Image storage: {:.2} MiB -> {:.2} MiB",
                report.input_bytes as f64 / 1_048_576.,
                report.output_bytes as f64 / 1_048_576.
            );
        }
        println!(
            "{}: {}",
            if report.dry_run {
                "Planned dataset"
            } else {
                "Dataset"
            },
            report.manifest.display()
        );
        if let Some(backup) = &report.backup {
            println!(
                "{}: {}",
                if report.dry_run {
                    "Planned recovery backup"
                } else {
                    "Original dataset saved"
                },
                backup.display()
            );
        }
        if !report.dry_run {
            println!(
                "Next: montgomery data --path \"{}\"",
                report.manifest.display()
            );
        }
    }
    Ok(())
}
