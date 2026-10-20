use super::{
    Index, Object, Record, error,
    inspect::index,
    manifest::{DatasetError, DatasetFormat},
    parallel_map,
};
use clap::ValueEnum;
use image::{Rgb, RgbImage, codecs::jpeg::JpegEncoder};
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    str::FromStr,
    sync::atomic::{AtomicUsize, Ordering},
};

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum Conversion {
    /// Replace polygons/masks with their bounding boxes; retain background images.
    #[value(alias = "detection", alias = "bbox")]
    Detect,
    /// Write one class-folder crop per object; background images produce no crops.
    #[value(alias = "classification", alias = "cls")]
    Classify,
}

#[derive(Debug, Clone, Copy, ValueEnum, Default, PartialEq, Eq)]
pub enum ResizeMode {
    /// Fit within the requested dimensions, preserving aspect ratio (no padding).
    #[default]
    Fit,
    /// Fit and center on an exact-size canvas filled with RGB 114.
    Letterbox,
    /// Resize to exact dimensions, changing the aspect ratio.
    Stretch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSize {
    pub width: u32,
    pub height: u32,
}

impl FromStr for ImageSize {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let value = value.to_ascii_lowercase();
        let (w, h) = value.split_once('x').unwrap_or((&value, &value));
        let parse = |v: &str| {
            v.parse::<u32>()
                .ok()
                .filter(|n| (1..=16384).contains(n))
                .ok_or_else(|| {
                    "imgsz must be N or WIDTHxHEIGHT, each between 1 and 16384".to_string()
                })
        };
        Ok(Self {
            width: parse(w)?,
            height: parse(h)?,
        })
    }
}

/// Explicit destination: a new directory, or replacement of the source with a recovery backup.
#[derive(Debug, Clone)]
pub struct TransformOptions {
    pub out_dir: Option<PathBuf>,
    pub override_source: bool,
    pub size: Option<ImageSize>,
    pub mode: ResizeMode,
    pub upscale: bool,
    pub jpeg_quality: u8,
    pub workers: usize,
    pub dry_run: bool,
    /// Extra space on each side, expressed as a fraction of the object's width/height.
    pub padding: f64,
    pub include_crowd: bool,
}

impl Default for TransformOptions {
    fn default() -> Self {
        Self {
            out_dir: None,
            override_source: false,
            size: None,
            mode: ResizeMode::Fit,
            upscale: false,
            jpeg_quality: 90,
            workers: super::default_workers(),
            dry_run: false,
            padding: 0.,
            include_crowd: false,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct TransformReport {
    pub operation: String,
    pub out_dir: PathBuf,
    pub backup: Option<PathBuf>,
    pub manifest: PathBuf,
    pub input_images: usize,
    pub output_images: usize,
    pub output_labels: usize,
    pub skipped_background_images: usize,
    pub skipped_crowd_objects: usize,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub dry_run: bool,
}

pub fn convert(
    path: impl AsRef<Path>,
    to: Conversion,
    options: &TransformOptions,
) -> Result<TransformReport, DatasetError> {
    if to == Conversion::Detect && options.size.is_some() {
        return Err(DatasetError::new(
            "--imgsz applies to classification crops; use data resize for detection images",
        ));
    }
    if to == Conversion::Detect
        && (options.padding != 0.
            || options.include_crowd
            || options.mode != ResizeMode::Fit
            || options.upscale
            || options.jpeg_quality != 90)
    {
        return Err(DatasetError::new(
            "detection conversion copies original images; crop and image encoding options apply only to --to classify",
        ));
    }
    run(path.as_ref(), Some(to), options)
}

pub fn resize(
    path: impl AsRef<Path>,
    options: &TransformOptions,
) -> Result<TransformReport, DatasetError> {
    if options.size.is_none() {
        return Err(DatasetError::new("resize requires --imgsz"));
    }
    run(path.as_ref(), None, options)
}

#[derive(Debug, Clone, Copy)]
struct Geometry {
    resized: [u32; 2],
    canvas: [u32; 2],
    pad: [u32; 2],
}

impl Geometry {
    fn new(source: [u32; 2], options: &TransformOptions) -> Self {
        let Some(size) = options.size else {
            return Self {
                resized: source,
                canvas: source,
                pad: [0, 0],
            };
        };
        let target = [size.width, size.height];
        let resized = if options.mode == ResizeMode::Stretch {
            target
        } else {
            let mut scale =
                (target[0] as f64 / source[0] as f64).min(target[1] as f64 / source[1] as f64);
            if !options.upscale {
                scale = scale.min(1.);
            }
            [
                (source[0] as f64 * scale).round().max(1.) as u32,
                (source[1] as f64 * scale).round().max(1.) as u32,
            ]
        };
        let canvas = if options.mode == ResizeMode::Letterbox {
            target
        } else {
            resized
        };
        Self {
            resized,
            canvas,
            pad: [(canvas[0] - resized[0]) / 2, (canvas[1] - resized[1]) / 2],
        }
    }
    fn point(self, point: [f64; 2]) -> [f64; 2] {
        [
            (point[0] * self.resized[0] as f64 + self.pad[0] as f64) / self.canvas[0] as f64,
            (point[1] * self.resized[1] as f64 + self.pad[1] as f64) / self.canvas[1] as f64,
        ]
    }
    fn bbox(self, b: [f64; 4]) -> [f64; 4] {
        let a = self.point([b[0], b[1]]);
        let z = self.point([b[2], b[3]]);
        [a[0], a[1], z[0], z[1]]
    }
    fn apply(self, source: RgbImage) -> Result<RgbImage, DatasetError> {
        let resized = if [source.width(), source.height()] == self.resized {
            source
        } else {
            use fast_image_resize as fir;
            let mut destination =
                fir::images::Image::new(self.resized[0], self.resized[1], fir::PixelType::U8x3);
            fir::Resizer::new()
                .resize(
                    &source,
                    &mut destination,
                    &fir::ResizeOptions::new()
                        .resize_alg(fir::ResizeAlg::Convolution(fir::FilterType::Bilinear)),
                )
                .map_err(|e| error("resize", e))?;
            RgbImage::from_raw(self.resized[0], self.resized[1], destination.into_vec())
                .expect("RGB resize buffer")
        };
        if self.canvas == self.resized {
            return Ok(resized);
        }
        let mut canvas = RgbImage::from_pixel(self.canvas[0], self.canvas[1], Rgb([114, 114, 114]));
        image::imageops::replace(
            &mut canvas,
            &resized,
            self.pad[0] as i64,
            self.pad[1] as i64,
        );
        Ok(canvas)
    }
}

fn crop_box(record: &Record, object: &Object, padding: f64) -> [u32; 4] {
    let [x1, y1, x2, y2] = object.bbox;
    let dx = (x2 - x1) * padding;
    let dy = (y2 - y1) * padding;
    [
        ((x1 - dx).clamp(0., 1.) * record.size[0] as f64).floor() as u32,
        ((y1 - dy).clamp(0., 1.) * record.size[1] as f64).floor() as u32,
        ((x2 + dx).clamp(0., 1.) * record.size[0] as f64)
            .ceil()
            .min(record.size[0] as f64) as u32,
        ((y2 + dy).clamp(0., 1.) * record.size[1] as f64)
            .ceil()
            .min(record.size[1] as f64) as u32,
    ]
}

fn filename(record: &Record, id: usize, copied: bool) -> String {
    let stem = record
        .image
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .chars()
        .take(48)
        .collect::<String>();
    let ext = if copied {
        record
            .image
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("jpg")
    } else {
        "jpg"
    };
    format!("{id:08}-{stem}.{ext}")
}

fn validate_options(options: &TransformOptions) -> Result<(), DatasetError> {
    if options.out_dir.is_some() == options.override_source {
        return Err(DatasetError::new(
            "specify exactly one of --out-dir DIRECTORY or --override",
        ));
    }
    if options.workers == 0
        || options.jpeg_quality == 0
        || options.jpeg_quality > 100
        || !options.padding.is_finite()
        || !(0.0..=1.0).contains(&options.padding)
    {
        return Err(DatasetError::new(
            "workers must be positive, quality 1..100, and padding 0..1",
        ));
    }
    if options
        .size
        .is_some_and(|s| s.width == 0 || s.height == 0 || s.width > 16384 || s.height > 16384)
    {
        return Err(DatasetError::new(
            "imgsz dimensions must be between 1 and 16384",
        ));
    }
    Ok(())
}

fn run(
    path: &Path,
    conversion: Option<Conversion>,
    options: &TransformOptions,
) -> Result<TransformReport, DatasetError> {
    validate_options(options)?;
    let index = index(path, false, options.workers)?;
    if index.report.invalid_images > 0 {
        return Err(DatasetError::new(format!(
            "dataset has {} invalid images/labels; run data inspect --path {} --json to review them; no output written",
            index.report.invalid_images,
            path.display()
        )));
    }
    let classify = conversion == Some(Conversion::Classify);
    let classification = classify || index.dataset.format == DatasetFormat::ClassificationFolders;
    if conversion.is_some() && index.dataset.format == DatasetFormat::ClassificationFolders {
        return Err(DatasetError::new(
            "dataset is already classification; use data resize to change its images",
        ));
    }
    if classification {
        let mut folders = BTreeSet::new();
        for name in &index.dataset.class_names {
            validate_class_folder(name)?;
            if !folders.insert(name.to_lowercase()) {
                return Err(DatasetError::new(format!(
                    "class names collide on case-insensitive filesystems: {name:?}; rename classes in the YAML before classification export"
                )));
            }
        }
    }
    let operation = match conversion {
        Some(Conversion::Detect) => "detect",
        Some(Conversion::Classify) => "classify",
        None => "resize",
    };
    let (output, backup) = if options.override_source {
        validate_override(&index)?;
        (
            index.dataset.root.clone(),
            Some(backup_path(&index.dataset.root)?),
        )
    } else {
        (
            prepare_output_path(
                &index,
                options.out_dir.as_ref().expect("validated destination"),
            )?,
            None,
        )
    };
    let manifest_name = if options.override_source {
        index
            .dataset
            .manifest
            .file_name()
            .expect("validated root manifest")
    } else {
        std::ffi::OsStr::new("data.yaml")
    };
    let mut report = TransformReport {
        operation: operation.into(),
        manifest: output.join(manifest_name),
        out_dir: output.clone(),
        backup: backup.clone(),
        input_images: index.records.len(),
        output_images: 0,
        output_labels: 0,
        skipped_background_images: 0,
        skipped_crowd_objects: 0,
        input_bytes: index.report.image_bytes,
        output_bytes: 0,
        dry_run: options.dry_run,
    };
    let mut per_split = BTreeMap::<&str, usize>::new();
    for record in &index.records {
        let count = if classify {
            if record.objects.is_empty() {
                report.skipped_background_images += 1;
            }
            let count = record
                .objects
                .iter()
                .filter(|o| options.include_crowd || !o.crowd)
                .count();
            report.skipped_crowd_objects += record.objects.len() - count;
            count
        } else {
            1
        };
        *per_split.entry(record.split).or_default() += count;
        report.output_images += count;
        report.output_labels += if classify || classification {
            count
        } else {
            record.objects.len()
        };
    }
    if per_split.get("train").copied().unwrap_or(0) == 0 {
        return Err(DatasetError::new(
            "conversion would produce an empty training split; no output written",
        ));
    }
    if options.dry_run {
        return Ok(report);
    }
    let transaction = OutputTransaction::new(&output)?;
    let staging = &transaction.staging;
    let coco_output = conversion.is_none() && index.dataset.format == DatasetFormat::Coco;
    for (split, count) in &per_split {
        if *count == 0 {
            continue;
        }
        if classification {
            for name in &index.dataset.class_names {
                create_dir(&staging.join(split).join(name))?;
            }
        } else {
            create_dir(&staging.join("images").join(split))?;
            if !coco_output {
                create_dir(&staging.join("labels").join(split))?;
            }
        }
    }
    if coco_output {
        create_dir(&staging.join("annotations"))?;
    }
    let results = parallel_map(
        &index.records.iter().enumerate().collect::<Vec<_>>(),
        options.workers,
        |&(id, record)| {
            transform_record(
                record,
                id,
                &index,
                conversion,
                options,
                staging,
                coco_output,
            )
        },
    );
    let mut provenance = BufWriter::new(
        fs::File::create(staging.join("sources.jsonl")).map_err(|e| error("sources.jsonl", e))?,
    );
    let mut coco_by_split = BTreeMap::<&str, (Vec<Value>, Vec<Value>)>::new();
    for (record, result) in index.records.iter().zip(results) {
        let result = result?;
        report.output_bytes += result.bytes;
        for mut source in result.sources {
            if let Some(backup) = &backup {
                // The original files now live under the retained backup after
                // publication, so provenance should resolve there, not into the
                // replacement dataset that may reuse entirely different names.
                let original = Path::new(source["source"].as_str().expect("source image path"));
                let relative = original
                    .strip_prefix(&index.dataset.root)
                    .expect("validated source within dataset root");
                let retained = backup.join(relative);
                source["original_source"] = source["source"].take();
                source["source"] = json!(retained);
            }
            serde_json::to_writer(&mut provenance, &source)
                .map_err(|e| error("sources.jsonl", e))?;
            writeln!(provenance).map_err(|e| error("sources.jsonl", e))?;
        }
        if let Some((image, annotations)) = result.coco {
            let entry = coco_by_split.entry(record.split).or_default();
            entry.0.push(image);
            entry.1.extend(annotations);
        }
    }
    provenance.flush().map_err(|e| error("sources.jsonl", e))?;
    drop(provenance);
    for (split, (images, mut annotations)) in coco_by_split {
        for (id, annotation) in annotations.iter_mut().enumerate() {
            annotation["id"] = json!(id);
        }
        let categories = index
            .dataset
            .class_names
            .iter()
            .enumerate()
            .map(|(id, name)| json!({"id":id,"name":name}))
            .collect::<Vec<_>>();
        write_json(
            &staging.join("annotations").join(format!("{split}.json")),
            &json!({"images":images,"annotations":annotations,"categories":categories}),
        )?;
    }
    let format = if classification {
        "classification-folders"
    } else if coco_output {
        "coco"
    } else {
        "yolo"
    };
    let mut manifest = serde_yaml::Mapping::new();
    manifest.insert("path".into(), ".".into());
    manifest.insert("format".into(), format.into());
    manifest.insert(
        "names".into(),
        serde_yaml::to_value(&index.dataset.class_names).map_err(|e| error("names", e))?,
    );
    for (split, count) in per_split {
        if count == 0 {
            continue;
        }
        manifest.insert(
            split.into(),
            if classification {
                split.to_string()
            } else {
                format!("images/{split}")
            }
            .into(),
        );
        if coco_output {
            manifest.insert(
                format!("{split}_annotations").into(),
                format!("annotations/{split}.json").into(),
            );
        }
    }
    fs::write(
        staging.join(manifest_name),
        serde_yaml::to_string(&manifest).map_err(|e| error("data.yaml", e))?,
    )
    .map_err(|e| error("data.yaml", e))?;
    write_json(
        &staging.join("transform.json"),
        &serde_json::to_value(&report).map_err(|e| error("transform report", e))?,
    )?;
    // Verify that the exact produced manifest is loadable before making the output visible.
    super::manifest::DatasetManifest::load(staging.join(manifest_name))?;
    transaction.commit(backup.as_deref())?;
    Ok(report)
}

struct RecordOutput {
    bytes: u64,
    sources: Vec<Value>,
    coco: Option<(Value, Vec<Value>)>,
}

fn transform_record(
    record: &Record,
    id: usize,
    index: &Index,
    conversion: Option<Conversion>,
    options: &TransformOptions,
    staging: &Path,
    coco_output: bool,
) -> Result<RecordOutput, DatasetError> {
    let mut output = RecordOutput {
        bytes: 0,
        sources: Vec::new(),
        coco: None,
    };
    if conversion == Some(Conversion::Classify) {
        let objects = record
            .objects
            .iter()
            .enumerate()
            .filter(|(_, o)| options.include_crowd || !o.crowd)
            .collect::<Vec<_>>();
        if objects.is_empty() {
            return Ok(output);
        }
        let image = decode(record)?;
        for (object_id, object) in objects {
            let [x1, y1, x2, y2] = crop_box(record, object, options.padding);
            let crop = image::imageops::crop_imm(&image, x1, y1, x2 - x1, y2 - y1).to_image();
            let geometry = Geometry::new([crop.width(), crop.height()], options);
            let crop = geometry.apply(crop)?;
            let relative = PathBuf::from(record.split)
                .join(&index.dataset.class_names[object.class_id])
                .join(format!("{id:08}-{object_id:05}.jpg"));
            output.bytes += save_jpeg(&crop, &staging.join(&relative), options.jpeg_quality)?;
            output.sources.push(json!({"output":relative.to_string_lossy().replace('\\',"/"),"source":record.image,"object":object_id,"class_id":object.class_id,"crop_xyxy":[x1,y1,x2,y2],"size":geometry.canvas}));
        }
        return Ok(output);
    }
    let copied = conversion == Some(Conversion::Detect);
    let name = filename(record, id, copied);
    let relative = if let Some(class) = record.class_id {
        PathBuf::from(record.split)
            .join(&index.dataset.class_names[class])
            .join(&name)
    } else {
        PathBuf::from("images").join(record.split).join(&name)
    };
    let geometry = Geometry::new(record.size, options);
    if copied {
        output.bytes = fs::copy(&record.image, staging.join(&relative))
            .map_err(|e| error(record.image.display(), e))?;
    } else {
        let image = geometry.apply(decode(record)?)?;
        output.bytes = save_jpeg(&image, &staging.join(&relative), options.jpeg_quality)?;
    }
    output.sources.push(json!({"output":relative.to_string_lossy().replace('\\',"/"),"source":record.image,"source_size":record.size,"size":geometry.canvas,"resized":geometry.resized,"pad":geometry.pad}));
    if coco_output {
        let mut annotations = Vec::new();
        for object in &record.objects {
            let b = geometry.bbox(object.bbox);
            let bbox = [
                b[0] * geometry.canvas[0] as f64,
                b[1] * geometry.canvas[1] as f64,
                (b[2] - b[0]) * geometry.canvas[0] as f64,
                (b[3] - b[1]) * geometry.canvas[1] as f64,
            ];
            let segmentation = object
                .segmentation
                .as_ref()
                .map(|s| resize_segmentation(s, record.size, geometry))
                .transpose()?;
            let area = segmentation
                .as_ref()
                .map(segmentation_area)
                .unwrap_or(bbox[2] * bbox[3]);
            let mut annotation = json!({"image_id":id,"category_id":object.class_id,"bbox":bbox,"iscrowd":u8::from(object.crowd),"area":area});
            if let Some(segmentation) = segmentation {
                annotation["segmentation"] = segmentation;
            }
            annotations.push(annotation);
        }
        output.coco = Some((
            json!({"id":id,"file_name":name,"width":geometry.canvas[0],"height":geometry.canvas[1]}),
            annotations,
        ));
    } else if record.class_id.is_none() {
        let mut labels = String::new();
        use std::fmt::Write;
        for object in &record.objects {
            let _ = write!(labels, "{}", object.class_id);
            if !copied && let Some(points) = &object.polygon {
                for point in points {
                    let p = geometry.point(*point);
                    let _ = write!(labels, " {:.10} {:.10}", p[0], p[1]);
                }
            } else {
                let b = geometry.bbox(object.bbox);
                let _ = write!(
                    labels,
                    " {:.10} {:.10} {:.10} {:.10}",
                    (b[0] + b[2]) / 2.,
                    (b[1] + b[3]) / 2.,
                    b[2] - b[0],
                    b[3] - b[1]
                );
            }
            labels.push('\n');
        }
        let label_path = staging
            .join("labels")
            .join(record.split)
            .join(Path::new(&name).with_extension("txt"));
        fs::write(&label_path, labels).map_err(|e| error(label_path.display(), e))?;
    }
    Ok(output)
}

fn decode(record: &Record) -> Result<RgbImage, DatasetError> {
    let image = image::open(&record.image).map_err(|e| error(record.image.display(), e))?;
    if [image.width(), image.height()] != record.size {
        return Err(error(
            record.image.display(),
            "image dimensions changed during transformation",
        ));
    }
    Ok(image.into_rgb8())
}

fn save_jpeg(image: &RgbImage, path: &Path, quality: u8) -> Result<u64, DatasetError> {
    let file = fs::File::create(path).map_err(|e| error(path.display(), e))?;
    let mut writer = BufWriter::new(file);
    JpegEncoder::new_with_quality(&mut writer, quality)
        .encode_image(image)
        .map_err(|e| error(path.display(), e))?;
    writer.flush().map_err(|e| error(path.display(), e))?;
    Ok(writer
        .get_ref()
        .metadata()
        .map_err(|e| error(path.display(), e))?
        .len())
}

fn validate_class_folder(name: &str) -> Result<(), DatasetError> {
    let upper = name
        .split('.')
        .next()
        .unwrap_or_default()
        .to_ascii_uppercase();
    let reserved = matches!(upper.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (upper.len() == 4
            && (upper.starts_with("COM") || upper.starts_with("LPT"))
            && upper.as_bytes()[3].is_ascii_digit());
    if name == "."
        || name == ".."
        || name.ends_with(['.', ' '])
        || name
            .chars()
            .any(|c| c.is_control() || "<>:\"/\\|?*".contains(c))
        || reserved
    {
        return Err(DatasetError::new(format!(
            "class name {name:?} cannot be a portable class folder; rename it in the YAML before classification export"
        )));
    }
    Ok(())
}

fn create_dir(path: &Path) -> Result<(), DatasetError> {
    fs::create_dir_all(path).map_err(|e| error(path.display(), e))
}
fn write_json(path: &Path, value: &Value) -> Result<(), DatasetError> {
    let mut writer = BufWriter::new(fs::File::create(path).map_err(|e| error(path.display(), e))?);
    serde_json::to_writer_pretty(&mut writer, value).map_err(|e| error(path.display(), e))?;
    writer.flush().map_err(|e| error(path.display(), e))
}

fn prepare_output_path(index: &Index, requested: &Path) -> Result<PathBuf, DatasetError> {
    if requested.exists() || requested.symlink_metadata().is_ok() {
        return Err(error(
            requested.display(),
            "directory already exists; choose a new --out-dir directory, or use --override to replace the source dataset",
        ));
    }
    let name = requested
        .file_name()
        .ok_or_else(|| DatasetError::new("output must name a new directory"))?;
    // Resolve existing ancestors without creating anything (also used by --dry-run).
    let absolute = if requested.is_absolute() {
        requested.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| error("current directory", e))?
            .join(requested)
    };
    let mut ancestor = absolute
        .parent()
        .ok_or_else(|| DatasetError::new("output has no parent"))?;
    let mut tail = Vec::new();
    while !ancestor.exists() {
        tail.push(
            ancestor
                .file_name()
                .ok_or_else(|| DatasetError::new("invalid output path"))?
                .to_os_string(),
        );
        ancestor = ancestor
            .parent()
            .ok_or_else(|| DatasetError::new("invalid output parent"))?;
    }
    let mut resolved = fs::canonicalize(ancestor).map_err(|e| error(ancestor.display(), e))?;
    for component in tail.into_iter().rev() {
        resolved.push(component);
    }
    resolved.push(name);
    if resolved.starts_with(&index.dataset.root)
        || index.records.iter().any(|r| r.image.starts_with(&resolved))
    {
        return Err(DatasetError::new(
            "output must be outside the source dataset root; choose a sibling directory",
        ));
    }
    Ok(resolved)
}

fn validate_override(index: &Index) -> Result<(), DatasetError> {
    let root = &index.dataset.root;
    if root.file_name().is_none() || root.join(".git").exists() {
        return Err(DatasetError::new(
            "--override requires a dedicated dataset directory, not a filesystem or Git repository root; use --out-dir",
        ));
    }
    if index.dataset.manifest.parent() != Some(root.as_path()) {
        return Err(DatasetError::new(
            "--override requires the dataset YAML directly inside its dataset root; use --out-dir for external or nested manifests",
        ));
    }
    if index
        .records
        .iter()
        .any(|record| !record.image.starts_with(root))
        || [
            &index.dataset.train_annotations,
            &index.dataset.val_annotations,
            &index.dataset.test_annotations,
        ]
        .into_iter()
        .flatten()
        .any(|path| !path.starts_with(root))
    {
        return Err(DatasetError::new(
            "--override requires all source images and annotations inside the dataset root; use --out-dir for datasets referencing external files",
        ));
    }
    if index.dataset.format == DatasetFormat::Yolo {
        for record in &index.records {
            let label = super::manifest::yolo_label_path(&record.image);
            if label.exists()
                && !fs::canonicalize(&label)
                    .map_err(|e| error(label.display(), e))?
                    .starts_with(root)
            {
                return Err(DatasetError::new(
                    "--override requires labels inside the dataset root; use --out-dir for external labels",
                ));
            }
        }
    }
    Ok(())
}

fn backup_path(root: &Path) -> Result<PathBuf, DatasetError> {
    let name = root
        .file_name()
        .expect("validated dataset directory")
        .to_string_lossy();
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| error("system clock", e))?
        .as_nanos();
    Ok(root.with_file_name(format!("{name}.backup-{timestamp}")))
}

struct OutputTransaction {
    staging: PathBuf,
    output: PathBuf,
    committed: bool,
}
impl OutputTransaction {
    fn new(output: &Path) -> Result<Self, DatasetError> {
        static SEQUENCE: AtomicUsize = AtomicUsize::new(0);
        let parent = output.parent().expect("validated output parent");
        create_dir(parent)?;
        let staging = parent.join(format!(
            ".montgomery-data-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&staging).map_err(|e| error(staging.display(), e))?;
        Ok(Self {
            staging,
            output: output.to_path_buf(),
            committed: false,
        })
    }
    fn commit(mut self, backup: Option<&Path>) -> Result<(), DatasetError> {
        if let Some(backup) = backup {
            if backup.symlink_metadata().is_ok() {
                return Err(error(
                    backup.display(),
                    "recovery backup already exists; source unchanged",
                ));
            }
            fs::rename(&self.output, backup).map_err(|e| {
                error(
                    self.output.display(),
                    format!("cannot move source to recovery backup: {e}; source unchanged; close processes using the dataset and retry from its parent directory, or use --out-dir"),
                )
            })?;
            if let Err(publish) = fs::rename(&self.staging, &self.output) {
                return match fs::rename(backup, &self.output) {
                    Ok(()) => Err(error(
                        self.output.display(),
                        format!(
                            "cannot publish transformed dataset: {publish}; original dataset restored"
                        ),
                    )),
                    Err(restore) => Err(error(
                        self.output.display(),
                        format!(
                            "cannot publish transformed dataset: {publish}; restore failed: {restore}; ORIGINAL DATA IS SAFE AT {}",
                            backup.display()
                        ),
                    )),
                };
            }
            self.committed = true;
            return Ok(());
        }
        if self.output.exists() || self.output.symlink_metadata().is_ok() {
            return Err(error(
                self.output.display(),
                "output appeared during transformation; refusing to overwrite",
            ));
        }
        fs::rename(&self.staging, &self.output).map_err(|e| error(self.output.display(), e))?;
        self.committed = true;
        Ok(())
    }
}
impl Drop for OutputTransaction {
    fn drop(&mut self) {
        if !self.committed {
            let _ = fs::remove_dir_all(&self.staging);
        }
    }
}

fn resize_segmentation(
    seg: &Value,
    source: [u32; 2],
    geometry: Geometry,
) -> Result<Value, DatasetError> {
    if let Some(polygons) = seg.as_array() {
        let mut output = Vec::new();
        for polygon in polygons {
            let mut points = Vec::new();
            for pair in polygon
                .as_array()
                .expect("validated polygon")
                .as_chunks::<2>()
                .0
            {
                let p = geometry.point([
                    pair[0].as_f64().expect("validated coordinate") / source[0] as f64,
                    pair[1].as_f64().expect("validated coordinate") / source[1] as f64,
                ]);
                points.extend([
                    p[0] * geometry.canvas[0] as f64,
                    p[1] * geometry.canvas[1] as f64,
                ]);
            }
            output.push(points);
        }
        return Ok(json!(output));
    }
    let counts = super::rle::counts(seg, source)?;
    // Column-major binary mask, matching COCO's run ordering. Decode one mask per object.
    let mut mask = vec![0u8; source[0] as usize * source[1] as usize];
    let mut cursor = 0;
    for (i, count) in counts.iter().enumerate() {
        mask[cursor..cursor + *count as usize].fill((i % 2) as u8);
        cursor += *count as usize;
    }
    let mut resized_counts = Vec::<u64>::new();
    let mut last = 0;
    let mut run = 0;
    for x in 0..geometry.canvas[0] {
        for y in 0..geometry.canvas[1] {
            let value = if x >= geometry.pad[0]
                && y >= geometry.pad[1]
                && x < geometry.pad[0] + geometry.resized[0]
                && y < geometry.pad[1] + geometry.resized[1]
            {
                let sx = ((x - geometry.pad[0]) as u64 * source[0] as u64
                    / geometry.resized[0] as u64) as usize;
                let sy = ((y - geometry.pad[1]) as u64 * source[1] as u64
                    / geometry.resized[1] as u64) as usize;
                mask[sx * source[1] as usize + sy]
            } else {
                0
            };
            if value != last {
                resized_counts.push(run);
                run = 0;
                last = value;
            }
            run += 1;
        }
    }
    resized_counts.push(run);
    Ok(json!({"size":[geometry.canvas[1],geometry.canvas[0]],"counts":resized_counts}))
}

fn segmentation_area(seg: &Value) -> f64 {
    if let Some(polygons) = seg.as_array() {
        polygons
            .iter()
            .map(|polygon| {
                let coordinates = polygon.as_array().expect("polygon");
                let points = coordinates
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|p| [p[0].as_f64().unwrap(), p[1].as_f64().unwrap()])
                    .collect::<Vec<_>>();
                points
                    .iter()
                    .zip(points.iter().cycle().skip(1))
                    .take(points.len())
                    .map(|(a, b)| a[0] * b[1] - b[0] * a[1])
                    .sum::<f64>()
                    .abs()
                    / 2.
            })
            .sum()
    } else {
        seg["counts"]
            .as_array()
            .expect("resized RLE")
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 1)
            .map(|(_, v)| v.as_u64().unwrap() as f64)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_override_publication_restores_original_directory() {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/dataset-transaction-tests")
            .join(format!("{}-{timestamp}", std::process::id()));
        let source = root.join("source");
        let backup = root.join("backup");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("original.txt"), "keep original").unwrap();
        let transaction = OutputTransaction::new(&source).unwrap();
        let staging = transaction.staging.clone();
        // A missing staging directory forces publication to fail after the
        // original directory has successfully moved to its recovery location.
        fs::remove_dir(&staging).unwrap();
        let error = transaction.commit(Some(&backup)).unwrap_err();
        assert!(error.to_string().contains("original dataset restored"));
        assert_eq!(
            fs::read_to_string(source.join("original.txt")).unwrap(),
            "keep original"
        );
        assert!(!backup.exists());
        assert!(!staging.exists());
        fs::remove_dir_all(root).unwrap();
    }
}
