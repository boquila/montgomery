use super::{
    Index, Record, annotations, error,
    manifest::{DatasetError, DatasetFormat, DatasetManifest},
    parallel_map,
};
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

/// A contextual problem found while reading an image or its annotations.
#[derive(Debug, Serialize)]
pub struct Issue {
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClassCount {
    pub id: usize,
    pub name: String,
    pub images: usize,
    pub labels: usize,
}

#[derive(Debug, Serialize)]
pub struct SplitReport {
    pub images: usize,
    pub labels: usize,
    pub background_images: usize,
    pub missing_label_files: usize,
    pub empty_label_files: usize,
    pub image_bytes: u64,
    pub label_types: BTreeMap<String, usize>,
    pub classes: Vec<ClassCount>,
}

/// Counts include each split membership; unique_images deduplicates canonical paths.
#[derive(Debug, Serialize)]
pub struct DatasetReport {
    pub path: String,
    pub format: DatasetFormat,
    pub task: String,
    pub images: usize,
    pub unique_images: usize,
    pub labels: usize,
    pub image_bytes: u64,
    pub label_types: BTreeMap<String, usize>,
    pub classes: Vec<ClassCount>,
    pub splits: BTreeMap<String, SplitReport>,
    pub dimensions: BTreeMap<String, usize>,
    pub cross_split_images: usize,
    pub invalid_images: usize,
    pub issues: Vec<Issue>,
    pub verified_pixels: bool,
}

impl DatasetReport {
    /// Compact human-readable summary, with a stable JSON alternative for scripts.
    pub fn text(&self) -> String {
        use std::fmt::Write;
        let mut out = format!(
            "Dataset: {}\nFormat: {} | Labels: {}\n{} images ({} unique), {} labels, {:.2} MiB of images\n",
            self.path,
            self.format,
            self.task,
            self.images,
            self.unique_images,
            self.labels,
            self.image_bytes as f64 / 1_048_576.
        );
        let _ = writeln!(
            out,
            "Label types: {}",
            if self.label_types.is_empty() {
                "none".into()
            } else {
                self.label_types
                    .iter()
                    .map(|(kind, count)| format!("{kind}: {count}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        );
        out.push_str("\nSplit       Images     Labels  Background  Missing labels\n");
        for (name, split) in &self.splits {
            let _ = writeln!(
                out,
                "{name:<10} {:>7} {:>10} {:>11} {:>15}",
                split.images, split.labels, split.background_images, split.missing_label_files
            );
        }
        out.push_str("\nClass                       Images     Labels\n");
        for class in &self.classes {
            let _ = writeln!(
                out,
                "{:>3} {:<24} {:>7} {:>10}",
                class.id, class.name, class.images, class.labels
            );
        }
        let mut dimensions = self.dimensions.iter().collect::<Vec<_>>();
        dimensions.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        let _ = writeln!(
            out,
            "\nImage sizes ({} distinct): {}",
            dimensions.len(),
            dimensions
                .iter()
                .take(8)
                .map(|(size, n)| format!("{size} ({n})"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        if self.cross_split_images > 0 {
            let _ = writeln!(
                out,
                "Warning: {} images occur in multiple splits (data leakage).",
                self.cross_split_images
            );
        }
        let missing: usize = self.splits.values().map(|s| s.missing_label_files).sum();
        if missing > 0 {
            let _ = writeln!(
                out,
                "Warning: {missing} missing label files are counted as background; verify this is intentional."
            );
        }
        if !self.issues.is_empty() {
            let _ = writeln!(
                out,
                "\n{} invalid images / annotation records:",
                self.invalid_images
            );
            for issue in self.issues.iter().take(20) {
                let _ = writeln!(out, "  {}: {}", issue.path, issue.message);
            }
            if self.issues.len() > 20 {
                let _ = writeln!(
                    out,
                    "  ... {} more (use --json for all issues)",
                    self.issues.len() - 20
                );
            }
        }
        if self.verified_pixels {
            out.push_str("Pixel decoding verified.\n");
        }
        out
    }
}

/// Inspect all splits. Header reads are cheap; verify_images additionally decodes pixels.
pub fn inspect(
    path: impl AsRef<Path>,
    verify_images: bool,
    workers: usize,
) -> Result<DatasetReport, DatasetError> {
    Ok(index(path.as_ref(), verify_images, workers)?.report)
}

pub(super) fn index(path: &Path, verify: bool, workers: usize) -> Result<Index, DatasetError> {
    if workers == 0 {
        return Err(DatasetError::new("workers must be at least 1"));
    }
    let dataset = DatasetManifest::load(path)?;
    let classes = dataset
        .class_names
        .iter()
        .enumerate()
        .map(|(id, name)| ClassCount {
            id,
            name: name.clone(),
            images: 0,
            labels: 0,
        })
        .collect::<Vec<_>>();
    let mut report = DatasetReport {
        path: dataset.manifest.display().to_string(),
        format: dataset.format,
        task: String::new(),
        images: 0,
        unique_images: 0,
        labels: 0,
        image_bytes: 0,
        label_types: BTreeMap::new(),
        classes,
        splits: BTreeMap::new(),
        dimensions: BTreeMap::new(),
        cross_split_images: 0,
        invalid_images: 0,
        issues: Vec::new(),
        verified_pixels: verify,
    };
    let mut memberships = BTreeMap::<_, BTreeSet<&str>>::new();
    let mut records = Vec::new();
    let class_ids = dataset
        .class_names
        .iter()
        .enumerate()
        .map(|(id, name)| (name.as_str(), id))
        .collect::<BTreeMap<_, _>>();
    for (split, paths, annotation) in [
        ("train", &dataset.train_images, &dataset.train_annotations),
        ("val", &dataset.val_images, &dataset.val_annotations),
        ("test", &dataset.test_images, &dataset.test_annotations),
    ] {
        if paths.is_empty() {
            continue;
        }
        let coco = if dataset.format == DatasetFormat::Coco {
            annotations::coco(
                annotation.as_ref().expect("resolved COCO annotation"),
                paths,
                &dataset,
            )?
        } else {
            BTreeMap::new()
        };
        let mut summary = SplitReport {
            images: paths.len(),
            labels: 0,
            background_images: 0,
            missing_label_files: 0,
            empty_label_files: 0,
            image_bytes: 0,
            label_types: BTreeMap::new(),
            classes: report
                .classes
                .iter()
                .map(|c| ClassCount {
                    id: c.id,
                    name: c.name.clone(),
                    images: 0,
                    labels: 0,
                })
                .collect(),
        };
        let results = parallel_map(paths, workers, |image| -> Result<Record, DatasetError> {
            let size = if verify {
                let decoded = image::open(image).map_err(|e| error(image.display(), e))?;
                [decoded.width(), decoded.height()]
            } else {
                let (w, h) =
                    image::image_dimensions(image).map_err(|e| error(image.display(), e))?;
                [w, h]
            };
            if size.contains(&0) {
                return Err(error(image.display(), "zero-sized image"));
            }
            let bytes = fs::metadata(image)
                .map_err(|e| error(image.display(), e))?
                .len();
            let (objects, missing_label, class_id) = match dataset.format {
                DatasetFormat::Yolo => {
                    let (objects, missing) = annotations::yolo(image, dataset.class_names.len())?;
                    (objects, missing, None)
                }
                DatasetFormat::Coco => {
                    let record = coco
                        .get(image)
                        .ok_or_else(|| error(image.display(), "no matching COCO image record"))?;
                    if record.size != size {
                        return Err(error(
                            image.display(),
                            "COCO dimensions differ from actual image dimensions",
                        ));
                    }
                    (record.objects.clone(), false, None)
                }
                DatasetFormat::ClassificationFolders => {
                    let class_id = image
                        .ancestors()
                        .skip(1)
                        .find_map(|p| dataset.class_folders.get(p).copied())
                        .or_else(|| {
                            // Individual-image and image-list splits have no declared
                            // class root; their names table identifies ancestor folders.
                            dataset
                                .class_folders
                                .is_empty()
                                .then(|| {
                                    image
                                        .ancestors()
                                        .skip(1)
                                        .take_while(|p| *p != dataset.root)
                                        .filter_map(|p| {
                                            p.file_name()
                                                .and_then(|name| name.to_str())
                                                .and_then(|name| class_ids.get(name).copied())
                                        })
                                        .last()
                                })
                                .flatten()
                        })
                        .ok_or_else(|| {
                            error(
                                image.display(),
                                "image has no class folder matching YAML names",
                            )
                        })?;
                    (Vec::new(), false, Some(class_id))
                }
            };
            Ok(Record {
                image: image.clone(),
                split,
                size,
                bytes,
                objects,
                class_id,
                missing_label,
            })
        });
        report.images += paths.len();
        for (path, result) in paths.iter().zip(results) {
            memberships.entry(path.clone()).or_default().insert(split);
            match result {
                Ok(record) => {
                    summary.image_bytes += record.bytes;
                    *report
                        .dimensions
                        .entry(format!("{}x{}", record.size[0], record.size[1]))
                        .or_default() += 1;
                    let mut seen_classes = BTreeSet::new();
                    if let Some(class) = record.class_id {
                        summary.classes[class].labels += 1;
                        summary.labels += 1;
                        *summary
                            .label_types
                            .entry("classification".into())
                            .or_default() += 1;
                        seen_classes.insert(class);
                    } else {
                        if record.objects.is_empty() {
                            summary.background_images += 1;
                            if record.missing_label {
                                summary.missing_label_files += 1;
                            } else if dataset.format == DatasetFormat::Yolo {
                                summary.empty_label_files += 1;
                            }
                        }
                        for object in &record.objects {
                            summary.classes[object.class_id].labels += 1;
                            summary.labels += 1;
                            seen_classes.insert(object.class_id);
                            let kind = if object.polygon.is_some()
                                || object.segmentation.as_ref().is_some_and(|v| v.is_array())
                            {
                                "polygon"
                            } else if object.segmentation.is_some() {
                                "rle"
                            } else {
                                "bounding-box"
                            };
                            *summary.label_types.entry(kind.into()).or_default() += 1;
                        }
                    }
                    for class in seen_classes {
                        summary.classes[class].images += 1;
                    }
                    records.push(record);
                }
                Err(e) => {
                    report.invalid_images += 1;
                    report.issues.push(Issue {
                        path: path.display().to_string(),
                        message: e.to_string(),
                    });
                }
            }
        }
        report.labels += summary.labels;
        report.image_bytes += summary.image_bytes;
        for (kind, count) in &summary.label_types {
            *report.label_types.entry(kind.clone()).or_default() += count;
        }
        for (total, class) in report.classes.iter_mut().zip(&summary.classes) {
            total.labels += class.labels;
            total.images += class.images;
        }
        report.splits.insert(split.into(), summary);
    }
    report.unique_images = memberships.len();
    report.cross_split_images = memberships.values().filter(|s| s.len() > 1).count();
    report.task = if dataset.format == DatasetFormat::ClassificationFolders {
        "classification"
    } else if report.label_types.contains_key("polygon") || report.label_types.contains_key("rle") {
        if report.label_types.contains_key("bounding-box") {
            "mixed detection / instance segmentation"
        } else {
            "instance segmentation"
        }
    } else if report.labels > 0 {
        "detection"
    } else {
        "unlabeled / background only"
    }
    .into();
    Ok(Index {
        dataset,
        records,
        report,
    })
}
