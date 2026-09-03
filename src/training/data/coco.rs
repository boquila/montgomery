use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use serde::Deserialize;

use crate::training::{
    data::{DetectionTarget, SegmentationSource, VisionSample, manifest::DatasetError},
    geometry::BoxXyxy,
};

#[derive(Deserialize)]
struct CocoFile {
    images: Vec<CocoImage>,
    annotations: Vec<CocoAnnotation>,
    categories: Vec<CocoCategory>,
}
#[derive(Deserialize)]
struct CocoImage {
    id: u64,
    file_name: String,
    width: u32,
    height: u32,
}
#[derive(Deserialize)]
struct CocoCategory {
    id: u64,
    name: String,
}
#[derive(Deserialize)]
struct CocoAnnotation {
    id: u64,
    image_id: u64,
    category_id: u64,
    bbox: [f32; 4],
    #[serde(default)]
    iscrowd: u8,
    segmentation: Option<CocoSegmentation>,
}
#[derive(Deserialize)]
#[serde(untagged)]
enum CocoSegmentation {
    Polygons(Vec<Vec<f32>>),
    Rle { size: [u32; 2], counts: CocoCounts },
}
#[derive(Deserialize)]
#[serde(untagged)]
enum CocoCounts {
    Compressed(String),
    Uncompressed(Vec<u32>),
}

#[derive(Debug, Clone)]
pub struct CocoDataset {
    pub samples: Vec<VisionSample>,
    pub sample_paths: Vec<PathBuf>,
    pub class_names: Vec<String>,
    pub category_to_class: BTreeMap<u64, usize>,
    pub dropped_invalid: usize,
}

#[derive(Debug, Clone)]
pub struct CocoSampleRecord {
    pub path: PathBuf,
    pub targets: Vec<DetectionTarget>,
    pub image_id: String,
    pub source_size: [u32; 2],
}

#[derive(Debug, Clone)]
pub struct CocoIndex {
    pub records: Vec<CocoSampleRecord>,
    pub class_names: Vec<String>,
    pub category_to_class: BTreeMap<u64, usize>,
    pub dropped_invalid: usize,
}

impl CocoIndex {
    pub fn load_sample(&self, index: usize, training: bool) -> Result<VisionSample, DatasetError> {
        let record = self
            .records
            .get(index)
            .ok_or_else(|| DatasetError::new(format!("COCO sample index {index} out of range")))?;
        let image = image::open(&record.path).map_err(|error| {
            DatasetError::new(format!(
                "cannot decode COCO image {} (image {}): {error}",
                record.path.display(),
                record.image_id
            ))
        })?;
        if [image.width(), image.height()] != record.source_size {
            return Err(DatasetError::new(format!(
                "COCO image {} dimensions changed after indexing",
                record.path.display()
            )));
        }
        let mut targets = record.targets.clone();
        if training {
            targets.retain(|target| !target.crowd);
        }
        Ok(VisionSample {
            image,
            targets,
            image_id: record.image_id.clone(),
            source_size: record.source_size,
        })
    }
}

pub fn load(
    annotation_json: impl AsRef<Path>,
    images_root: impl AsRef<Path>,
) -> Result<CocoDataset, DatasetError> {
    let index = load_index(annotation_json, images_root)?;
    let samples = (0..index.records.len())
        .map(|sample| index.load_sample(sample, false))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CocoDataset {
        sample_paths: index
            .records
            .iter()
            .map(|record| record.path.clone())
            .collect(),
        samples,
        class_names: index.class_names,
        category_to_class: index.category_to_class,
        dropped_invalid: index.dropped_invalid,
    })
}

pub fn load_index(
    annotation_json: impl AsRef<Path>,
    images_root: impl AsRef<Path>,
) -> Result<CocoIndex, DatasetError> {
    let annotation_json = annotation_json.as_ref();
    let bytes = fs::read(annotation_json).map_err(|error| {
        DatasetError::new(format!(
            "cannot read COCO annotations {}: {error}",
            annotation_json.display()
        ))
    })?;
    let parsed: CocoFile = serde_json::from_slice(&bytes).map_err(|error| {
        DatasetError::new(format!(
            "invalid COCO JSON {}: {error}",
            annotation_json.display()
        ))
    })?;
    let mut categories = parsed.categories;
    categories.sort_by_key(|category| category.id);
    let category_to_class = categories
        .iter()
        .enumerate()
        .map(|(index, category)| (category.id, index))
        .collect::<BTreeMap<_, _>>();
    let class_names = categories
        .into_iter()
        .map(|category| category.name)
        .collect::<Vec<_>>();
    if class_names.is_empty() {
        return Err(DatasetError::new("COCO dataset has no categories"));
    }
    let mut by_image = BTreeMap::<u64, Vec<CocoAnnotation>>::new();
    for annotation in parsed.annotations {
        by_image
            .entry(annotation.image_id)
            .or_default()
            .push(annotation);
    }
    let mut records = Vec::with_capacity(parsed.images.len());
    let mut dropped_invalid = 0;
    for image_record in parsed.images {
        let mut path = images_root.as_ref().join(&image_record.file_name);
        if !path.exists()
            && let Some(file_name) = Path::new(&image_record.file_name).file_name()
        {
            path = images_root.as_ref().join(file_name);
        }
        let path = fs::canonicalize(&path).map_err(|error| {
            DatasetError::new(format!(
                "cannot resolve COCO image {}: {error}",
                path.display()
            ))
        })?;
        let mut targets = Vec::new();
        for annotation in by_image.remove(&image_record.id).unwrap_or_default() {
            let Some(&class_id) = category_to_class.get(&annotation.category_id) else {
                return Err(DatasetError::new(format!(
                    "COCO annotation {} references unknown category {}",
                    annotation.id, annotation.category_id
                )));
            };
            let [x, y, width, height] = annotation.bbox;
            let bbox = BoxXyxy::new([x, y, x + width, y + height]).and_then(|value| {
                value
                    .clip(image_record.width as f32, image_record.height as f32)
                    .ok_or("box is empty after clipping")
            });
            let Ok(bbox) = bbox else {
                dropped_invalid += 1;
                continue;
            };
            let segmentation = annotation
                .segmentation
                .map(|segmentation| match segmentation {
                    CocoSegmentation::Polygons(polygons) => SegmentationSource::Polygons(
                        polygons
                            .into_iter()
                            .map(|polygon| {
                                polygon
                                    .as_chunks::<2>()
                                    .0
                                    .iter()
                                    .map(|point| [point[0], point[1]])
                                    .collect()
                            })
                            .collect(),
                    ),
                    CocoSegmentation::Rle {
                        size,
                        counts: CocoCounts::Compressed(counts),
                    } => SegmentationSource::CompressedRle { size, counts },
                    CocoSegmentation::Rle {
                        size,
                        counts: CocoCounts::Uncompressed(counts),
                    } => SegmentationSource::UncompressedRle { size, counts },
                });
            targets.push(DetectionTarget {
                class_id,
                bbox,
                segmentation,
                crowd: annotation.iscrowd != 0,
                source_annotation_id: Some(annotation.id),
            });
        }
        records.push(CocoSampleRecord {
            path,
            targets,
            image_id: image_record.id.to_string(),
            source_size: [image_record.width, image_record.height],
        });
    }
    if !by_image.is_empty() {
        return Err(DatasetError::new(
            "COCO annotations reference missing image records",
        ));
    }
    Ok(CocoIndex {
        records,
        class_names,
        category_to_class,
        dropped_invalid,
    })
}
