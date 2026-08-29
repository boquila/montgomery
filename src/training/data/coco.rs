use std::{collections::BTreeMap, fs, path::Path};

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
    pub class_names: Vec<String>,
    pub category_to_class: BTreeMap<u64, usize>,
    pub dropped_invalid: usize,
}

pub fn load(
    annotation_json: impl AsRef<Path>,
    images_root: impl AsRef<Path>,
) -> Result<CocoDataset, DatasetError> {
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
    let mut samples = Vec::with_capacity(parsed.images.len());
    let mut dropped_invalid = 0;
    for image_record in parsed.images {
        let path = images_root.as_ref().join(&image_record.file_name);
        let image = image::open(&path).map_err(|error| {
            DatasetError::new(format!(
                "cannot decode COCO image {} (image {}): {error}",
                path.display(),
                image_record.id
            ))
        })?;
        if image.width() != image_record.width || image.height() != image_record.height {
            return Err(DatasetError::new(format!(
                "COCO image {} dimensions disagree with record {}",
                path.display(),
                image_record.id
            )));
        }
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
                                    .chunks_exact(2)
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
        samples.push(VisionSample {
            image,
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
    Ok(CocoDataset {
        samples,
        class_names,
        category_to_class,
        dropped_invalid,
    })
}
