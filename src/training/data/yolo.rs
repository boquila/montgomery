use std::{fs, path::Path};

use crate::training::{
    data::{
        manifest::DatasetError,
        sample::{DetectionTarget, SegmentationSource},
    },
    geometry::BoxXyxy,
};

#[derive(Debug, Clone, Copy)]
pub struct YoloParseOptions {
    pub num_classes: usize,
    pub clipping_tolerance: f32,
}

impl YoloParseOptions {
    pub fn new(num_classes: usize) -> Self {
        Self {
            num_classes,
            clipping_tolerance: 1e-3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParsedLabels {
    pub targets: Vec<DetectionTarget>,
    pub clipped_coordinates: usize,
}

/// Parse detection or polygon YOLO labels. A missing file is a valid background image.
pub fn parse_labels(
    path: impl AsRef<Path>,
    image_id: &str,
    image_size: [u32; 2],
    options: YoloParseOptions,
) -> Result<ParsedLabels, DatasetError> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(ParsedLabels {
            targets: Vec::new(),
            clipped_coordinates: 0,
        });
    }
    let text = fs::read_to_string(path).map_err(|error| contextual(path, image_id, 0, error))?;
    let mut result = ParsedLabels {
        targets: Vec::new(),
        clipped_coordinates: 0,
    };
    for (line_index, line) in text.lines().enumerate() {
        let line_number = line_index + 1;
        if line.trim().is_empty() {
            continue;
        }
        let values: Vec<f32> = line
            .split_ascii_whitespace()
            .map(|value| value.parse::<f32>())
            .collect::<Result<_, _>>()
            .map_err(|error| contextual(path, image_id, line_number, error))?;
        if values.len() != 5 && (values.len() < 7 || values.len().is_multiple_of(2)) {
            return Err(contextual(
                path,
                image_id,
                line_number,
                "expected box or polygon record",
            ));
        }
        let class_value = values[0];
        if !class_value.is_finite() || class_value.fract() != 0.0 || class_value < 0.0 {
            return Err(contextual(
                path,
                image_id,
                line_number,
                "class ID must be a non-negative integer",
            ));
        }
        let class_id = class_value as usize;
        if class_id >= options.num_classes {
            return Err(contextual(
                path,
                image_id,
                line_number,
                format!("class ID {class_id} is outside 0..{}", options.num_classes),
            ));
        }
        let mut coordinates = values[1..].to_vec();
        for coordinate in &mut coordinates {
            if !coordinate.is_finite()
                || *coordinate < -options.clipping_tolerance
                || *coordinate > 1.0 + options.clipping_tolerance
            {
                return Err(contextual(
                    path,
                    image_id,
                    line_number,
                    "normalized coordinate outside [0, 1]",
                ));
            }
            let clipped = coordinate.clamp(0.0, 1.0);
            result.clipped_coordinates += usize::from(clipped != *coordinate);
            *coordinate = clipped;
        }
        let (bbox, segmentation) = if values.len() == 5 {
            let [width, height] = [image_size[0] as f32, image_size[1] as f32];
            let cx = coordinates[0] * width;
            let cy = coordinates[1] * height;
            let half_width = coordinates[2] * width * 0.5;
            let half_height = coordinates[3] * height * 0.5;
            let bbox = BoxXyxy::new([
                cx - half_width,
                cy - half_height,
                cx + half_width,
                cy + half_height,
            ])
            .and_then(|bbox| {
                bbox.clip(width, height)
                    .ok_or("box is empty after clipping")
            })
            .map_err(|error| contextual(path, image_id, line_number, error))?;
            (bbox, None)
        } else {
            let polygon: Vec<[f32; 2]> = coordinates
                .chunks_exact(2)
                .map(|point| {
                    [
                        point[0] * image_size[0] as f32,
                        point[1] * image_size[1] as f32,
                    ]
                })
                .collect();
            let xmin = polygon
                .iter()
                .map(|point| point[0])
                .fold(f32::INFINITY, f32::min);
            let ymin = polygon
                .iter()
                .map(|point| point[1])
                .fold(f32::INFINITY, f32::min);
            let xmax = polygon
                .iter()
                .map(|point| point[0])
                .fold(f32::NEG_INFINITY, f32::max);
            let ymax = polygon
                .iter()
                .map(|point| point[1])
                .fold(f32::NEG_INFINITY, f32::max);
            let bbox = BoxXyxy::new([xmin, ymin, xmax, ymax])
                .map_err(|error| contextual(path, image_id, line_number, error))?;
            (bbox, Some(SegmentationSource::Polygons(vec![polygon])))
        };
        result.targets.push(DetectionTarget {
            class_id,
            bbox,
            segmentation,
            crowd: false,
            source_annotation_id: Some(line_number as u64),
        });
    }
    Ok(result)
}

fn contextual(
    path: &Path,
    image_id: &str,
    line: usize,
    error: impl std::fmt::Display,
) -> DatasetError {
    DatasetError::new(format!(
        "{} (image {image_id}, annotation line {line}): {error}",
        path.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_label_is_valid_background() {
        let path =
            std::env::temp_dir().join(format!("boquilens-missing-{}.txt", std::process::id()));
        let parsed = parse_labels(path, "empty", [100, 50], YoloParseOptions::new(3)).unwrap();
        assert!(parsed.targets.is_empty());
    }
}
