use burn::tensor::{Bool, Int, Tensor, TensorData, backend::Backend};

use super::sample::ImageMeta;
use crate::data::augmentation::AugRng;
use crate::data::augmentation::{FormattedClassificationSample, FormattedDetectionSample};

#[derive(Debug, Clone, PartialEq)]
pub struct FormattedDetectionBatch {
    pub images_nchw_u8: Vec<u8>,
    pub image_shape: [usize; 4],
    pub batch_indexes: Vec<usize>,
    pub classes: Vec<u32>,
    pub boxes_xywh_normalized: Vec<[f32; 4]>,
}

impl FormattedDetectionBatch {
    pub fn collate(samples: &[FormattedDetectionSample]) -> Result<Self, String> {
        let first = samples
            .first()
            .ok_or_else(|| "cannot collate an empty batch".to_string())?;
        if samples.iter().any(|s| s.image_shape != first.image_shape) {
            return Err("formatted images have incompatible shapes".into());
        }
        let [channels, height, width] = first.image_shape;
        let mut images = Vec::with_capacity(samples.len() * channels * height * width);
        let mut batch_indexes = Vec::new();
        let mut classes = Vec::new();
        let mut boxes = Vec::new();
        for (batch_index, sample) in samples.iter().enumerate() {
            images.extend_from_slice(&sample.image_chw_u8);
            batch_indexes.extend(std::iter::repeat_n(batch_index, sample.classes.len()));
            classes.extend_from_slice(&sample.classes);
            boxes.extend_from_slice(&sample.boxes_xywh_normalized);
        }
        Ok(Self {
            images_nchw_u8: images,
            image_shape: [samples.len(), channels, height, width],
            batch_indexes,
            classes,
            boxes_xywh_normalized: boxes,
        })
    }

    pub fn images_f32(&self) -> Vec<f32> {
        self.images_nchw_u8
            .iter()
            .map(|v| *v as f32 / 255.0)
            .collect()
    }

    /// CPU compatibility fallback for trainer-side multi-scale bilinear resize.
    /// Normalized targets and batch indexes are deliberately not modified.
    pub fn images_f32_multi_scale(
        &self,
        sampled_side: usize,
        stride: usize,
    ) -> Result<(Vec<f32>, [usize; 4]), String> {
        let [batch, channels, height, width] = self.image_shape;
        let [new_height, new_width] = multi_scale_shape([height, width], sampled_side, stride)?;
        let input = self.images_f32();
        if [new_height, new_width] == [height, width] {
            return Ok((input, self.image_shape));
        }
        let mut output = vec![0.0; batch * channels * new_height * new_width];
        for n in 0..batch {
            for c in 0..channels {
                for y in 0..new_height {
                    let source_y = (y as f32 + 0.5) * height as f32 / new_height as f32 - 0.5;
                    let y0 = source_y.floor().clamp(0.0, (height - 1) as f32) as usize;
                    let y1 = (y0 + 1).min(height - 1);
                    let wy = (source_y - y0 as f32).clamp(0.0, 1.0);
                    for x in 0..new_width {
                        let source_x = (x as f32 + 0.5) * width as f32 / new_width as f32 - 0.5;
                        let x0 = source_x.floor().clamp(0.0, (width - 1) as f32) as usize;
                        let x1 = (x0 + 1).min(width - 1);
                        let wx = (source_x - x0 as f32).clamp(0.0, 1.0);
                        let at = |yy: usize, xx: usize| {
                            input[((n * channels + c) * height + yy) * width + xx]
                        };
                        let top = at(y0, x0) * (1.0 - wx) + at(y0, x1) * wx;
                        let bottom = at(y1, x0) * (1.0 - wx) + at(y1, x1) * wx;
                        output[((n * channels + c) * new_height + y) * new_width + x] =
                            top * (1.0 - wy) + bottom * wy;
                    }
                }
            }
        }
        Ok((output, [batch, channels, new_height, new_width]))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FormattedClassificationBatch {
    pub images_nchw_f32: Vec<f32>,
    pub image_shape: [usize; 4],
    pub classes: Vec<u32>,
}

impl FormattedClassificationBatch {
    pub fn collate(samples: &[FormattedClassificationSample]) -> Result<Self, String> {
        let first = samples
            .first()
            .ok_or_else(|| "cannot collate an empty batch".to_string())?;
        if samples.iter().any(|s| s.image_shape != first.image_shape) {
            return Err("formatted classification images have incompatible shapes".into());
        }
        let mut images = Vec::with_capacity(samples.len() * first.image_chw_f32.len());
        let mut classes = Vec::with_capacity(samples.len());
        for sample in samples {
            images.extend_from_slice(&sample.image_chw_f32);
            classes.push(sample.class_id);
        }
        Ok(Self {
            images_nchw_f32: images,
            image_shape: [
                samples.len(),
                first.image_shape[0],
                first.image_shape[1],
                first.image_shape[2],
            ],
            classes,
        })
    }
}

/// Reproduce trainer-side stride rounding for a sampled multi-scale side.
pub fn multi_scale_shape(
    shape: [usize; 2],
    sampled_side: usize,
    stride: usize,
) -> Result<[usize; 2], String> {
    if stride == 0 || sampled_side == 0 || shape.contains(&0) {
        return Err("multi-scale dimensions and stride must be positive".into());
    }
    let target = sampled_side / stride * stride;
    if target == 0 {
        return Err("sampled multi-scale side is smaller than model stride".into());
    }
    let scale = target as f32 / shape[0].max(shape[1]) as f32;
    Ok(shape.map(|dimension| {
        (((dimension as f32 * scale / stride as f32).ceil() as usize) * stride).max(stride)
    }))
}

pub fn sample_multi_scale_side(
    imgsz: usize,
    variation: f32,
    stride: usize,
    rng: &mut AugRng,
) -> Result<usize, String> {
    if imgsz == 0 || stride == 0 || !variation.is_finite() || !(0.0..=1.0).contains(&variation) {
        return Err("invalid multi-scale configuration".into());
    }
    if variation == 0.0 {
        return Ok(imgsz / stride * stride);
    }
    let minimum = ((imgsz as f32 * (1.0 - variation)) as usize).max(stride);
    let maximum = ((imgsz as f32 * (1.0 + variation)) as usize).max(minimum);
    let sampled = rng.uniform_inclusive_i32(minimum as i32, maximum as i32) as usize;
    Ok(sampled / stride * stride)
}

pub struct DetectionBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub classes: Tensor<B, 2, Int>,
    pub boxes_xyxy: Tensor<B, 3>,
    pub valid: Tensor<B, 2, Bool>,
    pub metadata: Vec<ImageMeta>,
}

pub struct SegmentationBatch<B: Backend> {
    pub detection: DetectionBatch<B>,
    pub masks: Tensor<B, 4>,
    pub semantic_class_map: Tensor<B, 3, Int>,
    /// Explicit foreground gate: class zero is a valid semantic target.
    pub semantic_coverage: Tensor<B, 3, Bool>,
}

pub struct ClassificationBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub classes: Tensor<B, 1, Int>,
    pub metadata: Vec<ImageMeta>,
}

impl FormattedDetectionBatch {
    /// Upload a formatted detection batch using padded targets and an explicit validity mask.
    pub fn into_device<B: Backend>(
        self,
        metadata: Vec<ImageMeta>,
        device: &B::Device,
    ) -> Result<DetectionBatch<B>, String> {
        let [batch, channels, height, width] = self.image_shape;
        if metadata.len() != batch || channels != 3 {
            return Err("detection metadata/image batch shape mismatch".into());
        }
        let max_objects = self
            .batch_indexes
            .iter()
            .copied()
            .fold(vec![0_usize; batch], |mut counts, index| {
                if index < batch {
                    counts[index] += 1;
                }
                counts
            })
            .into_iter()
            .max()
            .unwrap_or(0)
            .max(1);
        if self.classes.len() != self.batch_indexes.len()
            || self.classes.len() != self.boxes_xywh_normalized.len()
            || self.batch_indexes.iter().any(|index| *index >= batch)
        {
            return Err("formatted detection target arrays are inconsistent".into());
        }
        let mut classes = vec![0_i64; batch * max_objects];
        let mut boxes = vec![0_f32; batch * max_objects * 4];
        let mut valid = vec![false; batch * max_objects];
        let mut offsets = vec![0_usize; batch];
        for ((batch_index, class), xywh) in self
            .batch_indexes
            .into_iter()
            .zip(self.classes)
            .zip(self.boxes_xywh_normalized)
        {
            let object = offsets[batch_index];
            offsets[batch_index] += 1;
            let flat = batch_index * max_objects + object;
            classes[flat] = i64::from(class);
            valid[flat] = true;
            let [cx, cy, w, h] = xywh;
            boxes[flat * 4..flat * 4 + 4].copy_from_slice(&[
                (cx - w * 0.5) * width as f32,
                (cy - h * 0.5) * height as f32,
                (cx + w * 0.5) * width as f32,
                (cy + h * 0.5) * height as f32,
            ]);
        }
        Ok(DetectionBatch {
            images: Tensor::from_data(
                TensorData::new(
                    self.images_nchw_u8
                        .into_iter()
                        .map(|v| v as f32 / 255.0)
                        .collect(),
                    [batch, channels, height, width],
                ),
                device,
            ),
            classes: Tensor::from_data(TensorData::new(classes, [batch, max_objects]), device),
            boxes_xyxy: Tensor::from_data(TensorData::new(boxes, [batch, max_objects, 4]), device),
            valid: Tensor::from_data(TensorData::new(valid, [batch, max_objects]), device),
            metadata,
        })
    }
}

/// Collate explicit instance masks and the YOLO26 smallest-area-wins semantic map.
pub fn segmentation_into_device<B: Backend>(
    samples: &[FormattedDetectionSample],
    metadata: Vec<ImageMeta>,
    device: &B::Device,
) -> Result<SegmentationBatch<B>, String> {
    use crate::data::augmentation::MaskTargets;

    let formatted = FormattedDetectionBatch::collate(samples)?;
    let batch = samples.len();
    let max_objects = samples
        .iter()
        .map(|sample| sample.classes.len())
        .max()
        .unwrap_or(0)
        .max(1);
    let shape = samples
        .iter()
        .find_map(|sample| {
            sample.masks.as_ref().map(|masks| match masks {
                MaskTargets::OverlapU8 { shape, .. }
                | MaskTargets::OverlapU16 { shape, .. }
                | MaskTargets::Separate { shape, .. } => *shape,
            })
        })
        .ok_or_else(|| "segmentation batch contains no mask targets".to_string())?;
    let [mask_height, mask_width] = shape;
    let mut masks = vec![0_f32; batch * max_objects * mask_height * mask_width];
    let mut semantic = vec![0_i64; batch * mask_height * mask_width];
    let mut coverage = vec![false; batch * mask_height * mask_width];
    for (image, sample) in samples.iter().enumerate() {
        let Some(mask_targets) = &sample.masks else {
            return Err("segmentation sample is missing masks".into());
        };
        let separate: Vec<Vec<u8>> = match mask_targets {
            MaskTargets::Separate {
                data,
                shape: current,
            } => {
                if *current != shape || data.len() != sample.classes.len() {
                    return Err("separate mask shape/count mismatch".into());
                }
                data.clone()
            }
            MaskTargets::OverlapU8 {
                data,
                shape: current,
            } => {
                if *current != shape {
                    return Err("overlap mask shape mismatch".into());
                }
                (0..sample.classes.len())
                    .map(|object| {
                        data.iter()
                            .map(|value| u8::from(*value as usize == object + 1))
                            .collect()
                    })
                    .collect()
            }
            MaskTargets::OverlapU16 {
                data,
                shape: current,
            } => {
                if *current != shape {
                    return Err("overlap mask shape mismatch".into());
                }
                (0..sample.classes.len())
                    .map(|object| {
                        data.iter()
                            .map(|value| u8::from(*value as usize == object + 1))
                            .collect()
                    })
                    .collect()
            }
        };
        let mut by_area = (0..separate.len()).collect::<Vec<_>>();
        by_area.sort_by_key(|index| {
            separate[*index]
                .iter()
                .map(|v| usize::from(*v != 0))
                .sum::<usize>()
        });
        for (object, mask) in separate.iter().enumerate() {
            if mask.len() != mask_height * mask_width {
                return Err("instance mask buffer has invalid length".into());
            }
            let start = (image * max_objects + object) * mask_height * mask_width;
            for (destination, value) in masks[start..start + mask.len()].iter_mut().zip(mask) {
                *destination = f32::from(*value != 0);
            }
        }
        // Iterate large to small so the smallest covering instance is written last.
        for object in by_area.into_iter().rev() {
            for (pixel, value) in separate[object].iter().enumerate() {
                if *value != 0 {
                    let flat = image * mask_height * mask_width + pixel;
                    semantic[flat] = i64::from(sample.classes[object]);
                    coverage[flat] = true;
                }
            }
        }
    }
    let detection = formatted.into_device(metadata, device)?;
    Ok(SegmentationBatch {
        detection,
        masks: Tensor::from_data(
            TensorData::new(masks, [batch, max_objects, mask_height, mask_width]),
            device,
        ),
        semantic_class_map: Tensor::from_data(
            TensorData::new(semantic, [batch, mask_height, mask_width]),
            device,
        ),
        semantic_coverage: Tensor::from_data(
            TensorData::new(coverage, [batch, mask_height, mask_width]),
            device,
        ),
    })
}

impl FormattedClassificationBatch {
    pub fn into_device<B: Backend>(
        self,
        metadata: Vec<ImageMeta>,
        device: &B::Device,
    ) -> Result<ClassificationBatch<B>, String> {
        let [batch, channels, height, width] = self.image_shape;
        if metadata.len() != batch || self.classes.len() != batch {
            return Err("classification metadata/target batch mismatch".into());
        }
        Ok(ClassificationBatch {
            images: Tensor::from_data(
                TensorData::new(self.images_nchw_f32, [batch, channels, height, width]),
                device,
            ),
            classes: Tensor::from_data(
                TensorData::new(self.classes.into_iter().map(i64::from).collect(), [batch]),
                device,
            ),
            metadata,
        })
    }
}

#[cfg(test)]
mod formatted_tests {
    use super::*;
    #[test]
    fn multi_scale_rounds_each_dimension_up_to_stride() {
        assert_eq!(multi_scale_shape([320, 640], 608, 32).unwrap(), [320, 608]);
    }

    #[test]
    fn multi_scale_resizes_images_but_not_targets() {
        let batch = FormattedDetectionBatch {
            images_nchw_u8: vec![0, 255, 255, 0],
            image_shape: [1, 1, 2, 2],
            batch_indexes: vec![0],
            classes: vec![3],
            boxes_xywh_normalized: vec![[0.5, 0.5, 1.0, 1.0]],
        };
        let (images, shape) = batch.images_f32_multi_scale(4, 1).unwrap();
        assert_eq!(shape, [1, 1, 4, 4]);
        assert_eq!(images.len(), 16);
        assert_eq!(batch.boxes_xywh_normalized, [[0.5, 0.5, 1.0, 1.0]]);
    }
}
