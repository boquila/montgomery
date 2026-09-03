use serde::{Deserialize, Serialize};
use std::{error::Error, fmt};

use super::Instances;
use super::{BBox, BoxFormat, Interpolation, resize};
use crate::training::data::{SegmentationSource, VisionSample};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ColorOrder {
    Bgr,
    Rgb,
    Gray,
    MultiChannel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteImage {
    width: usize,
    height: usize,
    channels: usize,
    color: ColorOrder,
    data: Vec<u8>,
}

impl ByteImage {
    pub fn new(
        width: usize,
        height: usize,
        channels: usize,
        color: ColorOrder,
        data: Vec<u8>,
    ) -> Result<Self, AugmentationError> {
        if width == 0 || height == 0 || channels == 0 {
            return Err(AugmentationError::new(
                "image dimensions and channels must be positive",
            ));
        }
        let expected = width
            .checked_mul(height)
            .and_then(|v| v.checked_mul(channels))
            .ok_or_else(|| AugmentationError::new("image dimensions overflow"))?;
        if data.len() != expected {
            return Err(AugmentationError::new(format!(
                "image buffer has {} bytes, expected {expected}",
                data.len()
            )));
        }
        if matches!(color, ColorOrder::Bgr | ColorOrder::Rgb) && channels != 3 {
            return Err(AugmentationError::new(
                "BGR/RGB images must have three channels",
            ));
        }
        if color == ColorOrder::Gray && channels != 1 {
            return Err(AugmentationError::new("gray images must have one channel"));
        }
        Ok(Self {
            width,
            height,
            channels,
            color,
            data,
        })
    }

    pub fn filled(
        width: usize,
        height: usize,
        channels: usize,
        color: ColorOrder,
        value: u8,
    ) -> Self {
        Self::new(
            width,
            height,
            channels,
            color,
            vec![value; width * height * channels],
        )
        .expect("positive internal image shape")
    }

    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }
    pub fn channels(&self) -> usize {
        self.channels
    }
    pub fn color(&self) -> ColorOrder {
        self.color
    }
    pub fn set_color(&mut self, color: ColorOrder) {
        self.color = color;
    }
    pub fn data(&self) -> &[u8] {
        &self.data
    }
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
    pub fn offset(&self, x: usize, y: usize, channel: usize) -> usize {
        (y * self.width + x) * self.channels + channel
    }
    pub fn pixel(&self, x: usize, y: usize) -> &[u8] {
        let start = self.offset(x, y, 0);
        &self.data[start..start + self.channels]
    }
    pub fn pixel_mut(&mut self, x: usize, y: usize) -> &mut [u8] {
        let start = self.offset(x, y, 0);
        &mut self.data[start..start + self.channels]
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceMetadata {
    pub primary_id: String,
    pub primary_index: usize,
    pub mixed_indexes: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeometryMetadata {
    pub original_shape: [usize; 2],
    pub current_shape: [usize; 2],
    pub ratio: [f32; 2],
    pub pad: [f32; 2],
    pub reversible: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AugSample {
    pub image: ByteImage,
    pub classes: Vec<u32>,
    pub instances: Instances,
    pub source: SourceMetadata,
    pub geometry: GeometryMetadata,
}

impl AugSample {
    /// Decode the training dataset contract into immutable-loader-compatible HWC BGR bytes.
    pub fn from_vision(
        sample: VisionSample,
        source_index: usize,
        imgsz: usize,
        stretch: bool,
    ) -> Result<Self, AugmentationError> {
        if imgsz == 0 {
            return Err(AugmentationError::new("loader image size must be positive"));
        }
        let rgb = sample.image.to_rgb8();
        let original = [rgb.height() as usize, rgb.width() as usize];
        let mut bgr = Vec::with_capacity(rgb.as_raw().len());
        for pixel in rgb.as_raw().as_chunks::<3>().0 {
            bgr.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]);
        }
        let source_image = ByteImage::new(original[1], original[0], 3, ColorOrder::Bgr, bgr)?;
        let (width, height, ratio) = if stretch {
            (
                imgsz,
                imgsz,
                [
                    imgsz as f32 / original[1] as f32,
                    imgsz as f32 / original[0] as f32,
                ],
            )
        } else {
            let scale = (imgsz as f32 / original[0].max(original[1]) as f32).min(1.0);
            let width = ((original[1] as f32 * scale).ceil() as usize).clamp(1, imgsz);
            let height = ((original[0] as f32 * scale).ceil() as usize).clamp(1, imgsz);
            (
                width,
                height,
                [
                    width as f32 / original[1] as f32,
                    height as f32 / original[0] as f32,
                ],
            )
        };
        let image = resize::resize(&source_image, width, height, Interpolation::Bilinear)?;
        let mut boxes = Vec::with_capacity(sample.targets.len());
        let mut classes = Vec::with_capacity(sample.targets.len());
        let has_segments = sample
            .targets
            .iter()
            .any(|target| target.segmentation.is_some());
        if has_segments
            && sample
                .targets
                .iter()
                .any(|target| target.segmentation.is_none())
        {
            return Err(AugmentationError::new(
                "a segmentation sample must carry one polygon set per instance",
            ));
        }
        let mut segments = has_segments.then(Vec::new);
        for target in sample.targets {
            boxes.push(BBox([
                target.bbox.xmin * ratio[0],
                target.bbox.ymin * ratio[1],
                target.bbox.xmax * ratio[0],
                target.bbox.ymax * ratio[1],
            ]));
            classes.push(u32::try_from(target.class_id).map_err(|_| {
                AugmentationError::new("class ID does not fit the augmentation contract")
            })?);
            if let Some(destination) = &mut segments {
                let polygons = match target.segmentation.expect("presence checked above") {
                    SegmentationSource::Polygons(polygons) => polygons,
                    SegmentationSource::UncompressedRle { .. }
                    | SegmentationSource::CompressedRle { .. } => {
                        return Err(AugmentationError::new(
                            "RLE segmentation must be converted to polygons before augmentation",
                        ));
                    }
                };
                // Ultralytics carries one outline per instance. Multiple COCO rings are joined in
                // source order; dataset-specific hole semantics require an explicit conversion.
                destination.push(
                    polygons
                        .into_iter()
                        .flatten()
                        .map(|point| [point[0] * ratio[0], point[1] * ratio[1]])
                        .collect(),
                );
            }
        }
        let instances = Instances::new(boxes, BoxFormat::Xyxy, false, segments)?;
        let result = Self {
            image,
            classes,
            instances,
            source: SourceMetadata {
                primary_id: sample.image_id,
                primary_index: source_index,
                mixed_indexes: vec![],
            },
            geometry: GeometryMetadata {
                original_shape: original,
                current_shape: [height, width],
                ratio,
                pad: [0.0, 0.0],
                reversible: true,
            },
        };
        result.validate()?;
        Ok(result)
    }

    pub fn validate(&self) -> Result<(), AugmentationError> {
        if self.classes.len() != self.instances.len() {
            return Err(AugmentationError::new(format!(
                "sample {} has {} classes but {} instances",
                self.source.primary_id,
                self.classes.len(),
                self.instances.len()
            )));
        }
        self.instances.validate()?;
        if self.geometry.current_shape != [self.image.height(), self.image.width()] {
            return Err(AugmentationError::new(
                "geometry current shape does not match image",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AugmentationError {
    message: String,
    trace: Option<Box<super::trace::AugmentationTrace>>,
}

impl AugmentationError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            trace: None,
        }
    }
    pub fn with_trace(mut self, trace: super::trace::AugmentationTrace) -> Self {
        self.trace = Some(Box::new(trace));
        self
    }
    pub fn trace(&self) -> Option<&super::trace::AugmentationTrace> {
        self.trace.as_deref()
    }
    pub fn context(self, transform: &str, sample: &AugSample) -> Self {
        Self {
            message: format!(
                "{transform} failed for {} ({}x{}x{}, {} instances): {}",
                sample.source.primary_id,
                sample.image.width(),
                sample.image.height(),
                sample.image.channels(),
                sample.instances.len(),
                self.message
            ),
            trace: self.trace,
        }
    }
}

impl fmt::Display for AugmentationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for AugmentationError {}
