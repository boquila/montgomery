use image::DynamicImage;
use serde::{Deserialize, Serialize};

use crate::training::geometry::BoxXyxy;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SegmentationSource {
    Polygons(Vec<Vec<[f32; 2]>>),
    UncompressedRle { size: [u32; 2], counts: Vec<u32> },
    CompressedRle { size: [u32; 2], counts: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectionTarget {
    pub class_id: usize,
    pub bbox: BoxXyxy,
    pub segmentation: Option<SegmentationSource>,
    pub crowd: bool,
    pub source_annotation_id: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct VisionSample {
    pub image: DynamicImage,
    pub targets: Vec<DetectionTarget>,
    pub image_id: String,
    pub source_size: [u32; 2],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageMeta {
    pub image_id: String,
    pub source_size: [u32; 2],
    pub canvas_size: [u32; 2],
    pub scale: [f32; 2],
    pub pad: [f32; 2],
}
