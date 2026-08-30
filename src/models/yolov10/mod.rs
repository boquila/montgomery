//! Native Burn implementation of the Ultralytics YOLOv10 detector family (n/s/m/b/l/x).
//!
//! [`body`] implements the complete backbone/neck for every scale and produces the P3/P4/P5
//! tensors consumed by the Ultralytics v10Detect head. Default builds contain the NMS-free
//! one-to-one inference branch; the `training` feature restores and loads the one-to-many branch.
//! The scale variants are not mere width/depth rescalings: the official per-scale YAMLs swap
//! stage module flavors, so each scale declares its own body graph.

pub mod blocks;
pub mod body;
pub mod head;
pub mod model;
pub mod weights;

pub use model::{
    Yolov10B, Yolov10BConfig, Yolov10L, Yolov10LConfig, Yolov10M, Yolov10MConfig, Yolov10N,
    Yolov10NConfig, Yolov10S, Yolov10SConfig, Yolov10X, Yolov10XConfig,
};
