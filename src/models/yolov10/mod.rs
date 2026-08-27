//! Native Burn implementation of the Ultralytics YOLOv10 detector family (n/s/m/b/l/x).
//!
//! [`body`] implements the complete backbone/neck for every scale and produces the P3/P4/P5
//! tensors consumed by the Ultralytics v10Detect head. Only the NMS-free one2one inference branch
//! is implemented; the training-only one2many branch is not loaded from official checkpoints.
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
