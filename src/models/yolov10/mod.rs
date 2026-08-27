//! Native Burn implementation of the YOLOv10n detector.
//!
//! [`body::Yolov10Body`] implements the complete backbone/neck and produces the P3/P4/P5 tensors
//! consumed by the Ultralytics v10Detect head. Only the NMS-free one2one inference branch is
//! implemented; the training-only one2many branch is not loaded from official checkpoints.

pub mod blocks;
pub mod body;
pub mod head;
pub mod model;
pub mod weights;

pub use model::{Yolov10, Yolov10Config};
