//! Native Burn implementation of the YOLOv3-Tiny-Ultralytics detector.
//!
//! The model is being landed vertically. [`body::Yolov3TinyBody`] implements the complete
//! backbone/neck and produces the P4/P5 tensors consumed by the Ultralytics split detection head.

pub mod body;
pub mod head;
pub mod model;
pub mod weights;

pub use model::{Yolov3Tiny, Yolov3TinyConfig};
