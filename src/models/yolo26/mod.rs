//! Native Burn implementation of the YOLO26n detector.
//!
//! [`body::Yolo26Body`] implements the complete backbone/neck and produces the P3/P4/P5 tensors
//! consumed by the Ultralytics Detect head. YOLO26 is DFL-free (`reg_max = 1`) and end-to-end
//! (`end2end = True`), so only the NMS-free one2one inference branch is implemented; the
//! training-only one2many branch is not loaded from official checkpoints.

pub mod blocks;
pub mod body;
pub mod head;
pub mod model;
pub mod weights;

pub use model::{Yolo26, Yolo26Config};
