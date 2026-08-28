//! Native Burn implementation of the Ultralytics YOLO26 detector family (n/s/m/l/x).
//!
//! [`body`] implements the complete backbone/neck for every scale and produces the P3/P4/P5
//! tensors consumed by the Ultralytics Detect head. YOLO26 is DFL-free (`reg_max = 1`) and
//! end-to-end (`end2end = True`), so only the NMS-free one2one inference branch is implemented;
//! the training-only one2many branch is not loaded from official checkpoints. For the m/l/x
//! scales `parse_model` forces the C3k chain onto the early backbone stages, so those variants
//! declare a structurally different body graph.

pub mod blocks;
pub mod body;
pub mod classification;
pub mod head;
pub mod model;
pub mod weights;

pub use classification::{
    Yolo26ClsL, Yolo26ClsLConfig, Yolo26ClsM, Yolo26ClsMConfig, Yolo26ClsN, Yolo26ClsNConfig,
    Yolo26ClsS, Yolo26ClsSConfig, Yolo26ClsX, Yolo26ClsXConfig,
};
pub use model::{
    Yolo26L, Yolo26LConfig, Yolo26M, Yolo26MConfig, Yolo26N, Yolo26NConfig, Yolo26S, Yolo26SConfig,
    Yolo26X, Yolo26XConfig,
};
