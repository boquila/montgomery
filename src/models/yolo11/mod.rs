//! Native Burn implementation of the Ultralytics YOLO11 detector family (n/s/m/l/x) and the
//! YOLO11-seg instance-segmentation variants (n/s).
//!
//! [`body`] implements the complete backbone/neck for every scale and produces the P3/P4/P5
//! tensors consumed by the Ultralytics Detect head. YOLO11 keeps the classic DFL detection head
//! (`reg_max = 16`) and is **not** end-to-end: its per-anchor predictions require DFL projection,
//! anchor-grid decoding, and external class-aware non-maximum suppression, which the runtime
//! applies with Ultralytics' default thresholds. At the m/l/x scales `parse_model` forces the
//! C3k chain onto every C3k2 stage, so those variants declare a structurally different body graph.
//! The `-seg` variants reuse the same bodies and detection decode and add Ultralytics' Segment
//! head ([`segment_head`]): a stride-4 Proto module plus 32 raw mask coefficients per anchor that
//! ride along through the same class-aware NMS.

pub mod blocks;
pub mod body;
pub mod head;
pub mod model;
pub mod segment_head;
pub mod weights;

pub use model::{
    Yolo11L, Yolo11LConfig, Yolo11M, Yolo11MConfig, Yolo11N, Yolo11NConfig, Yolo11S, Yolo11SConfig,
    Yolo11SegN, Yolo11SegNConfig, Yolo11SegS, Yolo11SegSConfig, Yolo11X, Yolo11XConfig,
};
pub use segment_head::SegmentOutput;
