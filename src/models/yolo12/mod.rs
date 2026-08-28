//! Native Burn implementation of the Ultralytics YOLO12 detector family (n/s/m/l/x).
//!
//! [`body`] implements the complete backbone/neck for every scale and produces the P3/P4/P5
//! tensors consumed by the Ultralytics Detect head. YOLO12 keeps the classic DFL detection head
//! (`reg_max = 16`) and is **not** end-to-end: its per-anchor predictions require DFL projection,
//! anchor-grid decoding, and external class-aware non-maximum suppression, which the runtime
//! applies with Ultralytics' default thresholds. Its `Detect` head is byte-identical to YOLO11's
//! (light DWConv classification towers — verified from the checkpoints), so the head module is
//! shared from [`crate::models::yolo11::head`] instead of duplicated.
//!
//! The family's distinguishing blocks are the area-attention stages: `A2C2f` pairs a C2f-style
//! split shell with either two `ABlock`s (backbone stages 6/8, YAML `a2=True`, area 4/1) or a
//! C3k chain (neck stages 11/14/17, YAML `a2=False`). The l/x scales extend the YAML args of the
//! attention stages with `residual=True, mlp_ratio=1.2`, adding a learnable per-channel gamma to
//! the residual around the whole block; the m/l/x scales additionally force the C3k chain onto
//! the early backbone C3k2 stages (layers 2/4) at 0.25 expansion.

pub mod blocks;
pub mod body;
pub mod model;
pub mod weights;

pub use model::{
    Yolo12L, Yolo12LConfig, Yolo12M, Yolo12MConfig, Yolo12N, Yolo12NConfig, Yolo12S, Yolo12SConfig,
    Yolo12X, Yolo12XConfig,
};
