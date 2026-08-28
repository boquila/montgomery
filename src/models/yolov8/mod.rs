//! Native Burn implementation of the Ultralytics YOLOv8 detector family (n/s/m/l/x) and its
//! `-seg` and `-cls` task variants.
//!
//! [`body`] implements the complete backbone/neck for every scale and produces the P3/P4/P5
//! tensors consumed by the Ultralytics Detect head. YOLOv8 keeps the classic DFL detection head
//! (`reg_max = 16`) and is **not** end-to-end: its per-anchor predictions require DFL projection,
//! anchor-grid decoding, and external class-aware non-maximum suppression, which the runtime
//! applies with Ultralytics' default thresholds. Unlike YOLO11's light DWConv classification
//! towers, the YOLOv8 checkpoints predate that refactor and build the legacy full-3x3-conv `cv3`
//! towers ([`head`]). The family was verified to be a pure width/depth rescaling of one graph.
//!
//! The `-seg` variants reuse the same bodies and detection decode and add Ultralytics' Segment
//! head ([`segmentation`]): a stride-4 Proto module plus 32 raw mask coefficients per anchor that
//! ride along through the same class-aware NMS; the runtime-side output type is shared with
//! YOLO11-seg. The `-cls` variants ([`classification`]) run the ImageNet-1k classify graph, whose
//! backbone is a C2f chain without the C2PSA stage and whose batch norms carry plain PyTorch
//! defaults (see the BnFlavor invariant in AGENTS.md).

pub mod blocks;
pub mod body;
pub mod classification;
pub mod head;
pub mod model;
pub mod segmentation;
pub mod weights;

pub use classification::{
    Yolov8ClsL, Yolov8ClsLConfig, Yolov8ClsM, Yolov8ClsMConfig, Yolov8ClsN, Yolov8ClsNConfig,
    Yolov8ClsS, Yolov8ClsSConfig, Yolov8ClsX, Yolov8ClsXConfig,
};
pub use model::{
    Yolov8L, Yolov8LConfig, Yolov8M, Yolov8MConfig, Yolov8N, Yolov8NConfig, Yolov8S, Yolov8SConfig,
    Yolov8SegL, Yolov8SegLConfig, Yolov8SegM, Yolov8SegMConfig, Yolov8SegN, Yolov8SegNConfig,
    Yolov8SegS, Yolov8SegSConfig, Yolov8SegX, Yolov8SegXConfig, Yolov8X, Yolov8XConfig,
};
