//! Versioned metadata for boquilens-native YOLOv3-Tiny-U artifacts.

/// Native artifact schema written by `boquilens pack-weights`.
pub const ARTIFACT_FORMAT: &str = "yolov3-tinyu-v1";

/// Recommended filename for the first COCO-80 artifact.
pub const COCO_ARTIFACT_FILENAME: &str = "yolov3-tinyu-coco-ultralytics-v8.4-boquilens-v1.bpk";

/// The verified local release candidate produced during the v1 parity pass.
///
/// A distribution URL is intentionally absent until the AGPL artifact is published from a
/// maintained release channel.
pub const COCO_ARTIFACT_BYTES: u64 = 24_411_296;
pub const COCO_ARTIFACT_SHA256: &str =
    "52AD28C04D234F500387E9C874A52447F6A107490968BF9A23C653DDCB14DBBA";
