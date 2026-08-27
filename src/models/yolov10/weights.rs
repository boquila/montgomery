//! Versioned metadata for boquilens-native YOLOv10n artifacts.

/// Native artifact schema written by `boquilens pack-weights`.
pub const ARTIFACT_FORMAT: &str = "yolov10n-v1";

/// Recommended filename for the first COCO-80 artifact.
pub const COCO_ARTIFACT_FILENAME: &str = "yolov10n-coco-ultralytics-v8.4-boquilens-v1.bpk";

/// The verified local release candidate produced during the v1 parity pass.
///
/// A distribution URL is intentionally absent until the AGPL artifact is published from a
/// maintained release channel.
pub const COCO_ARTIFACT_BYTES: u64 = 4_779_424;
pub const COCO_ARTIFACT_SHA256: &str =
    "8A672F4924F52E89F7DF95C689C66CF157A96674CE1ADF3C2CF6A025D5C9C44B";
