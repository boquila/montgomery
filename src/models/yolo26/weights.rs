//! Versioned metadata for boquilens-native YOLO26n artifacts.

/// Native artifact schema written by `boquilens pack-weights`.
pub const ARTIFACT_FORMAT: &str = "yolo26n-v1";

/// Recommended filename for the first COCO-80 artifact.
pub const COCO_ARTIFACT_FILENAME: &str = "yolo26n-coco-ultralytics-v8.4-boquilens-v1.bpk";

/// The verified local release candidate produced during the v1 parity pass.
///
/// A distribution URL is intentionally absent until the AGPL artifact is published from a
/// maintained release channel.
pub const COCO_ARTIFACT_BYTES: u64 = 5_016_992;
pub const COCO_ARTIFACT_SHA256: &str =
    "5FB09D89850E2ECB75C0580893239DEF9BB130E95A228FB319675F267B5B24C6";
