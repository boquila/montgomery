//! Versioned metadata and simple names for native YOLOv8 artifacts.

const ARTIFACT_SCHEMA: &str = "v1";

pub fn artifact_format(model: &str) -> String {
    format!("{model}-{ARTIFACT_SCHEMA}")
}

pub fn coco_artifact_filename(model: &str) -> String {
    format!("{model}.bpk")
}

pub fn artifact_filename(model: &str) -> String {
    format!("{model}.bpk")
}
