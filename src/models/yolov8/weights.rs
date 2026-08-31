//! Versioned names for montgomery-native YOLOv8 artifacts.

const ARTIFACT_SCHEMA: &str = "v1";

pub fn artifact_format(model: &str) -> String {
    format!("{model}-{ARTIFACT_SCHEMA}")
}

pub fn coco_artifact_filename(model: &str) -> String {
    format!("{model}-coco-ultralytics-v8.4-montgomery-v1.bpk")
}

pub fn dataset_tag(model: &str) -> &'static str {
    if model.ends_with("-cls") {
        "imagenet1k"
    } else if model.ends_with("-obb") {
        "dotav1"
    } else {
        "coco"
    }
}

pub fn artifact_filename(model: &str) -> String {
    format!(
        "{model}-{}-ultralytics-v8.4-montgomery-v1.bpk",
        dataset_tag(model)
    )
}
