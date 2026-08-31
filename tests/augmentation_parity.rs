#![cfg(feature = "training")]

use montgomery::training::{
    TaskKind,
    data::augmentation::{
        AugmentationConfig, AugmentationTrace, TRACE_SCHEMA_VERSION, ULTRALYTICS_SOURCE_COMMIT,
    },
};

#[test]
fn compatibility_metadata_and_trace_schema_are_pinned() {
    let trace = AugmentationTrace::new("synthetic");
    assert_eq!(trace.schema_version, TRACE_SCHEMA_VERSION);
    assert_eq!(trace.source_commit, ULTRALYTICS_SOURCE_COMMIT);
    let encoded = trace.to_json().unwrap();
    assert_eq!(AugmentationTrace::from_json(&encoded).unwrap(), trace);
}

#[test]
fn validation_resolution_has_no_random_training_augmentation() {
    let resolved = AugmentationConfig::default()
        .resolve(TaskKind::Segment, false)
        .unwrap();
    assert_eq!(resolved.config.mosaic, 0.0);
    assert_eq!(resolved.config.copy_paste, 0.0);
    assert_eq!(resolved.config.mixup, 0.0);
    assert_eq!(resolved.config.cutmix, 0.0);
    assert_eq!(resolved.config.fliplr, 0.0);
    assert_eq!(resolved.config.erasing, 0.0);
}
