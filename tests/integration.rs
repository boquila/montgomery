//! Behavior-level contracts for Montgomery's public library and command-line interfaces.
//!
//! These tests intentionally avoid model internals. Expanding the public model catalog or
//! changing an implementation should generally require edits here only when observable behavior
//! changes.

use std::collections::HashSet;
use std::process::{Command, Output};

use burn::tensor::Tensor;
use burn_flex::Flex;
use image::{DynamicImage, Rgb};
use montgomery::models::{
    yolo11::{Yolo11ClsNConfig, Yolo11NConfig, Yolo11SegNConfig},
    yolo12::Yolo12NConfig,
    yolo26::{Yolo26ClsNConfig, Yolo26NConfig, Yolo26SegNConfig},
    yolov3_tiny::Yolov3TinyConfig,
    yolov8::{Yolov8ClsNConfig, Yolov8NConfig, Yolov8SegNConfig},
    yolov10::Yolov10NConfig,
    yolox::Yolox,
};
use montgomery::{
    CLASSIFICATION_TOP_K, CLASSIFY_INPUT_SIZE, COCO_CLASSES, Detection, INPUT_SIZE, InstanceMask,
    Model, ModelId, ModelTask, PredictOptions, Prediction, Predictor, SegmentationDetection,
    annotate, annotate_segmentation, pack_weights_to,
};

fn montgomery(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_montgomery"))
        .args(args)
        .output()
        .expect("Montgomery CLI should start")
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn with_model_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("integration-model-smoke".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(body)
        .expect("model smoke-test worker should start")
        .join()
        .expect("model smoke-test worker should not panic");
}

#[test]
fn detection_families_run_complete_public_graphs() {
    with_model_stack(|| {
        let device = Default::default();
        let input = Tensor::<Flex, 4>::zeros([1, 3, 64, 64], &device);

        let yolox = Yolox::<Flex>::yolox_nano(COCO_CLASSES.len(), &device).forward(input.clone());
        assert_eq!(yolox.dims(), [1, 84, 85]);

        let yolov3 = Yolov3TinyConfig
            .init::<Flex>(&device)
            .forward(input.clone());
        assert_eq!(yolov3.boxes.dims(), [1, 20, 4]);
        assert_eq!(yolov3.scores.dims(), [1, 20, 80]);

        let yolov8 = Yolov8NConfig.init::<Flex>(&device).forward(input.clone());
        assert_eq!(yolov8.boxes.dims(), [1, 84, 4]);
        assert_eq!(yolov8.scores.dims(), [1, 84, 80]);

        let yolov10 = Yolov10NConfig.init::<Flex>(&device).forward(input.clone());
        assert_eq!(yolov10.boxes.dims(), [1, 84, 4]);
        assert_eq!(yolov10.scores.dims(), [1, 84, 80]);

        let yolo11 = Yolo11NConfig.init::<Flex>(&device).forward(input.clone());
        assert_eq!(yolo11.boxes.dims(), [1, 84, 4]);
        assert_eq!(yolo11.scores.dims(), [1, 84, 80]);

        let yolo12 = Yolo12NConfig.init::<Flex>(&device).forward(input.clone());
        assert_eq!(yolo12.boxes.dims(), [1, 84, 4]);
        assert_eq!(yolo12.scores.dims(), [1, 84, 80]);

        let yolo26 = Yolo26NConfig.init::<Flex>(&device).forward(input);
        assert_eq!(yolo26.boxes.dims(), [1, 84, 4]);
        assert_eq!(yolo26.scores.dims(), [1, 84, 80]);
    });
}

#[test]
fn segmentation_families_run_complete_public_graphs() {
    with_model_stack(|| {
        let device = Default::default();
        let input = Tensor::<Flex, 4>::zeros([1, 3, 64, 64], &device);

        let yolov8 = Yolov8SegNConfig
            .init::<Flex>(&device)
            .forward(input.clone());
        assert_eq!(yolov8.boxes.dims(), [1, 84, 4]);
        assert_eq!(yolov8.scores.dims(), [1, 84, 80]);
        assert_eq!(yolov8.coefficients.dims(), [1, 32, 84]);
        assert_eq!(yolov8.prototypes.dims(), [1, 32, 16, 16]);

        let yolo11 = Yolo11SegNConfig
            .init::<Flex>(&device)
            .forward(input.clone());
        assert_eq!(yolo11.boxes.dims(), [1, 84, 4]);
        assert_eq!(yolo11.scores.dims(), [1, 84, 80]);
        assert_eq!(yolo11.coefficients.dims(), [1, 32, 84]);
        assert_eq!(yolo11.prototypes.dims(), [1, 32, 16, 16]);

        let yolo26 = Yolo26SegNConfig.init::<Flex>(&device).forward(input);
        assert_eq!(yolo26.decoded.boxes.dims(), [1, 84, 4]);
        assert_eq!(yolo26.decoded.scores.dims(), [1, 84, 80]);
        assert_eq!(yolo26.coefficients.dims(), [1, 32, 84]);
        assert_eq!(yolo26.prototypes.dims(), [1, 32, 16, 16]);
    });
}

#[test]
fn classification_families_run_complete_public_graphs() {
    with_model_stack(|| {
        let device = Default::default();
        let input = Tensor::<Flex, 4>::zeros([1, 3, 64, 64], &device);

        let yolov8 = Yolov8ClsNConfig
            .init::<Flex>(&device)
            .forward(input.clone());
        assert_eq!(yolov8.logits.dims(), [1, 1000]);
        assert_eq!(yolov8.probs.dims(), [1, 1000]);

        let yolo11 = Yolo11ClsNConfig
            .init::<Flex>(&device)
            .forward(input.clone());
        assert_eq!(yolo11.logits.dims(), [1, 1000]);
        assert_eq!(yolo11.probs.dims(), [1, 1000]);

        let yolo26 = Yolo26ClsNConfig.init::<Flex>(&device).forward(input);
        assert_eq!(yolo26.logits.dims(), [1, 1000]);
        assert_eq!(yolo26.probs.dims(), [1, 1000]);
    });
}

#[test]
fn model_catalog_is_a_consistent_public_registry() {
    assert_eq!(ModelId::ALL.len(), 63);
    assert_eq!(ModelId::default(), ModelId::YoloxNano);

    let mut canonical_names = HashSet::new();
    for model in ModelId::ALL {
        let name = model.as_str();
        assert!(canonical_names.insert(name), "duplicate model name {name}");
        assert_eq!(name.parse::<ModelId>(), Ok(model), "parse {name}");
        assert_eq!(model.to_string(), name);
        assert_eq!(model.artifact_filename(), format!("{name}.bpk"));

        let encoded = serde_json::to_string(&model).unwrap();
        assert_eq!(serde_json::from_str::<ModelId>(&encoded).unwrap(), model);

        let expected_size = if name.ends_with("-cls") {
            CLASSIFY_INPUT_SIZE
        } else if matches!(model, ModelId::YoloxNano | ModelId::YoloxTiny) {
            416
        } else {
            INPUT_SIZE
        };
        assert_eq!(
            model.default_input_size(),
            expected_size,
            "input size {name}"
        );
        let expected_task = if name.ends_with("-cls") {
            ModelTask::Classification
        } else if name.ends_with("-seg") {
            ModelTask::Segmentation
        } else {
            ModelTask::Detection
        };
        assert_eq!(model.task(), expected_task, "task {name}");
    }

    for (alias, model) in [
        ("nano", ModelId::YoloxNano),
        ("yolox_nano", ModelId::YoloxNano),
        ("yolov10-balanced", ModelId::Yolov10B),
        ("yolo11n_seg", ModelId::Yolo11NSeg),
        ("yolov8x_cls", ModelId::Yolov8XCls),
        ("yolo26-xlarge", ModelId::Yolo26X),
    ] {
        assert_eq!(alias.parse(), Ok(model), "alias {alias}");
    }
    assert!("yolo26".parse::<ModelId>().is_err());

    assert_eq!(COCO_CLASSES.len(), 80);
    assert_eq!(COCO_CLASSES[0], "person");
    assert_eq!(COCO_CLASSES[79], "toothbrush");
    assert_eq!(CLASSIFICATION_TOP_K, 5);
}

#[test]
fn simple_model_api_has_a_concrete_cpu_default_and_task_aware_results() {
    let error = Model::new("does-not-exist.pth")
        .err()
        .expect("upstream checkpoint suffix should be rejected")
        .to_string();
    assert!(error.contains("native .bpk artifact"), "{error}");

    let detections = Prediction::Detections(Vec::new());
    assert_eq!(detections.detections(), Some([].as_slice()));
    assert!(detections.segmentations().is_none());
    assert!(detections.classifications().is_none());
    assert!(detections.is_empty());
}

#[test]
fn prediction_options_enforce_the_public_threshold_contract() {
    for value in [0.0, 0.25, 1.0] {
        assert!(
            PredictOptions {
                confidence: value,
                iou: value,
            }
            .validate()
            .is_ok()
        );
    }

    for value in [f32::NEG_INFINITY, -0.1, 1.1, f32::INFINITY, f32::NAN] {
        assert!(
            PredictOptions {
                confidence: value,
                iou: 0.5,
            }
            .validate()
            .is_err(),
            "confidence {value}"
        );
        assert!(
            PredictOptions {
                confidence: 0.5,
                iou: value,
            }
            .validate()
            .is_err(),
            "IoU {value}"
        );
    }
}

#[test]
fn native_artifact_boundaries_reject_upstream_formats_before_io() {
    for model in ModelId::ALL {
        let error = pack_weights_to(model, "does-not-exist.pt", "target/not-a-burnpack.bin")
            .unwrap_err()
            .to_string();
        assert!(error.contains(".bpk extension"), "{model}: {error}");
    }

    let error = Predictor::<Flex>::from_checkpoint(
        ModelId::YoloxNano,
        "does-not-exist.pth",
        PredictOptions::default(),
    )
    .err()
    .expect("upstream checkpoints must be rejected")
    .to_string();
    assert!(error.contains("native .bpk artifact"), "{error}");

    let error = pack_weights_to(
        ModelId::YoloxNano,
        "upstream.pth",
        "target/rejected-direct-yolox.bpk",
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("tensor-only .pt state"), "{error}");
}

#[test]
fn renderers_preserve_source_geometry_and_public_result_schema() {
    let black = Rgb([0, 0, 0]);
    let source = DynamicImage::new_rgb8(10, 10);
    let box_detection = Detection {
        class_id: 0,
        class_name: "person".into(),
        confidence: 0.9,
        xmin: 2.0,
        ymin: 3.0,
        xmax: 7.0,
        ymax: 8.0,
    };
    let boxed = annotate(&source, std::slice::from_ref(&box_detection)).to_rgb8();
    let class_color = *boxed.get_pixel(2, 3);
    assert_ne!(class_color, black);
    assert_eq!(*boxed.get_pixel(7, 8), class_color);
    assert_eq!(*boxed.get_pixel(4, 5), black);
    assert_eq!(
        *source.to_rgb8().get_pixel(2, 3),
        black,
        "input was mutated"
    );

    let mut coverage = vec![false; 100];
    for y in 1..=3 {
        for x in 1..=3 {
            coverage[y * 10 + x] = true;
        }
    }
    let segmentation = SegmentationDetection {
        class_id: 0,
        class_name: "person".into(),
        confidence: 0.8,
        xmin: 6.0,
        ymin: 6.0,
        xmax: 9.0,
        ymax: 9.0,
        mask: InstanceMask {
            width: 10,
            height: 10,
            data: coverage,
        },
    };
    let segmented = annotate_segmentation(&source, std::slice::from_ref(&segmentation)).to_rgb8();
    assert_eq!(*segmented.get_pixel(1, 1), class_color, "mask boundary");
    assert_eq!(*segmented.get_pixel(2, 2), black, "mask interior");
    assert_eq!(*segmented.get_pixel(6, 6), class_color, "box overlays mask");

    let json = serde_json::to_value((&box_detection, &segmentation)).unwrap();
    assert_eq!(json[0]["xmin"], 2.0);
    assert_eq!(json[0]["xmax"], 7.0);
    assert_eq!(json[1]["mask"]["width"], 10);
    assert_eq!(json[1]["mask"]["data"].as_array().unwrap().len(), 100);
}

#[test]
fn cli_help_exposes_the_supported_workflows_and_coordinate_contract() {
    let output = montgomery(&["--help"]);
    assert!(output.status.success(), "{}", stderr(&output));
    let help = String::from_utf8_lossy(&output.stdout);
    for command in ["predict", "pack-weights", "export-onnx", "train"] {
        assert!(help.contains(command), "missing {command} in:\n{help}");
    }

    let output = montgomery(&["predict", "--help"]);
    assert!(output.status.success(), "{}", stderr(&output));
    let help = String::from_utf8_lossy(&output.stdout);
    for contract in ["continuous XYXY pixel edges", "[0, width] x [0, height]"] {
        assert!(help.contains(contract), "missing {contract} in:\n{help}");
    }
    assert!(help.contains("--model <MODEL.bpk>"), "{help}");

    let output = montgomery(&["pack-weights", "--help"]);
    assert!(output.status.success(), "{}", stderr(&output));
    let help = String::from_utf8_lossy(&output.stdout);
    assert!(help.contains("--architecture <ARCHITECTURE>"), "{help}");
    assert!(help.contains("--state <STATE.pt>"), "{help}");
}

#[cfg(feature = "training")]
#[test]
fn training_cli_requires_one_explicit_initialization_mode() {
    let output = montgomery(&["train", "--help"]);
    assert!(output.status.success(), "{}", stderr(&output));
    let help = String::from_utf8_lossy(&output.stdout);
    for selector in [
        "--architecture <ARCHITECTURE>",
        "--model <MODEL.bpk>",
        "--resume <CHECKPOINT>",
    ] {
        assert!(help.contains(selector), "missing {selector} in:\n{help}");
    }

    let output = montgomery(&[
        "train",
        "--architecture",
        "yolo26n",
        "--model",
        "yolo26n.bpk",
        "--data",
        "missing.yaml",
    ]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("cannot be used with"));

    let output = montgomery(&["train", "--model", "upstream.pt", "--data", "missing.yaml"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("native .bpk artifact"));
}

#[test]
fn cli_rejects_bad_requests_before_loading_models() {
    let cases: &[(&[&str], &str)] = &[
        (
            &[
                "predict",
                "--source",
                "missing.png",
                "--model",
                "upstream.pth",
            ],
            "native .bpk artifact",
        ),
        (
            &[
                "predict",
                "--source",
                "missing.png",
                "--model",
                "missing.bpk",
                "--confidence",
                "1.1",
            ],
            "confidence threshold must be between 0 and 1",
        ),
        (
            &[
                "pack-weights",
                "--architecture",
                "not-a-model",
                "--state",
                "missing.pt",
            ],
            "unknown model 'not-a-model'",
        ),
    ];

    for (args, expected) in cases {
        let output = montgomery(args);
        assert!(!output.status.success(), "{args:?} unexpectedly succeeded");
        let error = stderr(&output);
        assert!(
            error.contains(expected),
            "{args:?}: expected {expected:?} in {error:?}"
        );
    }
}

#[cfg(not(feature = "gpu"))]
#[test]
fn cli_reports_the_gpu_feature_boundary() {
    let output = montgomery(&[
        "predict",
        "--source",
        "missing.png",
        "--model",
        "missing.bpk",
        "--device",
        "gpu",
    ]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("requires building Montgomery with the gpu feature"));
}

#[cfg(feature = "training")]
mod augmentation {
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
}
