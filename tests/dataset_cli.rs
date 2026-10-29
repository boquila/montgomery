//! Dataset workflows exercise the real CLI, disk layout, labels, and image geometry.
#![cfg(feature = "pretrained")]

use image::{Rgb, RgbImage};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicUsize, Ordering},
};

struct Fixture {
    root: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        static SEQUENCE: AtomicUsize = AtomicUsize::new(0);
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/dataset-cli-tests")
            .join(format!(
                "{}-{}",
                std::process::id(),
                SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ));
        fs::create_dir_all(root.parent().unwrap()).unwrap();
        fs::create_dir(&root).unwrap();
        Self { root }
    }
    fn file(&self, relative: &str, text: &str) {
        let path = self.root.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }
    fn image(&self, relative: &str, size: [u32; 2]) {
        let path = self.root.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        RgbImage::from_fn(size[0], size[1], |x, y| {
            if x >= size[0] / 4 && x < size[0] * 3 / 4 && y >= size[1] / 4 && y < size[1] * 3 / 4 {
                Rgb([240, 0, 0])
            } else {
                Rgb([0, 0, 240])
            }
        })
        .save(path)
        .unwrap();
    }
    fn yolo(&self) {
        self.file(
            "source/data.yaml",
            "path: .\ntrain: images/train\nval: images/val\nnames: {0: object, 1: unused}\n",
        );
        self.image("source/images/train/nested/same.png", [8, 4]);
        self.image("source/images/train/background.png", [8, 4]);
        self.image("source/images/val/same.png", [8, 4]);
        self.file(
            "source/labels/train/nested/same.txt",
            "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n",
        );
        self.file("source/labels/val/same.txt", "0 0.5 0.5 0.5 0.5\n");
    }
    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_montgomery"))
            .current_dir(&self.root)
            .args(args)
            .output()
            .unwrap()
    }
    fn json(&self, args: &[&str]) -> Value {
        let result = self.run(args);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        serde_json::from_slice(&result.stdout).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).unwrap();
    }
}

fn images(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for entry in fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            files.extend(images(&path));
        } else if path
            .extension()
            .is_some_and(|ext| ext == "jpg" || ext == "png")
        {
            files.push(path);
        }
    }
    files.sort();
    files
}

#[test]
fn inspect_and_segmentation_box_classification_pipeline() {
    let f = Fixture::new();
    f.yolo();
    let report = f.json(&["data", "--path", "source/data.yaml", "--json"]);
    assert_eq!(report["images"], 3);
    assert_eq!(report["labels"], 2);
    assert_eq!(report["label_types"], json!({"bounding-box":1,"polygon":1}));
    assert_eq!(report["splits"]["train"]["missing_label_files"], 1);
    assert_eq!(report["classes"][1]["labels"], 0);
    let preview = f.json(&[
        "data",
        "convert",
        "--path",
        "source/data.yaml",
        "--to",
        "detect",
        "--out-dir",
        "boxes",
        "--dry-run",
        "--json",
    ]);
    assert_eq!(preview["output_images"], 3);
    assert!(!f.root.join("boxes").exists());
    f.json(&[
        "data",
        "convert",
        "--path",
        "source/data.yaml",
        "--to",
        "detect",
        "--out-dir",
        "boxes",
        "--json",
    ]);
    let converted = f.json(&[
        "data",
        "inspect",
        "--path",
        "boxes/data.yaml",
        "--json",
        "--verify-images",
        "--check",
    ]);
    assert_eq!(converted["task"], "detection");
    assert_eq!(converted["labels"], 2);
    let source_pixels = images(&f.root.join("source/images"));
    let copied_pixels = images(&f.root.join("boxes/images"));
    assert_eq!(source_pixels.len(), copied_pixels.len());
    for copy in copied_pixels {
        assert!(
            source_pixels
                .iter()
                .any(|source| fs::read(source).unwrap() == fs::read(&copy).unwrap())
        );
    }
    let crops = f.json(&[
        "data",
        "convert",
        "--path",
        "boxes/data.yaml",
        "--to",
        "classify",
        "--out-dir",
        "crops",
        "--json",
    ]);
    assert_eq!(crops["output_images"], 2);
    assert_eq!(crops["skipped_background_images"], 1);
    assert!(f.root.join("crops/train/unused").is_dir());
    assert!(f.root.join("crops/val/object").is_dir());
    for crop in images(&f.root.join("crops")) {
        let image = image::open(crop).unwrap().to_rgb8();
        assert_eq!([image.width(), image.height()], [4, 2]);
        assert!(image.pixels().all(|p| p[0] > 200 && p[2] < 30));
    }
    let classified = f.json(&["data", "--path", "crops/data.yaml", "--json"]);
    assert_eq!(classified["format"], "classification-folders");
    assert_eq!(classified["labels"], 2);
    f.json(&[
        "data",
        "resize",
        "--path",
        "crops/data.yaml",
        "--imgsz",
        "224",
        "--mode",
        "letterbox",
        "--out-dir",
        "small",
        "--json",
    ]);
    for file in images(&f.root.join("small")) {
        assert_eq!(image::image_dimensions(file).unwrap(), (224, 224));
    }
    assert_eq!(
        fs::read_to_string(f.root.join("source/labels/train/nested/same.txt")).unwrap(),
        "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n"
    );
}

#[test]
fn detection_to_classification_is_one_step_and_preserves_objects_classes_and_splits() {
    let f = Fixture::new();
    f.file(
        "source/data.yaml",
        "train: images/train\nval: images/val\ntest: images/test\nnames: [zebra, ant, unused]\n",
    );
    for split in ["train/nested", "train/other", "val", "test"] {
        let path = f.root.join(format!("source/images/{split}/same.png"));
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        RgbImage::from_fn(120, 80, |x, _| {
            if x < 60 {
                Rgb([240, 0, 0])
            } else {
                Rgb([0, 0, 240])
            }
        })
        .save(path)
        .unwrap();
    }
    f.file(
        "source/labels/train/nested/same.txt",
        "0 0.25 0.5 0.5 1\n1 0.75 0.5 0.5 1\n0 0 0.25 0.25 0.5\n",
    );
    f.file("source/labels/train/other/same.txt", "1 0.75 0.5 0.5 1\n");
    f.file("source/labels/val/same.txt", "0 0.25 0.5 0.5 1\n");
    f.file("source/labels/test/same.txt", "1 0.75 0.5 0.5 1\n");
    f.image("source/images/train/background.png", [120, 80]);
    let original_images = images(&f.root.join("source"))
        .into_iter()
        .map(|path| {
            let bytes = fs::read(&path).unwrap();
            (path, bytes)
        })
        .collect::<Vec<_>>();
    let original_yaml = fs::read(f.root.join("source/data.yaml")).unwrap();
    for (destination, workers) in [("classification", "1"), ("parallel", "4")] {
        let report = f.json(&[
            "data",
            "convert",
            "--path",
            "source/data.yaml",
            "--to",
            "classify",
            "--out-dir",
            destination,
            "--workers",
            workers,
            "--json",
        ]);
        assert_eq!(report["input_images"], 5);
        assert_eq!(report["output_images"], 6);
        assert_eq!(report["output_labels"], 6);
        assert_eq!(report["skipped_background_images"], 1);
    }
    let root = f.root.join("classification");
    let dataset =
        montgomery::dataset::manifest::DatasetManifest::load(root.join("data.yaml")).unwrap();
    assert_eq!(dataset.class_names, ["zebra", "ant", "unused"]);
    assert_eq!(dataset.train_images.len(), 4);
    assert_eq!(dataset.val_images.len(), 1);
    assert_eq!(dataset.test_images.len(), 1);
    for split in ["train", "val", "test"] {
        for class in &dataset.class_names {
            assert!(root.join(split).join(class).is_dir());
        }
        #[cfg(feature = "training")]
        {
            let items = montgomery::training::data::classification::enumerate(
                root.join(split),
                &dataset.class_names,
            )
            .unwrap();
            assert!(items.iter().all(|item| {
                dataset.class_names[item.class_id]
                    == item
                        .image
                        .parent()
                        .unwrap()
                        .file_name()
                        .unwrap()
                        .to_str()
                        .unwrap()
            }));
        }
    }
    let sources = fs::read_to_string(root.join("sources.jsonl")).unwrap();
    assert_eq!(sources.lines().count(), 6);
    assert_eq!(
        sources,
        fs::read_to_string(f.root.join("parallel/sources.jsonl")).unwrap()
    );
    let mut clipped = 0;
    for line in sources.lines() {
        let row: Value = serde_json::from_str(line).unwrap();
        let relative = Path::new(row["output"].as_str().unwrap());
        let crop = image::open(root.join(relative)).unwrap().to_rgb8();
        let dimensions = if row["crop_xyxy"] == json!([0, 0, 15, 40]) {
            clipped += 1;
            (15, 40)
        } else {
            (60, 80)
        };
        assert_eq!(crop.dimensions(), dimensions);
        let class = row["class_id"].as_u64().unwrap() as usize;
        assert_eq!(
            relative.parent().unwrap().file_name().unwrap(),
            dataset.class_names[class].as_str()
        );
        assert!(crop.pixels().all(|p| if class == 0 {
            p[0] > 230 && p[2] < 10
        } else {
            p[2] > 230 && p[0] < 10
        }));
        assert_eq!(
            fs::read(root.join(relative)).unwrap(),
            fs::read(f.root.join("parallel").join(relative)).unwrap()
        );
    }
    assert_eq!(clipped, 1);
    let report = f.json(&[
        "data",
        "inspect",
        "--path",
        "classification/data.yaml",
        "--verify-images",
        "--check",
        "--json",
    ]);
    assert_eq!(report["task"], "classification");
    assert_eq!(report["classes"][0]["labels"], 3);
    assert_eq!(report["classes"][1]["labels"], 3);
    assert_eq!(report["classes"][2]["labels"], 0);
    for (path, bytes) in original_images {
        assert_eq!(fs::read(path).unwrap(), bytes);
    }
    assert_eq!(
        fs::read(f.root.join("source/data.yaml")).unwrap(),
        original_yaml
    );
}

#[test]
fn resize_keeps_polygons_and_boxes_aligned_and_can_fit_or_stretch() {
    let f = Fixture::new();
    f.yolo();
    f.json(&[
        "data",
        "resize",
        "--path",
        "source/data.yaml",
        "--imgsz",
        "8",
        "--mode",
        "letterbox",
        "--out-dir",
        "padded",
        "--json",
    ]);
    let dataset =
        montgomery::dataset::manifest::DatasetManifest::load(f.root.join("padded/data.yaml"))
            .unwrap();
    let polygon = dataset
        .train_images
        .iter()
        .find(|path| path.file_name().unwrap().to_string_lossy().contains("same"))
        .unwrap();
    assert_eq!(image::image_dimensions(polygon).unwrap(), (8, 8));
    let text = fs::read_to_string(montgomery::dataset::manifest::yolo_label_path(polygon)).unwrap();
    let coords = text
        .split_whitespace()
        .skip(1)
        .map(|v| v.parse::<f64>().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        coords,
        vec![0.25, 0.375, 0.75, 0.375, 0.75, 0.625, 0.25, 0.625]
    );
    let label = fs::read_to_string(montgomery::dataset::manifest::yolo_label_path(
        &dataset.val_images[0],
    ))
    .unwrap();
    assert_eq!(
        label.trim(),
        "0 0.5000000000 0.5000000000 0.5000000000 0.2500000000"
    );
    f.json(&[
        "data",
        "resize",
        "--path",
        "source/data.yaml",
        "--imgsz",
        "4",
        "--out-dir",
        "fitted",
        "--json",
    ]);
    for file in images(&f.root.join("fitted")) {
        assert_eq!(image::image_dimensions(file).unwrap(), (4, 2));
    }
    f.json(&[
        "data",
        "resize",
        "--path",
        "source/data.yaml",
        "--imgsz",
        "3x5",
        "--mode",
        "stretch",
        "--out-dir",
        "stretched",
        "--json",
    ]);
    for file in images(&f.root.join("stretched")) {
        assert_eq!(image::image_dimensions(file).unwrap(), (3, 5));
    }
}

#[test]
fn invalid_labels_leakage_and_outputs_are_actionable_and_do_not_overwrite() {
    let f = Fixture::new();
    f.yolo();
    f.file("source/labels/train/nested/same.txt", "4 0.5 0.5 0.5 0.5\n");
    let result = f.run(&[
        "data",
        "inspect",
        "--path",
        "source/data.yaml",
        "--check",
        "--json",
    ]);
    assert!(!result.status.success());
    let report: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["invalid_images"], 1);
    let result = f.run(&[
        "data",
        "convert",
        "--path",
        "source/data.yaml",
        "--to",
        "detect",
        "--out-dir",
        "invalid",
    ]);
    assert!(!result.status.success());
    assert!(!f.root.join("invalid").exists());
    f.file("source/labels/train/nested/same.txt", "0 0.5 0.5 0.5 0.5\n");
    f.file("occupied/sentinel", "keep me");
    let result = f.run(&[
        "data",
        "resize",
        "--path",
        "source/data.yaml",
        "--imgsz",
        "4",
        "--out-dir",
        "occupied",
    ]);
    assert!(!result.status.success());
    assert_eq!(
        fs::read_to_string(f.root.join("occupied/sentinel")).unwrap(),
        "keep me"
    );
    let result = f.run(&[
        "data",
        "resize",
        "--path",
        "source/data.yaml",
        "--imgsz",
        "4",
        "--out-dir",
        "source/new",
    ]);
    assert!(!result.status.success());
    assert!(!f.root.join("source/new").exists());
    f.file(
        "source/data.yaml",
        "train: images/train\nval: images/train\nnames: [object]\nformat: yolo\n",
    );
    let result = f.run(&[
        "data",
        "inspect",
        "--path",
        "source/data.yaml",
        "--check",
        "--json",
    ]);
    assert!(!result.status.success());
    let report: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["cross_split_images"], 2);
    assert!(
        !f.run(&[
            "data",
            "resize",
            "--path",
            "source/data.yaml",
            "--imgsz",
            "0"
        ])
        .status
        .success()
    );
    assert!(!f.run(&["data", "inspect"]).status.success());
    assert!(
        !f.run(&["data", "--path", "source/data.yaml", "--workers", "0"])
            .status
            .success()
    );
}

#[test]
fn lists_and_empty_labels_work_from_an_unrelated_working_directory() {
    let f = Fixture::new();
    f.image("source/images/train/a.png", [8, 4]);
    f.image("source/images/train/b.png", [8, 4]);
    f.file("source/labels/train/a.txt", "");
    f.file(
        "source/lists/train.txt",
        "../images/train/a.png\n../images/train/a.png\n../images/train/b.png\n",
    );
    f.file(
        "source/data.yaml",
        "path: .\ntrain: [lists/train.txt]\nnames: [object]\nformat: yolo\n",
    );
    let report = f.json(&["data", "info", "--data", "source/data.yaml", "--json"]);
    assert_eq!(report["images"], 2);
    assert_eq!(report["splits"]["train"]["empty_label_files"], 1);
    assert_eq!(report["splits"]["train"]["missing_label_files"], 1);
}

#[test]
fn coco_polygons_and_compressed_rle_survive_resize_and_convert_to_crops() {
    let f = Fixture::new();
    f.image("source/images/train/nested/a.png", [4, 2]);
    f.file(
        "source/data.yaml",
        "train: images/train\nnames: [object]\ntrain_annotations: annotations/train.json\n",
    );
    let coco = json!({"images":[{"id":10,"file_name":"nested/a.png","width":4,"height":2}],"categories":[{"id":7,"name":"object"}],"annotations":[
        {"id":1,"image_id":10,"category_id":7,"bbox":[0,0,4,2],"segmentation":[[0,0,4,0,4,2,0,2]]},
        {"id":2,"image_id":10,"category_id":7,"bbox":[0,0,4,2],"iscrowd":1,"segmentation":{"size":[2,4],"counts":"0160"}}
    ]});
    f.file("source/annotations/train.json", &coco.to_string());
    let report = f.json(&["data", "--path", "source/data.yaml", "--json"]);
    assert_eq!(report["label_types"], json!({"polygon":1,"rle":1}));
    f.json(&[
        "data",
        "resize",
        "--path",
        "source/data.yaml",
        "--imgsz",
        "4",
        "--mode",
        "letterbox",
        "--out-dir",
        "resized",
        "--json",
    ]);
    let transformed: Value = serde_json::from_str(
        &fs::read_to_string(f.root.join("resized/annotations/train.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(transformed["images"][0]["width"], 4);
    assert_eq!(transformed["images"][0]["height"], 4);
    assert_eq!(
        transformed["annotations"][0]["bbox"],
        json!([0.0, 1.0, 4.0, 2.0])
    );
    assert_eq!(
        transformed["annotations"][1]["segmentation"]["size"],
        json!([4, 4])
    );
    assert_eq!(transformed["annotations"][1]["area"], 2.0);
    f.json(&[
        "data",
        "inspect",
        "--path",
        "resized/data.yaml",
        "--json",
        "--check",
    ]);
    let cropped = f.json(&[
        "data",
        "convert",
        "--path",
        "source/data.yaml",
        "--to",
        "classify",
        "--out-dir",
        "crops",
        "--json",
    ]);
    assert_eq!(cropped["output_images"], 1);
    assert_eq!(cropped["skipped_crowd_objects"], 1);
    let boxes = f.json(&[
        "data",
        "convert",
        "--path",
        "source/data.yaml",
        "--to",
        "detect",
        "--out-dir",
        "boxes",
        "--json",
    ]);
    assert_eq!(boxes["output_labels"], 2);
    let boxes = f.json(&["data", "--path", "boxes/data.yaml", "--json"]);
    assert_eq!(boxes["task"], "detection");
}

#[test]
fn classification_infers_configured_folders_and_ignores_nested_class_names() {
    let f = Fixture::new();
    f.file(
        "source/data.yaml",
        "train: custom-training\nval: custom-validation\n",
    );
    f.image("source/custom-training/cat/dog/a.png", [8, 4]);
    f.image("source/custom-training/dog/b.png", [8, 4]);
    f.image("source/custom-training/train/c.png", [8, 4]);
    f.image("source/custom-validation/cat/d.png", [8, 4]);
    fs::create_dir(f.root.join("source/custom-training/unused")).unwrap();
    let report = f.json(&["data", "--path", "source/data.yaml", "--json"]);
    assert_eq!(report["format"], "classification-folders");
    assert_eq!(report["invalid_images"], 0);
    assert_eq!(report["classes"][0]["name"], "cat");
    assert_eq!(report["classes"][0]["labels"], 2);
    assert_eq!(report["classes"][1]["labels"], 1);
    assert_eq!(report["classes"][2]["labels"], 1);
    assert_eq!(report["classes"][3]["labels"], 0);
    f.json(&[
        "data",
        "resize",
        "--path",
        "source/data.yaml",
        "--imgsz",
        "4",
        "--out-dir",
        "small",
        "--json",
    ]);
    assert_eq!(images(&f.root.join("small/train/cat")).len(), 1);
    assert_eq!(images(&f.root.join("small/train/dog")).len(), 1);
    assert_eq!(images(&f.root.join("small/train/train")).len(), 1);
    let resized = f.json(&["data", "--path", "small/data.yaml", "--json"]);
    assert_eq!(resized["classes"][0]["labels"], 2);
    assert_eq!(resized["classes"][2]["labels"], 1);
}

#[test]
fn classification_export_rejects_case_collisions_and_detection_encoding_flags() {
    let f = Fixture::new();
    f.yolo();
    f.file(
        "source/data.yaml",
        "train: images/train\nnames: [Object, object]\n",
    );
    let result = f.run(&[
        "data",
        "convert",
        "--path",
        "source/data.yaml",
        "--to",
        "classify",
        "--out-dir",
        "crops",
    ]);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("case-insensitive"));
    assert!(!f.root.join("crops").exists());
    let result = f.run(&[
        "data",
        "convert",
        "--path",
        "source/data.yaml",
        "--to",
        "detect",
        "--quality",
        "70",
        "--out-dir",
        "boxes",
    ]);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("copies original images"));
    assert!(!f.root.join("boxes").exists());
}

#[test]
fn coco_nested_names_resolve_repeated_basenames_but_ambiguous_names_fail() {
    let f = Fixture::new();
    f.image("source/images/train/alpha/a.png", [8, 4]);
    f.image("source/images/train/beta/a.png", [8, 4]);
    f.file(
        "source/data.yaml",
        "train: images/train\nnames: [object]\ntrain_annotations: train.json\n",
    );
    let mut coco = json!({"images":[
        {"id":1,"file_name":"alpha/a.png","width":8,"height":4},
        {"id":2,"file_name":"beta/a.png","width":8,"height":4}
    ],"categories":[{"id":0,"name":"object"}],"annotations":[
        {"id":1,"image_id":1,"category_id":0,"bbox":[0,0,4,2]}
    ]});
    f.file("source/train.json", &coco.to_string());
    let report = f.json(&["data", "--path", "source/data.yaml", "--json"]);
    assert_eq!(report["invalid_images"], 0);
    assert_eq!(report["labels"], 1);
    assert_eq!(report["splits"]["train"]["background_images"], 1);
    coco["images"][0]["file_name"] = json!("a.png");
    f.file("source/train.json", &coco.to_string());
    let result = f.run(&["data", "--path", "source/data.yaml"]);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("ambiguous COCO file_name"));
}

#[test]
fn pixel_decode_failure_removes_private_staging_and_keeps_sources() {
    let f = Fixture::new();
    f.yolo();
    let image = f.root.join("source/images/train/nested/same.png");
    let mut bytes = fs::read(&image).unwrap();
    let idat = bytes.windows(4).position(|bytes| bytes == b"IDAT").unwrap();
    bytes[idat + 4] ^= 0xff;
    fs::write(&image, &bytes).unwrap();
    let report = f.json(&["data", "--path", "source/data.yaml", "--json"]);
    assert_eq!(
        report["invalid_images"], 0,
        "header preflight should succeed"
    );
    let result = f.run(&[
        "data",
        "resize",
        "--path",
        "source/data.yaml",
        "--imgsz",
        "4",
        "--out-dir",
        "failed",
    ]);
    assert!(!result.status.success());
    assert!(!f.root.join("failed").exists());
    assert!(!fs::read_dir(&f.root).unwrap().any(|entry| {
        entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".montgomery-data-")
    }));
    assert_eq!(fs::read(image).unwrap(), bytes);
}

#[test]
fn override_pixel_decode_failure_preserves_source_without_creating_a_backup() {
    let f = Fixture::new();
    f.yolo();
    let image = f.root.join("source/images/train/nested/same.png");
    let mut bytes = fs::read(&image).unwrap();
    let idat = bytes.windows(4).position(|bytes| bytes == b"IDAT").unwrap();
    bytes[idat + 4] ^= 0xff;
    fs::write(&image, &bytes).unwrap();
    let yaml = fs::read(f.root.join("source/data.yaml")).unwrap();
    let result = f.run(&[
        "data",
        "resize",
        "--path",
        "source/data.yaml",
        "--imgsz",
        "4",
        "--override",
    ]);
    assert!(!result.status.success());
    assert_eq!(fs::read(image).unwrap(), bytes);
    assert_eq!(fs::read(f.root.join("source/data.yaml")).unwrap(), yaml);
    assert_eq!(fs::read_dir(&f.root).unwrap().count(), 1);
}

#[test]
fn override_works_when_launched_from_inside_the_dataset() {
    let f = Fixture::new();
    f.yolo();
    let result = Command::new(env!("CARGO_BIN_EXE_montgomery"))
        .current_dir(f.root.join("source"))
        .args([
            "data",
            "resize",
            "--path",
            "data.yaml",
            "--imgsz",
            "4",
            "--override",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert!(Path::new(report["backup"].as_str().unwrap()).is_dir());
    assert_eq!(
        f.json(&["data", "--path", "source/data.yaml", "--json"])["images"],
        3
    );
}

#[test]
fn transforms_require_an_explicit_unambiguous_destination() {
    let f = Fixture::new();
    f.yolo();
    for args in [
        vec![
            "data",
            "resize",
            "--path",
            "source/data.yaml",
            "--imgsz",
            "4",
        ],
        vec![
            "data",
            "convert",
            "--path",
            "source/data.yaml",
            "--to",
            "detect",
        ],
        vec![
            "data",
            "resize",
            "--path",
            "source/data.yaml",
            "--imgsz",
            "4",
            "--out-dir",
            "new",
            "--override",
        ],
        vec![
            "data",
            "resize",
            "--path",
            "source/data.yaml",
            "--imgsz",
            "4",
            "--output",
            "new",
        ],
    ] {
        let result = f.run(&args);
        assert!(!result.status.success(), "{args:?} should fail");
    }
    assert!(!f.root.join("new").exists());
    assert_eq!(
        image::image_dimensions(f.root.join("source/images/train/nested/same.png")).unwrap(),
        (8, 4)
    );
    let help = f.run(&["data", "resize", "--help"]);
    let text = String::from_utf8_lossy(&help.stdout);
    assert!(text.contains("--out-dir"));
    assert!(text.contains("--override"));
    assert!(!text.contains("--output"));
}

#[test]
fn override_resizes_and_converts_in_place_with_the_same_yaml_and_recovery_backups() {
    let f = Fixture::new();
    f.yolo();
    fs::rename(
        f.root.join("source/data.yaml"),
        f.root.join("source/dataset.yaml"),
    )
    .unwrap();
    f.file("source/notes.txt", "original notes");
    let source = f.root.join("source/images/train/nested/same.png");
    let original = fs::read(&source).unwrap();
    let preview = f.json(&[
        "data",
        "resize",
        "--path",
        "source/dataset.yaml",
        "--imgsz",
        "4",
        "--override",
        "--dry-run",
        "--json",
    ]);
    assert!(preview["dry_run"].as_bool().unwrap());
    assert!(!Path::new(preview["backup"].as_str().unwrap()).exists());
    assert_eq!(fs::read(&source).unwrap(), original);
    let resized = f.json(&[
        "data",
        "resize",
        "--path",
        "source/dataset.yaml",
        "--imgsz",
        "4",
        "--override",
        "--json",
    ]);
    assert_eq!(
        Path::new(resized["out_dir"].as_str().unwrap()),
        fs::canonicalize(f.root.join("source")).unwrap()
    );
    assert!(f.root.join("source/dataset.yaml").is_file());
    let backup = Path::new(resized["backup"].as_str().unwrap());
    assert_eq!(
        fs::read(backup.join("images/train/nested/same.png")).unwrap(),
        original
    );
    assert_eq!(
        fs::read_to_string(backup.join("notes.txt")).unwrap(),
        "original notes"
    );
    let provenance = fs::read_to_string(f.root.join("source/sources.jsonl")).unwrap();
    for row in provenance.lines() {
        let row: Value = serde_json::from_str(row).unwrap();
        assert!(Path::new(row["source"].as_str().unwrap()).is_file());
        assert!(Path::new(row["source"].as_str().unwrap()).starts_with(backup));
        assert!(
            Path::new(row["original_source"].as_str().unwrap())
                .starts_with(fs::canonicalize(f.root.join("source")).unwrap())
        );
    }
    assert!(!f.root.join("source/notes.txt").exists());
    let restored = montgomery::dataset::inspect(backup.join("dataset.yaml"), true, 1).unwrap();
    assert_eq!(restored.images, 3);
    assert_eq!(restored.labels, 2);
    for file in images(&f.root.join("source")) {
        assert_eq!(image::image_dimensions(file).unwrap(), (4, 2));
    }
    f.json(&[
        "data",
        "convert",
        "--path",
        "source/dataset.yaml",
        "--to",
        "detect",
        "--override",
        "--json",
    ]);
    let detection = f.json(&["data", "--path", "source/dataset.yaml", "--json"]);
    assert_eq!(detection["task"], "detection");
    f.json(&[
        "data",
        "convert",
        "--path",
        "source/dataset.yaml",
        "--to",
        "classify",
        "--override",
        "--json",
    ]);
    let classification = f.json(&["data", "--path", "source/dataset.yaml", "--json"]);
    assert_eq!(classification["task"], "classification");
    assert_eq!(classification["images"], 2);
}

#[test]
fn override_preflight_errors_preserve_the_entire_source() {
    let f = Fixture::new();
    f.yolo();
    f.file("source/labels/train/nested/same.txt", "9 0.5 0.5 0.5 0.5\n");
    let result = f.run(&[
        "data",
        "resize",
        "--path",
        "source/data.yaml",
        "--imgsz",
        "4",
        "--override",
    ]);
    assert!(!result.status.success());
    assert_eq!(
        fs::read_to_string(f.root.join("source/labels/train/nested/same.txt")).unwrap(),
        "9 0.5 0.5 0.5 0.5\n"
    );
    assert_eq!(fs::read_dir(&f.root).unwrap().count(), 1);
    f.file("source/labels/train/nested/same.txt", "0 0.5 0.5 0.5 0.5\n");
    fs::create_dir(f.root.join("source/.git")).unwrap();
    let result = f.run(&[
        "data",
        "resize",
        "--path",
        "source/data.yaml",
        "--imgsz",
        "4",
        "--override",
    ]);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("dedicated dataset"));
    assert!(f.root.join("source/data.yaml").exists());
}
