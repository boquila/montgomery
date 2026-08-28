//! Versioned inverse checkpoint mappings used by the SafeTensors handoff.
//!
//! Importers map upstream numeric graph paths into descriptive Burn fields. Export applies the
//! exact inverse patterns. Rules are anchored and ordered from the most specific head paths to the
//! generic body rule, making collisions visible to the strict Python state-dict audit.

use super::spec::{ExportFamily, ExportSpec, ExportTask};

pub(crate) fn reverse_rules(spec: ExportSpec) -> Vec<(String, String)> {
    if spec.family == ExportFamily::Yolox {
        return yolox_rules();
    }

    let mut rules = Vec::new();
    match spec.task {
        ExportTask::Classify => classification_rules(spec, &mut rules),
        ExportTask::Detect => detection_rules(spec, "head", &mut rules),
        ExportTask::Segment => {
            detection_rules(spec, "head\\.detect", &mut rules);
            segmentation_rules(spec, &mut rules);
        }
    }
    rules.push((
        "^body\\.model_([0-9]+)\\.(.+)$".into(),
        "model.$1.$2".into(),
    ));
    rules
}

fn head_index(spec: ExportSpec) -> usize {
    match spec.family {
        ExportFamily::Yolov3Tiny => 20,
        ExportFamily::Yolov8 => 22,
        ExportFamily::Yolo12 => 21,
        ExportFamily::Yolov10 | ExportFamily::Yolo11 | ExportFamily::Yolo26 => 23,
        ExportFamily::Yolox => unreachable!(),
    }
}

fn classification_rules(spec: ExportSpec, rules: &mut Vec<(String, String)>) {
    let head = if spec.family == ExportFamily::Yolov8 { 9 } else { 10 };
    rules.extend([
        ("^head\\.conv\\.conv\\.(.+)$".into(), format!("model.{head}.conv.conv.$1")),
        ("^head\\.conv\\.bn\\.(.+)$".into(), format!("model.{head}.conv.bn.$1")),
        ("^head\\.linear\\.(.+)$".into(), format!("model.{head}.linear.$1")),
    ]);
}

fn detection_rules(spec: ExportSpec, burn_head: &str, rules: &mut Vec<(String, String)>) {
    let head = head_index(spec);
    let upstream_box = if matches!(spec.family, ExportFamily::Yolov10 | ExportFamily::Yolo26) {
        "one2one_cv2"
    } else {
        "cv2"
    };
    let upstream_cls = if matches!(spec.family, ExportFamily::Yolov10 | ExportFamily::Yolo26) {
        "one2one_cv3"
    } else {
        "cv3"
    };
    let levels = [("p3", 0usize), ("p4", 1), ("p5", 2)];
    let available = if spec.family == ExportFamily::Yolov3Tiny {
        &levels[1..]
    } else {
        &levels[..]
    };
    for (level, index) in available {
        let box_names: &[(&str, usize)] = if spec.family == ExportFamily::Yolov3Tiny {
            &[("box_0", 0), ("box_1", 1), ("box_2", 2)]
        } else {
            &[("box_0", 0), ("box_1", 1), ("box_out", 2)]
        };
        for (burn, layer) in box_names {
            rules.push((
                format!("^{burn_head}\\.{level}\\.{burn}\\.(.+)$"),
                format!("model.{head}.{upstream_box}.{index}.{layer}.$1"),
            ));
        }

        if matches!(spec.family, ExportFamily::Yolov8 | ExportFamily::Yolov3Tiny) {
            let names: &[(&str, usize)] = if spec.family == ExportFamily::Yolov3Tiny {
                &[("cls_0", 0), ("cls_1", 1), ("cls_2", 2)]
            } else {
                &[("cls_0", 0), ("cls_1", 1), ("cls_out", 2)]
            };
            for (burn, layer) in names {
                rules.push((
                    format!("^{burn_head}\\.{level}\\.{burn}\\.(.+)$"),
                    format!("model.{head}.{upstream_cls}.{index}.{layer}.$1"),
                ));
            }
        } else {
            for (burn, path) in [
                ("cls_dw_0", "0.0"),
                ("cls_pw_0", "0.1"),
                ("cls_dw_1", "1.0"),
                ("cls_pw_1", "1.1"),
                ("cls_out", "2"),
            ] {
                rules.push((
                    format!("^{burn_head}\\.{level}\\.{burn}\\.(.+)$"),
                    format!("model.{head}.{upstream_cls}.{index}.{path}.$1"),
                ));
            }
        }
    }
}

fn segmentation_rules(spec: ExportSpec, rules: &mut Vec<(String, String)>) {
    let head = head_index(spec);
    let cv4 = if spec.family == ExportFamily::Yolo26 {
        "one2one_cv4"
    } else {
        "cv4"
    };
    for (level, index) in [("p3", 0usize), ("p4", 1), ("p5", 2)] {
        for (burn, layer) in [("mask_0", 0usize), ("mask_1", 1), ("mask_out", 2)] {
            rules.push((
                format!("^head\\.{level}_mask\\.{burn}\\.(.+)$"),
                format!("model.{head}.{cv4}.{index}.{layer}.$1"),
            ));
        }
    }
    for (burn, upstream) in [
        ("cv1", "cv1"),
        ("upsample", "upsample"),
        ("cv2", "cv2"),
        ("cv3", "cv3"),
        ("feat_refine_0", "feat_refine.0"),
        ("feat_refine_1", "feat_refine.1"),
        ("feat_fuse", "feat_fuse"),
    ] {
        rules.push((
            format!("^head\\.proto\\.{burn}\\.(.+)$"),
            format!("model.{head}.proto.{upstream}.$1"),
        ));
    }
}

fn yolox_rules() -> Vec<(String, String)> {
    vec![
        ("^backbone\\.c3_(.+)$".into(), "backbone.C3_$1".into()),
        (
            "^(backbone\\.backbone\\.dark[2-5])\\.conv\\.(.+)$".into(),
            "$1.0.$2".into(),
        ),
        (
            "^(backbone\\.backbone\\.dark[2-4])\\.c3\\.(.+)$".into(),
            "$1.1.$2".into(),
        ),
        (
            "^(backbone\\.backbone\\.dark5)\\.spp\\.(.+)$".into(),
            "$1.1.$2".into(),
        ),
        (
            "^(backbone\\.backbone\\.dark5)\\.c3\\.(.+)$".into(),
            "$1.2.$2".into(),
        ),
        (
            "^(head\\.(?:cls|reg)_convs\\.[0-9]+)\\.conv([0-9]+)\\.(.+)$".into(),
            "$1.$2.$3".into(),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ModelId;

    #[test]
    fn mappings_are_versioned_and_every_family_has_rules() {
        for id in ModelId::ALL {
            let spec = ExportSpec::for_model(id);
            assert!(!spec.key_map_version.is_empty());
            assert!(!reverse_rules(spec).is_empty());
        }
    }
}
