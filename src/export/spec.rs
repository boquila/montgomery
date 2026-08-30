use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::ModelId;

pub const EXPORT_SPEC_VERSION: &str = "boquilens-export-spec-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExportFamily {
    Yolox,
    Yolov3Tiny,
    Yolov10,
    Yolo11,
    Yolov8,
    Yolo12,
    Yolo26,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExportTask {
    Detect,
    Segment,
    Classify,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum OnnxProfile {
    Portable,
    Ultralytics,
    End2end,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum OnnxPrecision {
    Fp32,
    Fp16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum ExternalDataPolicy {
    Never,
    Auto,
    Always,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BoxFormat {
    Xywh,
    Xyxy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct OutputTensorSpec {
    pub name: &'static str,
    pub rank: usize,
    pub semantic: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ExportSpec {
    pub model_id: ModelId,
    pub family: ExportFamily,
    pub task: ExportTask,
    pub scale: &'static str,
    pub default_input: [usize; 4],
    pub stride: usize,
    pub num_classes: usize,
    pub box_format: Option<BoxFormat>,
    pub outputs: &'static [OutputTensorSpec],
    pub graph_config: &'static str,
    pub graph_source: &'static str,
    pub key_map_version: &'static str,
    pub license: &'static str,
    pub nms: bool,
}

const DETECT_OUTPUTS: &[OutputTensorSpec] = &[
    OutputTensorSpec {
        name: "boxes",
        rank: 3,
        semantic: "decoded boxes in model-input pixels",
    },
    OutputTensorSpec {
        name: "scores",
        rank: 3,
        semantic: "per-class detection scores",
    },
];
const SEGMENT_OUTPUTS: &[OutputTensorSpec] = &[
    OutputTensorSpec {
        name: "boxes",
        rank: 3,
        semantic: "decoded boxes in model-input pixels",
    },
    OutputTensorSpec {
        name: "scores",
        rank: 3,
        semantic: "per-class sigmoid probabilities",
    },
    OutputTensorSpec {
        name: "coefficients",
        rank: 3,
        semantic: "raw mask coefficients",
    },
    OutputTensorSpec {
        name: "prototypes",
        rank: 4,
        semantic: "stride-4 mask prototypes",
    },
];
const CLASSIFY_OUTPUTS: &[OutputTensorSpec] = &[
    OutputTensorSpec {
        name: "logits",
        rank: 2,
        semantic: "unnormalized class logits",
    },
    OutputTensorSpec {
        name: "probabilities",
        rank: 2,
        semantic: "softmax class probabilities",
    },
];

impl ExportSpec {
    pub fn for_model(model_id: ModelId) -> Self {
        use ModelId::*;

        let (family, task, scale, graph_config) = match model_id {
            YoloxNano => (
                ExportFamily::Yolox,
                ExportTask::Detect,
                "nano",
                "yolox-nano",
            ),
            YoloxTiny => (
                ExportFamily::Yolox,
                ExportTask::Detect,
                "tiny",
                "yolox-tiny",
            ),
            YoloxS => (ExportFamily::Yolox, ExportTask::Detect, "s", "yolox-s"),
            YoloxM => (ExportFamily::Yolox, ExportTask::Detect, "m", "yolox-m"),
            YoloxL => (ExportFamily::Yolox, ExportTask::Detect, "l", "yolox-l"),
            YoloxX => (ExportFamily::Yolox, ExportTask::Detect, "x", "yolox-x"),
            Yolov3TinyU => (
                ExportFamily::Yolov3Tiny,
                ExportTask::Detect,
                "tiny",
                "yolov3-tiny.yaml",
            ),
            Yolov10N => (
                ExportFamily::Yolov10,
                ExportTask::Detect,
                "n",
                "yolov10n.yaml",
            ),
            Yolov10S => (
                ExportFamily::Yolov10,
                ExportTask::Detect,
                "s",
                "yolov10s.yaml",
            ),
            Yolov10M => (
                ExportFamily::Yolov10,
                ExportTask::Detect,
                "m",
                "yolov10m.yaml",
            ),
            Yolov10B => (
                ExportFamily::Yolov10,
                ExportTask::Detect,
                "b",
                "yolov10b.yaml",
            ),
            Yolov10L => (
                ExportFamily::Yolov10,
                ExportTask::Detect,
                "l",
                "yolov10l.yaml",
            ),
            Yolov10X => (
                ExportFamily::Yolov10,
                ExportTask::Detect,
                "x",
                "yolov10x.yaml",
            ),
            Yolo11N => (
                ExportFamily::Yolo11,
                ExportTask::Detect,
                "n",
                "yolo11n.yaml",
            ),
            Yolo11S => (
                ExportFamily::Yolo11,
                ExportTask::Detect,
                "s",
                "yolo11s.yaml",
            ),
            Yolo11M => (
                ExportFamily::Yolo11,
                ExportTask::Detect,
                "m",
                "yolo11m.yaml",
            ),
            Yolo11L => (
                ExportFamily::Yolo11,
                ExportTask::Detect,
                "l",
                "yolo11l.yaml",
            ),
            Yolo11X => (
                ExportFamily::Yolo11,
                ExportTask::Detect,
                "x",
                "yolo11x.yaml",
            ),
            Yolo11NSeg => (
                ExportFamily::Yolo11,
                ExportTask::Segment,
                "n",
                "yolo11n-seg.yaml",
            ),
            Yolo11SSeg => (
                ExportFamily::Yolo11,
                ExportTask::Segment,
                "s",
                "yolo11s-seg.yaml",
            ),
            Yolo11MSeg => (
                ExportFamily::Yolo11,
                ExportTask::Segment,
                "m",
                "yolo11m-seg.yaml",
            ),
            Yolo11LSeg => (
                ExportFamily::Yolo11,
                ExportTask::Segment,
                "l",
                "yolo11l-seg.yaml",
            ),
            Yolo11XSeg => (
                ExportFamily::Yolo11,
                ExportTask::Segment,
                "x",
                "yolo11x-seg.yaml",
            ),
            Yolo11NCls => (
                ExportFamily::Yolo11,
                ExportTask::Classify,
                "n",
                "yolo11n-cls.yaml",
            ),
            Yolo11SCls => (
                ExportFamily::Yolo11,
                ExportTask::Classify,
                "s",
                "yolo11s-cls.yaml",
            ),
            Yolo11MCls => (
                ExportFamily::Yolo11,
                ExportTask::Classify,
                "m",
                "yolo11m-cls.yaml",
            ),
            Yolo11LCls => (
                ExportFamily::Yolo11,
                ExportTask::Classify,
                "l",
                "yolo11l-cls.yaml",
            ),
            Yolo11XCls => (
                ExportFamily::Yolo11,
                ExportTask::Classify,
                "x",
                "yolo11x-cls.yaml",
            ),
            Yolov8N => (
                ExportFamily::Yolov8,
                ExportTask::Detect,
                "n",
                "yolov8n.yaml",
            ),
            Yolov8S => (
                ExportFamily::Yolov8,
                ExportTask::Detect,
                "s",
                "yolov8s.yaml",
            ),
            Yolov8M => (
                ExportFamily::Yolov8,
                ExportTask::Detect,
                "m",
                "yolov8m.yaml",
            ),
            Yolov8L => (
                ExportFamily::Yolov8,
                ExportTask::Detect,
                "l",
                "yolov8l.yaml",
            ),
            Yolov8X => (
                ExportFamily::Yolov8,
                ExportTask::Detect,
                "x",
                "yolov8x.yaml",
            ),
            Yolov8NSeg => (
                ExportFamily::Yolov8,
                ExportTask::Segment,
                "n",
                "yolov8n-seg.yaml",
            ),
            Yolov8SSeg => (
                ExportFamily::Yolov8,
                ExportTask::Segment,
                "s",
                "yolov8s-seg.yaml",
            ),
            Yolov8MSeg => (
                ExportFamily::Yolov8,
                ExportTask::Segment,
                "m",
                "yolov8m-seg.yaml",
            ),
            Yolov8LSeg => (
                ExportFamily::Yolov8,
                ExportTask::Segment,
                "l",
                "yolov8l-seg.yaml",
            ),
            Yolov8XSeg => (
                ExportFamily::Yolov8,
                ExportTask::Segment,
                "x",
                "yolov8x-seg.yaml",
            ),
            Yolov8NCls => (
                ExportFamily::Yolov8,
                ExportTask::Classify,
                "n",
                "yolov8n-cls.yaml",
            ),
            Yolov8SCls => (
                ExportFamily::Yolov8,
                ExportTask::Classify,
                "s",
                "yolov8s-cls.yaml",
            ),
            Yolov8MCls => (
                ExportFamily::Yolov8,
                ExportTask::Classify,
                "m",
                "yolov8m-cls.yaml",
            ),
            Yolov8LCls => (
                ExportFamily::Yolov8,
                ExportTask::Classify,
                "l",
                "yolov8l-cls.yaml",
            ),
            Yolov8XCls => (
                ExportFamily::Yolov8,
                ExportTask::Classify,
                "x",
                "yolov8x-cls.yaml",
            ),
            Yolo12N => (
                ExportFamily::Yolo12,
                ExportTask::Detect,
                "n",
                "yolo12n.yaml",
            ),
            Yolo12S => (
                ExportFamily::Yolo12,
                ExportTask::Detect,
                "s",
                "yolo12s.yaml",
            ),
            Yolo12M => (
                ExportFamily::Yolo12,
                ExportTask::Detect,
                "m",
                "yolo12m.yaml",
            ),
            Yolo12L => (
                ExportFamily::Yolo12,
                ExportTask::Detect,
                "l",
                "yolo12l.yaml",
            ),
            Yolo12X => (
                ExportFamily::Yolo12,
                ExportTask::Detect,
                "x",
                "yolo12x.yaml",
            ),
            Yolo26N => (
                ExportFamily::Yolo26,
                ExportTask::Detect,
                "n",
                "yolo26n.yaml",
            ),
            Yolo26S => (
                ExportFamily::Yolo26,
                ExportTask::Detect,
                "s",
                "yolo26s.yaml",
            ),
            Yolo26M => (
                ExportFamily::Yolo26,
                ExportTask::Detect,
                "m",
                "yolo26m.yaml",
            ),
            Yolo26L => (
                ExportFamily::Yolo26,
                ExportTask::Detect,
                "l",
                "yolo26l.yaml",
            ),
            Yolo26X => (
                ExportFamily::Yolo26,
                ExportTask::Detect,
                "x",
                "yolo26x.yaml",
            ),
            Yolo26NSeg => (
                ExportFamily::Yolo26,
                ExportTask::Segment,
                "n",
                "yolo26n-seg.yaml",
            ),
            Yolo26SSeg => (
                ExportFamily::Yolo26,
                ExportTask::Segment,
                "s",
                "yolo26s-seg.yaml",
            ),
            Yolo26MSeg => (
                ExportFamily::Yolo26,
                ExportTask::Segment,
                "m",
                "yolo26m-seg.yaml",
            ),
            Yolo26LSeg => (
                ExportFamily::Yolo26,
                ExportTask::Segment,
                "l",
                "yolo26l-seg.yaml",
            ),
            Yolo26XSeg => (
                ExportFamily::Yolo26,
                ExportTask::Segment,
                "x",
                "yolo26x-seg.yaml",
            ),
            Yolo26NCls => (
                ExportFamily::Yolo26,
                ExportTask::Classify,
                "n",
                "yolo26n-cls.yaml",
            ),
            Yolo26SCls => (
                ExportFamily::Yolo26,
                ExportTask::Classify,
                "s",
                "yolo26s-cls.yaml",
            ),
            Yolo26MCls => (
                ExportFamily::Yolo26,
                ExportTask::Classify,
                "m",
                "yolo26m-cls.yaml",
            ),
            Yolo26LCls => (
                ExportFamily::Yolo26,
                ExportTask::Classify,
                "l",
                "yolo26l-cls.yaml",
            ),
            Yolo26XCls => (
                ExportFamily::Yolo26,
                ExportTask::Classify,
                "x",
                "yolo26x-cls.yaml",
            ),
        };

        let default_size = model_id.default_input_size();
        let outputs = match task {
            ExportTask::Detect => DETECT_OUTPUTS,
            ExportTask::Segment => SEGMENT_OUTPUTS,
            ExportTask::Classify => CLASSIFY_OUTPUTS,
        };
        let box_format = match (family, task) {
            (_, ExportTask::Classify) => None,
            (ExportFamily::Yolo11 | ExportFamily::Yolov8 | ExportFamily::Yolo12, _) => {
                Some(BoxFormat::Xywh)
            }
            _ => Some(BoxFormat::Xyxy),
        };
        let graph_source = if family == ExportFamily::Yolox {
            "yolox@0.1.1rc0"
        } else {
            "ultralytics@461196cf09175b64c9b9bd8babebf081c0540520"
        };
        let license = if family == ExportFamily::Yolox {
            "Apache-2.0"
        } else {
            "AGPL-3.0"
        };

        Self {
            model_id,
            family,
            task,
            scale,
            default_input: [1, 3, default_size, default_size],
            stride: 32,
            num_classes: if task == ExportTask::Classify {
                1000
            } else {
                80
            },
            box_format,
            outputs,
            graph_config,
            graph_source,
            key_map_version: "boquilens-pytorch-v1",
            license,
            nms: matches!(
                family,
                ExportFamily::Yolox
                    | ExportFamily::Yolov3Tiny
                    | ExportFamily::Yolo11
                    | ExportFamily::Yolov8
                    | ExportFamily::Yolo12
            ),
        }
    }

    pub fn supports_profile(self, profile: OnnxProfile) -> bool {
        match profile {
            OnnxProfile::Portable => true,
            OnnxProfile::Ultralytics => self.family != ExportFamily::Yolox,
            OnnxProfile::End2end => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn registry_is_exhaustive_and_unique() {
        let specs = ModelId::ALL.map(ExportSpec::for_model);
        let ids = specs
            .map(|spec| spec.model_id.as_str())
            .into_iter()
            .collect::<HashSet<_>>();
        assert_eq!(ids.len(), ModelId::ALL.len());
        for spec in specs {
            assert_eq!(spec.default_input[1], 3);
            assert_eq!(spec.default_input[2] % spec.stride, 0);
            assert!(!spec.outputs.is_empty());
        }
    }

    #[test]
    fn yolox_uses_the_portable_detection_contract() {
        let spec = ExportSpec::for_model(ModelId::YoloxNano);
        assert_eq!(spec.default_input, [1, 3, 416, 416]);
        assert_eq!(
            spec.outputs
                .iter()
                .map(|output| output.name)
                .collect::<Vec<_>>(),
            ["boxes", "scores"]
        );
        assert_eq!(spec.box_format, Some(BoxFormat::Xyxy));
    }
}
