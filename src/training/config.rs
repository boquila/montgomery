use std::{error::Error, fmt, path::PathBuf};

use serde::{Deserialize, Serialize};

use crate::data::augmentation::AugmentationConfig;
use crate::{CLASSIFY_INPUT_SIZE, INPUT_SIZE, ModelId};

/// Task represented by a trainable model graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TaskKind {
    Detect,
    Segment,
    Classify,
}

impl TaskKind {
    pub const fn for_model(model: ModelId) -> Self {
        use ModelId::*;
        match model {
            Yolo11NSeg | Yolo11SSeg | Yolo11MSeg | Yolo11LSeg | Yolo11XSeg | Yolov8NSeg
            | Yolov8SSeg | Yolov8MSeg | Yolov8LSeg | Yolov8XSeg | Yolo26NSeg | Yolo26SSeg
            | Yolo26MSeg | Yolo26LSeg | Yolo26XSeg => Self::Segment,
            Yolo11NCls | Yolo11SCls | Yolo11MCls | Yolo11LCls | Yolo11XCls | Yolov8NCls
            | Yolov8SCls | Yolov8MCls | Yolov8LCls | Yolov8XCls | Yolo26NCls | Yolo26SCls
            | Yolo26MCls | Yolo26LCls | Yolo26XCls => Self::Classify,
            _ => Self::Detect,
        }
    }
}

/// Architecture plus dataset-dependent output contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSpec {
    pub architecture: ModelId,
    pub task: TaskKind,
    pub num_classes: usize,
    pub class_names: Vec<String>,
    pub input_size: [usize; 2],
}

impl ModelSpec {
    pub fn new(
        architecture: ModelId,
        class_names: Vec<String>,
        input_size: Option<[usize; 2]>,
    ) -> Result<Self, ConfigError> {
        let task = TaskKind::for_model(architecture);
        let default = if task == TaskKind::Classify {
            CLASSIFY_INPUT_SIZE
        } else if matches!(architecture, ModelId::YoloxNano | ModelId::YoloxTiny) {
            416
        } else {
            INPUT_SIZE
        };
        let spec = Self {
            architecture,
            task,
            num_classes: class_names.len(),
            class_names,
            input_size: input_size.unwrap_or([default, default]),
        };
        spec.validate()?;
        Ok(spec)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.task != TaskKind::for_model(self.architecture) {
            return Err(ConfigError::new("model suffix and task disagree"));
        }
        if self.num_classes == 0 {
            return Err(ConfigError::new("num_classes must be greater than zero"));
        }
        if self.class_names.len() != self.num_classes {
            return Err(ConfigError::new(format!(
                "class_names contains {} entries but num_classes is {}",
                self.class_names.len(),
                self.num_classes
            )));
        }
        let mut names = std::collections::BTreeSet::new();
        for name in &self.class_names {
            if name.trim().is_empty() {
                return Err(ConfigError::new("class names must not be empty"));
            }
            if !names.insert(name) {
                return Err(ConfigError::new(format!("duplicate class name {name:?}")));
            }
        }
        let [height, width] = self.input_size;
        if height == 0 || width == 0 {
            return Err(ConfigError::new("input dimensions must be positive"));
        }
        if self.task == TaskKind::Classify {
            if height != width {
                return Err(ConfigError::new("classification input must be square"));
            }
        } else if height % 32 != 0 || width % 32 != 0 {
            return Err(ConfigError::new(
                "detection and segmentation input dimensions must be multiples of 32",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OptimizerKind {
    Sgd,
    AdamW,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ScheduleKind {
    Cosine,
    Linear,
    YoloxWarmCosine,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationConfig {
    pub confidence: f32,
    pub iou: f32,
    pub max_detections: usize,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            confidence: 0.001,
            iou: 0.7,
            max_detections: 300,
        }
    }
}

/// Fully resolved settings persisted in native checkpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub model: ModelSpec,
    pub data: PathBuf,
    pub run_root: PathBuf,
    pub epochs: usize,
    pub batch_size: usize,
    pub accumulation: usize,
    pub workers: usize,
    pub prefetch: usize,
    pub seed: u64,
    pub optimizer: OptimizerKind,
    pub schedule: ScheduleKind,
    pub initial_lr: f64,
    pub final_lr_ratio: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    pub warmup_steps: u64,
    pub gradient_clip: f64,
    pub validation_interval: usize,
    pub patience: usize,
    #[serde(default)]
    pub validation: ValidationConfig,
    #[serde(default)]
    pub augmentation: AugmentationConfig,
}

impl TrainingConfig {
    pub fn yolox(model: ModelSpec, data: PathBuf, run_root: PathBuf) -> Self {
        let imgsz = model.input_size[0];
        Self {
            model,
            data,
            run_root,
            epochs: 300,
            batch_size: 8,
            accumulation: 1,
            workers: 4,
            prefetch: 2,
            seed: 0,
            optimizer: OptimizerKind::Sgd,
            schedule: ScheduleKind::YoloxWarmCosine,
            initial_lr: 0.01,
            final_lr_ratio: 0.05,
            momentum: 0.9,
            weight_decay: 5e-4,
            warmup_steps: 0,
            gradient_clip: 10.0,
            validation_interval: 1,
            patience: 50,
            validation: ValidationConfig::default(),
            augmentation: AugmentationConfig {
                imgsz,
                ..AugmentationConfig::default()
            },
        }
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        self.model.validate()?;
        if self.epochs == 0 || self.batch_size == 0 || self.accumulation == 0 {
            return Err(ConfigError::new(
                "epochs, batch_size, and accumulation must be greater than zero",
            ));
        }
        if self.prefetch == 0 || self.validation_interval == 0 {
            return Err(ConfigError::new(
                "prefetch and validation_interval must be greater than zero",
            ));
        }
        if !self.initial_lr.is_finite() || self.initial_lr <= 0.0 {
            return Err(ConfigError::new("initial_lr must be finite and positive"));
        }
        if !(0.0..=1.0).contains(&self.final_lr_ratio) {
            return Err(ConfigError::new("final_lr_ratio must be in [0, 1]"));
        }
        if !self.gradient_clip.is_finite() || self.gradient_clip <= 0.0 {
            return Err(ConfigError::new(
                "gradient_clip must be finite and positive",
            ));
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            return Err(ConfigError::new(
                "weight_decay must be finite and non-negative",
            ));
        }
        if !self.validation.confidence.is_finite()
            || !(0.0..=1.0).contains(&self.validation.confidence)
            || !self.validation.iou.is_finite()
            || !(0.0..=1.0).contains(&self.validation.iou)
            || self.validation.max_detections == 0
        {
            return Err(ConfigError::new(
                "validation confidence/IoU must be in [0, 1] and max_detections must be positive",
            ));
        }
        self.augmentation
            .resolve(self.model.task, true)
            .map_err(|error| ConfigError::new(error.to_string()))?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError(String);

impl ConfigError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_spec_enforces_task_names_and_geometry() {
        assert!(ModelSpec::new(ModelId::Yolo11N, vec!["cat".into()], Some([640, 640])).is_ok());
        assert!(ModelSpec::new(ModelId::Yolo11N, vec!["cat".into()], Some([641, 640])).is_err());
        assert!(ModelSpec::new(ModelId::Yolo11NCls, vec!["cat".into()], Some([224, 225])).is_err());
        assert!(ModelSpec::new(ModelId::Yolo11N, vec!["cat".into(), "cat".into()], None).is_err());
    }

    #[test]
    fn nano_and_tiny_default_to_official_416_canvas() {
        let spec = ModelSpec::new(ModelId::YoloxNano, vec!["object".into()], None).unwrap();
        assert_eq!(spec.input_size, [416, 416]);
    }
}
