use serde::{Deserialize, Serialize};
use std::{error::Error, fmt};

use crate::training::TaskKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Compatibility {
    Ultralytics84117,
    Native,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MosaicGrid {
    Four,
    Nine,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CopyPasteMode {
    Flip,
    Mixup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AutoAugmentPolicy {
    None,
    Randaugment,
    Autoaugment,
    Augmix,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Interpolation {
    Bilinear,
    Nearest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TraceMode {
    Off,
    Failures,
    All,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "name", rename_all = "kebab-case")]
pub enum PhotometricTransformConfig {
    Blur {
        probability: f32,
        kernel: usize,
    },
    MedianBlur {
        probability: f32,
        kernel: usize,
    },
    ToGray {
        probability: f32,
    },
    Clahe {
        probability: f32,
        clip_limit: f32,
    },
    BrightnessContrast {
        probability: f32,
        brightness: f32,
        contrast: f32,
    },
    Gamma {
        probability: f32,
        range: [f32; 2],
    },
    ImageCompression {
        probability: f32,
        quality: [u8; 2],
    },
}

impl PhotometricTransformConfig {
    pub fn probability(&self) -> f32 {
        match *self {
            Self::Blur { probability, .. }
            | Self::MedianBlur { probability, .. }
            | Self::ToGray { probability }
            | Self::Clahe { probability, .. }
            | Self::BrightnessContrast { probability, .. }
            | Self::Gamma { probability, .. }
            | Self::ImageCompression { probability, .. } => probability,
        }
    }
}

fn default_photometric() -> Vec<PhotometricTransformConfig> {
    vec![
        PhotometricTransformConfig::Blur {
            probability: 0.01,
            kernel: 3,
        },
        PhotometricTransformConfig::MedianBlur {
            probability: 0.01,
            kernel: 3,
        },
        PhotometricTransformConfig::ToGray { probability: 0.01 },
        PhotometricTransformConfig::Clahe {
            probability: 0.01,
            clip_limit: 4.0,
        },
        PhotometricTransformConfig::BrightnessContrast {
            probability: 0.0,
            brightness: 0.2,
            contrast: 0.2,
        },
        PhotometricTransformConfig::Gamma {
            probability: 0.0,
            range: [0.8, 1.2],
        },
        PhotometricTransformConfig::ImageCompression {
            probability: 0.0,
            quality: [75, 100],
        },
    ]
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AugmentationConfig {
    pub imgsz: usize,
    pub rect: bool,
    pub close_mosaic: usize,
    pub multi_scale: f32,
    #[serde(alias = "overlap_mask")]
    pub mask_overlap: bool,
    pub mask_ratio: usize,
    pub hsv_h: f32,
    pub hsv_s: f32,
    pub hsv_v: f32,
    pub degrees: f32,
    pub translate: f32,
    pub scale: f32,
    pub shear: f32,
    pub perspective: f32,
    pub flipud: f32,
    pub fliplr: f32,
    pub bgr: f32,
    pub mosaic: f32,
    pub mixup: f32,
    pub cutmix: f32,
    pub copy_paste: f32,
    pub copy_paste_mode: CopyPasteMode,
    pub mosaic_grid: MosaicGrid,
    pub auto_augment: AutoAugmentPolicy,
    pub erasing: f32,
    pub classification_crop_scale: [f32; 2],
    pub classification_crop_ratio: [f32; 2],
    pub classification_force_color_jitter: bool,
    pub classification_mean: [f32; 3],
    pub classification_std: [f32; 3],
    pub interpolation: Interpolation,
    pub compatibility: Compatibility,
    pub trace: TraceMode,
    #[serde(default = "default_photometric")]
    pub photometric: Vec<PhotometricTransformConfig>,
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
            imgsz: 640,
            rect: false,
            close_mosaic: 10,
            multi_scale: 0.0,
            mask_overlap: true,
            mask_ratio: 4,
            hsv_h: 0.015,
            hsv_s: 0.7,
            hsv_v: 0.4,
            degrees: 0.0,
            translate: 0.1,
            scale: 0.5,
            shear: 0.0,
            perspective: 0.0,
            flipud: 0.0,
            fliplr: 0.5,
            bgr: 0.0,
            mosaic: 1.0,
            mixup: 0.0,
            cutmix: 0.0,
            copy_paste: 0.0,
            copy_paste_mode: CopyPasteMode::Flip,
            mosaic_grid: MosaicGrid::Four,
            auto_augment: AutoAugmentPolicy::Randaugment,
            erasing: 0.4,
            classification_crop_scale: [0.5, 1.0],
            classification_crop_ratio: [0.75, 4.0 / 3.0],
            classification_force_color_jitter: false,
            classification_mean: [0.0; 3],
            classification_std: [1.0; 3],
            interpolation: Interpolation::Bilinear,
            compatibility: Compatibility::Ultralytics84117,
            trace: if cfg!(test) {
                TraceMode::Failures
            } else {
                TraceMode::Off
            },
            photometric: default_photometric(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedAugmentationConfig {
    pub task: TaskKind,
    pub training: bool,
    pub config: AugmentationConfig,
}

impl AugmentationConfig {
    pub fn for_task(task: TaskKind) -> Self {
        Self {
            imgsz: if task == TaskKind::Classify { 224 } else { 640 },
            ..Self::default()
        }
    }

    pub fn resolve(
        &self,
        task: TaskKind,
        training: bool,
    ) -> Result<ResolvedAugmentationConfig, AugmentationConfigError> {
        self.validate(task)?;
        let mut config = self.clone();
        if task == TaskKind::Classify {
            // The default detector size is not meaningful for classification.
            if config.imgsz == 640 {
                config.imgsz = 224;
            }
        }
        if !training {
            config.mosaic = 0.0;
            config.mixup = 0.0;
            config.cutmix = 0.0;
            config.copy_paste = 0.0;
            config.hsv_h = 0.0;
            config.hsv_s = 0.0;
            config.hsv_v = 0.0;
            config.flipud = 0.0;
            config.fliplr = 0.0;
            config.erasing = 0.0;
        }
        if config.rect {
            config.mosaic = 0.0;
            config.mixup = 0.0;
            config.cutmix = 0.0;
        }
        Ok(ResolvedAugmentationConfig {
            task,
            training,
            config,
        })
    }

    pub fn validate(&self, task: TaskKind) -> Result<(), AugmentationConfigError> {
        if self.imgsz == 0 {
            return Err(AugmentationConfigError::new("imgsz must be positive"));
        }
        if self.mask_ratio == 0 || self.mask_ratio > self.imgsz {
            return Err(AugmentationConfigError::new(
                "mask_ratio must be in 1..=imgsz",
            ));
        }
        for (name, value) in [
            ("flipud", self.flipud),
            ("fliplr", self.fliplr),
            ("bgr", self.bgr),
            ("mosaic", self.mosaic),
            ("mixup", self.mixup),
            ("cutmix", self.cutmix),
            ("copy_paste", self.copy_paste),
            ("erasing", self.erasing),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(AugmentationConfigError::new(format!(
                    "{name} must be finite and in [0, 1]"
                )));
            }
        }
        for (name, value) in [
            ("hsv_h", self.hsv_h),
            ("hsv_s", self.hsv_s),
            ("hsv_v", self.hsv_v),
            ("degrees", self.degrees),
            ("translate", self.translate),
            ("scale", self.scale),
            ("shear", self.shear),
            ("perspective", self.perspective),
            ("multi_scale", self.multi_scale),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(AugmentationConfigError::new(format!(
                    "{name} must be finite and non-negative"
                )));
            }
        }
        if self.translate > 1.0 || self.perspective > 1.0 || self.multi_scale > 1.0 {
            return Err(AugmentationConfigError::new(
                "translate, perspective, and multi_scale must not exceed 1",
            ));
        }
        if task == TaskKind::Classify {
            validate_range("classification_crop_scale", self.classification_crop_scale)?;
            validate_range("classification_crop_ratio", self.classification_crop_ratio)?;
            if self.classification_crop_scale[0] <= 0.0 {
                return Err(AugmentationConfigError::new(
                    "classification crop lower scale must be positive",
                ));
            }
            if self
                .classification_std
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.0)
            {
                return Err(AugmentationConfigError::new(
                    "classification standard deviations must be finite and positive",
                ));
            }
            if matches!(
                self.auto_augment,
                AutoAugmentPolicy::Autoaugment | AutoAugmentPolicy::Augmix
            ) {
                return Err(AugmentationConfigError::new(
                    "autoaugment and augmix are not supported by the pinned compatibility profile; use randaugment or none",
                ));
            }
        }
        for transform in &self.photometric {
            let probability = transform.probability();
            if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
                return Err(AugmentationConfigError::new(
                    "photometric probabilities must be finite and in [0, 1]",
                ));
            }
            match transform {
                PhotometricTransformConfig::Blur { kernel, .. }
                | PhotometricTransformConfig::MedianBlur { kernel, .. }
                    if *kernel == 0 || kernel.is_multiple_of(2) =>
                {
                    return Err(AugmentationConfigError::new(
                        "photometric blur kernels must be positive and odd",
                    ));
                }
                PhotometricTransformConfig::Clahe { clip_limit, .. }
                    if !clip_limit.is_finite() || *clip_limit <= 0.0 =>
                {
                    return Err(AugmentationConfigError::new(
                        "CLAHE clip limit must be finite and positive",
                    ));
                }
                PhotometricTransformConfig::Gamma { range, .. }
                    if range[0] <= 0.0
                        || range[0] > range[1]
                        || range.iter().any(|v| !v.is_finite()) =>
                {
                    return Err(AugmentationConfigError::new("invalid gamma range"));
                }
                PhotometricTransformConfig::BrightnessContrast {
                    brightness,
                    contrast,
                    ..
                } if !brightness.is_finite()
                    || !contrast.is_finite()
                    || *brightness < 0.0
                    || *contrast < 0.0 =>
                {
                    return Err(AugmentationConfigError::new(
                        "brightness and contrast limits must be finite and non-negative",
                    ));
                }
                PhotometricTransformConfig::ImageCompression { quality, .. }
                    if quality[0] == 0 || quality[0] > quality[1] || quality[1] > 100 =>
                {
                    return Err(AugmentationConfigError::new(
                        "JPEG quality must be an ascending range in 1..=100",
                    ));
                }
                _ => {}
            }
        }
        Ok(())
    }
}

fn validate_range(name: &str, values: [f32; 2]) -> Result<(), AugmentationConfigError> {
    if values.iter().any(|v| !v.is_finite()) || values[0] > values[1] {
        return Err(AugmentationConfigError::new(format!(
            "{name} must be a finite ascending range"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AugmentationConfigError(String);

impl AugmentationConfigError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for AugmentationConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for AugmentationConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_rect_resolution_match_contract() {
        let default = AugmentationConfig::default();
        assert_eq!(default.mosaic, 1.0);
        assert_eq!(default.classification_crop_scale, [0.5, 1.0]);
        let resolved = AugmentationConfig {
            rect: true,
            mixup: 0.5,
            cutmix: 0.5,
            ..default
        }
        .resolve(TaskKind::Detect, true)
        .unwrap();
        assert_eq!(resolved.config.mosaic, 0.0);
        assert_eq!(resolved.config.mixup, 0.0);
        assert_eq!(resolved.config.cutmix, 0.0);
    }

    #[test]
    fn unsupported_classification_policies_fail_loudly() {
        let config = AugmentationConfig {
            auto_augment: AutoAugmentPolicy::Augmix,
            ..AugmentationConfig::for_task(TaskKind::Classify)
        };
        assert!(config.resolve(TaskKind::Classify, true).is_err());
    }
}
