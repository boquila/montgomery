//! Native data augmentation compatible with Ultralytics 8.4.117.
//!
//! This is a modified Rust adaptation of the Ultralytics data pipeline at commit
//! `461196cf09175b64c9b9bd8babebf081c0540520` (AGPL-3.0).

mod classify;
mod compose;
mod config;
pub mod copy_paste;
pub mod cutmix;
pub mod flip;
mod format;
pub mod hsv;
mod instances;
pub mod letterbox;
pub mod mask;
pub mod mixup;
pub mod mosaic;
pub mod perspective;
pub mod photometric;
pub mod resize;
mod rng;
mod sample;
mod trace;

pub use classify::{ClassificationPipeline, FormattedClassificationSample};
pub use compose::{
    AugmentationCounters, AugmentationPipeline, PartnerProvider, PipelinePhase, Transform,
    TransformContext, TransformKind, TransformParams,
};
pub use config::{
    AugmentationConfig, AutoAugmentPolicy, Compatibility, CopyPasteMode, Interpolation, MosaicGrid,
    PhotometricTransformConfig, ResolvedAugmentationConfig, TraceMode,
};
pub use format::{FormattedDetectionSample, MaskTargets};
pub use instances::{BBox, BoxFormat, Instances, Polygon};
pub use rng::{AugRng, SeedKey, python_round};
pub use sample::{
    AugSample, AugmentationError, ByteImage, ColorOrder, GeometryMetadata, SourceMetadata,
};
pub use trace::{AugmentationTrace, TraceEvent, TraceValue};

pub const ULTRALYTICS_SOURCE_COMMIT: &str = "461196cf09175b64c9b9bd8babebf081c0540520";
pub const TRACE_SCHEMA_VERSION: u32 = 1;
