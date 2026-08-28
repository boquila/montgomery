//! Task-correct pipeline construction and deterministic mixed-sample orchestration.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::{
    AugRng, AugSample, AugmentationTrace, CopyPasteMode, FormattedDetectionSample, MosaicGrid,
    PhotometricTransformConfig, ResolvedAugmentationConfig, SeedKey, TraceEvent, TraceValue,
    config::TraceMode,
    copy_paste, cutmix, flip, format, hsv,
    letterbox::{self, LetterBoxParams},
    mixup, mosaic,
    perspective::{self, PerspectiveParams, PerspectiveRanges},
    photometric,
    sample::AugmentationError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PipelinePhase {
    Open,
    Closed,
}
impl PipelinePhase {
    pub fn for_epoch(epoch: usize, epochs: usize, close_mosaic: usize) -> Self {
        if close_mosaic > 0 && epoch >= epochs.saturating_sub(close_mosaic) {
            Self::Closed
        } else {
            Self::Open
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TransformKind {
    Mosaic,
    CopyPasteFlip,
    RandomPerspective,
    CopyPasteMixup,
    Mixup,
    Cutmix,
    Blur,
    MedianBlur,
    ToGray,
    Clahe,
    RandomHsv,
    FlipVertical,
    FlipHorizontal,
    LetterBox,
    Format,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TransformParams(pub BTreeMap<String, TraceValue>);

pub struct TransformContext<'a> {
    pub rng: &'a mut AugRng,
    pub trace_path: &'a str,
}

/// Parameter-separated transform contract used by oracle fixtures and replay tests.
pub trait Transform: Send + Sync {
    fn kind(&self) -> TransformKind;
    fn sample_params(
        &self,
        sample: &AugSample,
        context: &mut TransformContext<'_>,
    ) -> Result<TransformParams, AugmentationError>;
    fn apply(
        &self,
        sample: AugSample,
        params: &TransformParams,
    ) -> Result<AugSample, AugmentationError>;
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AugmentationCounters {
    pub applied: BTreeMap<String, u64>,
    pub skipped: BTreeMap<String, u64>,
    pub samples: u64,
    pub surviving_instances: u64,
}

fn record_event(
    trace: &mut Option<AugmentationTrace>,
    path: &str,
    transform: &str,
    applied: bool,
    before: usize,
    after: usize,
    params: impl IntoIterator<Item = (String, TraceValue)>,
) {
    if let Some(trace) = trace {
        let mut event = TraceEvent::new(path, transform, applied, before);
        event.after_instances = after;
        event.params.extend(params);
        trace.events.push(event);
    }
}

impl AugmentationCounters {
    pub fn observe(&mut self, trace: &AugmentationTrace, surviving_instances: usize) {
        self.samples += 1;
        self.surviving_instances += surviving_instances as u64;
        for event in &trace.events {
            let destination = if event.applied {
                &mut self.applied
            } else {
                &mut self.skipped
            };
            *destination.entry(event.transform.clone()).or_default() += 1;
        }
    }
    pub fn average_surviving_instances(&self) -> f64 {
        if self.samples == 0 {
            0.0
        } else {
            self.surviving_instances as f64 / self.samples as f64
        }
    }
}

pub trait PartnerProvider {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn get(&mut self, index: usize) -> Result<AugSample, AugmentationError>;
    fn candidate_index(&self, logical_position: usize, draw: usize) -> usize {
        if self.len() == 0 {
            0
        } else {
            logical_position.wrapping_add(draw.wrapping_mul(0x9e3779b9)) % self.len()
        }
    }
}

#[derive(Debug, Clone)]
pub struct AugmentationPipeline {
    resolved: ResolvedAugmentationConfig,
    phase: PipelinePhase,
    order: Vec<TransformKind>,
}

impl AugmentationPipeline {
    pub fn for_epoch(
        resolved: ResolvedAugmentationConfig,
        epoch: usize,
        epochs: usize,
    ) -> Result<Self, AugmentationError> {
        let phase = PipelinePhase::for_epoch(epoch, epochs, resolved.config.close_mosaic);
        Self::new(resolved, phase)
    }

    pub fn new(
        resolved: ResolvedAugmentationConfig,
        phase: PipelinePhase,
    ) -> Result<Self, AugmentationError> {
        if resolved.task == crate::training::TaskKind::Classify {
            return Err(AugmentationError::new(
                "use ClassificationPipeline for classification",
            ));
        }
        let mut order = Vec::new();
        if resolved.training {
            order.push(TransformKind::Mosaic);
            if resolved.config.copy_paste_mode == CopyPasteMode::Flip {
                order.push(TransformKind::CopyPasteFlip);
            }
            order.push(TransformKind::RandomPerspective);
            if resolved.config.copy_paste_mode == CopyPasteMode::Mixup {
                order.push(TransformKind::CopyPasteMixup);
            }
            order.extend([
                TransformKind::Mixup,
                TransformKind::Cutmix,
                TransformKind::Blur,
                TransformKind::MedianBlur,
                TransformKind::ToGray,
                TransformKind::Clahe,
                TransformKind::RandomHsv,
                TransformKind::FlipVertical,
                TransformKind::FlipHorizontal,
                TransformKind::Format,
            ]);
        } else {
            order.extend([TransformKind::LetterBox, TransformKind::Format]);
        }
        Ok(Self {
            resolved,
            phase,
            order,
        })
    }
    pub fn order(&self) -> &[TransformKind] {
        &self.order
    }
    pub fn phase(&self) -> PipelinePhase {
        self.phase
    }
    pub fn apply(
        &self,
        sample: AugSample,
        provider: &mut dyn PartnerProvider,
        key: SeedKey<'_>,
    ) -> Result<(FormattedDetectionSample, Option<AugmentationTrace>), AugmentationError> {
        let trace_enabled = self.resolved.config.trace != TraceMode::Off;
        let mut trace =
            trace_enabled.then(|| AugmentationTrace::new(sample.source.primary_id.clone()));
        let mut rng = AugRng::new(key.clone());
        let result: Result<FormattedDetectionSample, AugmentationError> = (|| {
            let mut current = sample;
            if self.resolved.training {
                current = self.pre_transform(
                    current,
                    provider,
                    &mut rng,
                    &mut trace,
                    key.logical_position as usize,
                )?;
                let mixed_enabled = self.phase == PipelinePhase::Open && !self.resolved.config.rect;
                let copy_paste_mixup = self.resolved.config.copy_paste_mode == CopyPasteMode::Mixup
                    && mixed_enabled
                    && rng.gate(self.resolved.config.copy_paste);
                let before = current.instances.len();
                if copy_paste_mixup {
                    let partner = self.pre_transform(
                        self.partner(provider, key.logical_position as usize, 101)?,
                        provider,
                        &mut rng,
                        &mut trace,
                        key.logical_position as usize + 101,
                    )?;
                    current = copy_paste::mixup(current, partner, self.resolved.config.copy_paste)?;
                }
                record_event(
                    &mut trace,
                    "mixed/copy-paste",
                    "copy-paste-mixup",
                    copy_paste_mixup,
                    before,
                    current.instances.len(),
                    [(
                        "fraction".into(),
                        TraceValue::Float(self.resolved.config.copy_paste as f64),
                    )],
                );
                let mixup_applied = mixed_enabled && rng.gate(self.resolved.config.mixup);
                let before = current.instances.len();
                let mut mixup_ratio = None;
                if mixup_applied {
                    let partner = self.pre_transform(
                        self.partner(provider, key.logical_position as usize, 201)?,
                        provider,
                        &mut rng,
                        &mut trace,
                        key.logical_position as usize + 201,
                    )?;
                    let ratio = rng.beta(32., 32.);
                    mixup_ratio = Some(ratio);
                    current = mixup::apply(current, partner, ratio)?;
                }
                record_event(
                    &mut trace,
                    "mixed/mixup",
                    "mixup",
                    mixup_applied,
                    before,
                    current.instances.len(),
                    mixup_ratio
                        .into_iter()
                        .map(|ratio| ("ratio".into(), TraceValue::Float(ratio as f64))),
                );
                let cutmix_applied = mixed_enabled && rng.gate(self.resolved.config.cutmix);
                let before = current.instances.len();
                let mut cutmix_rect = None;
                if cutmix_applied {
                    let partner = self.pre_transform(
                        self.partner(provider, key.logical_position as usize, 301)?,
                        provider,
                        &mut rng,
                        &mut trace,
                        key.logical_position as usize + 301,
                    )?;
                    let mut rectangles = Vec::new();
                    for _ in 0..3 {
                        let lambda = rng.beta(1., 1.);
                        rectangles.push(cutmix::candidate_rect(
                            current.image.width(),
                            current.image.height(),
                            lambda,
                            [
                                rng.uniform_inclusive_i32(0, current.image.width() as i32 - 1),
                                rng.uniform_inclusive_i32(0, current.image.height() as i32 - 1),
                            ],
                        ));
                    }
                    rectangles.retain(|rect| !cutmix::overlaps_primary(&current, *rect));
                    if !rectangles.is_empty() {
                        let chosen = rectangles[rng.index(rectangles.len())];
                        cutmix_rect = Some(chosen);
                        current = cutmix::apply(
                            current,
                            partner,
                            chosen,
                            self.resolved.task == crate::training::TaskKind::Segment,
                        )?;
                    }
                }
                record_event(
                    &mut trace,
                    "mixed/cutmix",
                    "cutmix",
                    cutmix_applied,
                    before,
                    current.instances.len(),
                    cutmix_rect.into_iter().map(|rect| {
                        (
                            "rectangle".into(),
                            TraceValue::Integers(rect.into_iter().map(|v| v as i64).collect()),
                        )
                    }),
                );
                for transform in &self.resolved.config.photometric {
                    let applied = rng.gate(transform.probability());
                    let before = current.instances.len();
                    if !applied {
                        record_event(
                            &mut trace,
                            "photometric",
                            &format!("{transform:?}"),
                            false,
                            before,
                            before,
                            [],
                        );
                        continue;
                    }
                    current = match *transform {
                        PhotometricTransformConfig::Blur { kernel, .. } => {
                            photometric::blur(current, kernel)?
                        }
                        PhotometricTransformConfig::MedianBlur { kernel, .. } => {
                            photometric::median_blur(current, kernel)?
                        }
                        PhotometricTransformConfig::ToGray { .. } => {
                            photometric::grayscale(current)?
                        }
                        PhotometricTransformConfig::Clahe { clip_limit, .. } => {
                            photometric::clahe(current, clip_limit)?
                        }
                        PhotometricTransformConfig::BrightnessContrast {
                            brightness,
                            contrast,
                            ..
                        } => photometric::brightness_contrast(
                            current,
                            1.0 + rng.uniform(-contrast, contrast),
                            rng.uniform(-brightness, brightness),
                        ),
                        PhotometricTransformConfig::Gamma { range, .. } => {
                            photometric::gamma(current, rng.uniform(range[0], range[1]))
                        }
                        PhotometricTransformConfig::ImageCompression { quality, .. } => {
                            photometric::image_compression(
                                current,
                                rng.uniform_inclusive_i32(quality[0] as i32, quality[1] as i32)
                                    as u8,
                            )?
                        }
                    };
                    record_event(
                        &mut trace,
                        "photometric",
                        &format!("{transform:?}"),
                        true,
                        before,
                        current.instances.len(),
                        [],
                    );
                }
                let gains = [
                    rng.uniform(-1., 1.) * self.resolved.config.hsv_h,
                    rng.uniform(-1., 1.) * self.resolved.config.hsv_s,
                    rng.uniform(-1., 1.) * self.resolved.config.hsv_v,
                ];
                if gains != [0.; 3] {
                    current = hsv::apply(current, gains)?;
                }
                record_event(
                    &mut trace,
                    "color/hsv",
                    "random-hsv",
                    gains != [0.; 3],
                    current.instances.len(),
                    current.instances.len(),
                    [(
                        "gains".into(),
                        TraceValue::Floats(gains.into_iter().map(|v| v as f64).collect()),
                    )],
                );
                let flip_vertical = rng.gate(self.resolved.config.flipud);
                let before = current.instances.len();
                if flip_vertical {
                    current = flip::vertical(current)?;
                }
                record_event(
                    &mut trace,
                    "flip/vertical",
                    "flip-vertical",
                    flip_vertical,
                    before,
                    current.instances.len(),
                    [],
                );
                let flip_horizontal = rng.gate(self.resolved.config.fliplr);
                let before = current.instances.len();
                if flip_horizontal {
                    current = flip::horizontal(current)?;
                }
                record_event(
                    &mut trace,
                    "flip/horizontal",
                    "flip-horizontal",
                    flip_horizontal,
                    before,
                    current.instances.len(),
                    [],
                );
            } else {
                let before = current.instances.len();
                current = letterbox::apply(
                    current,
                    LetterBoxParams::validation(self.resolved.config.imgsz),
                )?;
                record_event(
                    &mut trace,
                    "validation/letterbox",
                    "letterbox",
                    true,
                    before,
                    current.instances.len(),
                    [(
                        "shape".into(),
                        TraceValue::Integers(vec![self.resolved.config.imgsz as i64; 2]),
                    )],
                );
            }
            let retain_bgr = rng.unit() < self.resolved.config.bgr;
            let before = current.instances.len();
            let formatted = format::apply(
                current,
                self.resolved.config.mask_ratio,
                self.resolved.config.mask_overlap,
                self.resolved.task == crate::training::TaskKind::Segment,
                retain_bgr,
            )?;
            record_event(
                &mut trace,
                "format",
                "format",
                true,
                before,
                formatted.classes.len(),
                [("retain_bgr".into(), TraceValue::Bool(retain_bgr))],
            );
            Ok(formatted)
        })();
        match result {
            Ok(v) => Ok((v, trace)),
            Err(e) => {
                if let Some(t) = trace.as_mut() {
                    let mut event = TraceEvent::new("failure", "pipeline", false, 0);
                    event
                        .params
                        .insert("error".into(), TraceValue::Text(e.to_string()));
                    t.events.push(event);
                }
                if let Some(trace) = trace {
                    Err(e.with_trace(trace))
                } else {
                    Err(e)
                }
            }
        }
    }
    fn partner(
        &self,
        provider: &mut dyn PartnerProvider,
        position: usize,
        draw: usize,
    ) -> Result<AugSample, AugmentationError> {
        if provider.is_empty() {
            return Err(AugmentationError::new(
                "mixed transform requires a non-empty partner provider",
            ));
        }
        let index = provider.candidate_index(position, draw);
        provider.get(index)
    }
    fn pre_transform(
        &self,
        mut current: AugSample,
        provider: &mut dyn PartnerProvider,
        rng: &mut AugRng,
        trace: &mut Option<AugmentationTrace>,
        position: usize,
    ) -> Result<AugSample, AugmentationError> {
        let mixed = self.phase == PipelinePhase::Open && !self.resolved.config.rect;
        let mosaic_applied = mixed && rng.gate(self.resolved.config.mosaic);
        if mosaic_applied {
            let count = match self.resolved.config.mosaic_grid {
                MosaicGrid::Four => 3,
                MosaicGrid::Nine => 8,
            };
            let mut partners = Vec::with_capacity(count);
            let mut indexes = Vec::new();
            for draw in 0..count {
                let p = self.partner(provider, position, draw + 1)?;
                indexes.push(p.source.primary_index);
                partners.push(p);
            }
            let center = [
                rng.uniform_inclusive_i32(
                    (self.resolved.config.imgsz / 2) as i32,
                    (self.resolved.config.imgsz * 3 / 2) as i32,
                ),
                rng.uniform_inclusive_i32(
                    (self.resolved.config.imgsz / 2) as i32,
                    (self.resolved.config.imgsz * 3 / 2) as i32,
                ),
            ];
            let before = current.instances.len();
            current = mosaic::apply(
                current,
                partners,
                self.resolved.config.imgsz,
                self.resolved.config.mosaic_grid,
                center,
            )?;
            if let Some(t) = trace {
                let mut e = TraceEvent::new("pre/mosaic", "mosaic", true, before);
                e.after_instances = current.instances.len();
                e.partners = indexes;
                e.params.insert(
                    "center".into(),
                    TraceValue::Integers(vec![center[0] as i64, center[1] as i64]),
                );
                t.events.push(e);
            }
        } else {
            record_event(
                trace,
                "pre/mosaic",
                "mosaic",
                false,
                current.instances.len(),
                current.instances.len(),
                [(
                    "probability".into(),
                    TraceValue::Float(self.resolved.config.mosaic as f64),
                )],
            );
        }
        if self.resolved.config.copy_paste_mode == CopyPasteMode::Flip
            && self.phase == PipelinePhase::Open
        {
            let before = current.instances.len();
            current = copy_paste::flip(current, self.resolved.config.copy_paste)?;
            record_event(
                trace,
                "pre/copy-paste",
                "copy-paste-flip",
                self.resolved.config.copy_paste > 0.0,
                before,
                current.instances.len(),
                [(
                    "fraction".into(),
                    TraceValue::Float(self.resolved.config.copy_paste as f64),
                )],
            );
        }
        let output = [self.resolved.config.imgsz; 2];
        let params = PerspectiveParams::sample(
            rng,
            [current.image.height(), current.image.width()],
            output,
            PerspectiveRanges {
                degrees: self.resolved.config.degrees,
                scale: self.resolved.config.scale,
                shear: self.resolved.config.shear,
                perspective: self.resolved.config.perspective,
                translate: self.resolved.config.translate,
            },
        );
        let before = current.instances.len();
        let matrix = perspective::matrix([current.image.height(), current.image.width()], params);
        current = perspective::apply(current, params)?;
        if let Some(trace) = trace {
            let mut event =
                TraceEvent::new("pre/random-perspective", "random-perspective", true, before);
            event.after_instances = current.instances.len();
            event.params.insert(
                "matrix".into(),
                TraceValue::Floats(matrix.into_iter().flatten().map(f64::from).collect()),
            );
            event.params.insert(
                "output".into(),
                TraceValue::Integers(output.into_iter().map(|v| v as i64).collect()),
            );
            trace.events.push(event);
        }
        Ok(current)
    }
}

#[cfg(test)]
mod tests {
    use super::super::AugmentationConfig;
    use super::super::{
        BBox, BoxFormat, ByteImage, ColorOrder, GeometryMetadata, Instances, SourceMetadata,
    };
    use super::*;

    #[derive(Clone)]
    struct Pool(Vec<AugSample>);

    impl PartnerProvider for Pool {
        fn len(&self) -> usize {
            self.0.len()
        }
        fn get(&mut self, index: usize) -> Result<AugSample, AugmentationError> {
            Ok(self.0[index].clone())
        }
    }

    fn sample() -> AugSample {
        AugSample {
            image: ByteImage::filled(8, 8, 3, ColorOrder::Bgr, 42),
            classes: vec![0],
            instances: Instances::new(vec![BBox([1., 1., 7., 7.])], BoxFormat::Xyxy, false, None)
                .unwrap(),
            source: SourceMetadata {
                primary_id: "sample".into(),
                primary_index: 0,
                mixed_indexes: vec![],
            },
            geometry: GeometryMetadata {
                original_shape: [8, 8],
                current_shape: [8, 8],
                ratio: [1., 1.],
                pad: [0., 0.],
                reversible: true,
            },
        }
    }
    #[test]
    fn topology_and_close_boundary() {
        let c = AugmentationConfig::default()
            .resolve(crate::training::TaskKind::Detect, true)
            .unwrap();
        let p = AugmentationPipeline::new(c, PipelinePhase::Open).unwrap();
        assert_eq!(
            &p.order()[0..3],
            &[
                TransformKind::Mosaic,
                TransformKind::CopyPasteFlip,
                TransformKind::RandomPerspective
            ]
        );
        assert_eq!(PipelinePhase::for_epoch(89, 100, 10), PipelinePhase::Open);
        assert_eq!(PipelinePhase::for_epoch(90, 100, 10), PipelinePhase::Closed);
        assert_eq!(PipelinePhase::for_epoch(99, 100, 0), PipelinePhase::Open);
    }

    #[test]
    fn validation_pipeline_is_deterministic_and_formatted() {
        let config = AugmentationConfig {
            imgsz: 8,
            mask_ratio: 1,
            ..AugmentationConfig::default()
        };
        let resolved = config
            .resolve(crate::training::TaskKind::Detect, false)
            .unwrap();
        let pipeline = AugmentationPipeline::new(resolved, PipelinePhase::Open).unwrap();
        let mut pool = Pool(vec![]);
        let key = SeedKey {
            run_seed: 1,
            epoch: 0,
            logical_position: 0,
            sample_index: 0,
            rank: 0,
            path: "validation",
        };
        let (formatted, _) = pipeline.apply(sample(), &mut pool, key).unwrap();
        assert_eq!(formatted.image_shape, [3, 8, 8]);
        assert_eq!(formatted.boxes_xywh_normalized, [[0.5, 0.5, 0.75, 0.75]]);
    }

    #[test]
    fn seeded_training_pipeline_replays_independent_of_provider_state() {
        let config = AugmentationConfig {
            imgsz: 8,
            mask_ratio: 1,
            mosaic: 0.0,
            scale: 0.0,
            translate: 0.0,
            hsv_h: 0.0,
            hsv_s: 0.0,
            hsv_v: 0.0,
            ..AugmentationConfig::default()
        };
        let resolved = config
            .resolve(crate::training::TaskKind::Detect, true)
            .unwrap();
        let pipeline = AugmentationPipeline::new(resolved, PipelinePhase::Open).unwrap();
        let mut first_pool = Pool(vec![sample()]);
        let mut second_pool = Pool(vec![sample()]);
        let key = || SeedKey {
            run_seed: 9,
            epoch: 3,
            logical_position: 17,
            sample_index: 0,
            rank: 0,
            path: "train",
        };
        let first = pipeline.apply(sample(), &mut first_pool, key()).unwrap();
        let second = pipeline.apply(sample(), &mut second_pool, key()).unwrap();
        assert_eq!(first, second);
    }
}
