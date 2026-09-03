//! Native classification transforms following the pinned torchvision behavior.
//!
//! RandAugment policy structure is adapted from torchvision (BSD-3-Clause).

use super::{
    AugRng, AugmentationTrace, AutoAugmentPolicy, ByteImage, ColorOrder, Interpolation,
    ResolvedAugmentationConfig, SeedKey, TraceEvent, TraceValue,
    resize::{mat_mul, resize, warp},
    sample::AugmentationError,
};

#[derive(Debug, Clone, PartialEq)]
pub struct FormattedClassificationSample {
    pub image_chw_f32: Vec<f32>,
    pub image_shape: [usize; 3],
    pub class_id: u32,
}
#[derive(Debug, Clone)]
pub struct ClassificationPipeline {
    config: ResolvedAugmentationConfig,
}

impl ClassificationPipeline {
    pub fn new(config: ResolvedAugmentationConfig) -> Result<Self, AugmentationError> {
        if config.task != crate::training::TaskKind::Classify {
            return Err(AugmentationError::new(
                "classification pipeline requires classify task",
            ));
        }
        Ok(Self { config })
    }
    pub fn apply(
        &self,
        image: ByteImage,
        class_id: u32,
        key: SeedKey<'_>,
    ) -> Result<FormattedClassificationSample, AugmentationError> {
        self.apply_traced(image, class_id, key).map(|value| value.0)
    }

    pub fn apply_traced(
        &self,
        mut image: ByteImage,
        class_id: u32,
        key: SeedKey<'_>,
    ) -> Result<(FormattedClassificationSample, AugmentationTrace), AugmentationError> {
        if image.channels() != 3 {
            return Err(AugmentationError::new(
                "classification requires a three-channel image",
            ));
        }
        if image.color() == ColorOrder::Bgr {
            for px in image.data_mut().as_chunks_mut::<3>().0 {
                px.swap(0, 2);
            }
            image.set_color(ColorOrder::Rgb);
        }
        let mut trace = AugmentationTrace::new(format!("class-{class_id}"));
        let mut rng = AugRng::new(key);
        let side = self.config.config.imgsz;
        if self.config.training {
            let rect = random_resized_crop_rect(
                &mut rng,
                image.width(),
                image.height(),
                self.config.config.classification_crop_scale,
                self.config.config.classification_crop_ratio,
            );
            let mut crop_event = TraceEvent::new("classify/crop", "random-resized-crop", true, 1);
            crop_event.params.insert(
                "rectangle".into(),
                TraceValue::Integers(rect.into_iter().map(|v| v as i64).collect()),
            );
            trace.events.push(crop_event);
            image = crop(&image, rect)?;
            image = resize(&image, side, side, Interpolation::Bilinear)?;
            let horizontal = rng.gate(0.5);
            if horizontal {
                flip_image_horizontal(&mut image);
            }
            trace.events.push(TraceEvent::new(
                "classify/hflip",
                "horizontal-flip",
                horizontal,
                1,
            ));
            let vertical = self.config.config.flipud > 0. && rng.gate(self.config.config.flipud);
            if vertical {
                flip_image_vertical(&mut image);
            }
            trace.events.push(TraceEvent::new(
                "classify/vflip",
                "vertical-flip",
                vertical,
                1,
            ));
            if self.config.config.auto_augment == AutoAugmentPolicy::Randaugment {
                randaugment(&mut image, &mut rng, 2, 9, 31)?;
            }
            let mut policy = TraceEvent::new(
                "classify/policy",
                "auto-augment",
                self.config.config.auto_augment != AutoAugmentPolicy::None,
                1,
            );
            policy.params.insert(
                "policy".into(),
                TraceValue::Text(format!("{:?}", self.config.config.auto_augment)),
            );
            policy
                .params
                .insert("num_ops".into(), TraceValue::Integer(2));
            policy
                .params
                .insert("magnitude".into(), TraceValue::Integer(9));
            trace.events.push(policy);
            if self.config.config.auto_augment == AutoAugmentPolicy::None
                || self.config.config.classification_force_color_jitter
            {
                color_jitter(
                    &mut image,
                    &mut rng,
                    self.config.config.hsv_h,
                    self.config.config.hsv_s,
                    self.config.config.hsv_v,
                )?;
            }
        } else {
            let scale = side as f32 / image.width().min(image.height()) as f32;
            image = resize(
                &image,
                (image.width() as f32 * scale).round() as usize,
                (image.height() as f32 * scale).round() as usize,
                Interpolation::Bilinear,
            )?;
            let left = (image.width() - side) / 2;
            let top = (image.height() - side) / 2;
            image = crop(&image, [left, top, left + side, top + side])?;
            let mut event = TraceEvent::new("classify/validation", "resize-center-crop", true, 1);
            event.params.insert(
                "crop".into(),
                TraceValue::Integers(vec![
                    left as i64,
                    top as i64,
                    (left + side) as i64,
                    (top + side) as i64,
                ]),
            );
            trace.events.push(event);
        }
        let mut chw = to_normalized_chw(
            &image,
            self.config.config.classification_mean,
            self.config.config.classification_std,
        )?;
        let erase_gate = self.config.training && rng.gate(self.config.config.erasing);
        let erased = if erase_gate {
            random_erasing(
                &mut chw,
                [3, side, side],
                &mut rng,
                [0.02, 0.33],
                [0.3, 3.3],
                10,
            )
        } else {
            false
        };
        let mut erase_event = TraceEvent::new("classify/erase", "random-erasing", erased, 1);
        erase_event
            .params
            .insert("gate".into(), TraceValue::Bool(erase_gate));
        trace.events.push(erase_event);
        Ok((
            FormattedClassificationSample {
                image_chw_f32: chw,
                image_shape: [3, side, side],
                class_id,
            },
            trace,
        ))
    }
}

pub fn random_resized_crop_rect(
    rng: &mut AugRng,
    width: usize,
    height: usize,
    scale: [f32; 2],
    ratio: [f32; 2],
) -> [usize; 4] {
    let area = (width * height) as f32;
    for _ in 0..10 {
        let target = area * rng.uniform(scale[0], scale[1]);
        let aspect = rng.uniform(ratio[0].ln(), ratio[1].ln()).exp();
        let w = super::python_round((target * aspect).sqrt()).max(1) as usize;
        let h = super::python_round((target / aspect).sqrt()).max(1) as usize;
        if w <= width && h <= height {
            let left = rng.uniform_inclusive_i32(0, (width - w) as i32) as usize;
            let top = rng.uniform_inclusive_i32(0, (height - h) as i32) as usize;
            return [left, top, left + w, top + h];
        }
    }
    let source = width as f32 / height as f32;
    let (w, h) = if source < ratio[0] {
        (
            width,
            super::python_round(width as f32 / ratio[0]).max(1) as usize,
        )
    } else if source > ratio[1] {
        (
            super::python_round(height as f32 * ratio[1]).max(1) as usize,
            height,
        )
    } else {
        (width, height)
    };
    [
        (width - w) / 2,
        (height - h) / 2,
        (width + w) / 2,
        (height + h) / 2,
    ]
}

fn crop(image: &ByteImage, r: [usize; 4]) -> Result<ByteImage, AugmentationError> {
    if r[2] <= r[0] || r[3] <= r[1] || r[2] > image.width() || r[3] > image.height() {
        return Err(AugmentationError::new("crop rectangle outside image"));
    }
    let mut out = ByteImage::filled(r[2] - r[0], r[3] - r[1], image.channels(), image.color(), 0);
    for y in r[1]..r[3] {
        for x in r[0]..r[2] {
            out.pixel_mut(x - r[0], y - r[1])
                .copy_from_slice(image.pixel(x, y));
        }
    }
    Ok(out)
}
fn flip_image_horizontal(image: &mut ByteImage) {
    let w = image.width();
    for y in 0..image.height() {
        for x in 0..w / 2 {
            for c in 0..image.channels() {
                let a = image.offset(x, y, c);
                let b = image.offset(w - 1 - x, y, c);
                image.data_mut().swap(a, b);
            }
        }
    }
}
fn flip_image_vertical(image: &mut ByteImage) {
    let row = image.width() * image.channels();
    for y in 0..image.height() / 2 {
        let other = image.height() - 1 - y;
        for x in 0..row {
            image.data_mut().swap(y * row + x, other * row + x);
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum RandOp {
    Identity,
    ShearX,
    ShearY,
    TranslateX,
    TranslateY,
    Rotate,
    Brightness,
    Color,
    Contrast,
    Sharpness,
    Posterize,
    Solarize,
    AutoContrast,
    Equalize,
    Invert,
}
const OPS: [RandOp; 15] = [
    RandOp::Identity,
    RandOp::ShearX,
    RandOp::ShearY,
    RandOp::TranslateX,
    RandOp::TranslateY,
    RandOp::Rotate,
    RandOp::Brightness,
    RandOp::Color,
    RandOp::Contrast,
    RandOp::Sharpness,
    RandOp::Posterize,
    RandOp::Solarize,
    RandOp::AutoContrast,
    RandOp::Equalize,
    RandOp::Invert,
];
fn magnitude(op: RandOp, index: usize, bins: usize) -> f32 {
    let t = index as f32 / (bins - 1).max(1) as f32;
    match op {
        RandOp::ShearX | RandOp::ShearY => 0.3 * t,
        RandOp::TranslateX | RandOp::TranslateY => 150.0 / 331.0 * t,
        RandOp::Rotate => 30. * t,
        RandOp::Brightness | RandOp::Color | RandOp::Contrast | RandOp::Sharpness => 0.9 * t,
        RandOp::Posterize => (8. - (4. * t).round()).max(1.),
        RandOp::Solarize => 255. * (1. - t),
        _ => 0.,
    }
}
fn randaugment(
    image: &mut ByteImage,
    rng: &mut AugRng,
    num_ops: usize,
    index: usize,
    bins: usize,
) -> Result<(), AugmentationError> {
    for _ in 0..num_ops {
        let op = OPS[rng.index(OPS.len())];
        let mut m = magnitude(op, index, bins);
        if matches!(
            op,
            RandOp::ShearX
                | RandOp::ShearY
                | RandOp::TranslateX
                | RandOp::TranslateY
                | RandOp::Rotate
                | RandOp::Brightness
                | RandOp::Color
                | RandOp::Contrast
                | RandOp::Sharpness
        ) {
            m *= rng.sign();
        }
        apply_rand_op(image, op, m)?;
    }
    Ok(())
}
fn apply_rand_op(image: &mut ByteImage, op: RandOp, m: f32) -> Result<(), AugmentationError> {
    match op {
        RandOp::Identity => {}
        RandOp::ShearX
        | RandOp::ShearY
        | RandOp::TranslateX
        | RandOp::TranslateY
        | RandOp::Rotate => {
            let cx = image.width() as f32 / 2.;
            let cy = image.height() as f32 / 2.;
            let matrix = match op {
                RandOp::ShearX => [[1., m, 0.], [0., 1., 0.], [0., 0., 1.]],
                RandOp::ShearY => [[1., 0., 0.], [m, 1., 0.], [0., 0., 1.]],
                RandOp::TranslateX => [
                    [1., 0., m * image.width() as f32],
                    [0., 1., 0.],
                    [0., 0., 1.],
                ],
                RandOp::TranslateY => [
                    [1., 0., 0.],
                    [0., 1., m * image.height() as f32],
                    [0., 0., 1.],
                ],
                _ => {
                    let a = m.to_radians();
                    mat_mul(
                        [[1., 0., cx], [0., 1., cy], [0., 0., 1.]],
                        mat_mul(
                            [
                                [a.cos(), a.sin(), 0.],
                                [-a.sin(), a.cos(), 0.],
                                [0., 0., 1.],
                            ],
                            [[1., 0., -cx], [0., 1., -cy], [0., 0., 1.]],
                        ),
                    )
                }
            };
            *image = warp(image, matrix, image.width(), image.height(), false, 0)?;
        }
        RandOp::Brightness => enhance(image, m, Enhance::Brightness),
        RandOp::Color => enhance(image, m, Enhance::Color),
        RandOp::Contrast => enhance(image, m, Enhance::Contrast),
        RandOp::Sharpness => enhance(image, m, Enhance::Sharpness),
        RandOp::Posterize => {
            let shift = 8 - m as u8;
            for v in image.data_mut() {
                *v = (*v >> shift) << shift;
            }
        }
        RandOp::Solarize => {
            for v in image.data_mut() {
                if *v >= m as u8 {
                    *v = 255 - *v;
                }
            }
        }
        RandOp::AutoContrast => autocontrast(image),
        RandOp::Equalize => equalize(image),
        RandOp::Invert => {
            for v in image.data_mut() {
                *v = 255 - *v;
            }
        }
    }
    Ok(())
}
enum Enhance {
    Brightness,
    Color,
    Contrast,
    Sharpness,
}
fn enhance(image: &mut ByteImage, m: f32, kind: Enhance) {
    let factor = 1.0 + m;
    match kind {
        Enhance::Brightness => {
            for v in image.data_mut() {
                *v = (*v as f32 * factor).round().clamp(0., 255.) as u8
            }
        }
        Enhance::Color => {
            for p in image.data_mut().as_chunks_mut::<3>().0 {
                let gray =
                    (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32).round();
                for v in p {
                    *v = (gray + (*v as f32 - gray) * factor).round().clamp(0., 255.) as u8;
                }
            }
        }
        Enhance::Contrast => {
            let mean =
                image.data().iter().map(|v| *v as f32).sum::<f32>() / image.data().len() as f32;
            for v in image.data_mut() {
                *v = (mean + (*v as f32 - mean) * factor).round().clamp(0., 255.) as u8;
            }
        }
        Enhance::Sharpness => {
            let old = image.clone();
            for y in 0..image.height() {
                for x in 0..image.width() {
                    for c in 0..3 {
                        let mut avg = 0.;
                        let mut n = 0.;
                        for dy in -1..=1 {
                            for dx in -1..=1 {
                                let xx =
                                    (x as isize + dx).clamp(0, image.width() as isize - 1) as usize;
                                let yy = (y as isize + dy).clamp(0, image.height() as isize - 1)
                                    as usize;
                                avg += old.pixel(xx, yy)[c] as f32;
                                n += 1.;
                            }
                        }
                        let original = old.pixel(x, y)[c] as f32;
                        image.pixel_mut(x, y)[c] = (avg / n + (original - avg / n) * factor)
                            .round()
                            .clamp(0., 255.)
                            as u8;
                    }
                }
            }
        }
    }
}
fn autocontrast(image: &mut ByteImage) {
    for c in 0..3 {
        let min = image
            .data()
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| p[c])
            .min()
            .unwrap_or(0);
        let max = image
            .data()
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| p[c])
            .max()
            .unwrap_or(255);
        if max > min {
            for p in image.data_mut().as_chunks_mut::<3>().0 {
                p[c] = ((p[c] - min) as f32 * 255. / (max - min) as f32).round() as u8;
            }
        }
    }
}
fn equalize(image: &mut ByteImage) {
    for c in 0..3 {
        let mut hist = [0usize; 256];
        for p in image.data().as_chunks::<3>().0 {
            hist[p[c] as usize] += 1;
        }
        let nonzero = hist.iter().copied().find(|v| *v > 0).unwrap_or(0);
        let step = (image.width() * image.height() - nonzero) / 255;
        if step == 0 {
            continue;
        }
        let mut lut = [0u8; 256];
        let mut sum = 0usize;
        for (i, h) in hist.into_iter().enumerate() {
            lut[i] = ((sum + step / 2) / step).min(255) as u8;
            sum += h;
        }
        for p in image.data_mut().as_chunks_mut::<3>().0 {
            p[c] = lut[p[c] as usize];
        }
    }
}
fn color_jitter(
    image: &mut ByteImage,
    rng: &mut AugRng,
    h: f32,
    s: f32,
    v: f32,
) -> Result<(), AugmentationError> {
    let mut order = [0usize, 1, 2, 3];
    rng.shuffle(&mut order);
    for op in order {
        match op {
            0 => enhance(
                image,
                rng.uniform((1. - v).max(0.), 1. + v) - 1.,
                Enhance::Brightness,
            ),
            1 => enhance(
                image,
                rng.uniform((1. - s).max(0.), 1. + s) - 1.,
                Enhance::Color,
            ),
            2 => enhance(
                image,
                rng.uniform((1. - v).max(0.), 1. + v) - 1.,
                Enhance::Contrast,
            ),
            _ => {
                let shift = rng.uniform(-h, h) * 255.;
                for p in image.data_mut().as_chunks_mut::<3>().0 {
                    let r = p[0] as f32;
                    let g = p[1] as f32;
                    let b = p[2] as f32;
                    p[0] = (r + shift).rem_euclid(256.) as u8;
                    p[1] = (g - shift / 2.).rem_euclid(256.) as u8;
                    p[2] = (b - shift / 2.).rem_euclid(256.) as u8;
                }
            }
        }
    }
    Ok(())
}
fn to_normalized_chw(
    image: &ByteImage,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<Vec<f32>, AugmentationError> {
    if std.iter().any(|v| *v <= 0. || !v.is_finite()) {
        return Err(AugmentationError::new(
            "normalization standard deviations must be positive",
        ));
    }
    let mut out = vec![0.; image.data().len()];
    for y in 0..image.height() {
        for x in 0..image.width() {
            for c in 0..3 {
                out[(c * image.height() + y) * image.width() + x] =
                    (image.pixel(x, y)[c] as f32 / 255. - mean[c]) / std[c];
            }
        }
    }
    Ok(out)
}
fn random_erasing(
    data: &mut [f32],
    shape: [usize; 3],
    rng: &mut AugRng,
    scale: [f32; 2],
    ratio: [f32; 2],
    attempts: usize,
) -> bool {
    let [c, h, w] = shape;
    let area = (h * w) as f32;
    for _ in 0..attempts {
        let target = area * rng.uniform(scale[0], scale[1]);
        let aspect = rng.uniform(ratio[0].ln(), ratio[1].ln()).exp();
        let eh = super::python_round((target * aspect).sqrt()).max(1) as usize;
        let ew = super::python_round((target / aspect).sqrt()).max(1) as usize;
        if eh < h && ew < w {
            let top = rng.uniform_inclusive_i32(0, (h - eh) as i32) as usize;
            let left = rng.uniform_inclusive_i32(0, (w - ew) as i32) as usize;
            for ch in 0..c {
                let fill = 0.0;
                for y in top..top + eh {
                    for x in left..left + ew {
                        data[(ch * h + y) * w + x] = fill;
                    }
                }
            }
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rng() -> AugRng {
        AugRng::new(SeedKey {
            run_seed: 1,
            epoch: 0,
            logical_position: 0,
            sample_index: 0,
            rank: 0,
            path: "crop",
        })
    }
    #[test]
    fn crop_always_fits() {
        let mut r = rng();
        for _ in 0..100 {
            let q = random_resized_crop_rect(&mut r, 13, 7, [0.5, 1.], [0.75, 4. / 3.]);
            assert!(q[2] <= 13 && q[3] <= 7 && q[0] < q[2] && q[1] < q[3]);
        }
    }
    #[test]
    fn rand_ops_cover_boundaries() {
        let base = ByteImage::filled(8, 8, 3, ColorOrder::Rgb, 128);
        for op in OPS {
            for i in [0, 15, 30] {
                let mut image = base.clone();
                apply_rand_op(&mut image, op, magnitude(op, i, 31)).unwrap();
                assert_eq!(image.data().len(), base.data().len());
            }
        }
    }
}
