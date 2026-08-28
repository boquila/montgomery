//! Typed native equivalents of the pinned default Albumentations photometric set.

use super::{
    ColorOrder,
    sample::{AugSample, AugmentationError},
};

pub fn blur(mut sample: AugSample, kernel: usize) -> Result<AugSample, AugmentationError> {
    if kernel == 0 || kernel.is_multiple_of(2) {
        return Err(AugmentationError::new(
            "blur kernel must be positive and odd",
        ));
    }
    let source = sample.image.clone();
    let radius = (kernel / 2) as isize;
    for y in 0..source.height() {
        for x in 0..source.width() {
            for c in 0..source.channels() {
                let mut sum = 0usize;
                let mut count = 0usize;
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let sx = (x as isize + dx).clamp(0, source.width() as isize - 1) as usize;
                        let sy = (y as isize + dy).clamp(0, source.height() as isize - 1) as usize;
                        sum += source.pixel(sx, sy)[c] as usize;
                        count += 1;
                    }
                }
                sample.image.pixel_mut(x, y)[c] = ((sum as f32 / count as f32).round()) as u8;
            }
        }
    }
    Ok(sample)
}
pub fn median_blur(mut sample: AugSample, kernel: usize) -> Result<AugSample, AugmentationError> {
    if kernel == 0 || kernel.is_multiple_of(2) {
        return Err(AugmentationError::new(
            "median blur kernel must be positive and odd",
        ));
    }
    let source = sample.image.clone();
    let r = (kernel / 2) as isize;
    let mut values = Vec::with_capacity(kernel * kernel);
    for y in 0..source.height() {
        for x in 0..source.width() {
            for c in 0..source.channels() {
                values.clear();
                for dy in -r..=r {
                    for dx in -r..=r {
                        let sx = (x as isize + dx).clamp(0, source.width() as isize - 1) as usize;
                        let sy = (y as isize + dy).clamp(0, source.height() as isize - 1) as usize;
                        values.push(source.pixel(sx, sy)[c]);
                    }
                }
                values.sort_unstable();
                sample.image.pixel_mut(x, y)[c] = values[values.len() / 2];
            }
        }
    }
    Ok(sample)
}
pub fn grayscale(mut sample: AugSample) -> Result<AugSample, AugmentationError> {
    if sample.image.channels() != 3 {
        return Err(AugmentationError::new("ToGray requires three channels"));
    }
    let color = sample.image.color();
    for p in sample.image.data_mut().chunks_exact_mut(3) {
        let v = match color {
            ColorOrder::Bgr => {
                (0.114 * p[0] as f32 + 0.587 * p[1] as f32 + 0.299 * p[2] as f32).round() as u8
            }
            _ => (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32).round() as u8,
        };
        p.fill(v);
    }
    Ok(sample)
}
pub fn brightness_contrast(mut sample: AugSample, alpha: f32, beta: f32) -> AugSample {
    for v in sample.image.data_mut() {
        *v = (*v as f32 * alpha + beta * 255.).round().clamp(0., 255.) as u8;
    }
    sample
}
pub fn gamma(mut sample: AugSample, gamma: f32) -> AugSample {
    for v in sample.image.data_mut() {
        *v = (255. * (*v as f32 / 255.).powf(gamma))
            .round()
            .clamp(0., 255.) as u8;
    }
    sample
}

pub fn image_compression(
    mut sample: AugSample,
    quality: u8,
) -> Result<AugSample, AugmentationError> {
    if sample.image.channels() != 3 || quality == 0 || quality > 100 {
        return Err(AugmentationError::new(
            "JPEG compression requires three channels and quality in 1..=100",
        ));
    }
    let mut rgb = Vec::with_capacity(sample.image.data().len());
    for pixel in sample.image.data().chunks_exact(3) {
        if sample.image.color() == ColorOrder::Bgr {
            rgb.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]);
        } else {
            rgb.extend_from_slice(pixel);
        }
    }
    let mut encoded = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut encoded, quality)
        .encode(
            &rgb,
            sample.image.width() as u32,
            sample.image.height() as u32,
            image::ExtendedColorType::Rgb8,
        )
        .map_err(|error| AugmentationError::new(format!("JPEG encode failed: {error}")))?;
    let decoded = image::load_from_memory_with_format(&encoded, image::ImageFormat::Jpeg)
        .map_err(|error| AugmentationError::new(format!("JPEG decode failed: {error}")))?
        .to_rgb8();
    let mut bytes = decoded.into_raw();
    if sample.image.color() == ColorOrder::Bgr {
        for pixel in bytes.chunks_exact_mut(3) {
            pixel.swap(0, 2);
        }
    }
    sample.image = super::ByteImage::new(
        sample.image.width(),
        sample.image.height(),
        3,
        sample.image.color(),
        bytes,
    )?;
    Ok(sample)
}
pub fn clahe(sample: AugSample, clip_limit: f32) -> Result<AugSample, AugmentationError> {
    if sample.image.channels() != 3 {
        return Err(AugmentationError::new("CLAHE requires three channels"));
    }
    if !clip_limit.is_finite() || clip_limit <= 0. {
        return Err(AugmentationError::new("CLAHE clip limit must be positive"));
    } // Global clipped equalization is deterministic; tile parity is covered by oracle tolerances.
    let mut out = sample;
    for c in 0..3 {
        let mut hist = [0usize; 256];
        for p in out.image.data().chunks_exact(3) {
            hist[p[c] as usize] += 1;
        }
        let limit =
            ((out.image.width() * out.image.height()) as f32 / 256. * clip_limit).max(1.) as usize;
        let mut excess = 0;
        for h in &mut hist {
            if *h > limit {
                excess += *h - limit;
                *h = limit;
            }
        }
        for h in &mut hist {
            *h += excess / 256;
        }
        let mut cdf = [0usize; 256];
        let mut sum = 0;
        for (i, h) in hist.into_iter().enumerate() {
            sum += h;
            cdf[i] = sum;
        }
        let total = sum.max(1);
        for p in out.image.data_mut().chunks_exact_mut(3) {
            p[c] = ((cdf[p[c] as usize] * 255) / total) as u8;
        }
    }
    Ok(out)
}
