//! OpenCV-style byte-domain HSV augmentation with the post-8.3.78 additive hue LUT.

use super::sample::{AugSample, AugmentationError, ColorOrder};

pub fn apply(mut sample: AugSample, gains: [f32; 3]) -> Result<AugSample, AugmentationError> {
    if sample.image.channels() != 3 || sample.image.color() != ColorOrder::Bgr {
        return Err(AugmentationError::new(
            "RandomHSV requires a three-channel BGR image",
        ));
    }
    let mut hue = [0u8; 256];
    let mut sat = [0u8; 256];
    let mut val = [0u8; 256];
    for x in 0..256 {
        hue[x] = ((x as f32 + gains[0] * 180.0).rem_euclid(180.0)) as u8;
        sat[x] = (x as f32 * (1.0 + gains[1])).clamp(0.0, 255.0) as u8;
        val[x] = (x as f32 * (1.0 + gains[2])).clamp(0.0, 255.0) as u8;
    }
    sat[0] = 0;
    for px in sample.image.data_mut().chunks_exact_mut(3) {
        let (h, s, v) = bgr_to_hsv(px[0], px[1], px[2]);
        let (b, g, r) = hsv_to_bgr(hue[h as usize], sat[s as usize], val[v as usize]);
        px.copy_from_slice(&[b, g, r]);
    }
    Ok(sample)
}

fn bgr_to_hsv(b: u8, g: u8, r: u8) -> (u8, u8, u8) {
    let b = b as f32 / 255.;
    let g = g as f32 / 255.;
    let r = r as f32 / 255.;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let h = if d == 0.0 {
        0.0
    } else if max == r {
        60.0 * ((g - b) / d).rem_euclid(6.0)
    } else if max == g {
        60.0 * ((b - r) / d + 2.0)
    } else {
        60.0 * ((r - g) / d + 4.0)
    };
    (
        (h / 2.0).round().rem_euclid(180.0) as u8,
        if max == 0.0 {
            0
        } else {
            (255.0 * d / max).round() as u8
        },
        (max * 255.0).round() as u8,
    )
}
fn hsv_to_bgr(h: u8, s: u8, v: u8) -> (u8, u8, u8) {
    let h = h as f32 * 2.0;
    let s = s as f32 / 255.;
    let v = v as f32 / 255.;
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0).rem_euclid(2.0) - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match (h / 60.0) as usize {
        0 => (c, x, 0.),
        1 => (x, c, 0.),
        2 => (0., c, x),
        3 => (0., x, c),
        4 => (x, 0., c),
        _ => (c, 0., x),
    };
    (
        ((b + m) * 255.).round() as u8,
        ((g + m) * 255.).round() as u8,
        ((r + m) * 255.).round() as u8,
    )
}

#[cfg(test)]
mod tests {
    use super::super::{AugSample, ByteImage, GeometryMetadata, Instances, SourceMetadata};
    use super::*;
    #[test]
    fn white_stays_white_under_saturation() {
        let s = AugSample {
            image: ByteImage::new(1, 1, 3, ColorOrder::Bgr, vec![255; 3]).unwrap(),
            classes: vec![],
            instances: Instances::empty(),
            source: SourceMetadata {
                primary_id: "w".into(),
                primary_index: 0,
                mixed_indexes: vec![],
            },
            geometry: GeometryMetadata {
                original_shape: [1, 1],
                current_shape: [1, 1],
                ratio: [1., 1.],
                pad: [0., 0.],
                reversible: true,
            },
        };
        assert_eq!(
            apply(s, [0., 1., 0.]).unwrap().image.data(),
            [255, 255, 255]
        );
    }
}
