//! Modified from Ultralytics `LetterBox` at the pinned compatibility commit.

use super::{
    Interpolation,
    resize::resize,
    rng::python_round,
    sample::{AugSample, AugmentationError, ByteImage},
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LetterBoxParams {
    pub new_shape: [usize; 2],
    pub auto: bool,
    pub scale_fill: bool,
    pub scaleup: bool,
    pub center: bool,
    pub stride: usize,
    pub padding: u8,
    pub interpolation: Interpolation,
}

impl LetterBoxParams {
    pub fn validation(side: usize) -> Self {
        Self {
            new_shape: [side, side],
            auto: false,
            scale_fill: false,
            scaleup: false,
            center: true,
            stride: 32,
            padding: 114,
            interpolation: Interpolation::Bilinear,
        }
    }
}

pub fn apply(mut sample: AugSample, p: LetterBoxParams) -> Result<AugSample, AugmentationError> {
    let [nh, nw] = p.new_shape;
    if nh == 0 || nw == 0 || p.stride == 0 {
        return Err(AugmentationError::new("invalid LetterBox shape/stride"));
    }
    let old_h = sample.image.height();
    let old_w = sample.image.width();
    let mut r = (nh as f32 / old_h as f32).min(nw as f32 / old_w as f32);
    if !p.scaleup {
        r = r.min(1.0);
    }
    let mut ratio = [r, r];
    let mut rw = python_round(old_w as f32 * r).max(1) as usize;
    let mut rh = python_round(old_h as f32 * r).max(1) as usize;
    let mut dw = nw.saturating_sub(rw);
    let mut dh = nh.saturating_sub(rh);
    if p.auto {
        dw %= p.stride;
        dh %= p.stride;
    } else if p.scale_fill {
        dw = 0;
        dh = 0;
        rw = nw;
        rh = nh;
        ratio = [nw as f32 / old_w as f32, nh as f32 / old_h as f32];
    }
    let (half_w, half_h) = if p.center {
        (dw as f32 / 2.0, dh as f32 / 2.0)
    } else {
        (0.0, 0.0)
    };
    let left = python_round(half_w - 0.1).max(0) as usize;
    let right = python_round(dw as f32 - half_w + 0.1).max(0) as usize;
    let top = python_round(half_h - 0.1).max(0) as usize;
    let bottom = python_round(dh as f32 - half_h + 0.1).max(0) as usize;
    let resized = resize(&sample.image, rw, rh, p.interpolation)?;
    let mut output = ByteImage::filled(
        rw + left + right,
        rh + top + bottom,
        resized.channels(),
        resized.color(),
        p.padding,
    );
    for y in 0..rh {
        for x in 0..rw {
            output
                .pixel_mut(x + left, y + top)
                .copy_from_slice(resized.pixel(x, y));
        }
    }
    let previous_ratio = sample.geometry.ratio;
    let previous_pad = sample.geometry.pad;
    sample.instances.denormalize(old_w as f32, old_h as f32);
    sample.instances.scale(ratio[0], ratio[1]);
    sample.instances.pad(left as f32, top as f32)?;
    sample.image = output;
    sample.geometry.current_shape = [sample.image.height(), sample.image.width()];
    sample.geometry.ratio = [previous_ratio[0] * ratio[0], previous_ratio[1] * ratio[1]];
    sample.geometry.pad = [
        previous_pad[0] * ratio[0] + left as f32,
        previous_pad[1] * ratio[1] + top as f32,
    ];
    sample.geometry.reversible = true;
    sample.validate()?;
    Ok(sample)
}

#[cfg(test)]
mod tests {
    use super::super::{BBox, BoxFormat, ColorOrder, GeometryMetadata, Instances, SourceMetadata};
    use super::*;
    #[test]
    fn validates_python_border_rounding() {
        let s = AugSample {
            image: ByteImage::filled(20, 10, 3, ColorOrder::Bgr, 0),
            classes: vec![0],
            instances: Instances::new(vec![BBox([0., 0., 20., 10.])], BoxFormat::Xyxy, false, None)
                .unwrap(),
            source: SourceMetadata {
                primary_id: "x".into(),
                primary_index: 0,
                mixed_indexes: vec![],
            },
            geometry: GeometryMetadata {
                original_shape: [10, 20],
                current_shape: [10, 20],
                ratio: [1., 1.],
                pad: [0., 0.],
                reversible: true,
            },
        };
        let o = apply(s, LetterBoxParams::validation(32)).unwrap();
        assert_eq!([o.image.height(), o.image.width()], [32, 32]);
        assert_eq!(o.instances.boxes()[0].0, [6., 11., 26., 21.]);
    }

    #[test]
    fn composes_loader_resize_with_letterbox_geometry() {
        let s = AugSample {
            image: ByteImage::filled(20, 10, 3, ColorOrder::Bgr, 0),
            classes: vec![],
            instances: Instances::new(vec![], BoxFormat::Xyxy, false, None).unwrap(),
            source: SourceMetadata {
                primary_id: "x".into(),
                primary_index: 0,
                mixed_indexes: vec![],
            },
            geometry: GeometryMetadata {
                original_shape: [20, 40],
                current_shape: [10, 20],
                ratio: [0.5, 0.5],
                pad: [0.0, 0.0],
                reversible: true,
            },
        };
        let o = apply(s, LetterBoxParams::validation(32)).unwrap();
        assert_eq!(o.geometry.ratio, [0.5, 0.5]);
        assert_eq!(o.geometry.pad, [6.0, 11.0]);
        assert_eq!(o.geometry.current_shape, [32, 32]);
    }
}
