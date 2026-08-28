use super::{
    mask::{self, IndexedMask},
    sample::{AugSample, AugmentationError, ColorOrder},
};

#[derive(Debug, Clone, PartialEq)]
pub enum MaskTargets {
    OverlapU8 {
        data: Vec<u8>,
        shape: [usize; 2],
    },
    OverlapU16 {
        data: Vec<u16>,
        shape: [usize; 2],
    },
    Separate {
        data: Vec<Vec<u8>>,
        shape: [usize; 2],
    },
}
#[derive(Debug, Clone, PartialEq)]
pub struct FormattedDetectionSample {
    pub image_chw_u8: Vec<u8>,
    pub image_shape: [usize; 3],
    pub classes: Vec<u32>,
    pub boxes_xywh_normalized: Vec<[f32; 4]>,
    pub masks: Option<MaskTargets>,
}

pub fn apply(
    mut sample: AugSample,
    mask_ratio: usize,
    mask_overlap: bool,
    segment: bool,
    retain_bgr: bool,
) -> Result<FormattedDetectionSample, AugmentationError> {
    sample.validate()?;
    let w = sample.image.width();
    let h = sample.image.height();
    sample.instances.denormalize(w as f32, h as f32);
    let masks = if segment {
        if mask_overlap {
            match mask::overlap(&mut sample.instances, &mut sample.classes, w, h, mask_ratio)? {
                IndexedMask::U8 {
                    data,
                    width,
                    height,
                } => Some(MaskTargets::OverlapU8 {
                    data,
                    shape: [height, width],
                }),
                IndexedMask::U16 {
                    data,
                    width,
                    height,
                } => Some(MaskTargets::OverlapU16 {
                    data,
                    shape: [height, width],
                }),
            }
        } else {
            let (data, mw, mh) = mask::separate(&sample.instances, w, h, mask_ratio)?;
            Some(MaskTargets::Separate {
                data,
                shape: [mh, mw],
            })
        }
    } else {
        None
    };
    sample.instances.convert(super::BoxFormat::Xywh);
    sample.instances.normalize(w as f32, h as f32);
    sample.instances.convert(super::BoxFormat::Xywh);
    let boxes = sample.instances.boxes().iter().map(|b| b.0).collect();
    let c = sample.image.channels();
    let mut chw = vec![0; sample.image.data().len()];
    for y in 0..h {
        for x in 0..w {
            for oc in 0..c {
                let sc = if !retain_bgr && c == 3 && sample.image.color() == ColorOrder::Bgr {
                    2 - oc
                } else {
                    oc
                };
                chw[(oc * h + y) * w + x] = sample.image.pixel(x, y)[sc];
            }
        }
    }
    Ok(FormattedDetectionSample {
        image_chw_u8: chw,
        image_shape: [c, h, w],
        classes: sample.classes,
        boxes_xywh_normalized: boxes,
        masks,
    })
}

#[cfg(test)]
mod tests {
    use super::super::{
        BBox, BoxFormat, ByteImage, ColorOrder, GeometryMetadata, Instances, SourceMetadata,
    };
    use super::*;
    #[test]
    fn bgr_to_rgb_chw_and_normalized_box() {
        let s = AugSample {
            image: ByteImage::new(1, 1, 3, ColorOrder::Bgr, vec![1, 2, 3]).unwrap(),
            classes: vec![7],
            instances: Instances::new(vec![BBox([0., 0., 1., 1.])], BoxFormat::Xyxy, false, None)
                .unwrap(),
            source: SourceMetadata {
                primary_id: "x".into(),
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
        let o = apply(s, 1, true, false, false).unwrap();
        assert_eq!(o.image_chw_u8, [3, 2, 1]);
        assert_eq!(o.boxes_xywh_normalized, [[0.5, 0.5, 1., 1.]]);
    }
}
