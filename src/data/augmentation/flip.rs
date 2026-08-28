use super::sample::{AugSample, AugmentationError};

pub fn horizontal(mut sample: AugSample) -> Result<AugSample, AugmentationError> {
    let w = sample.image.width();
    for y in 0..sample.image.height() {
        for x in 0..w / 2 {
            let opposite = w - 1 - x;
            for c in 0..sample.image.channels() {
                let a = sample.image.offset(x, y, c);
                let b = sample.image.offset(opposite, y, c);
                sample.image.data_mut().swap(a, b);
            }
        }
    }
    let extent = if sample.instances.normalized() {
        1.0
    } else {
        w as f32
    };
    sample.instances.flip_horizontal(extent);
    sample.validate()?;
    Ok(sample)
}
pub fn vertical(mut sample: AugSample) -> Result<AugSample, AugmentationError> {
    let h = sample.image.height();
    let row = sample.image.width() * sample.image.channels();
    for y in 0..h / 2 {
        let other = h - 1 - y;
        for x in 0..row {
            sample.image.data_mut().swap(y * row + x, other * row + x);
        }
    }
    let extent = if sample.instances.normalized() {
        1.0
    } else {
        h as f32
    };
    sample.instances.flip_vertical(extent);
    sample.validate()?;
    Ok(sample)
}

#[cfg(test)]
mod tests {
    use super::super::{
        BBox, BoxFormat, ByteImage, ColorOrder, GeometryMetadata, Instances, SourceMetadata,
    };
    use super::*;
    fn sample() -> AugSample {
        AugSample {
            image: ByteImage::new(2, 1, 1, ColorOrder::Gray, vec![1, 2]).unwrap(),
            classes: vec![0],
            instances: Instances::new(vec![BBox([0., 0., 1., 1.])], BoxFormat::Xyxy, false, None)
                .unwrap(),
            source: SourceMetadata {
                primary_id: "x".into(),
                primary_index: 0,
                mixed_indexes: vec![],
            },
            geometry: GeometryMetadata {
                original_shape: [1, 2],
                current_shape: [1, 2],
                ratio: [1., 1.],
                pad: [0., 0.],
                reversible: true,
            },
        }
    }
    #[test]
    fn flip_image_and_edges() {
        let s = horizontal(sample()).unwrap();
        assert_eq!(s.image.data(), [2, 1]);
        assert_eq!(
            s.instances.boxes()[0].xyxy(s.instances.format()),
            [1., 0., 2., 1.]
        );
    }
}
