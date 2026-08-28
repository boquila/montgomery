use super::{
    Instances,
    sample::{AugSample, AugmentationError},
};
pub fn apply(
    mut primary: AugSample,
    secondary: AugSample,
    ratio: f32,
) -> Result<AugSample, AugmentationError> {
    if primary.image.width() != secondary.image.width()
        || primary.image.height() != secondary.image.height()
        || primary.image.channels() != secondary.image.channels()
    {
        return Err(AugmentationError::new("MixUp image shapes/channels differ"));
    }
    for (a, b) in primary
        .image
        .data_mut()
        .iter_mut()
        .zip(secondary.image.data())
    {
        *a = ((*a as f32 * ratio + *b as f32 * (1.0 - ratio)).clamp(0., 255.)) as u8;
    }
    primary.instances = Instances::concatenate(&[primary.instances, secondary.instances])?;
    primary.classes.extend(secondary.classes);
    primary
        .source
        .mixed_indexes
        .push(secondary.source.primary_index);
    primary.geometry.reversible = false;
    primary.validate()?;
    Ok(primary)
}

#[cfg(test)]
mod tests {
    use super::super::{ByteImage, ColorOrder, GeometryMetadata, Instances, SourceMetadata};
    use super::*;
    fn s(v: u8, i: usize) -> AugSample {
        AugSample {
            image: ByteImage::filled(1, 1, 1, ColorOrder::Gray, v),
            classes: vec![],
            instances: Instances::empty(),
            source: SourceMetadata {
                primary_id: i.to_string(),
                primary_index: i,
                mixed_indexes: vec![],
            },
            geometry: GeometryMetadata {
                original_shape: [1, 1],
                current_shape: [1, 1],
                ratio: [1., 1.],
                pad: [0., 0.],
                reversible: true,
            },
        }
    }
    #[test]
    fn numpy_truncation() {
        assert_eq!(apply(s(0, 0), s(255, 1), 0.5).unwrap().image.data(), [127]);
    }
}
