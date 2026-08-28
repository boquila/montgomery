//! Polygon CopyPaste modes adapted from the pinned Ultralytics pipeline.

use super::{
    BoxFormat, Instances,
    mask::polygon_mask,
    rng::python_round,
    sample::{AugSample, AugmentationError},
};

fn intersection_over_candidate(candidate: [f32; 4], primary: [f32; 4]) -> f32 {
    let inter = (candidate[2].min(primary[2]) - candidate[0].max(primary[0])).max(0.)
        * (candidate[3].min(primary[3]) - candidate[1].max(primary[1])).max(0.);
    let area = (candidate[2] - candidate[0]).max(0.) * (candidate[3] - candidate[1]).max(0.);
    inter / (area + 1e-16)
}

pub fn flip(mut sample: AugSample, fraction: f32) -> Result<AugSample, AugmentationError> {
    if fraction <= 0. || sample.instances.segments().is_none() {
        return Ok(sample);
    }
    let mut candidates = sample.instances.clone();
    candidates.denormalize(sample.image.width() as f32, sample.image.height() as f32);
    candidates.flip_horizontal(sample.image.width() as f32);
    sample
        .instances
        .denormalize(sample.image.width() as f32, sample.image.height() as f32);
    paste(sample, candidates, None, fraction, true)
}

pub fn mixup(
    mut primary: AugSample,
    mut secondary: AugSample,
    fraction: f32,
) -> Result<AugSample, AugmentationError> {
    if fraction <= 0. || secondary.instances.segments().is_none() {
        return Ok(primary);
    }
    primary
        .instances
        .denormalize(primary.image.width() as f32, primary.image.height() as f32);
    secondary.instances.denormalize(
        secondary.image.width() as f32,
        secondary.image.height() as f32,
    );
    paste(
        primary,
        secondary.instances,
        Some((
            secondary.image,
            secondary.classes,
            secondary.source.primary_index,
        )),
        fraction,
        false,
    )
}

fn paste(
    mut primary: AugSample,
    mut candidates: Instances,
    source: Option<(super::ByteImage, Vec<u32>, usize)>,
    fraction: f32,
    flipped: bool,
) -> Result<AugSample, AugmentationError> {
    if primary.image.channels() != 3 {
        return Err(AugmentationError::new(
            "CopyPaste currently requires three-channel images",
        ));
    }
    primary.instances.convert(BoxFormat::Xyxy);
    candidates.convert(BoxFormat::Xyxy);
    let mut eligible: Vec<_> = candidates
        .boxes()
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            let max = primary
                .instances
                .boxes()
                .iter()
                .map(|p| intersection_over_candidate(c.0, p.0))
                .fold(0.0, f32::max);
            (max < 0.30).then_some((i, max))
        })
        .collect();
    eligible.sort_by(|a, b| a.1.total_cmp(&b.1));
    eligible.truncate(python_round(fraction * eligible.len() as f32).max(0) as usize);
    if eligible.is_empty() {
        return Ok(primary);
    }
    let indexes: Vec<_> = eligible.iter().map(|v| v.0).collect();
    let source_image = if let Some((image, _, _)) = &source {
        image.clone()
    } else {
        let mut image = primary.image.clone();
        let w = image.width();
        for y in 0..image.height() {
            for x in 0..w / 2 {
                let other = w - 1 - x;
                for c in 0..image.channels() {
                    let a = image.offset(x, y, c);
                    let b = image.offset(other, y, c);
                    image.data_mut().swap(a, b);
                }
            }
        }
        image
    };
    let polygons = candidates
        .segments()
        .expect("eligibility requires polygons");
    let mut mask = vec![0u8; primary.image.width() * primary.image.height()];
    for &i in &indexes {
        for (j, v) in polygon_mask(primary.image.width(), primary.image.height(), &polygons[i])
            .into_iter()
            .enumerate()
        {
            mask[j] |= v;
        }
    }
    for y in 0..primary.image.height() {
        for x in 0..primary.image.width() {
            if mask[y * primary.image.width() + x] != 0 {
                primary
                    .image
                    .pixel_mut(x, y)
                    .copy_from_slice(source_image.pixel(x, y));
            }
        }
    }
    candidates.select(&indexes);
    let classes: Vec<u32> = if let Some((_, classes, index)) = source {
        primary.source.mixed_indexes.push(index);
        indexes.iter().map(|&i| classes[i]).collect()
    } else {
        indexes.iter().map(|&i| primary.classes[i]).collect()
    };
    primary.instances = Instances::concatenate(&[primary.instances, candidates])?;
    primary.classes.extend(classes);
    primary.geometry.reversible = false;
    let _ = flipped;
    primary.validate()?;
    Ok(primary)
}

#[cfg(test)]
mod tests {
    use super::super::{BBox, ByteImage, ColorOrder, GeometryMetadata, SourceMetadata};
    use super::*;

    #[test]
    fn flip_mode_appends_selected_non_overlapping_polygon() {
        let sample = AugSample {
            image: ByteImage::filled(10, 4, 3, ColorOrder::Bgr, 7),
            classes: vec![3],
            instances: Instances::new(
                vec![BBox([0., 0., 2., 4.])],
                BoxFormat::Xyxy,
                false,
                Some(vec![vec![[0., 0.], [2., 0.], [2., 4.], [0., 4.]]]),
            )
            .unwrap(),
            source: SourceMetadata {
                primary_id: "x".into(),
                primary_index: 0,
                mixed_indexes: vec![],
            },
            geometry: GeometryMetadata {
                original_shape: [4, 10],
                current_shape: [4, 10],
                ratio: [1., 1.],
                pad: [0., 0.],
                reversible: true,
            },
        };
        let output = flip(sample, 1.0).unwrap();
        assert_eq!(output.classes, [3, 3]);
        assert_eq!(output.instances.len(), 2);
    }
}
