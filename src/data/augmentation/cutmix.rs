//! Ultralytics detector CutMix: paste only into a primary-object-free region.

use super::{
    BoxFormat, Instances,
    sample::{AugSample, AugmentationError},
};

fn ioa(rect: [f32; 4], b: [f32; 4]) -> f32 {
    let inter = (rect[2].min(b[2]) - rect[0].max(b[0])).max(0.)
        * (rect[3].min(b[3]) - rect[1].max(b[1])).max(0.);
    let area = (b[2] - b[0]).max(0.) * (b[3] - b[1]).max(0.);
    inter / (area + 1e-16)
}

pub fn overlaps_primary(sample: &AugSample, rect: [usize; 4]) -> bool {
    let rf = [
        rect[0] as f32,
        rect[1] as f32,
        rect[2] as f32,
        rect[3] as f32,
    ];
    sample.instances.boxes().iter().any(|bbox| {
        let mut box_xyxy = bbox.xyxy(sample.instances.format());
        if sample.instances.normalized() {
            box_xyxy[0] *= sample.image.width() as f32;
            box_xyxy[2] *= sample.image.width() as f32;
            box_xyxy[1] *= sample.image.height() as f32;
            box_xyxy[3] *= sample.image.height() as f32;
        }
        ioa(rf, box_xyxy) > 0.0
    })
}

pub fn apply(
    mut primary: AugSample,
    mut secondary: AugSample,
    rect: [usize; 4],
    segment_threshold: bool,
) -> Result<AugSample, AugmentationError> {
    if primary.image.width() != secondary.image.width()
        || primary.image.height() != secondary.image.height()
        || primary.image.channels() != secondary.image.channels()
    {
        return Err(AugmentationError::new(
            "CutMix image shapes/channels differ",
        ));
    }
    primary
        .instances
        .denormalize(primary.image.width() as f32, primary.image.height() as f32);
    secondary.instances.denormalize(
        secondary.image.width() as f32,
        secondary.image.height() as f32,
    );
    primary.instances.convert(BoxFormat::Xyxy);
    secondary.instances.convert(BoxFormat::Xyxy);
    let rf = [
        rect[0] as f32,
        rect[1] as f32,
        rect[2] as f32,
        rect[3] as f32,
    ];
    if primary.instances.boxes().iter().any(|b| ioa(rf, b.0) > 0.) {
        return Ok(primary);
    }
    let threshold = if segment_threshold { 0.01 } else { 0.10 };
    let keep: Vec<_> = secondary
        .instances
        .boxes()
        .iter()
        .enumerate()
        .filter_map(|(i, b)| (ioa(rf, b.0) >= threshold).then_some(i))
        .collect();
    if keep.is_empty() {
        return Ok(primary);
    }
    for y in rect[1]..rect[3].min(primary.image.height()) {
        for x in rect[0]..rect[2].min(primary.image.width()) {
            primary
                .image
                .pixel_mut(x, y)
                .copy_from_slice(secondary.image.pixel(x, y));
        }
    }
    secondary.instances.select(&keep);
    secondary.classes = keep.iter().map(|&i| secondary.classes[i]).collect();
    for b in &mut secondary.instances.boxes {
        b.0[0] = b.0[0].clamp(rf[0], rf[2]);
        b.0[1] = b.0[1].clamp(rf[1], rf[3]);
        b.0[2] = b.0[2].clamp(rf[0], rf[2]);
        b.0[3] = b.0[3].clamp(rf[1], rf[3]);
    }
    if let Some(polys) = &mut secondary.instances.segments {
        for p in polys.iter_mut().flatten() {
            p[0] = p[0].clamp(rf[0], rf[2]);
            p[1] = p[1].clamp(rf[1], rf[3]);
        }
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

pub fn candidate_rect(width: usize, height: usize, lambda: f32, center: [i32; 2]) -> [usize; 4] {
    let ratio = (1.0 - lambda).sqrt();
    let w = (width as f32 * ratio) as i32;
    let h = (height as f32 * ratio) as i32;
    [
        (center[0] - w / 2).clamp(0, width as i32) as usize,
        (center[1] - h / 2).clamp(0, height as i32) as usize,
        (center[0] + w / 2).clamp(0, width as i32) as usize,
        (center[1] + h / 2).clamp(0, height as i32) as usize,
    ]
}

#[cfg(test)]
mod tests {
    use super::super::{BBox, ByteImage, ColorOrder, GeometryMetadata, SourceMetadata};
    use super::*;
    #[test]
    fn candidate_is_clipped() {
        assert_eq!(candidate_rect(10, 10, 0., [0, 0]), [0, 0, 5, 5]);
    }

    fn sample(value: u8, boxes: Vec<BBox>, index: usize) -> AugSample {
        let classes = vec![index as u32; boxes.len()];
        AugSample {
            image: ByteImage::filled(8, 8, 3, ColorOrder::Bgr, value),
            classes,
            instances: Instances::new(boxes, BoxFormat::Xyxy, false, None).unwrap(),
            source: SourceMetadata {
                primary_id: index.to_string(),
                primary_index: index,
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
    fn detector_cutmix_adds_qualifying_secondary_object() {
        let output = apply(
            sample(1, vec![], 0),
            sample(9, vec![BBox([2., 2., 6., 6.])], 1),
            [1, 1, 7, 7],
            false,
        )
        .unwrap();
        assert_eq!(output.instances.len(), 1);
        assert_eq!(output.image.pixel(3, 3)[0], 9);
    }
}
