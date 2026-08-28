//! Modified Rust adaptation of Ultralytics Mosaic-4 and Mosaic-9 placement.

use super::{
    Instances, MosaicGrid,
    sample::{AugSample, AugmentationError, ByteImage},
};

pub fn apply(
    primary: AugSample,
    partners: Vec<AugSample>,
    side: usize,
    grid: MosaicGrid,
    center: [i32; 2],
) -> Result<AugSample, AugmentationError> {
    match grid {
        MosaicGrid::Four => four(primary, partners, side, center),
        MosaicGrid::Nine => nine(primary, partners, side, center),
    }
}

fn paste(canvas: &mut ByteImage, image: &ByteImage, dst: [i32; 4]) -> [i32; 4] {
    let [mut x1, mut y1, mut x2, mut y2] = dst;
    let sx1 = (-x1).max(0);
    let sy1 = (-y1).max(0);
    x1 = x1.max(0);
    y1 = y1.max(0);
    x2 = x2.min(canvas.width() as i32);
    y2 = y2.min(canvas.height() as i32);
    for y in y1..y2 {
        for x in x1..x2 {
            let sx = (sx1 + x - x1) as usize;
            let sy = (sy1 + y - y1) as usize;
            if sx < image.width() && sy < image.height() {
                canvas
                    .pixel_mut(x as usize, y as usize)
                    .copy_from_slice(image.pixel(sx, sy));
            }
        }
    }
    [x1 - sx1, y1 - sy1, x2, y2]
}

fn merge(
    mut primary: AugSample,
    placed: Vec<(AugSample, [i32; 2])>,
    canvas: ByteImage,
) -> Result<AugSample, AugmentationError> {
    let mut groups = Vec::new();
    let mut classes = Vec::new();
    let mut sources = Vec::new();
    for (mut s, pad) in placed {
        s.instances
            .denormalize(s.image.width() as f32, s.image.height() as f32);
        s.instances.pad(pad[0] as f32, pad[1] as f32)?;
        groups.push(s.instances);
        classes.extend(s.classes);
        sources.push(s.source.primary_index);
    }
    primary.image = canvas;
    primary.instances = Instances::concatenate(&groups)?;
    primary.classes = classes;
    primary
        .instances
        .clip(primary.image.width() as f32, primary.image.height() as f32);
    let keep = primary.instances.remove_zero_area();
    primary.classes = keep.iter().map(|&i| primary.classes[i]).collect();
    primary.source.mixed_indexes.extend(
        sources
            .into_iter()
            .filter(|i| *i != primary.source.primary_index),
    );
    primary.geometry.current_shape = [primary.image.height(), primary.image.width()];
    primary.geometry.reversible = false;
    primary.validate()?;
    Ok(primary)
}

fn four(
    primary: AugSample,
    partners: Vec<AugSample>,
    side: usize,
    center: [i32; 2],
) -> Result<AugSample, AugmentationError> {
    if partners.len() != 3 {
        return Err(AugmentationError::new(
            "Mosaic-4 requires exactly three partners",
        ));
    }
    let mut samples = vec![primary.clone()];
    samples.extend(partners);
    let mut canvas = ByteImage::filled(
        side * 2,
        side * 2,
        primary.image.channels(),
        primary.image.color(),
        114,
    );
    let mut placed = Vec::new();
    for (i, s) in samples.into_iter().enumerate() {
        if s.image.channels() != canvas.channels() {
            return Err(AugmentationError::new("Mosaic channel mismatch"));
        }
        let w = s.image.width() as i32;
        let h = s.image.height() as i32;
        let dst = match i {
            0 => [center[0] - w, center[1] - h, center[0], center[1]],
            1 => [center[0], center[1] - h, center[0] + w, center[1]],
            2 => [center[0] - w, center[1], center[0], center[1] + h],
            _ => [center[0], center[1], center[0] + w, center[1] + h],
        };
        let effective = paste(&mut canvas, &s.image, dst);
        placed.push((s, [effective[0], effective[1]]));
    }
    merge(primary, placed, canvas)
}

fn nine(
    primary: AugSample,
    partners: Vec<AugSample>,
    side: usize,
    _center: [i32; 2],
) -> Result<AugSample, AugmentationError> {
    if partners.len() != 8 {
        return Err(AugmentationError::new(
            "Mosaic-9 requires exactly eight partners",
        ));
    }
    let mut samples = vec![primary.clone()];
    samples.extend(partners);
    let mut canvas = ByteImage::filled(
        side * 3,
        side * 3,
        primary.image.channels(),
        primary.image.color(),
        114,
    );
    let mut placed = Vec::new();
    let mut previous = [-1_i32, -1_i32];
    let mut center_shape = [0_i32, 0_i32];
    for (index, s) in samples.into_iter().enumerate() {
        let width = s.image.width() as i32;
        let height = s.image.height() as i32;
        let [previous_height, previous_width] = previous;
        let [center_height, center_width] = center_shape;
        let x = side as i32;
        let y = side as i32;
        let dst = match index {
            0 => {
                center_shape = [height, width];
                [x, y, x + width, y + height]
            }
            1 => [x, y - height, x + width, y],
            2 => [
                x + previous_width,
                y - height,
                x + previous_width + width,
                y,
            ],
            3 => [x + center_width, y, x + center_width + width, y + height],
            4 => [
                x + center_width,
                y + previous_height,
                x + center_width + width,
                y + previous_height + height,
            ],
            5 => [
                x + center_width - width,
                y + center_height,
                x + center_width,
                y + center_height + height,
            ],
            6 => [
                x + center_width - previous_width - width,
                y + center_height,
                x + center_width - previous_width,
                y + center_height + height,
            ],
            7 => [x - width, y + center_height - height, x, y + center_height],
            _ => [
                x - width,
                y + center_height - previous_height - height,
                x,
                y + center_height - previous_height,
            ],
        };
        let effective = paste(&mut canvas, &s.image, dst);
        placed.push((
            s,
            [
                effective[0] - side as i32 / 2,
                effective[1] - side as i32 / 2,
            ],
        ));
        previous = [height, width];
    }
    let mut cropped = ByteImage::filled(side * 2, side * 2, canvas.channels(), canvas.color(), 114);
    for y in 0..side * 2 {
        for x in 0..side * 2 {
            cropped
                .pixel_mut(x, y)
                .copy_from_slice(canvas.pixel(x + side / 2, y + side / 2));
        }
    }
    merge(primary, placed, cropped)
}

#[cfg(test)]
mod tests {
    use super::super::{BBox, BoxFormat, ColorOrder, GeometryMetadata, SourceMetadata};
    use super::*;

    fn sample(index: usize, value: u8) -> AugSample {
        AugSample {
            image: ByteImage::filled(4, 4, 3, ColorOrder::Bgr, value),
            classes: vec![index as u32],
            instances: Instances::new(vec![BBox([0., 0., 4., 4.])], BoxFormat::Xyxy, false, None)
                .unwrap(),
            source: SourceMetadata {
                primary_id: index.to_string(),
                primary_index: index,
                mixed_indexes: vec![],
            },
            geometry: GeometryMetadata {
                original_shape: [4, 4],
                current_shape: [4, 4],
                ratio: [1., 1.],
                pad: [0., 0.],
                reversible: true,
            },
        }
    }

    #[test]
    fn mosaic_four_places_all_quadrants_without_mutating_sources() {
        let primary = sample(0, 1);
        let cached = primary.clone();
        let output = apply(
            primary,
            vec![sample(1, 2), sample(2, 3), sample(3, 4)],
            4,
            MosaicGrid::Four,
            [4, 4],
        )
        .unwrap();
        assert_eq!([output.image.height(), output.image.width()], [8, 8]);
        assert_eq!(output.classes, [0, 1, 2, 3]);
        assert_eq!(cached.image.data()[0], 1);
        assert_eq!(output.image.pixel(1, 1)[0], 1);
        assert_eq!(output.image.pixel(6, 1)[0], 2);
        assert_eq!(output.image.pixel(1, 6)[0], 3);
        assert_eq!(output.image.pixel(6, 6)[0], 4);
    }

    #[test]
    fn mosaic_nine_crops_to_double_side() {
        let output = apply(
            sample(0, 1),
            (1..9).map(|i| sample(i, i as u8)).collect(),
            4,
            MosaicGrid::Nine,
            [0, 0],
        )
        .unwrap();
        assert_eq!([output.image.height(), output.image.width()], [8, 8]);
    }
}
