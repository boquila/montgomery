use super::{Instances, Polygon, resize, sample::AugmentationError};

#[derive(Debug, Clone, PartialEq)]
pub enum IndexedMask {
    U8 {
        data: Vec<u8>,
        width: usize,
        height: usize,
    },
    U16 {
        data: Vec<u16>,
        width: usize,
        height: usize,
    },
}

pub fn polygon_mask(width: usize, height: usize, polygon: &Polygon) -> Vec<u8> {
    let mut out = vec![0; width * height];
    if polygon.len() < 3 {
        return out;
    }
    for y in 0..height {
        let scan = y as f32 + 0.5;
        let mut intersections = Vec::new();
        for i in 0..polygon.len() {
            let a = polygon[i];
            let b = polygon[(i + 1) % polygon.len()];
            if (a[1] <= scan && b[1] > scan) || (b[1] <= scan && a[1] > scan) {
                intersections.push(a[0] + (scan - a[1]) * (b[0] - a[0]) / (b[1] - a[1]));
            }
        }
        intersections.sort_by(|a, b| a.total_cmp(b));
        for pair in intersections.chunks_exact(2) {
            let start = pair[0].ceil().max(0.) as usize;
            let end = pair[1].floor().min(width as f32 - 1.) as usize;
            if start <= end && start < width {
                for x in start..=end.min(width - 1) {
                    out[y * width + x] = 1;
                }
            }
        }
    }
    out
}

fn downsample(mask: &[u8], width: usize, height: usize, ratio: usize) -> Vec<u8> {
    let image = super::ByteImage::new(
        width,
        height,
        1,
        super::ColorOrder::Gray,
        mask.iter().map(|v| v.saturating_mul(255)).collect(),
    )
    .expect("valid mask");
    resize::resize(
        &image,
        width / ratio,
        height / ratio,
        super::Interpolation::Bilinear,
    )
    .expect("positive validated mask target")
    .data()
    .iter()
    .map(|v| u8::from(*v > 0))
    .collect()
}

pub fn overlap(
    instances: &mut Instances,
    classes: &mut Vec<u32>,
    width: usize,
    height: usize,
    ratio: usize,
) -> Result<IndexedMask, AugmentationError> {
    if ratio == 0 || ratio > width.min(height) {
        return Err(AugmentationError::new("invalid mask ratio"));
    }
    if instances.is_empty() {
        return Ok(IndexedMask::U8 {
            data: vec![0; (width / ratio) * (height / ratio)],
            width: width / ratio,
            height: height / ratio,
        });
    }
    let segments = instances
        .segments()
        .ok_or_else(|| AugmentationError::new("segmentation masks require polygons"))?;
    let mut masks: Vec<_> = segments
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let m = downsample(&polygon_mask(width, height, p), width, height, ratio);
            let area = m.iter().map(|v| *v as usize).sum::<usize>();
            (i, m, area)
        })
        .collect();
    masks.sort_by_key(|entry| std::cmp::Reverse(entry.2));
    let order: Vec<_> = masks.iter().map(|m| m.0).collect();
    instances.select(&order);
    *classes = order.iter().map(|&i| classes[i]).collect();
    let w = width / ratio;
    let h = height / ratio;
    if masks.len() <= 255 {
        let mut out = vec![0u8; w * h];
        for (id, (_, m, _)) in masks.iter().enumerate() {
            for (i, v) in m.iter().enumerate() {
                if *v != 0 {
                    out[i] = out[i].max((id + 1) as u8);
                }
            }
        }
        Ok(IndexedMask::U8 {
            data: out,
            width: w,
            height: h,
        })
    } else {
        let mut out = vec![0u16; w * h];
        for (id, (_, m, _)) in masks.iter().enumerate() {
            for (i, v) in m.iter().enumerate() {
                if *v != 0 {
                    out[i] = out[i].max((id + 1) as u16);
                }
            }
        }
        Ok(IndexedMask::U16 {
            data: out,
            width: w,
            height: h,
        })
    }
}

pub fn separate(
    instances: &Instances,
    width: usize,
    height: usize,
    ratio: usize,
) -> Result<(Vec<Vec<u8>>, usize, usize), AugmentationError> {
    if ratio == 0 || ratio > width.min(height) {
        return Err(AugmentationError::new("invalid mask ratio"));
    }
    if instances.is_empty() {
        return Ok((Vec::new(), width / ratio, height / ratio));
    }
    let segments = instances
        .segments()
        .ok_or_else(|| AugmentationError::new("segmentation masks require polygons"))?;
    Ok((
        segments
            .iter()
            .map(|p| downsample(&polygon_mask(width, height, p), width, height, ratio))
            .collect(),
        width / ratio,
        height / ratio,
    ))
}

#[cfg(test)]
mod tests {
    use super::super::{BBox, BoxFormat};
    use super::*;
    #[test]
    fn overlap_reorders_and_widens() {
        let mut boxes = Vec::new();
        let mut polygons = Vec::new();
        let mut classes = Vec::new();
        for i in 0..256 {
            boxes.push(BBox([0., 0., 4., 4.]));
            polygons.push(vec![[0., 0.], [4., 0.], [4., 4.], [0., 4.]]);
            classes.push(i as u32);
        }
        let mut instances = Instances::new(boxes, BoxFormat::Xyxy, false, Some(polygons)).unwrap();
        assert!(matches!(
            overlap(&mut instances, &mut classes, 4, 4, 1).unwrap(),
            IndexedMask::U16 { .. }
        ));
    }

    #[test]
    fn empty_background_has_valid_mask_targets() {
        let mut instances = Instances::empty();
        let mut classes = Vec::new();
        assert_eq!(
            overlap(&mut instances, &mut classes, 8, 4, 2).unwrap(),
            IndexedMask::U8 {
                data: vec![0; 8],
                width: 4,
                height: 2,
            }
        );
        assert_eq!(separate(&instances, 8, 4, 2).unwrap(), (vec![], 4, 2));
    }
}
