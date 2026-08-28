use image::{DynamicImage, GenericImage, ImageBuffer, Rgb, imageops::FilterType};

use crate::training::{
    data::sample::{ImageMeta, SegmentationSource, VisionSample},
    geometry::BoxXyxy,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanvasPlacement {
    Center,
    TopLeft,
}

/// Joint deterministic resize used by the augmentation-free parity slice.
pub fn resize_to_canvas(
    mut sample: VisionSample,
    canvas: [u32; 2],
    placement: CanvasPlacement,
) -> VisionSample {
    let source = [sample.image.width(), sample.image.height()];
    let scale = (canvas[0] as f32 / source[0] as f32).min(canvas[1] as f32 / source[1] as f32);
    let resized = [
        (source[0] as f32 * scale).round().max(1.0) as u32,
        (source[1] as f32 * scale).round().max(1.0) as u32,
    ];
    let pad = match placement {
        CanvasPlacement::Center => [(canvas[0] - resized[0]) / 2, (canvas[1] - resized[1]) / 2],
        CanvasPlacement::TopLeft => [0, 0],
    };
    let resized_image = sample
        .image
        .resize_exact(resized[0], resized[1], FilterType::Triangle)
        .to_rgb8();
    let mut output = ImageBuffer::from_pixel(canvas[0], canvas[1], Rgb([114, 114, 114]));
    output
        .copy_from(&resized_image, pad[0], pad[1])
        .expect("resized image fits canvas");
    for target in &mut sample.targets {
        target.bbox = BoxXyxy::new([
            target.bbox.xmin * scale + pad[0] as f32,
            target.bbox.ymin * scale + pad[1] as f32,
            target.bbox.xmax * scale + pad[0] as f32,
            target.bbox.ymax * scale + pad[1] as f32,
        ])
        .expect("positive affine preserves a valid box");
        if let Some(SegmentationSource::Polygons(polygons)) = &mut target.segmentation {
            for point in polygons.iter_mut().flatten() {
                point[0] = point[0] * scale + pad[0] as f32;
                point[1] = point[1] * scale + pad[1] as f32;
            }
        }
    }
    sample.image = DynamicImage::ImageRgb8(output);
    sample
}

pub fn horizontal_flip(mut sample: VisionSample) -> VisionSample {
    let width = sample.image.width() as f32;
    sample.image = sample.image.fliph();
    for target in &mut sample.targets {
        target.bbox = BoxXyxy::new([
            width - target.bbox.xmax,
            target.bbox.ymin,
            width - target.bbox.xmin,
            target.bbox.ymax,
        ])
        .expect("flip preserves valid boxes");
        if let Some(SegmentationSource::Polygons(polygons)) = &mut target.segmentation {
            for point in polygons.iter_mut().flatten() {
                point[0] = width - point[0];
            }
        }
    }
    sample
}

pub fn metadata(sample: &VisionSample, source_size: [u32; 2]) -> ImageMeta {
    ImageMeta {
        image_id: sample.image_id.clone(),
        source_size,
        canvas_size: [sample.image.width(), sample.image.height()],
        scale: [
            sample.image.width() as f32 / source_size[0] as f32,
            sample.image.height() as f32 / source_size[1] as f32,
        ],
        pad: [0.0, 0.0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::data::sample::DetectionTarget;

    #[test]
    fn flip_preserves_source_edge_contract() {
        let sample = VisionSample {
            image: DynamicImage::new_rgb8(100, 50),
            targets: vec![DetectionTarget {
                class_id: 0,
                bbox: BoxXyxy::new([0.0, 5.0, 20.0, 50.0]).unwrap(),
                segmentation: None,
                crowd: false,
                source_annotation_id: None,
            }],
            image_id: "boundary".into(),
            source_size: [100, 50],
        };
        let flipped = horizontal_flip(sample);
        assert_eq!(flipped.targets[0].bbox.xmax, 100.0);
    }
}
