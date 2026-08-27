use image::{DynamicImage, ImageBuffer, Rgb, RgbImage, imageops};

fn resize_opencv_linear(source: &DynamicImage, width: u32, height: u32) -> RgbImage {
    let source = source.to_rgb8();
    let source_width = source.width();
    let source_height = source.height();
    let scale_x = source_width as f64 / width as f64;
    let scale_y = source_height as f64 / height as f64;

    ImageBuffer::from_fn(width, height, |x, y| {
        let source_x = (x as f64 + 0.5) * scale_x - 0.5;
        let source_y = (y as f64 + 0.5) * scale_y - 0.5;
        let x0 = source_x.floor().clamp(0.0, source_width as f64 - 1.0) as u32;
        let y0 = source_y.floor().clamp(0.0, source_height as f64 - 1.0) as u32;
        let x1 = (x0 + 1).min(source_width - 1);
        let y1 = (y0 + 1).min(source_height - 1);
        let weight_x = source_x.clamp(0.0, source_width as f64 - 1.0) - x0 as f64;
        let weight_y = source_y.clamp(0.0, source_height as f64 - 1.0) - y0 as f64;
        let top_left = source.get_pixel(x0, y0).0;
        let top_right = source.get_pixel(x1, y0).0;
        let bottom_left = source.get_pixel(x0, y1).0;
        let bottom_right = source.get_pixel(x1, y1).0;
        let mut output = [0_u8; 3];

        for channel in 0..3 {
            let top =
                top_left[channel] as f64 * (1.0 - weight_x) + top_right[channel] as f64 * weight_x;
            let bottom = bottom_left[channel] as f64 * (1.0 - weight_x)
                + bottom_right[channel] as f64 * weight_x;
            output[channel] = (top * (1.0 - weight_y) + bottom * weight_y)
                .round()
                .clamp(0.0, 255.0) as u8;
        }
        Rgb(output)
    })
}

/// An image fitted into a square model input without changing its aspect ratio.
pub(crate) struct LetterboxedImage {
    image: DynamicImage,
    scale: f32,
    pad_x: u32,
    pad_y: u32,
    source_width: u32,
    source_height: u32,
}

impl LetterboxedImage {
    /// Match YOLOX's inference transform: resize to fit, anchor at the top-left, and fill unused
    /// pixels with 114. The geometry is retained for mapping detections back to the source image.
    pub(crate) fn yolox(source: &DynamicImage, size: usize) -> Self {
        let source_width = source.width();
        let source_height = source.height();
        let scale = (size as f32 / source_width as f32).min(size as f32 / source_height as f32);
        let resized_width = (source_width as f32 * scale) as u32;
        let resized_height = (source_height as f32 * scale) as u32;
        let resized = source.resize_exact(
            resized_width,
            resized_height,
            imageops::FilterType::Triangle,
        );

        let mut canvas = ImageBuffer::from_pixel(size as u32, size as u32, Rgb([114, 114, 114]));
        imageops::replace(&mut canvas, &resized.to_rgb8(), 0, 0);

        Self {
            image: DynamicImage::ImageRgb8(canvas),
            scale,
            pad_x: 0,
            pad_y: 0,
            source_width,
            source_height,
        }
    }

    /// Match Ultralytics' single-image rectangular inference letterbox: round the resized
    /// dimensions, reduce padding to the configured stride, center the image, and fill with 114.
    pub(crate) fn ultralytics(source: &DynamicImage, size: usize, stride: usize) -> Self {
        let source_width = source.width();
        let source_height = source.height();
        let scale = (size as f32 / source_width as f32).min(size as f32 / source_height as f32);
        let resized_width = (source_width as f32 * scale).round() as u32;
        let resized_height = (source_height as f32 * scale).round() as u32;
        let resized = resize_opencv_linear(source, resized_width, resized_height);
        let total_pad_x = (size as u32 - resized_width) % stride as u32;
        let total_pad_y = (size as u32 - resized_height) % stride as u32;
        let pad_x = total_pad_x / 2;
        let pad_y = total_pad_y / 2;
        let canvas_width = resized_width + total_pad_x;
        let canvas_height = resized_height + total_pad_y;

        let mut canvas = ImageBuffer::from_pixel(canvas_width, canvas_height, Rgb([114, 114, 114]));
        imageops::replace(&mut canvas, &resized, pad_x.into(), pad_y.into());

        Self {
            image: DynamicImage::ImageRgb8(canvas),
            scale,
            pad_x,
            pad_y,
            source_width,
            source_height,
        }
    }

    pub(crate) fn image(&self) -> &DynamicImage {
        &self.image
    }

    pub(crate) fn to_source_box(&self, bbox: [f32; 4]) -> [f32; 4] {
        let map_x = |value: f32| {
            ((value - self.pad_x as f32) / self.scale).clamp(0.0, self.source_width as f32)
        };
        let map_y = |value: f32| {
            ((value - self.pad_y as f32) / self.scale).clamp(0.0, self.source_height as f32)
        };
        [
            map_x(bbox[0]),
            map_y(bbox[1]),
            map_x(bbox[2]),
            map_y(bbox[3]),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_aspect_ratio_and_pads_with_yolox_value() {
        let source = DynamicImage::new_rgb8(100, 50);
        let prepared = LetterboxedImage::yolox(&source, 640);
        let image = prepared.image().to_rgb8();

        assert_eq!(image.dimensions(), (640, 640));
        assert_eq!(*image.get_pixel(10, 10), Rgb([0, 0, 0]));
        assert_eq!(*image.get_pixel(10, 500), Rgb([114, 114, 114]));
    }

    #[test]
    fn maps_model_boxes_back_to_source_coordinates() {
        let source = DynamicImage::new_rgb8(100, 50);
        let prepared = LetterboxedImage::yolox(&source, 640);
        assert_eq!(
            prepared.to_source_box([64.0, 32.0, 320.0, 256.0]),
            [10.0, 5.0, 50.0, 40.0]
        );
        assert_eq!(
            prepared.to_source_box([0.0, 0.0, 640.0, 640.0]),
            [0.0, 0.0, 100.0, 50.0]
        );
    }

    #[test]
    fn centers_ultralytics_letterbox_and_tracks_padding() {
        let source = DynamicImage::new_rgb8(100, 50);
        let prepared = LetterboxedImage::ultralytics(&source, 640, 32);
        let image = prepared.image().to_rgb8();

        assert_eq!(image.dimensions(), (640, 320));
        assert_eq!(*image.get_pixel(10, 10), Rgb([0, 0, 0]));
        assert_eq!(
            prepared.to_source_box([64.0, 32.0, 320.0, 256.0]),
            [10.0, 5.0, 50.0, 40.0]
        );
    }

    #[test]
    #[ignore]
    fn measures_ultralytics_preprocessing_fixture_parity() {
        let source = image::open("assets/dog_bike_man.jpg").unwrap();
        let expected_source = image::open("target/yolov3-tinyu-source-reference.png")
            .expect("generate the reference with tools/export_ultralytics_fixtures.py")
            .to_rgb8();
        let actual_source = source.to_rgb8();
        let source_total_error: u64 = actual_source
            .as_raw()
            .iter()
            .zip(expected_source.as_raw())
            .map(|(a, b)| u64::from(a.abs_diff(*b)))
            .sum();
        let source_mean_error = source_total_error as f64 / actual_source.as_raw().len() as f64;
        eprintln!("source decoder parity: mean_abs_error={source_mean_error:.6}");
        assert!(source_mean_error <= 0.01);
        let actual = LetterboxedImage::ultralytics(&source, 640, 32)
            .image()
            .to_rgb8();
        actual
            .save("target/yolov3-tinyu-preprocessed-rust.png")
            .unwrap();
        let expected = image::open("target/yolov3-tinyu-preprocessed-reference.png")
            .expect("generate the reference with tools/export_ultralytics_fixtures.py")
            .to_rgb8();
        assert_eq!(actual.dimensions(), expected.dimensions());

        let (total_error, max_error, differing) = actual
            .as_raw()
            .iter()
            .zip(expected.as_raw())
            .fold((0_u64, 0_u8, 0_usize), |(sum, max, count), (a, b)| {
                let error = a.abs_diff(*b);
                (
                    sum + u64::from(error),
                    max.max(error),
                    count + usize::from(error != 0),
                )
            });
        let values = actual.as_raw().len();
        let mean_error = total_error as f64 / values as f64;
        eprintln!(
            "preprocessing parity: mean_abs_error={mean_error:.6}, max_abs_error={max_error}, differing={differing}/{values}"
        );
        assert!(mean_error <= 0.11);
        assert!(max_error <= 2);
    }
}
