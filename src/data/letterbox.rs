use fast_image_resize as fir;
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

/// Resize an RGB8 image with the `fast_image_resize` crate (runtime-dispatched SIMD kernels).
fn resize_fir(
    source: &DynamicImage,
    width: u32,
    height: u32,
    algorithm: fir::ResizeAlg,
) -> RgbImage {
    let source = source.to_rgb8();
    let mut destination = fir::images::Image::new(width, height, fir::PixelType::U8x3);
    let mut resizer = fir::Resizer::new();
    let options = fir::ResizeOptions::new()
        .resize_alg(algorithm)
        .use_alpha(false);
    resizer
        .resize(&source, &mut destination, &options)
        .expect("fir resize of an rgb8 image into a matching buffer");
    ImageBuffer::from_raw(width, height, destination.into_vec()).expect("fir destination buffer")
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
    ///
    /// Resizes with `fast_image_resize`'s adaptive-kernel bilinear convolution (runtime-dispatched
    /// SIMD), the counterpart of the `imageops::Triangle` resampling this used before: both apply
    /// an anti-aliased triangle kernel stretched by the downscale factor, and their outputs agree
    /// within +/-1 per channel. Source pixels are converted to RGB8 before resizing, matching the
    /// Ultralytics transform below.
    pub(crate) fn yolox(source: &DynamicImage, size: usize) -> Self {
        let source_width = source.width();
        let source_height = source.height();
        let scale = (size as f32 / source_width as f32).min(size as f32 / source_height as f32);
        let resized_width = (source_width as f32 * scale) as u32;
        let resized_height = (source_height as f32 * scale) as u32;
        let resized = resize_fir(
            source,
            resized_width,
            resized_height,
            fir::ResizeAlg::Convolution(fir::FilterType::Bilinear),
        );

        let mut canvas = ImageBuffer::from_pixel(size as u32, size as u32, Rgb([114, 114, 114]));
        imageops::replace(&mut canvas, &resized, 0, 0);

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
    ///
    /// The resize deliberately stays on the hand-rolled cv2-equivalent bilinear sampler:
    /// `fast_image_resize`'s U8 kernels quantize weights more coarsely than cv2 and fall outside
    /// the preprocessing-parity budget of `measures_ultralytics_preprocessing_fixture_parity`,
    /// while its parity-preserving U16 pipeline is slower than this loop.
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

    /// The letterbox scale (source pixels per model-input pixel) and centered padding.
    pub(crate) fn letterbox_geometry(&self) -> (f32, f32, f32) {
        (self.scale, self.pad_x as f32, self.pad_y as f32)
    }

    /// The original source-image dimensions.
    pub(crate) fn source_dimensions(&self) -> (u32, u32) {
        (self.source_width, self.source_height)
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

/// Match Ultralytics' classification inference transform (`classify_transforms`): resize the
/// shortest edge to `size` with anti-aliased bilinear filtering (torchvision `T.Resize` with an
/// int size truncates the long-edge dimension), then a centered `size x size` crop.
///
/// The normalization constants of the official transform are identity, so callers only need RGB
/// values scaled to `[0, 1]`. Classification models carry no geometry: nothing is padded, so
/// there is no source-image mapping.
pub(crate) fn classify_transform(source: &DynamicImage, size: usize) -> DynamicImage {
    let source_width = source.width();
    let source_height = source.height();
    let short = source_width.min(source_height);
    let long = source_width.max(source_height);
    let new_long = (size as f64 * long as f64 / short as f64) as u32;
    let (resized_width, resized_height) = if source_width <= source_height {
        (size as u32, new_long)
    } else {
        (new_long, size as u32)
    };
    let mut resized = resize_fir(
        source,
        resized_width,
        resized_height,
        fir::ResizeAlg::Convolution(fir::FilterType::Bilinear),
    );
    let crop = size.min(resized.width().min(resized.height()) as usize) as u32;
    let left = (resized.width() - crop) / 2;
    let top = (resized.height() - crop) / 2;
    DynamicImage::ImageRgb8(imageops::crop(&mut resized, left, top, crop, crop).to_image())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

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
        let source = image::open("docs/dog_bike_man.jpg").unwrap();
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

    /// Ignored microbenchmark for the letterbox preprocessing cost (release mode):
    /// `cargo test --locked --release measures_letterbox_resize_cost -- --ignored --nocapture`
    ///
    /// Times JPEG decoding, both letterbox constructors, and the raw scalers on the reference
    /// image plus two generated synthetic images under `target/`. It keeps the scaler candidates
    /// measurable: the Ultralytics transform stays on the hand-rolled cv2-linear sampler while the
    /// YOLOX transform uses `fast_image_resize`, so the bench times the production constructors,
    /// the rejected fir candidates, and the previous imageops Triangle scaler, diffs their
    /// canvases, and writes the reference-image canvases to `target/` for cv2.resize comparison.
    #[test]
    #[ignore]
    fn measures_letterbox_resize_cost() {
        write_synthetic_bench_image("target/letterbox-bench-1920x1080.jpg", 1920, 1080);
        write_synthetic_bench_image("target/letterbox-bench-3840x2160.jpg", 3840, 2160);
        let sources = [
            "docs/dog_bike_man.jpg",
            "target/letterbox-bench-1920x1080.jpg",
            "target/letterbox-bench-3840x2160.jpg",
        ];

        for path in sources {
            let decode_ms = median_ms(|| image::open(path).unwrap(), 3, 10);
            let source = image::open(path).unwrap();
            let (scale, resized_width, resized_height) = ultralytics_geometry(&source, 640);
            eprintln!(
                "{path}: {}x{} -> ultralytics {resized_width}x{resized_height} (scale {scale:.4})",
                source.width(),
                source.height()
            );
            eprintln!("  decode: {decode_ms:.3} ms");

            let ultralytics_ms =
                median_ms(|| LetterboxedImage::ultralytics(&source, 640, 32), 3, 10);
            let ultralytics_fir_ms = median_ms(|| fir_ultralytics_canvas(&source, 640, 32), 3, 10);
            let yolox_ms = median_ms(|| LetterboxedImage::yolox(&source, 640), 3, 10);
            let yolox_triangle_ms = median_ms(|| triangle_yolox_canvas(&source, 640), 3, 10);
            let ultralytics_fir_u16_ms =
                median_ms(|| fir_u16_ultralytics_canvas(&source, 640, 32), 3, 10);

            let opencv_linear_ms = median_ms(
                || resize_opencv_linear(&source, resized_width, resized_height),
                3,
                10,
            );
            let fir_interpolation_ms = median_ms(
                || {
                    resize_fir(
                        &source,
                        resized_width,
                        resized_height,
                        fir::ResizeAlg::Interpolation(fir::FilterType::Bilinear),
                    )
                },
                3,
                10,
            );
            let triangle_dimensions = yolox_scaled_dimensions(&source, 640);
            let triangle_ms = median_ms(
                || {
                    source.resize_exact(
                        triangle_dimensions.0,
                        triangle_dimensions.1,
                        imageops::FilterType::Triangle,
                    )
                },
                3,
                10,
            );
            let fir_convolution_ms = median_ms(
                || {
                    let (width, height) = yolox_scaled_dimensions(&source, 640);
                    resize_fir(
                        &source,
                        width,
                        height,
                        fir::ResizeAlg::Convolution(fir::FilterType::Bilinear),
                    )
                },
                3,
                10,
            );

            eprintln!(
                "  ultralytics letterbox: current {ultralytics_ms:.3} ms, fir {ultralytics_fir_ms:.3} ms, fir-u16 {ultralytics_fir_u16_ms:.3} ms"
            );
            eprintln!(
                "  yolox letterbox:       fir {yolox_ms:.3} ms, previous-triangle {yolox_triangle_ms:.3} ms"
            );
            eprintln!(
                "  resize only:           opencv-linear {opencv_linear_ms:.3} ms, fir-interp {fir_interpolation_ms:.3} ms, triangle {triangle_ms:.3} ms, fir-conv {fir_convolution_ms:.3} ms"
            );

            let current = LetterboxedImage::ultralytics(&source, 640, 32)
                .image()
                .to_rgb8();
            let fir_canvas = fir_ultralytics_canvas(&source, 640, 32);
            let (max, mean, fraction) = diff_stats(&fir_canvas, &current);
            eprintln!(
                "  fir vs current ultralytics canvas: max={max} mean={mean:.4} differing={fraction:.5}"
            );
            let current_yolox = LetterboxedImage::yolox(&source, 640).image().to_rgb8();
            let triangle_canvas_yolox = triangle_yolox_canvas(&source, 640);
            let (max, mean, fraction) = diff_stats(&current_yolox, &triangle_canvas_yolox);
            eprintln!(
                "  fir vs previous-triangle yolox canvas: max={max} mean={mean:.4} differing={fraction:.5}"
            );

            if path == "docs/dog_bike_man.jpg" {
                source
                    .to_rgb8()
                    .save("target/letterbox-bench-source.png")
                    .unwrap();
                current
                    .save("target/letterbox-bench-ultralytics-current.png")
                    .unwrap();
                fir_canvas
                    .save("target/letterbox-bench-ultralytics-fir.png")
                    .unwrap();
                current_yolox
                    .save("target/letterbox-bench-yolox-current.png")
                    .unwrap();
                triangle_canvas_yolox
                    .save("target/letterbox-bench-yolox-fir.png")
                    .unwrap();

                let fir_u16_canvas = fir_u16_ultralytics_canvas(&source, 640, 32);
                let (max, mean, fraction) = diff_stats(&fir_u16_canvas, &current);
                eprintln!(
                    "  fir-u16 vs current ultralytics canvas: max={max} mean={mean:.4} differing={fraction:.5}"
                );

                let fixture_path = "target/yolov3-tinyu-preprocessed-reference.png";
                if std::path::Path::new(fixture_path).exists() {
                    let fixture = image::open(fixture_path).unwrap().to_rgb8();
                    for (name, canvas) in [
                        ("current", &current),
                        ("fir-u8", &fir_canvas),
                        ("fir-u16", &fir_u16_canvas),
                    ] {
                        let (max, mean, fraction) = diff_stats(canvas, &fixture);
                        eprintln!(
                            "  fixture parity {name}: max={max} mean={mean:.4} differing={fraction:.5}"
                        );
                    }
                }
            }
        }
    }

    /// Evaluation candidate: resize through fir's U16x3 pixel type for higher fixed-point
    /// precision, widening u8 components by *257 and narrowing with round(value / 257).
    fn resize_fir_u16(
        source: &DynamicImage,
        width: u32,
        height: u32,
        algorithm: fir::ResizeAlg,
    ) -> RgbImage {
        let source = source.to_rgb8();
        let widened: Vec<u16> = source
            .as_raw()
            .iter()
            .map(|&byte| u16::from(byte) * 257)
            .collect();
        // A native-endian view of the u16 buffer is what fir's byte-slice constructors expect.
        let widened_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(widened.as_ptr().cast::<u8>(), widened.len() * 2) };
        let source_image = fir::images::ImageRef::new(
            source.width(),
            source.height(),
            widened_bytes,
            fir::PixelType::U16x3,
        )
        .expect("widened rgb8 buffer matches U16x3");
        let mut destination = fir::images::Image::new(width, height, fir::PixelType::U16x3);
        let mut resizer = fir::Resizer::new();
        let options = fir::ResizeOptions::new()
            .resize_alg(algorithm)
            .use_alpha(false);
        resizer
            .resize(&source_image, &mut destination, &options)
            .expect("fir u16 resize into a matching buffer");
        let wide_bytes = destination.into_vec();
        let narrowed: Vec<u8> = wide_bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| {
                let value = u16::from_ne_bytes([pair[0], pair[1]]);
                ((((value as u32 * 2 + 257) / 514) as usize).min(255)) as u8
            })
            .collect();
        ImageBuffer::from_raw(width, height, narrowed).expect("narrowed fir destination buffer")
    }

    fn write_synthetic_bench_image(path: &str, width: u32, height: u32) {
        if std::path::Path::new(path).exists() {
            return;
        }
        let image = ImageBuffer::from_fn(width, height, |x, y| {
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;
            let wave = ((fx * 60.0).sin() * (fy * 45.0).cos() * 0.5 + 0.5) * 255.0;
            let checker = if (x / 16 + y / 16) % 2 == 0 {
                45.0
            } else {
                210.0
            };
            Rgb([
                (fx * 255.0) as u8,
                wave as u8,
                ((wave * 0.5 + checker * 0.5).min(255.0)) as u8,
            ])
        });
        image.save(path).unwrap();
    }

    fn median_ms<T>(mut operation: impl FnMut() -> T, warmups: usize, iterations: usize) -> f64 {
        for _ in 0..warmups {
            operation();
        }
        let mut samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let started = Instant::now();
            operation();
            samples.push(started.elapsed().as_secs_f64() * 1e3);
        }
        samples.sort_by(|a, b| a.total_cmp(b));
        samples[iterations / 2]
    }

    fn diff_stats(actual: &RgbImage, expected: &RgbImage) -> (u8, f64, f64) {
        assert_eq!(actual.dimensions(), expected.dimensions());
        let (total, max, differing) = actual.as_raw().iter().zip(expected.as_raw()).fold(
            (0_u64, 0_u8, 0_usize),
            |(sum, max, count), (a, b)| {
                let error = a.abs_diff(*b);
                (
                    sum + u64::from(error),
                    max.max(error),
                    count + usize::from(error != 0),
                )
            },
        );
        let values = actual.as_raw().len() as f64;
        (max, total as f64 / values, differing as f64 / values)
    }

    fn ultralytics_geometry(source: &DynamicImage, size: usize) -> (f32, u32, u32) {
        let scale = (size as f32 / source.width() as f32).min(size as f32 / source.height() as f32);
        (
            scale,
            (source.width() as f32 * scale).round() as u32,
            (source.height() as f32 * scale).round() as u32,
        )
    }

    fn yolox_scaled_dimensions(source: &DynamicImage, size: usize) -> (u32, u32) {
        let scale = (size as f32 / source.width() as f32).min(size as f32 / source.height() as f32);
        (
            (source.width() as f32 * scale) as u32,
            (source.height() as f32 * scale) as u32,
        )
    }

    /// The ultralytics letterbox implemented on top of `fast_image_resize` with the fixed-kernel
    /// (OpenCV-like) bilinear algorithm; identical stride geometry and 114 fill.
    fn fir_ultralytics_canvas(source: &DynamicImage, size: usize, stride: usize) -> RgbImage {
        let (_, resized_width, resized_height) = ultralytics_geometry(source, size);
        let resized = resize_fir(
            source,
            resized_width,
            resized_height,
            fir::ResizeAlg::Interpolation(fir::FilterType::Bilinear),
        );
        let total_pad_x = (size as u32 - resized_width) % stride as u32;
        let total_pad_y = (size as u32 - resized_height) % stride as u32;
        let mut canvas = ImageBuffer::from_pixel(
            resized_width + total_pad_x,
            resized_height + total_pad_y,
            Rgb([114, 114, 114]),
        );
        imageops::replace(
            &mut canvas,
            &resized,
            (total_pad_x / 2).into(),
            (total_pad_y / 2).into(),
        );
        canvas
    }

    /// The ultralytics letterbox on top of fir's higher-precision U16x3 pipeline; identical
    /// stride geometry and 114 fill.
    fn fir_u16_ultralytics_canvas(source: &DynamicImage, size: usize, stride: usize) -> RgbImage {
        let (_, resized_width, resized_height) = ultralytics_geometry(source, size);
        let resized = resize_fir_u16(
            source,
            resized_width,
            resized_height,
            fir::ResizeAlg::Interpolation(fir::FilterType::Bilinear),
        );
        let total_pad_x = (size as u32 - resized_width) % stride as u32;
        let total_pad_y = (size as u32 - resized_height) % stride as u32;
        let mut canvas = ImageBuffer::from_pixel(
            resized_width + total_pad_x,
            resized_height + total_pad_y,
            Rgb([114, 114, 114]),
        );
        imageops::replace(
            &mut canvas,
            &resized,
            (total_pad_x / 2).into(),
            (total_pad_y / 2).into(),
        );
        canvas
    }

    /// The previous yolox letterbox (the `image` crate's anti-aliased Triangle resampling), kept
    /// as the A/B reference for the adopted `fast_image_resize` implementation.
    fn triangle_yolox_canvas(source: &DynamicImage, size: usize) -> RgbImage {
        let (resized_width, resized_height) = yolox_scaled_dimensions(source, size);
        let resized = source.resize_exact(
            resized_width,
            resized_height,
            imageops::FilterType::Triangle,
        );
        let mut canvas = ImageBuffer::from_pixel(size as u32, size as u32, Rgb([114, 114, 114]));
        imageops::replace(&mut canvas, &resized.to_rgb8(), 0, 0);
        canvas
    }
}
