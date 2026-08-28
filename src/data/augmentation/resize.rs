//! Explicit pixel-center resize/warp kernels used instead of inference preprocessing.

use super::{
    Interpolation,
    sample::{AugmentationError, ByteImage},
};

pub fn resize(
    image: &ByteImage,
    width: usize,
    height: usize,
    method: Interpolation,
) -> Result<ByteImage, AugmentationError> {
    if width == 0 || height == 0 {
        return Err(AugmentationError::new("resize output must be positive"));
    }
    if width == image.width() && height == image.height() {
        return Ok(image.clone());
    }
    let mut out = ByteImage::filled(width, height, image.channels(), image.color(), 0);
    for y in 0..height {
        for x in 0..width {
            match method {
                Interpolation::Nearest => {
                    let sx = ((x as f32 + 0.5) * image.width() as f32 / width as f32 - 0.5)
                        .round()
                        .clamp(0.0, (image.width() - 1) as f32)
                        as usize;
                    let sy = ((y as f32 + 0.5) * image.height() as f32 / height as f32 - 0.5)
                        .round()
                        .clamp(0.0, (image.height() - 1) as f32)
                        as usize;
                    out.pixel_mut(x, y).copy_from_slice(image.pixel(sx, sy));
                }
                Interpolation::Bilinear => {
                    let fx = (x as f32 + 0.5) * image.width() as f32 / width as f32 - 0.5;
                    let fy = (y as f32 + 0.5) * image.height() as f32 / height as f32 - 0.5;
                    let x0 = fx.floor().clamp(0.0, (image.width() - 1) as f32) as usize;
                    let y0 = fy.floor().clamp(0.0, (image.height() - 1) as f32) as usize;
                    let x1 = (x0 + 1).min(image.width() - 1);
                    let y1 = (y0 + 1).min(image.height() - 1);
                    let wx = (fx - x0 as f32).clamp(0.0, 1.0);
                    let wy = (fy - y0 as f32).clamp(0.0, 1.0);
                    for c in 0..image.channels() {
                        let a = image.pixel(x0, y0)[c] as f32 * (1.0 - wx)
                            + image.pixel(x1, y0)[c] as f32 * wx;
                        let b = image.pixel(x0, y1)[c] as f32 * (1.0 - wx)
                            + image.pixel(x1, y1)[c] as f32 * wx;
                        out.pixel_mut(x, y)[c] =
                            (a * (1.0 - wy) + b * wy).round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
    }
    Ok(out)
}

pub fn warp(
    image: &ByteImage,
    m: [[f32; 3]; 3],
    width: usize,
    height: usize,
    perspective: bool,
    fill: u8,
) -> Result<ByteImage, AugmentationError> {
    let inv = invert(m).ok_or_else(|| AugmentationError::new("singular perspective matrix"))?;
    let mut out = ByteImage::filled(width, height, image.channels(), image.color(), fill);
    for y in 0..height {
        for x in 0..width {
            let px = x as f32;
            let py = y as f32;
            let z = if perspective {
                inv[2][0] * px + inv[2][1] * py + inv[2][2]
            } else {
                1.0
            };
            if z.abs() < 1e-8 {
                continue;
            }
            let sx = (inv[0][0] * px + inv[0][1] * py + inv[0][2]) / z;
            let sy = (inv[1][0] * px + inv[1][1] * py + inv[1][2]) / z;
            if sx < 0.0
                || sy < 0.0
                || sx > (image.width() - 1) as f32
                || sy > (image.height() - 1) as f32
            {
                continue;
            }
            let x0 = sx.floor() as usize;
            let y0 = sy.floor() as usize;
            let x1 = (x0 + 1).min(image.width() - 1);
            let y1 = (y0 + 1).min(image.height() - 1);
            let wx = sx - x0 as f32;
            let wy = sy - y0 as f32;
            for c in 0..image.channels() {
                let a =
                    image.pixel(x0, y0)[c] as f32 * (1.0 - wx) + image.pixel(x1, y0)[c] as f32 * wx;
                let b =
                    image.pixel(x0, y1)[c] as f32 * (1.0 - wx) + image.pixel(x1, y1)[c] as f32 * wx;
                out.pixel_mut(x, y)[c] = (a * (1.0 - wy) + b * wy).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(out)
}

fn invert(a: [[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let d = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if d.abs() < 1e-12 {
        return None;
    }
    let id = 1.0 / d;
    Some([
        [
            (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * id,
            (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * id,
            (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * id,
        ],
        [
            (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * id,
            (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * id,
            (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * id,
        ],
        [
            (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * id,
            (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * id,
            (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * id,
        ],
    ])
}

pub fn mat_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                o[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    o
}

#[cfg(test)]
mod tests {
    use super::super::ColorOrder;
    use super::*;
    #[test]
    fn identity_warp() {
        let i = ByteImage::new(2, 2, 1, ColorOrder::Gray, vec![1, 2, 3, 4]).unwrap();
        assert_eq!(
            warp(
                &i,
                [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                2,
                2,
                false,
                114
            )
            .unwrap(),
            i
        );
    }
    #[test]
    fn resize_keeps_shape() {
        let i = ByteImage::filled(3, 4, 3, ColorOrder::Bgr, 7);
        let o = resize(&i, 5, 6, Interpolation::Bilinear).unwrap();
        assert_eq!((o.width(), o.height(), o.data()[0]), (5, 6, 7));
    }
}
