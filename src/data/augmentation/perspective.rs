//! Modified Rust adaptation of Ultralytics `RandomPerspective`.

use super::{
    BBox, BoxFormat,
    resize::{mat_mul, warp},
    rng::AugRng,
    sample::{AugSample, AugmentationError},
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerspectiveParams {
    pub perspective: [f32; 2],
    pub angle: f32,
    pub scale: f32,
    pub shear: [f32; 2],
    pub translate: [f32; 2],
    pub output: [usize; 2],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerspectiveRanges {
    pub degrees: f32,
    pub scale: f32,
    pub shear: f32,
    pub perspective: f32,
    pub translate: f32,
}

impl PerspectiveParams {
    pub fn sample(
        rng: &mut AugRng,
        input: [usize; 2],
        output: [usize; 2],
        ranges: PerspectiveRanges,
    ) -> Self {
        let _ = input;
        Self {
            perspective: [
                rng.uniform(-ranges.perspective, ranges.perspective),
                rng.uniform(-ranges.perspective, ranges.perspective),
            ],
            angle: rng.uniform(-ranges.degrees, ranges.degrees),
            scale: rng.uniform(1.0 - ranges.scale, 1.0 + ranges.scale),
            shear: [
                rng.uniform(-ranges.shear, ranges.shear),
                rng.uniform(-ranges.shear, ranges.shear),
            ],
            translate: [
                rng.uniform(0.5 - ranges.translate, 0.5 + ranges.translate) * output[1] as f32,
                rng.uniform(0.5 - ranges.translate, 0.5 + ranges.translate) * output[0] as f32,
            ],
            output,
        }
    }
}

pub fn matrix(input: [usize; 2], p: PerspectiveParams) -> [[f32; 3]; 3] {
    let c = [
        [1., 0., -(input[1] as f32) / 2.],
        [0., 1., -(input[0] as f32) / 2.],
        [0., 0., 1.],
    ];
    let pe = [
        [1., 0., 0.],
        [0., 1., 0.],
        [p.perspective[0], p.perspective[1], 1.],
    ];
    let a = p.angle.to_radians();
    let r = [
        [p.scale * a.cos(), p.scale * a.sin(), 0.],
        [-p.scale * a.sin(), p.scale * a.cos(), 0.],
        [0., 0., 1.],
    ];
    let s = [
        [1., p.shear[0].to_radians().tan(), 0.],
        [p.shear[1].to_radians().tan(), 1., 0.],
        [0., 0., 1.],
    ];
    let t = [
        [1., 0., p.translate[0]],
        [0., 1., p.translate[1]],
        [0., 0., 1.],
    ];
    mat_mul(t, mat_mul(s, mat_mul(r, mat_mul(pe, c))))
}

fn point(m: [[f32; 3]; 3], p: [f32; 2], divide: bool) -> Option<[f32; 2]> {
    let x = m[0][0] * p[0] + m[0][1] * p[1] + m[0][2];
    let y = m[1][0] * p[0] + m[1][1] * p[1] + m[1][2];
    let z = m[2][0] * p[0] + m[2][1] * p[1] + m[2][2];
    let q = if divide { [x / z, y / z] } else { [x, y] };
    q.iter().all(|v| v.is_finite()).then_some(q)
}

pub fn apply(mut sample: AugSample, p: PerspectiveParams) -> Result<AugSample, AugmentationError> {
    let old_h = sample.image.height();
    let old_w = sample.image.width();
    sample.instances.denormalize(old_w as f32, old_h as f32);
    sample.instances.convert(BoxFormat::Xyxy);
    let m = matrix([old_h, old_w], p);
    let perspective = p.perspective != [0., 0.];
    let image = warp(&sample.image, m, p.output[1], p.output[0], perspective, 114)?;
    let original = sample.instances.boxes.clone();
    let mut new_boxes = Vec::with_capacity(original.len());
    let mut new_segments = sample.instances.segments.clone();
    if let Some(segments) = &mut new_segments {
        for (index, poly) in segments.iter_mut().enumerate() {
            for pt in poly.iter_mut() {
                *pt = point(m, *pt, true)
                    .ok_or_else(|| AugmentationError::new("non-finite transformed segment"))?;
            }
            let visible: Vec<_> = poly
                .iter()
                .copied()
                .filter(|q| {
                    q[0] >= 0.
                        && q[0] <= p.output[1] as f32
                        && q[1] >= 0.
                        && q[1] <= p.output[0] as f32
                })
                .collect();
            if visible.is_empty() {
                new_boxes.push(BBox([0.; 4]));
            } else {
                let mut b = [
                    f32::INFINITY,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    f32::NEG_INFINITY,
                ];
                for q in visible {
                    b[0] = b[0].min(q[0]);
                    b[1] = b[1].min(q[1]);
                    b[2] = b[2].max(q[0]);
                    b[3] = b[3].max(q[1]);
                }
                for q in poly.iter_mut() {
                    q[0] = q[0].clamp(b[0], b[2]);
                    q[1] = q[1].clamp(b[1], b[3]);
                }
                new_boxes.push(BBox(b));
            }
            let _ = index;
        }
    } else {
        for b in &original {
            let [x1, y1, x2, y2] = b.0;
            let corners = [[x1, y1], [x2, y2], [x1, y2], [x2, y1]];
            let q: Vec<_> = corners
                .into_iter()
                .map(|v| {
                    point(m, v, perspective)
                        .ok_or_else(|| AugmentationError::new("non-finite transformed box"))
                })
                .collect::<Result<_, _>>()?;
            new_boxes.push(BBox([
                q.iter().map(|v| v[0]).fold(f32::INFINITY, f32::min),
                q.iter().map(|v| v[1]).fold(f32::INFINITY, f32::min),
                q.iter().map(|v| v[0]).fold(f32::NEG_INFINITY, f32::max),
                q.iter().map(|v| v[1]).fold(f32::NEG_INFINITY, f32::max),
            ]));
        }
    }
    for bbox in &mut new_boxes {
        bbox.0[0] = bbox.0[0].clamp(0.0, p.output[1] as f32);
        bbox.0[2] = bbox.0[2].clamp(0.0, p.output[1] as f32);
        bbox.0[1] = bbox.0[1].clamp(0.0, p.output[0] as f32);
        bbox.0[3] = bbox.0[3].clamp(0.0, p.output[0] as f32);
    }
    let threshold = if new_segments.is_some() { 0.01 } else { 0.10 };
    let mut keep = Vec::new();
    for (i, (old, new)) in original.iter().zip(&new_boxes).enumerate() {
        let [ox1, oy1, ox2, oy2] = old.0;
        let [nx1, ny1, nx2, ny2] = new.0;
        let ow = (ox2 - ox1) * p.scale;
        let oh = (oy2 - oy1) * p.scale;
        let nw = nx2 - nx1;
        let nh = ny2 - ny1;
        let ratio = nw * nh / (ow * oh + 1e-16);
        let aspect = (nw / (nh + 1e-16)).max(nh / (nw + 1e-16));
        if nw > 2. && nh > 2. && ratio > threshold && aspect < 100. {
            keep.push(i);
        }
    }
    sample.instances.boxes = new_boxes;
    sample.instances.segments = new_segments;
    sample
        .instances
        .clip(p.output[1] as f32, p.output[0] as f32);
    sample.instances.select(&keep);
    sample.classes = keep.iter().map(|&i| sample.classes[i]).collect();
    sample.image = image;
    sample.geometry.current_shape = p.output;
    sample.geometry.reversible = false;
    sample.validate()?;
    Ok(sample)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matrix_order_identity_center_translation() {
        let p = PerspectiveParams {
            perspective: [0., 0.],
            angle: 0.,
            scale: 1.,
            shear: [0., 0.],
            translate: [5., 5.],
            output: [10, 10],
        };
        assert_eq!(
            matrix([10, 10], p),
            [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        );
    }
}
