use std::f32::consts::PI;

use super::boxes::BoxXyxy;

pub fn iou(a: BoxXyxy, b: BoxXyxy) -> f32 {
    let intersection_width = (a.xmax.min(b.xmax) - a.xmin.max(b.xmin)).max(0.0);
    let intersection_height = (a.ymax.min(b.ymax) - a.ymin.max(b.ymin)).max(0.0);
    let intersection = intersection_width * intersection_height;
    intersection / (a.area() + b.area() - intersection + 1e-7)
}

/// Complete IoU in the form used by Ultralytics' detection loss.
pub fn ciou(a: BoxXyxy, b: BoxXyxy) -> f32 {
    let overlap = iou(a, b);
    let [ax, ay] = a.center();
    let [bx, by] = b.center();
    let center_distance = (bx - ax).powi(2) + (by - ay).powi(2);
    let enclosing_width = a.xmax.max(b.xmax) - a.xmin.min(b.xmin);
    let enclosing_height = a.ymax.max(b.ymax) - a.ymin.min(b.ymin);
    let diagonal = enclosing_width.powi(2) + enclosing_height.powi(2) + 1e-7;
    let v = 4.0 / PI.powi(2) * (b.width().atan2(b.height()) - a.width().atan2(a.height())).powi(2);
    let alpha = v / (1.0 - overlap + v + 1e-7);
    overlap - center_distance / diagonal - alpha * v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_primitives_cover_disjoint_and_identical_boxes() {
        let a = BoxXyxy::new([0.0, 0.0, 10.0, 10.0]).unwrap();
        let b = BoxXyxy::new([20.0, 20.0, 30.0, 30.0]).unwrap();
        assert!((iou(a, a) - 1.0).abs() < 1e-6);
        assert_eq!(iou(a, b), 0.0);
        assert!(ciou(a, b) < 0.0);
    }
}
