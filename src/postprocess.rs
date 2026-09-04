//! Shared detection post-processing primitives.

use alloc::vec::Vec;
use burn::tensor::{ElementConversion, Tensor, backend::Backend};

pub struct BoundingBox {
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32,
    pub confidence: f32,
}

/// Non-maximum suppression (NMS) filters overlapping bounding boxes that have an intersection-over-
/// union (IoU) greater or equal than the specified `iou_threshold` with previously selected boxes.
///
/// Boxes are filtered based on `score_threshold` and ranked based on their score. As such, lower
/// scoring boxes are removed when overlapping with another (higher scoring) box.
///
/// # Arguments
///
/// * `boxes`: Bounding box coordinates. Shape: `[batch_size, num_boxes, 4]`.
/// * `scores` - Classification scores for each box. Shape: `[batch_size, num_boxes, num_classes]`.
/// * `iou_threshold` - Scalar threshold for IoU.
/// * `score_threshold` - Scalar threshold for scores.
///
/// # Returns
///
/// Vector of bounding boxes grouped by class for each batch. The boxes are sorted in decreasing
/// order of scores for each class.
pub fn nms<B: Backend>(
    boxes: Tensor<B, 3>,
    scores: Tensor<B, 3>,
    iou_threshold: f32,
    score_threshold: f32,
) -> Vec<Vec<Vec<BoundingBox>>> {
    let [batch_size, num_boxes, num_classes] = scores.dims();

    // Bounding boxes grouped by batch and by (maximum) class index
    let mut bboxes = boxes
        .iter_dim(0)
        .zip(scores.iter_dim(0))
        // Per-batch
        .map(|(candidate_boxes, candidate_scores)| {
            // Keep max scoring boxes only ([num_boxes, 1], [num_boxes, 1])
            let (cls_score, cls_idx) = candidate_scores.squeeze_dim::<2>(0).max_dim_with_indices(1);
            let cls_score: Vec<_> = cls_score
                .into_data()
                .iter::<B::FloatElem>()
                .map(|v| v.elem::<f32>())
                .collect();
            let cls_idx: Vec<_> = cls_idx
                .into_data()
                .iter::<B::IntElem>()
                .map(|v| v.elem::<i64>() as usize)
                .collect();

            // [num_boxes, 4]
            let candidate_boxes: Vec<_> = candidate_boxes
                .into_data()
                .iter::<B::FloatElem>()
                .map(|v| v.elem::<f32>())
                .collect();

            // Per-class filtering based on score: single pass partitioned by argmax class.
            // (Previously this scanned all boxes once per class plus a wasted ascending sort
            // per class that `non_maximum_suppression` immediately re-sorted descending.)
            let mut per_class: Vec<Vec<BoundingBox>> =
                (0..num_classes).map(|_| Vec::new()).collect();
            // Reserve roughly even distribution to avoid regrowth; over-reserve slightly when
            // num_boxes is small.
            let per_class_cap = (num_boxes / num_classes.max(1)).max(4);
            for vec in per_class.iter_mut() {
                vec.reserve(per_class_cap);
            }
            for box_idx in 0..num_boxes {
                let box_cls_score = cls_score[box_idx];
                if box_cls_score < score_threshold {
                    continue;
                }
                let box_cls_idx = cls_idx[box_idx];
                if box_cls_idx >= num_classes {
                    continue;
                }
                let bbox = &candidate_boxes[box_idx * 4..box_idx * 4 + 4];
                per_class[box_cls_idx].push(BoundingBox {
                    xmin: bbox[0] - bbox[2] / 2.,
                    ymin: bbox[1] - bbox[3] / 2.,
                    xmax: bbox[0] + bbox[2] / 2.,
                    ymax: bbox[1] + bbox[3] / 2.,
                    confidence: box_cls_score,
                });
            }
            per_class
        })
        .collect::<Vec<_>>();

    for batch_bboxes in bboxes.iter_mut().take(batch_size) {
        non_maximum_suppression(batch_bboxes, iou_threshold);
    }

    bboxes
}

/// Intersection over union of two bounding boxes.
///
/// Retained as the shared definition (unit tests, segmentation path reference); the NMS hot
/// loop inlines the same math with precomputed areas and an AABB early-reject.
#[allow(dead_code)]
pub fn iou(b1: &BoundingBox, b2: &BoundingBox) -> f32 {
    let b1_area = (b1.xmax - b1.xmin).max(0.) * (b1.ymax - b1.ymin).max(0.);
    let b2_area = (b2.xmax - b2.xmin).max(0.) * (b2.ymax - b2.ymin).max(0.);
    let i_xmin = b1.xmin.max(b2.xmin);
    let i_xmax = b1.xmax.min(b2.xmax);
    let i_ymin = b1.ymin.max(b2.ymin);
    let i_ymax = b1.ymax.min(b2.ymax);
    let i_area = (i_xmax - i_xmin).max(0.) * (i_ymax - i_ymin).max(0.);
    i_area / (b1_area + b2_area - i_area)
}

/// Perform non-maximum suppression over boxes of the same class.
pub fn non_maximum_suppression(bboxes: &mut [Vec<BoundingBox>], threshold: f32) {
    for bboxes_for_class in bboxes.iter_mut() {
        if bboxes_for_class.len() < 2 {
            continue;
        }
        bboxes_for_class.sort_unstable_by(|a, b| b.confidence.total_cmp(&a.confidence));
        let len = bboxes_for_class.len();
        let mut areas = Vec::with_capacity(len);
        for bbox in bboxes_for_class.iter() {
            areas.push((bbox.xmax - bbox.xmin).max(0.) * (bbox.ymax - bbox.ymin).max(0.));
        }
        let mut current_index = 0;
        for index in 0..len {
            let (xmin, ymin, xmax, ymax) = {
                let current = &bboxes_for_class[index];
                (current.xmin, current.ymin, current.xmax, current.ymax)
            };
            let mut drop = false;
            for prev_index in 0..current_index {
                let kept = &bboxes_for_class[prev_index];
                // AABB early-reject before the full IoU.
                if kept.xmax <= xmin || kept.xmin >= xmax || kept.ymax <= ymin || kept.ymin >= ymax
                {
                    continue;
                }
                let i_xmin = kept.xmin.max(xmin);
                let i_xmax = kept.xmax.min(xmax);
                let i_ymin = kept.ymin.max(ymin);
                let i_ymax = kept.ymax.min(ymax);
                let i_area = (i_xmax - i_xmin).max(0.) * (i_ymax - i_ymin).max(0.);
                if i_area == 0.0 {
                    continue;
                }
                let union = areas[prev_index] + areas[index] - i_area;
                if union > 0.0 && i_area / union > threshold {
                    drop = true;
                    break;
                }
            }
            if !drop {
                if current_index != index {
                    bboxes_for_class.swap(current_index, index);
                    areas.swap(current_index, index);
                }
                current_index += 1;
            }
        }
        bboxes_for_class.truncate(current_index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bbox(xmin: f32, ymin: f32, xmax: f32, ymax: f32) -> BoundingBox {
        BoundingBox {
            xmin,
            ymin,
            xmax,
            ymax,
            confidence: 1.0,
        }
    }

    #[test]
    fn iou_uses_continuous_xyxy_box_edges() {
        let first = bbox(0.0, 0.0, 10.0, 10.0);
        let second = bbox(5.0, 0.0, 15.0, 10.0);

        assert!((iou(&first, &second) - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(iou(&first, &bbox(10.0, 0.0, 20.0, 10.0)), 0.0);
    }
}
