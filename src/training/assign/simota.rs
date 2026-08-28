use crate::training::{
    geometry::{BoxXywh, BoxXyxy, iou::iou},
    loss::common::{bce_with_logits, sigmoid},
};

#[derive(Debug, Clone)]
pub struct GroundTruth {
    pub class_id: usize,
    pub bbox: BoxXyxy,
}

#[derive(Debug, Clone)]
pub struct AnchorPrediction {
    pub box_xywh: BoxXywh,
    pub raw_box: [f32; 4],
    pub objectness_logit: f32,
    pub class_logits: Vec<f32>,
    pub center_xy: [f32; 2],
    pub stride: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Match {
    pub anchor_index: usize,
    pub gt_index: usize,
    pub overlap: f32,
}

/// Deterministic host oracle for official YOLOX SimOTA assignment.
///
/// Assignment is intentionally detached from autodiff; tensor loss code gathers the selected
/// differentiable predictions by the returned anchor indices.
pub fn assign(ground_truth: &[GroundTruth], predictions: &[AnchorPrediction]) -> Vec<Match> {
    if ground_truth.is_empty() || predictions.is_empty() {
        return Vec::new();
    }
    let candidate_indices: Vec<usize> = predictions
        .iter()
        .enumerate()
        .filter_map(|(anchor, prediction)| {
            ground_truth
                .iter()
                .any(|gt| in_box(gt.bbox, prediction.center_xy) || in_center(gt.bbox, prediction))
                .then_some(anchor)
        })
        .collect();
    if candidate_indices.is_empty() {
        return Vec::new();
    }

    let mut selected = Vec::<(usize, usize, f32, f32)>::new();
    for (gt_index, gt) in ground_truth.iter().enumerate() {
        let mut candidates: Vec<(usize, f32, f32)> = candidate_indices
            .iter()
            .map(|anchor_index| {
                let prediction = &predictions[*anchor_index];
                let overlap = iou(gt.bbox, prediction.box_xywh.to_xyxy());
                let class_cost = prediction
                    .class_logits
                    .iter()
                    .enumerate()
                    .map(|(class, class_logit)| {
                        let probability =
                            (sigmoid(*class_logit) * sigmoid(prediction.objectness_logit)).sqrt();
                        // BCE is specified on the probability's inverse-sigmoid value.
                        let probability = probability.clamp(1e-7, 1.0 - 1e-7);
                        let logit = (probability / (1.0 - probability)).ln();
                        bce_with_logits(logit, f32::from(class == gt.class_id))
                    })
                    .sum::<f32>();
                let valid = in_box(gt.bbox, prediction.center_xy) && in_center(gt.bbox, prediction);
                let cost =
                    class_cost - 3.0 * (overlap + 1e-8).ln() + if valid { 0.0 } else { 100_000.0 };
                (*anchor_index, overlap, cost)
            })
            .collect();
        let mut overlaps: Vec<f32> = candidates.iter().map(|item| item.1).collect();
        overlaps.sort_by(|a, b| b.total_cmp(a));
        let dynamic_k = overlaps.iter().take(10).sum::<f32>().floor().max(1.0) as usize;
        candidates.sort_by(|a, b| a.2.total_cmp(&b.2).then_with(|| a.0.cmp(&b.0)));
        selected.extend(
            candidates
                .into_iter()
                .take(dynamic_k)
                .map(|(anchor, overlap, cost)| (anchor, gt_index, overlap, cost)),
        );
    }

    // One anchor may be proposed by multiple GTs; official SimOTA retains the lowest-cost GT.
    selected.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.3.total_cmp(&b.3))
            .then_with(|| a.1.cmp(&b.1))
    });
    let mut result = Vec::new();
    for (anchor, gt, overlap, _) in selected {
        if result
            .last()
            .is_some_and(|item: &Match| item.anchor_index == anchor)
        {
            continue;
        }
        result.push(Match {
            anchor_index: anchor,
            gt_index: gt,
            overlap,
        });
    }
    result
}

fn in_box(bbox: BoxXyxy, point: [f32; 2]) -> bool {
    point[0] > bbox.xmin && point[0] < bbox.xmax && point[1] > bbox.ymin && point[1] < bbox.ymax
}

fn in_center(gt: BoxXyxy, prediction: &AnchorPrediction) -> bool {
    let [cx, cy] = gt.center();
    let radius = 2.5 * prediction.stride;
    prediction.center_xy[0] > cx - radius
        && prediction.center_xy[0] < cx + radius
        && prediction.center_xy[1] > cy - radius
        && prediction.center_xy[1] < cy + radius
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_targets_have_no_positive_anchors() {
        assert!(assign(&[], &[]).is_empty());
    }

    #[test]
    fn conflict_has_one_deterministic_owner() {
        let prediction = AnchorPrediction {
            box_xywh: BoxXywh {
                cx: 10.0,
                cy: 10.0,
                width: 10.0,
                height: 10.0,
            },
            raw_box: [0.0; 4],
            objectness_logit: 2.0,
            class_logits: vec![2.0, 2.0],
            center_xy: [10.0, 10.0],
            stride: 8.0,
        };
        let gt = vec![
            GroundTruth {
                class_id: 0,
                bbox: BoxXyxy::new([5.0, 5.0, 15.0, 15.0]).unwrap(),
            },
            GroundTruth {
                class_id: 1,
                bbox: BoxXyxy::new([6.0, 6.0, 16.0, 16.0]).unwrap(),
            },
        ];
        let matches = assign(&gt, &[prediction]);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].anchor_index, 0);
        assert_eq!(matches[0].gt_index, 0);
    }
}
