use crate::training::geometry::{AnchorPoint, BoxXyxy, iou::iou};

#[derive(Debug, Clone)]
pub struct TalGroundTruth {
    pub class_id: usize,
    pub bbox: BoxXyxy,
}

#[derive(Debug, Clone)]
pub struct TalPrediction {
    pub bbox: BoxXyxy,
    pub class_scores: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TalMatch {
    pub anchor_index: usize,
    pub gt_index: usize,
    pub target_score: f32,
    pub overlap: f32,
}

/// Deterministic Task-Aligned Assigner for fixture generation and CPU diagnosis.
pub fn assign(
    ground_truth: &[TalGroundTruth],
    predictions: &[TalPrediction],
    anchors: &[AnchorPoint],
    top_k: usize,
) -> Result<Vec<TalMatch>, &'static str> {
    if predictions.len() != anchors.len() {
        return Err("prediction and anchor counts differ");
    }
    if top_k == 0 {
        return Err("top_k must be positive");
    }
    let mut proposals = Vec::<(usize, usize, f32, f32)>::new();
    for (gt_index, gt) in ground_truth.iter().enumerate() {
        let mut candidates = Vec::new();
        for (anchor_index, (prediction, anchor)) in predictions.iter().zip(anchors).enumerate() {
            if gt.class_id >= prediction.class_scores.len() {
                return Err("ground-truth class outside prediction channels");
            }
            let point = [
                anchor.grid_xy[0] * anchor.stride,
                anchor.grid_xy[1] * anchor.stride,
            ];
            if point[0] <= gt.bbox.xmin
                || point[0] >= gt.bbox.xmax
                || point[1] <= gt.bbox.ymin
                || point[1] >= gt.bbox.ymax
            {
                continue;
            }
            let overlap = iou(gt.bbox, prediction.bbox);
            let alignment = prediction.class_scores[gt.class_id].max(0.0).powf(0.5)
                * overlap.max(0.0).powf(6.0);
            candidates.push((anchor_index, alignment, overlap));
        }
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        proposals.extend(
            candidates
                .into_iter()
                .take(top_k)
                .filter(|candidate| candidate.1 > 0.0)
                .map(|(anchor, alignment, overlap)| (anchor, gt_index, alignment, overlap)),
        );
    }

    let mut max_alignment = vec![0.0_f32; ground_truth.len()];
    let mut max_overlap = vec![0.0_f32; ground_truth.len()];
    for (_, gt, alignment, overlap) in &proposals {
        max_alignment[*gt] = max_alignment[*gt].max(*alignment);
        max_overlap[*gt] = max_overlap[*gt].max(*overlap);
    }

    // Resolve conflicts by overlap, with GT index as the stable secondary key.
    proposals.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| b.3.total_cmp(&a.3))
            .then_with(|| a.1.cmp(&b.1))
    });
    let mut owned = Vec::new();
    for proposal in proposals {
        if owned
            .last()
            .is_some_and(|item: &(usize, usize, f32, f32)| item.0 == proposal.0)
        {
            continue;
        }
        owned.push(proposal);
    }

    let mut result = Vec::with_capacity(owned.len());
    for (anchor, gt, alignment, overlap) in owned {
        let target_score = if max_alignment[gt] > 0.0 {
            alignment * max_overlap[gt] / (max_alignment[gt] + 1e-9)
        } else {
            0.0
        };
        result.push(TalMatch {
            anchor_index: anchor,
            gt_index: gt,
            target_score,
            overlap,
        });
    }
    result.sort_by_key(|item| item.anchor_index);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_two_level_arbitrary_class_count() {
        let anchors = crate::training::geometry::make_anchors(&[
            crate::training::geometry::FeatureLevelLayout {
                height: 1,
                width: 1,
                stride: 16,
            },
            crate::training::geometry::FeatureLevelLayout {
                height: 1,
                width: 1,
                stride: 32,
            },
        ]);
        let predictions = vec![
            TalPrediction {
                bbox: BoxXyxy::new([0.0, 0.0, 20.0, 20.0]).unwrap(),
                class_scores: vec![0.1, 0.8, 0.1],
            },
            TalPrediction {
                bbox: BoxXyxy::new([0.0, 0.0, 40.0, 40.0]).unwrap(),
                class_scores: vec![0.1, 0.7, 0.2],
            },
        ];
        let truth = [TalGroundTruth {
            class_id: 1,
            bbox: BoxXyxy::new([0.0, 0.0, 30.0, 30.0]).unwrap(),
        }];
        assert_eq!(assign(&truth, &predictions, &anchors, 10).unwrap().len(), 2);
    }
}
