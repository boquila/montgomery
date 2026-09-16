use crate::training::geometry::{AnchorPoint, BoxXyxy, iou::iou};

#[derive(Debug, Clone)]
pub struct TalGroundTruth {
    pub class_id: usize,
    pub bbox: BoxXyxy,
}

#[derive(Debug, Clone, Copy)]
pub struct TalPredictions<'a> {
    boxes: &'a [f32],
    class_scores: &'a [f32],
    class_count: usize,
}

impl<'a> TalPredictions<'a> {
    pub fn new(
        boxes: &'a [f32],
        class_scores: &'a [f32],
        class_count: usize,
    ) -> Result<Self, &'static str> {
        if class_count == 0 {
            return Err("prediction class count must be positive");
        }
        if !boxes.len().is_multiple_of(4) {
            return Err("prediction boxes are not packed XYXY values");
        }
        if class_scores.len() != boxes.len() / 4 * class_count {
            return Err("prediction score and box counts differ");
        }
        if boxes.iter().any(|value| !value.is_finite()) {
            return Err("decoded box edges are not finite");
        }
        Ok(Self {
            boxes,
            class_scores,
            class_count,
        })
    }

    fn len(self) -> usize {
        self.boxes.len() / 4
    }

    fn bbox(self, index: usize) -> BoxXyxy {
        let offset = index * 4;
        BoxXyxy {
            xmin: self.boxes[offset],
            ymin: self.boxes[offset + 1],
            xmax: self.boxes[offset + 2],
            ymax: self.boxes[offset + 3],
        }
    }

    fn class_score(self, index: usize, class_id: usize) -> f32 {
        self.class_scores[index * self.class_count + class_id]
    }
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
    predictions: TalPredictions<'_>,
    anchors: &[AnchorPoint],
    top_k: usize,
) -> Result<Vec<TalMatch>, &'static str> {
    if predictions.len() != anchors.len() {
        return Err("prediction and anchor counts differ");
    }
    if top_k == 0 {
        return Err("top_k must be positive");
    }
    let mut proposals =
        Vec::<(usize, usize, f32, f32)>::with_capacity(ground_truth.len().saturating_mul(top_k));
    let mut candidates = Vec::with_capacity(top_k);
    for (gt_index, gt) in ground_truth.iter().enumerate() {
        if gt.class_id >= predictions.class_count {
            return Err("ground-truth class outside prediction channels");
        }
        candidates.clear();
        for (anchor_index, anchor) in anchors.iter().enumerate() {
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
            let overlap = iou(gt.bbox, predictions.bbox(anchor_index));
            let alignment = predictions
                .class_score(anchor_index, gt.class_id)
                .max(0.0)
                .powf(0.5)
                * overlap.max(0.0).powf(6.0);
            let candidate = (anchor_index, alignment, overlap);
            let position = candidates
                .partition_point(|existing| candidate_order(existing, &candidate).is_lt());
            if position < top_k {
                candidates.insert(position, candidate);
                if candidates.len() > top_k {
                    candidates.pop();
                }
            }
        }
        proposals.extend(
            candidates
                .drain(..)
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

fn candidate_order(a: &(usize, f32, f32), b: &(usize, f32, f32)) -> std::cmp::Ordering {
    b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_selection_matches_full_sort() {
        let values = [0.2, 0.9, 0.5, 0.1, 0.9, 0.3, 0.7, 0.4, 0.8, 0.6, 0.05, 0.95];
        let top_k = 5;
        let mut bounded = Vec::with_capacity(top_k);
        let mut complete = Vec::new();
        for (anchor, alignment) in values.into_iter().enumerate() {
            let candidate = (anchor, alignment, alignment / 2.0);
            complete.push(candidate);
            let position =
                bounded.partition_point(|existing| candidate_order(existing, &candidate).is_lt());
            if position < top_k {
                bounded.insert(position, candidate);
                if bounded.len() > top_k {
                    bounded.pop();
                }
            }
        }
        complete.sort_by(candidate_order);
        assert_eq!(bounded, complete[..top_k]);
        assert_eq!(bounded[1].0, 1, "equal scores retain lower anchor first");
        assert_eq!(bounded[2].0, 4);
    }

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
        let boxes = [0.0, 0.0, 20.0, 20.0, 0.0, 0.0, 40.0, 40.0];
        let scores = [0.1, 0.8, 0.1, 0.1, 0.7, 0.2];
        let predictions = TalPredictions::new(&boxes, &scores, 3).unwrap();
        let truth = [TalGroundTruth {
            class_id: 1,
            bbox: BoxXyxy::new([0.0, 0.0, 30.0, 30.0]).unwrap(),
        }];
        assert_eq!(assign(&truth, predictions, &anchors, 10).unwrap().len(), 2);
    }
}
