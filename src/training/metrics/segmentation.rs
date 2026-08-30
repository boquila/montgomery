use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct MetricMaskPrediction {
    pub image_id: String,
    pub class_id: usize,
    pub confidence: f32,
    pub mask: Vec<bool>,
}

#[derive(Debug, Clone)]
pub struct MetricMaskTarget {
    pub image_id: String,
    pub class_id: usize,
    pub mask: Vec<bool>,
    pub crowd: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentationMetrics {
    pub map_50_95: f32,
    pub map_50: f32,
    pub per_class_support: BTreeMap<usize, usize>,
}

pub fn evaluate(
    predictions: &[MetricMaskPrediction],
    targets: &[MetricMaskTarget],
) -> SegmentationMetrics {
    let classes: BTreeSet<_> = targets
        .iter()
        .filter(|target| !target.crowd)
        .map(|target| target.class_id)
        .collect();
    let support = classes
        .iter()
        .map(|class| {
            (
                *class,
                targets
                    .iter()
                    .filter(|target| !target.crowd && target.class_id == *class)
                    .count(),
            )
        })
        .collect();
    if classes.is_empty() {
        return SegmentationMetrics {
            map_50_95: 0.0,
            map_50: 0.0,
            per_class_support: support,
        };
    }
    let thresholds = (0..10)
        .map(|index| 0.5 + index as f32 * 0.05)
        .collect::<Vec<_>>();
    let mut sums = vec![0.0; thresholds.len()];
    for class in &classes {
        for (index, threshold) in thresholds.iter().enumerate() {
            sums[index] += average_precision(*class, *threshold, predictions, targets);
        }
    }
    for value in &mut sums {
        *value /= classes.len() as f32;
    }
    SegmentationMetrics {
        map_50: sums[0],
        map_50_95: sums.iter().sum::<f32>() / sums.len() as f32,
        per_class_support: support,
    }
}

fn average_precision(
    class: usize,
    threshold: f32,
    predictions: &[MetricMaskPrediction],
    targets: &[MetricMaskTarget],
) -> f32 {
    let truth = targets
        .iter()
        .filter(|target| target.class_id == class)
        .collect::<Vec<_>>();
    let positive_count = truth.iter().filter(|target| !target.crowd).count();
    if positive_count == 0 {
        return 0.0;
    }
    let mut candidates = predictions
        .iter()
        .filter(|prediction| prediction.class_id == class)
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| {
        b.confidence
            .total_cmp(&a.confidence)
            .then_with(|| a.image_id.cmp(&b.image_id))
    });
    let mut matched = BTreeSet::<(String, usize)>::new();
    let mut tp = Vec::with_capacity(candidates.len());
    let mut fp = Vec::with_capacity(candidates.len());
    for prediction in candidates {
        let mut possible = truth
            .iter()
            .enumerate()
            .filter(|(_, target)| target.image_id == prediction.image_id)
            .map(|(index, target)| (index, *target, mask_iou(&prediction.mask, &target.mask)))
            .filter(|(_, _, overlap)| *overlap >= threshold)
            .collect::<Vec<_>>();
        possible.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
        if let Some((index, target, _)) = possible.iter().copied().find(|(index, target, _)| {
            !target.crowd && !matched.contains(&(target.image_id.clone(), *index))
        }) {
            matched.insert((target.image_id.clone(), index));
            tp.push(1.0_f32);
            fp.push(0.0_f32);
        } else if possible.iter().any(|(_, target, _)| target.crowd) {
            continue;
        } else {
            tp.push(0.0);
            fp.push(1.0);
        }
    }
    for index in 1..tp.len() {
        tp[index] += tp[index - 1];
        fp[index] += fp[index - 1];
    }
    let (precision, recall): (Vec<_>, Vec<_>) = tp
        .into_iter()
        .zip(fp)
        .map(|(tp, fp)| (tp / (tp + fp).max(1e-9), tp / positive_count as f32))
        .unzip();
    (0..=100)
        .map(|point| {
            let threshold = point as f32 / 100.0;
            recall
                .iter()
                .zip(&precision)
                .filter(|(recall, _)| **recall >= threshold)
                .map(|(_, precision)| *precision)
                .fold(0.0, f32::max)
        })
        .sum::<f32>()
        / 101.0
}

fn mask_iou(left: &[bool], right: &[bool]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    let mut intersection = 0_usize;
    let mut union = 0_usize;
    for (left, right) in left.iter().zip(right) {
        intersection += usize::from(*left && *right);
        union += usize::from(*left || *right);
    }
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_mask_prediction_has_unit_map() {
        let mask = vec![false, true, true, false];
        let metrics = evaluate(
            &[MetricMaskPrediction {
                image_id: "a".into(),
                class_id: 1,
                confidence: 0.9,
                mask: mask.clone(),
            }],
            &[MetricMaskTarget {
                image_id: "a".into(),
                class_id: 1,
                mask,
                crowd: false,
            }],
        );
        assert!((metrics.map_50_95 - 1.0).abs() < 1e-6);
    }
}
