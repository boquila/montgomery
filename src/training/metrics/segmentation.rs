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
    #[serde(default)]
    pub precision: f32,
    #[serde(default)]
    pub recall: f32,
    pub map_50_95: f32,
    pub map_50: f32,
    pub per_class_support: BTreeMap<usize, usize>,
}

#[derive(Debug, Clone)]
struct ScoredMaskMatch {
    class_id: usize,
    confidence: f32,
    order: usize,
    outcomes: [i8; 10],
}

/// Dataset accumulator that discards full-resolution masks after each image.
///
/// AP only needs confidence-ordered TP/FP decisions at each IoU threshold. Retaining those compact
/// decisions instead of every predicted mask keeps full-dataset validation memory bounded.
#[derive(Debug, Default)]
pub struct SegmentationEvaluator {
    support: BTreeMap<usize, usize>,
    matches: Vec<ScoredMaskMatch>,
    next_order: usize,
}

impl SegmentationEvaluator {
    pub fn update(&mut self, predictions: &[MetricMaskPrediction], targets: &[MetricMaskTarget]) {
        for target in targets.iter().filter(|target| !target.crowd) {
            *self.support.entry(target.class_id).or_default() += 1;
        }
        let classes = predictions
            .iter()
            .map(|prediction| prediction.class_id)
            .chain(targets.iter().map(|target| target.class_id))
            .collect::<BTreeSet<_>>();
        for class in classes {
            let truth = targets
                .iter()
                .filter(|target| target.class_id == class)
                .collect::<Vec<_>>();
            let mut candidates = predictions
                .iter()
                .filter(|prediction| prediction.class_id == class)
                .collect::<Vec<_>>();
            candidates.sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
            let mut matched = std::array::from_fn::<_, 10, _>(|_| BTreeSet::<usize>::new());
            for prediction in candidates {
                let mut outcomes = [0_i8; 10];
                for (threshold_index, owners) in matched.iter_mut().enumerate() {
                    let threshold = 0.5 + threshold_index as f32 * 0.05;
                    let mut possible = truth
                        .iter()
                        .enumerate()
                        .map(|(index, target)| {
                            (index, *target, mask_iou(&prediction.mask, &target.mask))
                        })
                        .filter(|(_, _, overlap)| *overlap >= threshold)
                        .collect::<Vec<_>>();
                    possible.sort_by(|left, right| {
                        right
                            .2
                            .total_cmp(&left.2)
                            .then_with(|| left.0.cmp(&right.0))
                    });
                    if let Some((index, _, _)) = possible
                        .iter()
                        .copied()
                        .find(|(index, target, _)| !target.crowd && !owners.contains(index))
                    {
                        owners.insert(index);
                        outcomes[threshold_index] = 1;
                    } else if possible.iter().any(|(_, target, _)| target.crowd) {
                        outcomes[threshold_index] = -1;
                    }
                }
                self.matches.push(ScoredMaskMatch {
                    class_id: class,
                    confidence: prediction.confidence,
                    order: self.next_order,
                    outcomes,
                });
                self.next_order += 1;
            }
        }
    }

    pub fn finish(self) -> SegmentationMetrics {
        if self.support.is_empty() {
            return SegmentationMetrics {
                precision: 0.0,
                recall: 0.0,
                map_50_95: 0.0,
                map_50: 0.0,
                per_class_support: self.support,
            };
        }
        let classes = self.support.keys().copied().collect::<Vec<_>>();
        let mut sums = [0.0_f32; 10];
        let mut precision = 0.0;
        let mut recall = 0.0;
        for class in &classes {
            for (threshold, sum) in sums.iter_mut().enumerate() {
                let curve = self.curve(*class, threshold);
                *sum += interpolated_ap(&curve);
                if threshold == 0 {
                    let (class_precision, class_recall) = best_precision_recall(curve);
                    precision += class_precision;
                    recall += class_recall;
                }
            }
        }
        let class_count = classes.len() as f32;
        for sum in &mut sums {
            *sum /= class_count;
        }
        SegmentationMetrics {
            precision: precision / class_count,
            recall: recall / class_count,
            map_50: sums[0],
            map_50_95: sums.iter().sum::<f32>() / sums.len() as f32,
            per_class_support: self.support,
        }
    }

    fn curve(&self, class: usize, threshold: usize) -> (Vec<f32>, Vec<f32>) {
        let positive_count = self.support.get(&class).copied().unwrap_or_default();
        if positive_count == 0 {
            return (Vec::new(), Vec::new());
        }
        let mut candidates = self
            .matches
            .iter()
            .filter(|prediction| prediction.class_id == class)
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            right
                .confidence
                .total_cmp(&left.confidence)
                .then_with(|| left.order.cmp(&right.order))
        });
        let mut tp = 0.0_f32;
        let mut fp = 0.0_f32;
        let mut precision = Vec::new();
        let mut recall = Vec::new();
        for candidate in candidates {
            match candidate.outcomes[threshold] {
                -1 => continue,
                1 => tp += 1.0,
                _ => fp += 1.0,
            }
            precision.push(tp / (tp + fp).max(1e-9));
            recall.push(tp / positive_count as f32);
        }
        (precision, recall)
    }
}

fn interpolated_ap(curve: &(Vec<f32>, Vec<f32>)) -> f32 {
    (0..=100)
        .map(|point| {
            let threshold = point as f32 / 100.0;
            curve
                .1
                .iter()
                .zip(&curve.0)
                .filter(|(recall, _)| **recall >= threshold)
                .map(|(_, precision)| *precision)
                .fold(0.0, f32::max)
        })
        .sum::<f32>()
        / 101.0
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
            precision: 0.0,
            recall: 0.0,
            map_50_95: 0.0,
            map_50: 0.0,
            per_class_support: support,
        };
    }
    let thresholds = (0..10)
        .map(|index| 0.5 + index as f32 * 0.05)
        .collect::<Vec<_>>();
    let mut sums = vec![0.0; thresholds.len()];
    let mut precision = 0.0;
    let mut recall = 0.0;
    for class in &classes {
        let (class_precision, class_recall) =
            best_precision_recall(precision_recall_curve(*class, 0.5, predictions, targets));
        precision += class_precision;
        recall += class_recall;
        for (index, threshold) in thresholds.iter().enumerate() {
            sums[index] += average_precision(*class, *threshold, predictions, targets);
        }
    }
    for value in &mut sums {
        *value /= classes.len() as f32;
    }
    SegmentationMetrics {
        precision: precision / classes.len() as f32,
        recall: recall / classes.len() as f32,
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
    let (precision, recall) = precision_recall_curve(class, threshold, predictions, targets);
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

fn precision_recall_curve(
    class: usize,
    threshold: f32,
    predictions: &[MetricMaskPrediction],
    targets: &[MetricMaskTarget],
) -> (Vec<f32>, Vec<f32>) {
    let truth = targets
        .iter()
        .filter(|target| target.class_id == class)
        .collect::<Vec<_>>();
    let positive_count = truth.iter().filter(|target| !target.crowd).count();
    if positive_count == 0 {
        return (Vec::new(), Vec::new());
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
    (precision, recall)
}

fn best_precision_recall(curve: (Vec<f32>, Vec<f32>)) -> (f32, f32) {
    curve
        .0
        .into_iter()
        .zip(curve.1)
        .max_by(
            |(left_precision, left_recall), (right_precision, right_recall)| {
                let left =
                    2.0 * left_precision * left_recall / (left_precision + left_recall).max(1e-9);
                let right = 2.0 * right_precision * right_recall
                    / (right_precision + right_recall).max(1e-9);
                left.total_cmp(&right)
            },
        )
        .unwrap_or((0.0, 0.0))
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
        assert_eq!((metrics.precision, metrics.recall), (1.0, 1.0));
    }

    #[test]
    fn streaming_evaluator_matches_dataset_evaluation() {
        let predictions = vec![
            MetricMaskPrediction {
                image_id: "a".into(),
                class_id: 1,
                confidence: 0.95,
                mask: vec![true, true, false, false],
            },
            MetricMaskPrediction {
                image_id: "a".into(),
                class_id: 1,
                confidence: 0.75,
                mask: vec![false, false, true, true],
            },
            MetricMaskPrediction {
                image_id: "b".into(),
                class_id: 2,
                confidence: 0.85,
                mask: vec![true, false, true, false],
            },
            MetricMaskPrediction {
                image_id: "b".into(),
                class_id: 2,
                confidence: 0.65,
                mask: vec![false, true, false, true],
            },
        ];
        let targets = vec![
            MetricMaskTarget {
                image_id: "a".into(),
                class_id: 1,
                mask: vec![true, true, false, false],
                crowd: false,
            },
            MetricMaskTarget {
                image_id: "a".into(),
                class_id: 1,
                mask: vec![false, false, true, true],
                crowd: true,
            },
            MetricMaskTarget {
                image_id: "b".into(),
                class_id: 2,
                mask: vec![true, false, true, false],
                crowd: false,
            },
        ];

        let expected = evaluate(&predictions, &targets);
        let mut streaming = SegmentationEvaluator::default();
        streaming.update(&predictions[..2], &targets[..2]);
        streaming.update(&predictions[2..], &targets[2..]);
        let actual = streaming.finish();

        assert_eq!(actual.per_class_support, expected.per_class_support);
        assert!((actual.precision - expected.precision).abs() < 1e-6);
        assert!((actual.recall - expected.recall).abs() < 1e-6);
        assert!((actual.map_50 - expected.map_50).abs() < 1e-6);
        assert!((actual.map_50_95 - expected.map_50_95).abs() < 1e-6);
    }
}
