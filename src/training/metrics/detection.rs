use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::training::geometry::{BoxXyxy, iou::iou};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPrediction {
    pub image_id: String,
    pub class_id: usize,
    pub confidence: f32,
    pub bbox: BoxXyxy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricTarget {
    pub image_id: String,
    pub class_id: usize,
    pub bbox: BoxXyxy,
    pub crowd: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectionMetrics {
    #[serde(default)]
    pub precision: f32,
    #[serde(default)]
    pub recall: f32,
    pub map_50_95: f32,
    pub map_50: f32,
    pub per_class_support: BTreeMap<usize, usize>,
}

pub fn evaluate(predictions: &[MetricPrediction], targets: &[MetricTarget]) -> DetectionMetrics {
    let classes: BTreeSet<usize> = targets
        .iter()
        .filter(|target| !target.crowd)
        .map(|target| target.class_id)
        .collect();
    let mut support = BTreeMap::new();
    for class in &classes {
        support.insert(
            *class,
            targets
                .iter()
                .filter(|target| !target.crowd && target.class_id == *class)
                .count(),
        );
    }
    if classes.is_empty() {
        return DetectionMetrics {
            precision: 0.0,
            recall: 0.0,
            map_50_95: 0.0,
            map_50: 0.0,
            per_class_support: support,
        };
    }
    let thresholds: Vec<f32> = (0..10).map(|index| 0.5 + index as f32 * 0.05).collect();
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
    DetectionMetrics {
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
    predictions: &[MetricPrediction],
    targets: &[MetricTarget],
) -> f32 {
    let (precision, recall) = precision_recall_curve(class, threshold, predictions, targets);
    // COCO-style 101-point AP.
    //
    // pycocotools first builds the precision envelope (`p[i] = max(p[i..])`) and then, for each of
    // the 101 recall thresholds, reads the envelope at the first recall at or above the threshold.
    // Taking the maximum over every point whose recall clears the threshold — what this fold does —
    // is the same number, because the envelope at that index is by definition that maximum. Do not
    // "fix" this into a linear interpolation between neighbouring points: pycocotools does not
    // interpolate, and adding it would move every reported mAP away from the reference.
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
    predictions: &[MetricPrediction],
    targets: &[MetricTarget],
) -> (Vec<f32>, Vec<f32>) {
    let truth: Vec<_> = targets
        .iter()
        .filter(|target| target.class_id == class)
        .collect();
    let positive_count = truth.iter().filter(|target| !target.crowd).count();
    if positive_count == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut candidates: Vec<_> = predictions
        .iter()
        .filter(|prediction| prediction.class_id == class)
        .collect();
    candidates.sort_by(|a, b| {
        b.confidence
            .total_cmp(&a.confidence)
            .then_with(|| a.image_id.cmp(&b.image_id))
    });
    let mut matched = BTreeSet::<(String, usize)>::new();
    let mut true_positives = Vec::with_capacity(candidates.len());
    let mut false_positives = Vec::with_capacity(candidates.len());
    for prediction in candidates {
        let mut possible: Vec<_> = truth
            .iter()
            .enumerate()
            .filter(|(_, target)| target.image_id == prediction.image_id)
            .map(|(index, target)| (index, *target, iou(prediction.bbox, target.bbox)))
            .filter(|(_, _, overlap)| *overlap >= threshold)
            .collect();
        possible.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
        let owner = possible.into_iter().find(|(index, target, _)| {
            !target.crowd && !matched.contains(&(target.image_id.clone(), *index))
        });
        if let Some((index, target, _)) = owner {
            matched.insert((target.image_id.clone(), index));
            true_positives.push(1.0_f32);
            false_positives.push(0.0_f32);
        } else if truth.iter().any(|target| {
            target.crowd
                && target.image_id == prediction.image_id
                && iou(prediction.bbox, target.bbox) >= threshold
        }) {
            // An otherwise-unmatched prediction on a crowd region is ignored.
            continue;
        } else {
            true_positives.push(0.0);
            false_positives.push(1.0);
        }
    }
    for index in 1..true_positives.len() {
        true_positives[index] += true_positives[index - 1];
        false_positives[index] += false_positives[index - 1];
    }
    let mut precision = Vec::with_capacity(true_positives.len());
    let mut recall = Vec::with_capacity(true_positives.len());
    for (tp, fp) in true_positives.into_iter().zip(false_positives) {
        precision.push(tp / (tp + fp).max(1e-9));
        recall.push(tp / positive_count as f32);
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn average_precision_reads_the_precision_envelope() {
        // One class, two ground truths, one hit and one miss. The PR curve has a single recall
        // value (0.5) at two precisions (1.0 then 0.5); the envelope keeps 1.0 for every threshold
        // up to 0.5 and drops to 0 beyond, so exactly 51 of the 101 points contribute.
        let hit = BoxXyxy::new([0.0, 0.0, 10.0, 10.0]).unwrap();
        let targets = [
            MetricTarget {
                image_id: "a".into(),
                class_id: 0,
                bbox: hit,
                crowd: false,
            },
            MetricTarget {
                image_id: "a".into(),
                class_id: 0,
                bbox: BoxXyxy::new([20.0, 20.0, 30.0, 30.0]).unwrap(),
                crowd: false,
            },
        ];
        let predictions = [
            MetricPrediction {
                image_id: "a".into(),
                class_id: 0,
                confidence: 0.9,
                bbox: hit,
            },
            MetricPrediction {
                image_id: "a".into(),
                class_id: 0,
                confidence: 0.5,
                bbox: BoxXyxy::new([50.0, 50.0, 60.0, 60.0]).unwrap(),
            },
        ];
        let ap = average_precision(0, 0.5, &predictions, &targets);
        assert!((ap - 51.0 / 101.0).abs() < 1e-5, "{ap}");
    }

    #[test]
    fn perfect_prediction_has_unit_map() {
        let bbox = BoxXyxy::new([0.0, 0.0, 10.0, 10.0]).unwrap();
        let metrics = evaluate(
            &[MetricPrediction {
                image_id: "a".into(),
                class_id: 2,
                confidence: 0.9,
                bbox,
            }],
            &[MetricTarget {
                image_id: "a".into(),
                class_id: 2,
                bbox,
                crowd: false,
            }],
        );
        assert!((metrics.map_50_95 - 1.0).abs() < 1e-6);
        assert_eq!((metrics.precision, metrics.recall), (1.0, 1.0));
    }
}
