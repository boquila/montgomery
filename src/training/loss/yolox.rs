use std::collections::BTreeMap;

use crate::training::{
    assign::simota::{AnchorPrediction, GroundTruth, assign},
    geometry::iou::iou,
};

use super::common::bce_with_logits;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YoloxLoss {
    pub iou: f32,
    pub objectness: f32,
    pub classification: f32,
    pub l1: f32,
    pub total: f32,
    pub foreground: usize,
    pub ground_truth: usize,
}

/// Detached scalar YOLOX reference loss.
///
/// The production tensor path follows the same equations; this host implementation is kept as a
/// deterministic assignment/loss oracle and powers the small exact fixtures.
pub fn loss(
    ground_truth: &[GroundTruth],
    predictions: &[AnchorPrediction],
    use_l1: bool,
) -> Result<YoloxLoss, &'static str> {
    if predictions
        .iter()
        .any(|prediction| prediction.class_logits.is_empty())
    {
        return Err("YOLOX predictions require at least one class");
    }
    let matches = assign(ground_truth, predictions);
    let owners: BTreeMap<usize, _> = matches
        .iter()
        .map(|matched| (matched.anchor_index, matched))
        .collect();
    let normalizer = matches.len().max(1) as f32;
    let objectness = predictions
        .iter()
        .enumerate()
        .map(|(anchor, prediction)| {
            bce_with_logits(
                prediction.objectness_logit,
                f32::from(owners.contains_key(&anchor)),
            )
        })
        .sum::<f32>()
        / normalizer;

    let mut iou_loss = 0.0;
    let mut classification = 0.0;
    let mut l1 = 0.0;
    for matched in &matches {
        let prediction = &predictions[matched.anchor_index];
        let truth = &ground_truth[matched.gt_index];
        let overlap = iou(prediction.box_xywh.to_xyxy(), truth.bbox);
        iou_loss += 1.0 - overlap * overlap;
        for (class, logit) in prediction.class_logits.iter().enumerate() {
            let target = if class == truth.class_id {
                matched.overlap
            } else {
                0.0
            };
            classification += bce_with_logits(*logit, target);
        }
        if use_l1 {
            let encoded = encode_l1(truth, prediction);
            l1 += prediction
                .raw_box
                .iter()
                .zip(encoded)
                .map(|(actual, target)| (*actual - target).abs())
                .sum::<f32>();
        }
    }
    iou_loss /= normalizer;
    classification /= normalizer;
    l1 /= normalizer;
    let total = 5.0 * iou_loss + objectness + classification + l1;
    Ok(YoloxLoss {
        iou: iou_loss,
        objectness,
        classification,
        l1,
        total,
        foreground: matches.len(),
        ground_truth: ground_truth.len(),
    })
}

fn encode_l1(gt: &GroundTruth, prediction: &AnchorPrediction) -> [f32; 4] {
    let xywh = gt.bbox.to_xywh();
    [
        xywh.cx / prediction.stride - (prediction.center_xy[0] / prediction.stride - 0.5),
        xywh.cy / prediction.stride - (prediction.center_xy[1] / prediction.stride - 0.5),
        (xywh.width / prediction.stride).max(1e-8).ln(),
        (xywh.height / prediction.stride).max(1e-8).ln(),
    ]
}

#[cfg(test)]
mod tests {
    use crate::training::geometry::BoxXywh;

    use super::*;

    #[test]
    fn empty_batch_has_only_finite_background_objectness() {
        let predictions = vec![AnchorPrediction {
            box_xywh: BoxXywh {
                cx: 4.0,
                cy: 4.0,
                width: 8.0,
                height: 8.0,
            },
            raw_box: [0.0; 4],
            objectness_logit: 100.0,
            class_logits: vec![-100.0, 100.0, 0.0],
            center_xy: [4.0, 4.0],
            stride: 8.0,
        }];
        let value = loss(&[], &predictions, false).unwrap();
        assert_eq!(value.iou, 0.0);
        assert_eq!(value.classification, 0.0);
        assert!(value.total.is_finite());
        assert!(value.objectness > 99.0);
    }
}
