use std::collections::BTreeMap;

use burn::tensor::{Tensor, TensorData, backend::Backend};

use crate::{
    models::yolox::RawPredictions,
    training::{
        assign::simota::{AnchorPrediction, GroundTruth, assign},
        geometry::{BoxXywh, iou::iou},
    },
};

use super::common::{LossOutput, bce_with_logits, bce_with_logits_tensor, scalar_values};

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

/// Differentiable YOLOX criterion with deterministic detached SimOTA assignment.
///
/// Assignment uses detached host values; loss terms stay on the original differentiable graph.
pub fn tensor_loss<B: Backend>(
    output: RawPredictions<B>,
    targets: &[Vec<GroundTruth>],
    use_l1: bool,
) -> Result<LossOutput<B>, &'static str> {
    let [batch, anchors, classes] = output.class_logits.dims();
    if targets.len() != batch || classes == 0 {
        return Err("YOLOX target batch or class count does not match predictions");
    }
    let [decoded_host, regression_host, objectness_host, classes_host] =
        burn::tensor::Transaction::default()
            .register(output.decoded_boxes.clone().detach())
            .register(output.regression.clone().detach())
            .register(output.objectness_logits.clone().detach())
            .register(output.class_logits.clone().detach())
            .execute()
            .try_into()
            .expect("YOLOX assignment transaction must preserve four tensors");
    let decoded = decoded_host
        .as_slice::<f32>()
        .map_err(|_| "YOLOX boxes are not f32")?;
    let regression = regression_host
        .as_slice::<f32>()
        .map_err(|_| "YOLOX deltas are not f32")?;
    let objectness = objectness_host
        .as_slice::<f32>()
        .map_err(|_| "YOLOX objectness is not f32")?;
    let class_logits = classes_host
        .as_slice::<f32>()
        .map_err(|_| "YOLOX classes are not f32")?;

    let mut dense_boxes = vec![0.0_f32; batch * anchors * 4];
    let mut dense_raw = vec![0.0_f32; batch * anchors * 4];
    let mut dense_classes = vec![0.0_f32; batch * anchors * classes];
    let mut foreground = vec![0.0_f32; batch * anchors];
    let mut foreground_count = 0;
    let mut gt_count = 0;
    let mut centers = Vec::with_capacity(anchors);
    for level in output.levels {
        for y in 0..level.height {
            for x in 0..level.width {
                centers.push([
                    (x as f32 + 0.5) * level.stride as f32,
                    (y as f32 + 0.5) * level.stride as f32,
                    level.stride as f32,
                ]);
            }
        }
    }
    if centers.len() != anchors {
        return Err("YOLOX feature layouts do not match anchor count");
    }
    for image in 0..batch {
        gt_count += targets[image].len();
        let predictions = (0..anchors)
            .map(|anchor| {
                let box_offset = (image * anchors + anchor) * 4;
                let class_offset = (image * anchors + anchor) * classes;
                AnchorPrediction {
                    box_xywh: BoxXywh {
                        cx: decoded[box_offset],
                        cy: decoded[box_offset + 1],
                        width: decoded[box_offset + 2],
                        height: decoded[box_offset + 3],
                    },
                    raw_box: regression[box_offset..box_offset + 4].try_into().unwrap(),
                    objectness_logit: objectness[image * anchors + anchor],
                    class_logits: class_logits[class_offset..class_offset + classes].to_vec(),
                    center_xy: [centers[anchor][0], centers[anchor][1]],
                    stride: centers[anchor][2],
                }
            })
            .collect::<Vec<_>>();
        for matched in assign(&targets[image], &predictions) {
            let truth = &targets[image][matched.gt_index];
            if truth.class_id >= classes {
                return Err("YOLOX target class is outside model channels");
            }
            let flat = image * anchors + matched.anchor_index;
            foreground[flat] = 1.0;
            foreground_count += 1;
            let box_offset = flat * 4;
            let xywh = truth.bbox.to_xywh();
            dense_boxes[box_offset..box_offset + 4].copy_from_slice(&[
                xywh.cx,
                xywh.cy,
                xywh.width,
                xywh.height,
            ]);
            dense_raw[box_offset..box_offset + 4]
                .copy_from_slice(&encode_l1(truth, &predictions[matched.anchor_index]));
            dense_classes[flat * classes + truth.class_id] = matched.overlap;
        }
    }

    let device = output.class_logits.device();
    let fg = Tensor::from_data(TensorData::new(foreground, [batch, anchors, 1]), &device);
    let target_boxes =
        Tensor::from_data(TensorData::new(dense_boxes, [batch, anchors, 4]), &device);
    let target_raw = Tensor::from_data(TensorData::new(dense_raw, [batch, anchors, 4]), &device);
    let target_classes = Tensor::from_data(
        TensorData::new(dense_classes, [batch, anchors, classes]),
        &device,
    );
    let normalizer = foreground_count.max(1) as f64;

    let objectness_loss =
        bce_with_logits_tensor(output.objectness_logits, fg.clone()).sum() / normalizer;
    let classification_loss =
        (bce_with_logits_tensor(output.class_logits, target_classes) * fg.clone()).sum()
            / normalizer;
    let predicted = output.decoded_boxes;
    let pred_xy = predicted.clone().slice([0..batch, 0..anchors, 0..2]);
    let pred_wh = predicted.clone().slice([0..batch, 0..anchors, 2..4]);
    let target_xy = target_boxes.clone().slice([0..batch, 0..anchors, 0..2]);
    let target_wh = target_boxes.slice([0..batch, 0..anchors, 2..4]);
    let pred_lt = pred_xy.clone() - pred_wh.clone() * 0.5;
    let pred_rb = pred_xy + pred_wh * 0.5;
    let target_lt = target_xy.clone() - target_wh.clone() * 0.5;
    let target_rb = target_xy + target_wh * 0.5;
    let intersection = (pred_rb.clone().min_pair(target_rb.clone())
        - pred_lt.clone().max_pair(target_lt.clone()))
    .clamp_min(0.0);
    let intersection = intersection.clone().slice([0..batch, 0..anchors, 0..1])
        * intersection.slice([0..batch, 0..anchors, 1..2]);
    let pred_size = pred_rb - pred_lt;
    let target_size = target_rb - target_lt;
    let pred_area = pred_size.clone().slice([0..batch, 0..anchors, 0..1])
        * pred_size.slice([0..batch, 0..anchors, 1..2]);
    let target_area = target_size.clone().slice([0..batch, 0..anchors, 0..1])
        * target_size.slice([0..batch, 0..anchors, 1..2]);
    let overlap = intersection.clone() / (pred_area + target_area - intersection + 1e-7);
    let iou_loss = ((overlap.clone() * overlap).neg() + 1.0) * fg.clone();
    let iou_loss = iou_loss.sum() / normalizer;
    let l1_loss = if use_l1 {
        ((output.regression - target_raw).abs() * fg).sum() / normalizer
    } else {
        output.regression.sum() * 0.0
    };
    let total = iou_loss.clone() * 5.0
        + objectness_loss.clone()
        + classification_loss.clone()
        + l1_loss.clone();
    let [
        iou_value,
        objectness_value,
        classification_value,
        l1_value,
        value,
    ] = scalar_values([
        iou_loss,
        objectness_loss,
        classification_loss,
        l1_loss,
        total.clone(),
    ]);
    let mut components = BTreeMap::new();
    components.insert("iou_loss".into(), iou_value);
    components.insert("objectness_loss".into(), objectness_value);
    components.insert("classification_loss".into(), classification_value);
    components.insert("l1_loss".into(), l1_value);
    Ok(LossOutput {
        total,
        total_value: value,
        deferred_component: None,
        components,
        targets: gt_count,
        foreground: foreground_count,
        finite: value.is_finite(),
    })
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
