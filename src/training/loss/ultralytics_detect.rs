use std::collections::BTreeMap;

use burn::tensor::{Tensor, TensorData, activation, backend::Backend};

use crate::training::{
    assign::tal::{TalGroundTruth, TalPrediction, assign},
    geometry::{FeatureLevelLayout, make_anchors},
};

use super::common::{LossOutput, bce_with_logits_tensor, log_softmax, scalar_value};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectionLossConfig {
    pub reg_max: usize,
    pub top_k: usize,
    pub box_gain: f32,
    pub class_gain: f32,
    pub regression_gain: f32,
    pub image_size: [usize; 2],
}

impl DetectionLossConfig {
    pub fn dfl(image_size: [usize; 2], top_k: usize) -> Self {
        Self {
            reg_max: 16,
            top_k,
            box_gain: 7.5,
            class_gain: 0.5,
            regression_gain: 1.5,
            image_size,
        }
    }

    pub fn direct(image_size: [usize; 2], top_k: usize) -> Self {
        Self {
            reg_max: 1,
            top_k,
            box_gain: 7.5,
            class_gain: 0.5,
            regression_gain: 1.5,
            image_size,
        }
    }
}

/// Encode one non-negative side distance as the two DFL bins and interpolation weights.
pub fn dfl_target(distance: f32, reg_max: usize) -> Result<(usize, usize, f32, f32), &'static str> {
    if reg_max < 2 || !distance.is_finite() || distance < 0.0 {
        return Err("DFL requires a finite non-negative distance and reg_max >= 2");
    }
    let target = distance.min(reg_max as f32 - 1.0 - 0.01);
    let left = target.floor() as usize;
    let right = left + 1;
    let right_weight = target - left as f32;
    Ok((left, right, 1.0 - right_weight, right_weight))
}

pub fn dfl_loss(logits: &[f32], distance: f32) -> Result<f32, &'static str> {
    let (left, right, left_weight, right_weight) = dfl_target(distance, logits.len())?;
    let log_prob = log_softmax(logits);
    Ok(-log_prob[left] * left_weight - log_prob[right] * right_weight)
}

/// Differentiable TAL/CIoU/DFL (or direct-side L1) criterion.
///
/// TAL uses detached host values to make discrete matches deterministic. Dense target tensors are
/// uploaded once and all numeric losses are evaluated against the connected model outputs.
pub fn tensor_loss<B: Backend>(
    raw_boxes: Tensor<B, 3>,
    raw_scores: Tensor<B, 3>,
    levels: &[FeatureLevelLayout],
    targets: &[Vec<TalGroundTruth>],
    config: DetectionLossConfig,
) -> Result<LossOutput<B>, &'static str> {
    let [batch, box_channels, anchor_count] = raw_boxes.dims();
    let [score_batch, classes, score_anchors] = raw_scores.dims();
    if score_batch != batch
        || score_anchors != anchor_count
        || targets.len() != batch
        || box_channels != 4 * config.reg_max
        || config.reg_max == 0
        || config.top_k == 0
    {
        return Err("invalid modern detection prediction/configuration shapes");
    }
    let anchors = make_anchors(levels);
    if anchors.len() != anchor_count {
        return Err("feature-level layout does not match prediction anchors");
    }
    let device = raw_boxes.device();
    let mut anchor_xy = Vec::with_capacity(anchor_count * 2);
    let mut stride_values = Vec::with_capacity(anchor_count);
    for anchor in &anchors {
        anchor_xy.extend(anchor.grid_xy);
        stride_values.push(anchor.stride);
    }
    let anchor_tensor = Tensor::<B, 2>::from_data(
        TensorData::new(anchor_xy.clone(), [anchor_count, 2]),
        &device,
    )
    .unsqueeze::<3>();
    let stride_tensor = Tensor::<B, 2>::from_data(
        TensorData::new(stride_values.clone(), [anchor_count, 1]),
        &device,
    )
    .unsqueeze::<3>();
    let distances = if config.reg_max > 1 {
        let projection = Tensor::<B, 4>::from_data(
            TensorData::new(
                (0..config.reg_max).map(|value| value as f32).collect(),
                [1, 1, config.reg_max, 1],
            ),
            &device,
        );
        (activation::softmax(
            raw_boxes
                .clone()
                .reshape([batch, 4, config.reg_max, anchor_count]),
            2,
        ) * projection)
            .sum_dim(2)
            .squeeze_dim::<3>(2)
            .swap_dims(1, 2)
    } else {
        raw_boxes.clone().swap_dims(1, 2)
    };
    let left_top = distances.clone().slice([0..batch, 0..anchor_count, 0..2]);
    let right_bottom = distances.clone().slice([0..batch, 0..anchor_count, 2..4]);
    let decoded = Tensor::cat(
        vec![
            anchor_tensor.clone() - left_top,
            anchor_tensor.clone() + right_bottom,
        ],
        2,
    ) * stride_tensor.clone();

    let decoded_data = decoded.clone().detach().into_data();
    let score_data = activation::sigmoid(raw_scores.clone().detach().swap_dims(1, 2)).into_data();
    let decoded_host = decoded_data
        .as_slice::<f32>()
        .map_err(|_| "decoded boxes are not f32")?;
    let score_host = score_data
        .as_slice::<f32>()
        .map_err(|_| "scores are not f32")?;
    let mut dense_boxes = vec![0_f32; batch * anchor_count * 4];
    let mut dense_scores = vec![0_f32; batch * anchor_count * classes];
    let mut dense_weights = vec![0_f32; batch * anchor_count];
    let mut dense_distribution = vec![0_f32; batch * anchor_count * 4 * config.reg_max];
    let mut foreground = 0;
    let mut target_count = 0;
    for image in 0..batch {
        target_count += targets[image].len();
        let predictions = (0..anchor_count)
            .map(|anchor| {
                let box_offset = (image * anchor_count + anchor) * 4;
                let score_offset = (image * anchor_count + anchor) * classes;
                Ok(TalPrediction {
                    bbox: crate::training::geometry::BoxXyxy::new(
                        decoded_host[box_offset..box_offset + 4].try_into().unwrap(),
                    )?,
                    class_scores: score_host[score_offset..score_offset + classes].to_vec(),
                })
            })
            .collect::<Result<Vec<_>, &'static str>>()?;
        for matched in assign(&targets[image], &predictions, &anchors, config.top_k)? {
            let truth = &targets[image][matched.gt_index];
            if truth.class_id >= classes {
                return Err("target class is outside prediction channels");
            }
            let flat = image * anchor_count + matched.anchor_index;
            dense_boxes[flat * 4..flat * 4 + 4].copy_from_slice(&[
                truth.bbox.xmin,
                truth.bbox.ymin,
                truth.bbox.xmax,
                truth.bbox.ymax,
            ]);
            dense_scores[flat * classes + truth.class_id] = matched.target_score;
            dense_weights[flat] = matched.target_score;
            foreground += 1;
            let anchor = anchors[matched.anchor_index];
            let distances = [
                anchor.grid_xy[0] - truth.bbox.xmin / anchor.stride,
                anchor.grid_xy[1] - truth.bbox.ymin / anchor.stride,
                truth.bbox.xmax / anchor.stride - anchor.grid_xy[0],
                truth.bbox.ymax / anchor.stride - anchor.grid_xy[1],
            ];
            if config.reg_max > 1 {
                for (side, distance) in distances.into_iter().enumerate() {
                    let (left, right, left_weight, right_weight) =
                        dfl_target(distance.max(0.0), config.reg_max)?;
                    let offset = (flat * 4 + side) * config.reg_max;
                    dense_distribution[offset + left] = left_weight;
                    dense_distribution[offset + right] = right_weight;
                }
            }
        }
    }
    let target_boxes = Tensor::from_data(
        TensorData::new(dense_boxes, [batch, anchor_count, 4]),
        &device,
    );
    let target_scores = Tensor::from_data(
        TensorData::new(dense_scores, [batch, anchor_count, classes]),
        &device,
    );
    let target_weights = Tensor::from_data(
        TensorData::new(dense_weights, [batch, anchor_count, 1]),
        &device,
    );
    let score_sum = scalar_value(target_scores.clone().sum()).max(1.0) as f64;
    let class_loss =
        bce_with_logits_tensor(raw_scores.swap_dims(1, 2), target_scores).sum() / score_sum;
    let ciou = ciou_tensor(decoded, target_boxes.clone());
    let box_loss = ((ciou.neg() + 1.0) * target_weights.clone()).sum() / score_sum;
    let regression_loss = if config.reg_max > 1 {
        let target_distribution = Tensor::from_data(
            TensorData::new(dense_distribution, [batch, anchor_count, 4, config.reg_max]),
            &device,
        );
        let logits = raw_boxes
            .reshape([batch, 4, config.reg_max, anchor_count])
            .permute([0, 3, 1, 2]);
        let per_side = -(activation::log_softmax(logits, 3) * target_distribution)
            .sum_dim(3)
            .squeeze_dim::<3>(3);
        (per_side * target_weights).sum() / score_sum
    } else {
        let width = config.image_size[1] as f64;
        let height = config.image_size[0] as f64;
        let target_left_top = anchor_tensor.clone()
            - target_boxes
                .clone()
                .slice([0..batch, 0..anchor_count, 0..2])
                / stride_tensor.clone();
        let target_right_bottom =
            target_boxes.slice([0..batch, 0..anchor_count, 2..4]) / stride_tensor - anchor_tensor;
        let target_distances = Tensor::cat(vec![target_left_top, target_right_bottom], 2);
        let normalization = Tensor::<B, 3>::from_data(
            TensorData::new(
                vec![
                    1.0 / width as f32,
                    1.0 / height as f32,
                    1.0 / width as f32,
                    1.0 / height as f32,
                ],
                [1, 1, 4],
            ),
            &device,
        );
        ((distances - target_distances).abs() * normalization * target_weights).sum()
            / (score_sum * 4.0)
    };
    let total = box_loss.clone() * config.box_gain as f64
        + class_loss.clone() * config.class_gain as f64
        + regression_loss.clone() * config.regression_gain as f64;
    let mut components = BTreeMap::new();
    components.insert("box_loss".into(), scalar_value(box_loss));
    components.insert("classification_loss".into(), scalar_value(class_loss));
    components.insert(
        if config.reg_max > 1 {
            "dfl_loss"
        } else {
            "l1_loss"
        }
        .into(),
        scalar_value(regression_loss),
    );
    let value = scalar_value(total.clone());
    Ok(LossOutput {
        total,
        components,
        targets: target_count,
        foreground,
        finite: value.is_finite(),
    })
}

fn ciou_tensor<B: Backend>(predicted: Tensor<B, 3>, target: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch, anchors, _] = predicted.dims();
    let pred_lt = predicted.clone().slice([0..batch, 0..anchors, 0..2]);
    let pred_rb = predicted.clone().slice([0..batch, 0..anchors, 2..4]);
    let target_lt = target.clone().slice([0..batch, 0..anchors, 0..2]);
    let target_rb = target.clone().slice([0..batch, 0..anchors, 2..4]);
    let intersection_size = (pred_rb.clone().min_pair(target_rb.clone())
        - pred_lt.clone().max_pair(target_lt.clone()))
    .clamp_min(0.0);
    let intersection = intersection_size
        .clone()
        .slice([0..batch, 0..anchors, 0..1])
        * intersection_size.slice([0..batch, 0..anchors, 1..2]);
    let pred_size = (pred_rb.clone() - pred_lt.clone()).clamp_min(0.0);
    let target_size = (target_rb.clone() - target_lt.clone()).clamp_min(0.0);
    let pred_area = pred_size.clone().slice([0..batch, 0..anchors, 0..1])
        * pred_size.clone().slice([0..batch, 0..anchors, 1..2]);
    let target_area = target_size.clone().slice([0..batch, 0..anchors, 0..1])
        * target_size.clone().slice([0..batch, 0..anchors, 1..2]);
    let overlap = intersection.clone() / (pred_area + target_area - intersection + 1e-7);
    let pred_center = (pred_lt.clone() + pred_rb.clone()) * 0.5;
    let target_center = (target_lt.clone() + target_rb.clone()) * 0.5;
    let center_delta = pred_center - target_center;
    let center_distance = center_delta
        .clone()
        .slice([0..batch, 0..anchors, 0..1])
        .powi_scalar(2)
        + center_delta
            .slice([0..batch, 0..anchors, 1..2])
            .powi_scalar(2);
    let enclosing = target_rb.max_pair(pred_rb) - target_lt.min_pair(pred_lt);
    let diagonal = enclosing
        .clone()
        .slice([0..batch, 0..anchors, 0..1])
        .powi_scalar(2)
        + enclosing.slice([0..batch, 0..anchors, 1..2]).powi_scalar(2)
        + 1e-7;
    let pred_angle = pred_size
        .clone()
        .slice([0..batch, 0..anchors, 0..1])
        .atan2(pred_size.slice([0..batch, 0..anchors, 1..2]));
    let target_angle = target_size
        .clone()
        .slice([0..batch, 0..anchors, 0..1])
        .atan2(target_size.slice([0..batch, 0..anchors, 1..2]));
    let v = (target_angle - pred_angle).powi_scalar(2) * (4.0 / std::f64::consts::PI.powi(2));
    let alpha = v.clone() / ((overlap.clone().neg() + 1.0) + v.clone() + 1e-7);
    overlap - center_distance / diagonal - alpha * v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dfl_clamps_upper_bin() {
        let (left, right, lw, rw) = dfl_target(100.0, 16).unwrap();
        assert_eq!((left, right), (14, 15));
        assert!((lw + rw - 1.0).abs() < 1e-6);
        assert!(dfl_loss(&[0.0; 16], 100.0).unwrap().is_finite());
    }
}
