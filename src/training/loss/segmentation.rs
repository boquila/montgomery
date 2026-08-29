use burn::tensor::{Bool, Int, Tensor, TensorData, activation, backend::Backend};

use super::common::{bce_with_logits_tensor, connected_zero};
use crate::training::geometry::BoxXyxy;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaskMatch {
    pub batch_index: usize,
    pub anchor_index: usize,
    pub target_index: usize,
    /// Target box normalized to the training canvas.
    pub normalized_box: BoxXyxy,
}

/// Cropped prototype-mask BCE used by YOLO11/YOLO26 segmentation.
pub fn instance_mask_loss<B: Backend>(
    coefficients: Tensor<B, 3>,
    prototypes: Tensor<B, 4>,
    target_masks: Tensor<B, 4>,
    matches: &[MaskMatch],
) -> Result<Tensor<B, 1>, &'static str> {
    let [batch, masks, anchors] = coefficients.dims();
    let [proto_batch, proto_masks, height, width] = prototypes.dims();
    let [target_batch, target_count, target_height, target_width] = target_masks.dims();
    if proto_batch != batch
        || target_batch != batch
        || proto_masks != masks
        || target_height != height
        || target_width != width
    {
        return Err("segmentation coefficient/prototype/target shapes disagree");
    }
    if matches.is_empty() {
        return Ok(connected_zero(coefficients) + connected_zero(prototypes));
    }
    let device = prototypes.device();
    let mut losses = Vec::with_capacity(matches.len());
    for matched in matches {
        if matched.batch_index >= batch
            || matched.anchor_index >= anchors
            || matched.target_index >= target_count
        {
            return Err("segmentation match index is outside tensors");
        }
        let b = matched.batch_index;
        let a = matched.anchor_index;
        let t = matched.target_index;
        let coefficient = coefficients
            .clone()
            .slice([b..b + 1, 0..masks, a..a + 1])
            .swap_dims(1, 2);
        let prototype = prototypes
            .clone()
            .slice([b..b + 1, 0..masks, 0..height, 0..width])
            .reshape([1, masks, height * width]);
        let logits = coefficient.matmul(prototype).reshape([1, 1, height, width]);
        let target = target_masks
            .clone()
            .slice([b..b + 1, t..t + 1, 0..height, 0..width]);
        let box_value = matched.normalized_box;
        let xmin = (box_value.xmin * width as f32)
            .floor()
            .clamp(0.0, width as f32) as usize;
        let ymin = (box_value.ymin * height as f32)
            .floor()
            .clamp(0.0, height as f32) as usize;
        let xmax = (box_value.xmax * width as f32)
            .ceil()
            .clamp(0.0, width as f32) as usize;
        let ymax = (box_value.ymax * height as f32)
            .ceil()
            .clamp(0.0, height as f32) as usize;
        let mut crop = vec![0_f32; height * width];
        for y in ymin..ymax {
            crop[y * width + xmin..y * width + xmax].fill(1.0);
        }
        let normalized_area = (box_value.xmax - box_value.xmin) * (box_value.ymax - box_value.ymin);
        if normalized_area <= 0.0 || !normalized_area.is_finite() {
            return Err("segmentation match has an invalid normalized box");
        }
        let crop = Tensor::<B, 4>::from_data(TensorData::new(crop, [1, 1, height, width]), &device);
        losses.push(
            (bce_with_logits_tensor(logits, target) * crop).sum()
                / ((height * width) as f64 * normalized_area as f64),
        );
    }
    Ok(Tensor::stack::<2>(losses, 0).mean())
}

/// YOLO26 semantic BCE/Dice term. Background remains an all-zero class target; the separate
/// coverage tensor disambiguates it from object class zero while constructing one-hot targets.
pub fn semantic_bce_dice_loss<B: Backend>(
    logits: Tensor<B, 4>,
    class_map: Tensor<B, 3, Int>,
    coverage: Tensor<B, 3, Bool>,
) -> Result<Tensor<B, 1>, &'static str> {
    let [batch, classes, height, width] = logits.dims();
    if class_map.dims() != [batch, height, width]
        || coverage.dims() != [batch, height, width]
        || classes == 0
    {
        return Err("semantic logits and target shapes disagree");
    }
    let one_hot: Tensor<B, 4, Int> = class_map.one_hot(classes);
    let target = one_hot.permute([0, 3, 1, 2]).float() * coverage.float().unsqueeze_dim::<4>(1);
    let bce = bce_with_logits_tensor(logits.clone(), target.clone()).mean();
    let probabilities = activation::sigmoid(logits);
    let intersection = (probabilities.clone() * target.clone()).sum();
    let dice = 1.0 - (intersection * 2.0 + 1.0) / (probabilities.sum() + target.sum() + 1.0);
    Ok(bce * 0.5 + dice * 0.5)
}
