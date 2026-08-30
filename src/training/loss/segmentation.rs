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

/// Differentiable 2x bilinear upsample with half-pixel centers and clamped borders.
///
/// Burn's WGPU JIT backend does not currently implement the backward pass for the generic
/// bilinear interpolation operator. YOLO26's semantic tower always needs the fixed P3-stride-8 to
/// mask-stride-4 resize, so expressing that exact resize with slices, arithmetic, and reshapes
/// keeps the official interpolation geometry while remaining differentiable on WGPU.
pub fn bilinear_upsample_2x<B: Backend>(input: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, channels, height, width] = input.dims();
    let horizontal_prev = if width == 1 {
        input.clone()
    } else {
        Tensor::cat(
            vec![
                input
                    .clone()
                    .slice([0..batch, 0..channels, 0..height, 0..1]),
                input
                    .clone()
                    .slice([0..batch, 0..channels, 0..height, 0..width - 1]),
            ],
            3,
        )
    };
    let horizontal_next = if width == 1 {
        input.clone()
    } else {
        Tensor::cat(
            vec![
                input
                    .clone()
                    .slice([0..batch, 0..channels, 0..height, 1..width]),
                input
                    .clone()
                    .slice([0..batch, 0..channels, 0..height, width - 1..width]),
            ],
            3,
        )
    };
    let horizontal_even = horizontal_prev * 0.25 + input.clone() * 0.75;
    let horizontal_odd = input * 0.75 + horizontal_next * 0.25;
    let horizontal = Tensor::stack::<5>(vec![horizontal_even, horizontal_odd], 4).reshape([
        batch,
        channels,
        height,
        width * 2,
    ]);

    let vertical_prev = if height == 1 {
        horizontal.clone()
    } else {
        Tensor::cat(
            vec![
                horizontal
                    .clone()
                    .slice([0..batch, 0..channels, 0..1, 0..width * 2]),
                horizontal
                    .clone()
                    .slice([0..batch, 0..channels, 0..height - 1, 0..width * 2]),
            ],
            2,
        )
    };
    let vertical_next = if height == 1 {
        horizontal.clone()
    } else {
        Tensor::cat(
            vec![
                horizontal
                    .clone()
                    .slice([0..batch, 0..channels, 1..height, 0..width * 2]),
                horizontal
                    .clone()
                    .slice([0..batch, 0..channels, height - 1..height, 0..width * 2]),
            ],
            2,
        )
    };
    let vertical_even = vertical_prev * 0.25 + horizontal.clone() * 0.75;
    let vertical_odd = horizontal * 0.75 + vertical_next * 0.25;
    Tensor::stack::<5>(vec![vertical_even, vertical_odd], 3).reshape([
        batch,
        channels,
        height * 2,
        width * 2,
    ])
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
    let coverage = coverage.float().unsqueeze_dim::<4>(1);
    let target = if classes == 1 {
        // Burn deliberately rejects one-hot encodings with fewer than two classes. A one-class
        // segmenter still has a valid semantic target: covered pixels belong to its sole object
        // class and uncovered pixels are background.
        coverage
    } else {
        let one_hot: Tensor<B, 4, Int> = class_map.one_hot(classes);
        one_hot.permute([0, 3, 1, 2]).float() * coverage
    };
    let bce = bce_with_logits_tensor(logits.clone(), target.clone()).mean();
    let probabilities = activation::sigmoid(logits);
    let intersection = (probabilities.clone() * target.clone()).sum();
    let dice = 1.0 - (intersection * 2.0 + 1.0) / (probabilities.sum() + target.sum() + 1.0);
    Ok(bce * 0.5 + dice * 0.5)
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Bool, Int, Tensor};
    use burn_flex::Flex;

    use super::{bilinear_upsample_2x, semantic_bce_dice_loss};

    #[test]
    fn manual_bilinear_upsample_matches_half_pixel_geometry() {
        let device = Default::default();
        let input = Tensor::<Flex, 4>::from_floats([[[[1.0, 2.0], [3.0, 4.0]]]], &device);
        let actual = bilinear_upsample_2x(input).into_data();
        let expected = [
            1.0, 1.25, 1.75, 2.0, 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3.0, 3.25, 3.75, 4.0,
        ];
        assert_eq!(actual.shape.dims::<4>(), [1, 1, 4, 4]);
        assert_eq!(actual.as_slice::<f32>().unwrap(), expected);
    }

    #[test]
    fn single_class_semantic_target_uses_coverage_without_one_hot() {
        let device = Default::default();
        let logits = Tensor::<Flex, 4>::zeros([1, 1, 2, 2], &device);
        let class_map = Tensor::<Flex, 3, Int>::from_ints([[[0, 0], [0, 0]]], &device);
        let coverage =
            Tensor::<Flex, 3, Bool>::from_bool([[[true, false], [false, true]]].into(), &device);

        let loss = semantic_bce_dice_loss(logits, class_map, coverage)
            .unwrap()
            .into_data();
        let value = loss.as_slice::<f32>().unwrap()[0];
        assert!(value.is_finite());
        assert!(value > 0.0);
    }
}
