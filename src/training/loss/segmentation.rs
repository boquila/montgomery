use burn::tensor::{Bool, Int, Tensor, TensorData, activation, backend::Backend};

use super::common::{bce_with_logits_tensor, connected_zero};
use crate::training::geometry::BoxXyxy;

/// Ultralytics' box gain is also applied to both instance-mask and semantic-mask terms.
pub const SEGMENTATION_GAIN: f64 = 7.5;

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
    let matched_count = matches.len();
    let mut matches_by_batch = vec![Vec::new(); batch];
    for matched in matches {
        if matched.batch_index >= batch
            || matched.anchor_index >= anchors
            || matched.target_index >= target_count
        {
            return Err("segmentation match index is outside tensors");
        }
        matches_by_batch[matched.batch_index].push(matched);
    }
    let matches_per_batch = matches_by_batch
        .iter()
        .map(Vec::len)
        .max()
        .expect("non-empty matches have a batch")
        .max(1);
    let slot_count = batch * matches_per_batch;
    let mut coefficient_indices = vec![0_i64; slot_count];
    let mut target_indices = vec![0_i64; slot_count];
    let mut valid_indices = Vec::with_capacity(matched_count);
    let mut crop = vec![0_f32; slot_count * height * width];
    let mut normalizers = Vec::with_capacity(matched_count);
    for (b, image_matches) in matches_by_batch.into_iter().enumerate() {
        for (slot, matched) in image_matches.into_iter().enumerate() {
            let a = matched.anchor_index;
            let t = matched.target_index;
            let box_value = matched.normalized_box;
            let xmin = (box_value.xmin * width as f32)
                .ceil()
                .clamp(0.0, width as f32) as usize;
            let ymin = (box_value.ymin * height as f32)
                .ceil()
                .clamp(0.0, height as f32) as usize;
            let xmax = (box_value.xmax * width as f32)
                .ceil()
                .clamp(0.0, width as f32) as usize;
            let ymax = (box_value.ymax * height as f32)
                .ceil()
                .clamp(0.0, height as f32) as usize;
            let flat_slot = b * matches_per_batch + slot;
            let crop_offset = flat_slot * height * width;
            for y in ymin..ymax {
                crop[crop_offset + y * width + xmin..crop_offset + y * width + xmax].fill(1.0);
            }
            let normalized_area =
                (box_value.xmax - box_value.xmin) * (box_value.ymax - box_value.ymin);
            if normalized_area <= 0.0 || !normalized_area.is_finite() {
                return Err("segmentation match has an invalid normalized box");
            }
            coefficient_indices[flat_slot] = (b * anchors + a) as i64;
            target_indices[flat_slot] = (b * target_count + t) as i64;
            valid_indices.push(flat_slot as i64);
            normalizers.push((1.0 / ((height * width) as f64 * normalized_area as f64)) as f32);
        }
    }

    // Pad only the positive coefficients per image, then multiply each image's coefficient matrix
    // by its prototype matrix once. This mirrors Ultralytics' per-image einsum without duplicating
    // a full `[masks, height, width]` prototype for every positive anchor. Padded slots are removed
    // before the final reduction and therefore contribute neither loss nor gradient.
    let coefficient_indices =
        Tensor::<B, 1, Int>::from_data(TensorData::new(coefficient_indices, [slot_count]), &device);
    let target_indices =
        Tensor::<B, 1, Int>::from_data(TensorData::new(target_indices, [slot_count]), &device);
    let coefficients = coefficients
        .swap_dims(1, 2)
        .reshape([batch * anchors, masks])
        .select(0, coefficient_indices)
        .reshape([batch, matches_per_batch, masks]);
    let prototypes = prototypes.reshape([batch, masks, height * width]);
    let logits = coefficients
        .matmul(prototypes)
        .reshape([slot_count, 1, height, width]);
    let targets = target_masks
        .reshape([batch * target_count, height, width])
        .select(0, target_indices)
        .unsqueeze_dim::<4>(1);
    let crop = Tensor::<B, 4>::from_data(
        TensorData::new(crop, [slot_count, 1, height, width]),
        &device,
    );
    let normalizers =
        Tensor::<B, 2>::from_data(TensorData::new(normalizers, [matched_count, 1]), &device);
    let per_match = (bce_with_logits_tensor(logits, targets) * crop)
        .reshape([slot_count, height * width])
        .sum_dim(1)
        .select(
            0,
            Tensor::<B, 1, Int>::from_data(
                TensorData::new(valid_indices, [matched_count]),
                &device,
            ),
        );
    Ok((per_match * normalizers).mean())
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

/// Compute YOLO26's shared semantic loss once and provide its detached one-to-one counterpart.
///
/// `Yolo26SegHead` deliberately exposes the same semantic logits on both branches, with only the
/// one-to-one view detached. Applying deterministic elementwise/reduction operations before that
/// detach is value-equivalent and keeps the one-to-one contribution disconnected from the graph.
pub fn dual_semantic_bce_dice_loss<B: Backend>(
    logits: Tensor<B, 4>,
    class_map: Tensor<B, 3, Int>,
    coverage: Tensor<B, 3, Bool>,
) -> Result<(Tensor<B, 1>, Tensor<B, 1>), &'static str> {
    let one_to_many = semantic_bce_dice_loss(logits, class_map, coverage)?;
    let one_to_one = one_to_many.clone().detach();
    Ok((one_to_many, one_to_one))
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Autodiff,
        tensor::{Bool, Int, Tensor, TensorData},
    };
    use burn_flex::Flex;

    use super::{
        MaskMatch, bilinear_upsample_2x, dual_semantic_bce_dice_loss, instance_mask_loss,
        semantic_bce_dice_loss,
    };
    use crate::training::{geometry::BoxXyxy, loss::common::bce_with_logits};

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

    #[test]
    fn dual_semantic_loss_reuses_value_but_detaches_one_to_one() {
        type B = Autodiff<Flex>;

        let device = Default::default();
        let logits =
            Tensor::<B, 4>::from_floats([[[[0.5, -0.25], [1.0, -2.0]]]], &device).require_grad();
        let class_map = Tensor::<B, 3, Int>::from_ints([[[0, 0], [0, 0]]], &device);
        let coverage =
            Tensor::<B, 3, Bool>::from_bool([[[true, false], [true, false]]].into(), &device);

        let (one_to_many, one_to_one) =
            dual_semantic_bce_dice_loss(logits.clone(), class_map.clone(), coverage.clone())
                .unwrap();
        let values =
            Tensor::cat(vec![one_to_many.clone().detach(), one_to_one.clone()], 0).into_data();
        let values = values.as_slice::<f32>().unwrap();
        assert_eq!(values[0], values[1]);

        // The detached copy contributes its value but must not double the connected gradient.
        let dual_gradient = logits
            .grad(&(one_to_many + one_to_one).backward())
            .unwrap()
            .into_data();
        let reference_logits =
            Tensor::<B, 4>::from_floats([[[[0.5, -0.25], [1.0, -2.0]]]], &device).require_grad();
        let reference_loss =
            semantic_bce_dice_loss(reference_logits.clone(), class_map, coverage).unwrap();
        let reference_gradient = reference_logits
            .grad(&reference_loss.backward())
            .unwrap()
            .into_data();
        for (actual, expected) in dual_gradient
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .zip(reference_gradient.as_slice::<f32>().unwrap())
        {
            assert!((actual - expected).abs() < 1e-7, "{actual} != {expected}");
        }
    }

    #[test]
    fn batched_instance_masks_match_scalar_reference_and_backpropagate() {
        type B = Autodiff<Flex>;

        let device = Default::default();
        let coefficients_data = vec![
            0.1, 0.2, 0.3, // batch 0, prototype channel 0
            0.4, 0.5, 0.6, // batch 0, prototype channel 1
            -0.1, -0.2, -0.3, // batch 1, prototype channel 0
            0.7, 0.8, 0.9, // batch 1, prototype channel 1
        ];
        let prototypes_data = vec![
            0.1, 0.2, 0.3, 0.4, // batch 0, channel 0
            0.5, 0.6, 0.7, 0.8, // batch 0, channel 1
            -0.1, 0.2, -0.3, 0.4, // batch 1, channel 0
            0.6, -0.5, 0.4, -0.3, // batch 1, channel 1
        ];
        let targets_data = vec![
            1.0, 0.0, 1.0, 0.0, // batch 0, target 0
            0.0, 1.0, 0.0, 1.0, // batch 0, target 1
            1.0, 1.0, 0.0, 0.0, // batch 1, target 0
            0.0, 0.0, 1.0, 1.0, // batch 1, target 1
        ];
        let matches = [
            MaskMatch {
                batch_index: 0,
                anchor_index: 1,
                target_index: 0,
                // Fractional lower and upper edges exercise Ultralytics' `r >= x1 && r < x2`
                // crop geometry: only column one is inside this box at width two.
                normalized_box: BoxXyxy::new([0.1, 0.0, 0.9, 1.0]).unwrap(),
            },
            MaskMatch {
                batch_index: 0,
                anchor_index: 1,
                target_index: 1,
                normalized_box: BoxXyxy::new([0.5, 0.0, 1.0, 1.0]).unwrap(),
            },
            MaskMatch {
                batch_index: 1,
                anchor_index: 2,
                target_index: 0,
                normalized_box: BoxXyxy::new([0.0, 0.0, 1.0, 1.0]).unwrap(),
            },
        ];
        let coefficients = Tensor::<B, 3>::from_data(
            TensorData::new(coefficients_data.clone(), [2, 2, 3]),
            &device,
        )
        .require_grad();
        let prototypes = Tensor::<B, 4>::from_data(
            TensorData::new(prototypes_data.clone(), [2, 2, 2, 2]),
            &device,
        )
        .require_grad();
        let targets =
            Tensor::<B, 4>::from_data(TensorData::new(targets_data.clone(), [2, 2, 2, 2]), &device);

        let loss = instance_mask_loss(coefficients.clone(), prototypes.clone(), targets, &matches)
            .unwrap();
        let actual = loss.clone().into_data().as_slice::<f32>().unwrap()[0];

        let mut expected = 0.0;
        for matched in matches {
            let area = (matched.normalized_box.xmax - matched.normalized_box.xmin)
                * (matched.normalized_box.ymax - matched.normalized_box.ymin);
            let xmin = (matched.normalized_box.xmin * 2.0).ceil() as usize;
            let xmax = (matched.normalized_box.xmax * 2.0).ceil() as usize;
            let ymin = (matched.normalized_box.ymin * 2.0).ceil() as usize;
            let ymax = (matched.normalized_box.ymax * 2.0).ceil() as usize;
            let mut matched_loss = 0.0;
            for y in ymin..ymax {
                for x in xmin..xmax {
                    let pixel = y * 2 + x;
                    let logit = (0..2)
                        .map(|channel| {
                            coefficients_data
                                [(matched.batch_index * 2 + channel) * 3 + matched.anchor_index]
                                * prototypes_data[(matched.batch_index * 2 + channel) * 4 + pixel]
                        })
                        .sum::<f32>();
                    let target =
                        targets_data[(matched.batch_index * 2 + matched.target_index) * 4 + pixel];
                    matched_loss += bce_with_logits(logit, target);
                }
            }
            expected += matched_loss / (4.0 * area);
        }
        expected /= matches.len() as f32;
        assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");

        let gradients = loss.backward();
        for gradient in [
            coefficients.grad(&gradients).unwrap().into_data(),
            prototypes.grad(&gradients).unwrap().into_data(),
        ] {
            let values = gradient.as_slice::<f32>().unwrap();
            assert!(values.iter().all(|value| value.is_finite()));
            assert!(values.iter().any(|value| value.abs() > 0.0));
        }
    }
}
