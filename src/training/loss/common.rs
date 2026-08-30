use std::collections::BTreeMap;

use burn::tensor::{Tensor, activation::log_sigmoid, backend::Backend};

/// Differentiable scalar plus detached diagnostics returned by every native criterion.
///
/// Assignment is deliberately allowed to synchronize detached predictions to the host, but the
/// `total` tensor always remains connected to the original model output.
pub struct LossOutput<B: Backend> {
    pub total: Tensor<B, 1>,
    /// Host diagnostic captured by the criterion's single scalar synchronization.
    pub total_value: f32,
    /// A component that can be read with the other deferred totals once the epoch is complete.
    pub deferred_component: Option<&'static str>,
    pub components: BTreeMap<String, f32>,
    pub targets: usize,
    pub foreground: usize,
    pub finite: bool,
}

pub fn scalar_value<B: Backend>(value: Tensor<B, 1>) -> f32 {
    value
        .detach()
        .into_data()
        .as_slice::<f32>()
        .expect("loss scalar must use f32 storage")[0]
}

/// Read several detached scalar diagnostics with a single backend synchronization.
pub fn scalar_values<B: Backend, const N: usize>(values: [Tensor<B, 1>; N]) -> [f32; N] {
    let values = Tensor::cat(
        values.into_iter().map(Tensor::detach).collect::<Vec<_>>(),
        0,
    )
    .into_data();
    let values = values
        .as_slice::<f32>()
        .expect("loss scalars must use f32 storage");
    assert_eq!(values.len(), N, "loss diagnostic count must be preserved");
    std::array::from_fn(|index| values[index])
}

pub fn connected_zero<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, 1> {
    tensor.sum() * 0.0
}

/// Numerically stable binary cross entropy from a raw logit.
pub fn bce_with_logits(logit: f32, target: f32) -> f32 {
    logit.max(0.0) - logit * target + (-logit.abs()).exp().ln_1p()
}

/// Elementwise differentiable BCE-with-logits for Burn training graphs.
pub fn bce_with_logits_tensor<B: Backend, const D: usize>(
    logits: Tensor<B, D>,
    targets: Tensor<B, D>,
) -> Tensor<B, D> {
    (targets.neg() + 1.0) * logits.clone() - log_sigmoid(logits)
}

pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum = logits
        .iter()
        .map(|value| (*value - max).exp())
        .sum::<f32>()
        .ln();
    logits.iter().map(|value| *value - max - log_sum).collect()
}

pub fn cross_entropy(logits: &[f32], class: usize) -> Result<f32, &'static str> {
    if class >= logits.len() {
        return Err("class index outside logits");
    }
    Ok(-log_softmax(logits)[class])
}

pub fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extreme_logits_remain_finite() {
        for logit in [-100.0, 100.0] {
            for target in [0.0, 1.0] {
                assert!(bce_with_logits(logit, target).is_finite());
            }
        }
        assert!(cross_entropy(&[-100.0, 100.0], 0).unwrap().is_finite());
    }

    #[test]
    fn tensor_bce_matches_scalar_reference() {
        use burn::tensor::Tensor;
        use burn_flex::Flex;

        let device = Default::default();
        let logits = [-100.0, -2.0, 0.0, 2.0, 100.0];
        let targets = [0.0, 1.0, 0.0, 1.0, 1.0];
        let actual = bce_with_logits_tensor(
            Tensor::<Flex, 1>::from_floats(logits, &device),
            Tensor::<Flex, 1>::from_floats(targets, &device),
        )
        .into_data();
        let actual = actual.as_slice::<f32>().unwrap();
        for ((logit, target), actual) in logits.into_iter().zip(targets).zip(actual) {
            let expected = bce_with_logits(logit, target);
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }
}
