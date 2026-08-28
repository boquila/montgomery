use burn::tensor::{Tensor, backend::Backend};

/// Numerically stable binary cross entropy from a raw logit.
pub fn bce_with_logits(logit: f32, target: f32) -> f32 {
    logit.max(0.0) - logit * target + (-logit.abs()).exp().ln_1p()
}

/// Elementwise differentiable BCE-with-logits for Burn training graphs.
pub fn bce_with_logits_tensor<B: Backend, const D: usize>(
    logits: Tensor<B, D>,
    targets: Tensor<B, D>,
) -> Tensor<B, D> {
    logits.clone().clamp_min(0.0) - logits.clone() * targets + (-logits.abs()).exp().log1p()
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
}
