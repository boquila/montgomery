use super::common::log_softmax;

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
