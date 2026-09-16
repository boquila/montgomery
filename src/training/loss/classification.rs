use std::collections::BTreeMap;

use burn::tensor::{Int, Tensor, activation};

use super::common::{DeferredScalar, LossOutput, cross_entropy};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassificationMetrics {
    pub mean_loss: f32,
    pub top1_correct: usize,
    pub top5_correct: usize,
    pub count: usize,
}

pub fn classification_loss(
    logits: &[Vec<f32>],
    classes: &[usize],
) -> Result<ClassificationMetrics, &'static str> {
    if logits.len() != classes.len() {
        return Err("logit and target batch sizes differ");
    }
    if logits.is_empty() {
        return Ok(ClassificationMetrics {
            mean_loss: 0.0,
            top1_correct: 0,
            top5_correct: 0,
            count: 0,
        });
    }
    let mut loss = 0.0;
    let mut top1 = 0;
    let mut top5 = 0;
    for (row, class) in logits.iter().zip(classes) {
        loss += cross_entropy(row, *class)?;
        let mut order: Vec<usize> = (0..row.len()).collect();
        order.sort_by(|a, b| row[*b].total_cmp(&row[*a]).then_with(|| a.cmp(b)));
        top1 += usize::from(order.first() == Some(class));
        top5 += usize::from(
            order
                .iter()
                .take(5.min(row.len()))
                .any(|item| item == class),
        );
    }
    Ok(ClassificationMetrics {
        mean_loss: loss / logits.len() as f32,
        top1_correct: top1,
        top5_correct: top5,
        count: logits.len(),
    })
}

/// Mean differentiable cross entropy from raw classifier logits.
pub fn tensor_loss(logits: Tensor<2>, classes: Tensor<1, Int>) -> Result<LossOutput, &'static str> {
    let [batch, num_classes] = logits.dims();
    if batch == 0 || num_classes == 0 || classes.dims() != [batch] {
        return Err("classification logits/classes have invalid shapes");
    }
    let targets: Tensor<2, Int> = classes.one_hot(num_classes);
    let targets = targets.float();
    let total = -(activation::log_softmax(logits, 1) * targets).sum() / batch as f64;
    Ok(LossOutput {
        deferred: vec![DeferredScalar::total_and_component(
            "classification_loss",
            total.clone(),
        )],
        total,
        total_value: 0.0,
        components: BTreeMap::new(),
        targets: batch,
        foreground: batch,
        finite: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top5_uses_available_classes() {
        let metrics = classification_loss(&[vec![0.0, 2.0, 1.0]], &[2]).unwrap();
        assert_eq!(metrics.top1_correct, 0);
        assert_eq!(metrics.top5_correct, 1);
    }
}
