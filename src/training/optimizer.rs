use std::collections::{BTreeMap, BTreeSet};

use burn::{
    module::{Module, ModuleMapper, Param},
    optim::{AdaptiveMomentumState, ModuleOptimizer, Optimizer, RecordState},
    tensor::{Device, Tensor},
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ParameterRole {
    ConvWeight,
    LinearWeight,
    NormalizationScale,
    NormalizationBias,
    Bias,
    OtherNoDecay,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterDescriptor {
    pub key: String,
    pub shape: Vec<usize>,
    pub role: ParameterRole,
    pub trainable: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ParameterGroup {
    Decay,
    NoDecay,
    Frozen,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterGroupManifest {
    pub parameters: BTreeMap<String, ParameterGroup>,
    pub elements: BTreeMap<String, usize>,
}

pub fn classify_parameters(
    descriptors: &[ParameterDescriptor],
) -> Result<ParameterGroupManifest, &'static str> {
    let mut seen = BTreeSet::new();
    let mut parameters = BTreeMap::new();
    let mut elements = BTreeMap::<String, usize>::new();
    for descriptor in descriptors {
        if descriptor.key.is_empty() || !seen.insert(&descriptor.key) {
            return Err("parameter keys must be non-empty and unique");
        }
        let group = if !descriptor.trainable {
            ParameterGroup::Frozen
        } else if matches!(
            descriptor.role,
            ParameterRole::ConvWeight | ParameterRole::LinearWeight
        ) {
            ParameterGroup::Decay
        } else {
            ParameterGroup::NoDecay
        };
        let count = descriptor
            .shape
            .iter()
            .try_fold(1_usize, |total, value| total.checked_mul(*value))
            .ok_or("parameter element count overflow")?;
        *elements.entry(format!("{group:?}")).or_default() += count;
        parameters.insert(descriptor.key.clone(), group);
    }
    Ok(ParameterGroupManifest {
        parameters,
        elements,
    })
}

/// Apply decoupled weight decay only to matrix/kernel parameters (`D >= 2`). Burn modules model
/// convolution and linear weights with rank two or greater, while biases and normalization scales
/// are rank one. This keeps AdamW decay off BN and bias tensors without graph-specific key lists.
pub fn apply_selective_weight_decay<M: Module>(model: M, learning_rate: f64, penalty: f64) -> M {
    if penalty == 0.0 {
        return model;
    }
    struct Decay {
        factor: f64,
    }
    impl ModuleMapper for Decay {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
            let (id, value, mapper) = param.consume();
            let value = if D >= 2 { value * self.factor } else { value };
            Param::from_mapped_value(id, value, mapper)
        }
    }
    model.map(&mut Decay {
        factor: 1.0 - learning_rate * penalty,
    })
}

/// AdamW with PyTorch's epsilon and Ultralytics-style selective decay. Burn's stock AdamW applies
/// decay to every parameter, while Ultralytics excludes rank-one normalization and bias tensors.
#[derive(Clone)]
pub struct SelectiveAdamW {
    weight_decay: f32,
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,
}

#[derive(RecordState, Clone)]
pub struct SelectiveAdamWState<const D: usize> {
    momentum: AdaptiveMomentumState<D>,
}

impl Optimizer for SelectiveAdamW {
    type State<const D: usize> = SelectiveAdamWState<D>;

    fn step<const D: usize>(
        &self,
        learning_rate: f64,
        tensor: Tensor<D>,
        grad: Tensor<D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>) {
        let factor_1 = 1.0 - self.beta_1;
        let factor_2 = 1.0 - self.beta_2;
        let momentum = if let Some(mut state) = state.map(|state| state.momentum) {
            state.moment_1 = state.moment_1 * self.beta_1 + grad.clone() * factor_1;
            state.moment_2 = state.moment_2 * self.beta_2 + grad.square() * factor_2;
            state.time += 1;
            state
        } else {
            AdaptiveMomentumState {
                time: 1,
                moment_1: grad.clone() * factor_1,
                moment_2: grad.square() * factor_2,
                max_moment_2: None,
            }
        };
        let time = momentum.time as i32;
        let moment_1 = momentum.moment_1.clone() / (1.0 - self.beta_1.powi(time));
        let moment_2 = momentum.moment_2.clone() / (1.0 - self.beta_2.powi(time));
        let update = moment_1 / (moment_2.sqrt() + self.epsilon);
        let tensor = if D >= 2 && self.weight_decay != 0.0 {
            tensor * (1.0 - learning_rate * f64::from(self.weight_decay))
        } else {
            tensor
        };
        (
            tensor - update * learning_rate,
            Some(SelectiveAdamWState { momentum }),
        )
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device) -> Self::State<D> {
        state.momentum = state.momentum.to_device(device);
        state
    }
}

pub fn selective_adamw(weight_decay: f32, gradient_clip: f32) -> ModuleOptimizer {
    use burn::grad_clipping::GradientClipping;

    ModuleOptimizer::from(SelectiveAdamW {
        weight_decay,
        beta_1: 0.9,
        beta_2: 0.999,
        epsilon: 1e-8,
    })
    .with_grad_clipping(GradientClipping::Norm(gradient_clip))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::nn::LinearConfig;

    #[test]
    fn decay_is_role_based_and_exclusive() {
        let manifest = classify_parameters(&[
            ParameterDescriptor {
                key: "conv.weight".into(),
                shape: vec![4, 3, 3, 3],
                role: ParameterRole::ConvWeight,
                trainable: true,
            },
            ParameterDescriptor {
                key: "bn.gamma".into(),
                shape: vec![4],
                role: ParameterRole::NormalizationScale,
                trainable: true,
            },
            ParameterDescriptor {
                key: "head.bias".into(),
                shape: vec![3],
                role: ParameterRole::Bias,
                trainable: true,
            },
        ])
        .unwrap();
        assert_eq!(manifest.parameters["conv.weight"], ParameterGroup::Decay);
        assert_eq!(manifest.parameters["bn.gamma"], ParameterGroup::NoDecay);
        assert_eq!(manifest.parameters.len(), 3);
    }

    #[test]
    fn selective_decay_changes_weight_but_not_bias() {
        let device = Default::default();
        let model = LinearConfig::new(2, 2).with_bias(true).init(&device);
        let zeros = Tensor::<2>::zeros([1, 2], &device);
        let ones = Tensor::<2>::ones([1, 2], &device);
        let before_bias_only = model.forward(zeros.clone()).into_data();
        let before_with_weight = model.forward(ones.clone()).into_data();
        let model = apply_selective_weight_decay(model, 0.1, 0.5);
        let after_bias_only = model.forward(zeros).into_data();
        let after_with_weight = model.forward(ones).into_data();
        assert_eq!(before_bias_only, after_bias_only);
        assert_ne!(before_with_weight, after_with_weight);
    }

    #[test]
    fn selective_adamw_decays_matrices_but_not_vectors() {
        let device = Default::default();
        let optimizer = SelectiveAdamW {
            weight_decay: 0.5,
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 1e-8,
        };
        let matrix = Tensor::<2>::ones([1, 1], &device);
        let matrix_grad = Tensor::<2>::ones([1, 1], &device);
        let vector = Tensor::<1>::ones([1], &device);
        let vector_grad = Tensor::<1>::ones([1], &device);
        let matrix = optimizer.step(0.1, matrix, matrix_grad, None).0;
        let vector = optimizer.step(0.1, vector, vector_grad, None).0;
        let matrix = matrix.into_data().as_slice::<f32>().unwrap()[0];
        let vector = vector.into_data().as_slice::<f32>().unwrap()[0];
        assert!((matrix - 0.85).abs() < 1e-5);
        assert!((vector - 0.9).abs() < 1e-5);
    }
}
