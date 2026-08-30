use std::collections::{BTreeMap, BTreeSet};

use burn::{
    module::{Module, ModuleMapper, Param},
    tensor::{Tensor, backend::Backend},
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
pub fn apply_selective_weight_decay<B: Backend, M: Module<B>>(
    model: M,
    learning_rate: f64,
    penalty: f64,
) -> M {
    if penalty == 0.0 {
        return model;
    }
    struct Decay {
        factor: f64,
    }
    impl<B: Backend> ModuleMapper<B> for Decay {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
            let (id, value, mapper) = param.consume();
            let value = if D >= 2 { value * self.factor } else { value };
            Param::from_mapped_value(id, value, mapper)
        }
    }
    model.map(&mut Decay {
        factor: 1.0 - learning_rate * penalty,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::nn::LinearConfig;
    use burn_flex::Flex;

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
        let model = LinearConfig::new(2, 2)
            .with_bias(true)
            .init::<Flex>(&device);
        let before_weight = model.weight.val().into_data();
        let before_bias = model.bias.as_ref().unwrap().val().into_data();
        let model = apply_selective_weight_decay(model, 0.1, 0.5);
        assert_ne!(before_weight, model.weight.val().into_data());
        assert_eq!(before_bias, model.bias.as_ref().unwrap().val().into_data());
    }
}
