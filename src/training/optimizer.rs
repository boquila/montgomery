use std::collections::{BTreeMap, BTreeSet};

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
