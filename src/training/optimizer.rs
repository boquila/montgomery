use std::collections::{BTreeMap, BTreeSet};

use burn::{
    grad_clipping::{GradientClipping, GradientClippingConfig},
    module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param},
    optim::{
        AdaptiveMomentumState, GradientsParams, MultiGradientsParams, Optimizer, SimpleOptimizer,
    },
    record::Record,
    tensor::{
        Tensor,
        backend::{AutodiffBackend, Backend},
        ops::Device,
    },
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

/// AdamW with PyTorch's epsilon and Ultralytics-style selective decay. Burn's stock AdamW applies
/// decay to every parameter, while Ultralytics excludes rank-one normalization and bias tensors.
#[derive(Clone)]
pub struct SelectiveAdamW {
    weight_decay: f32,
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,
}

#[derive(Record, Clone)]
pub struct SelectiveAdamWState<B: Backend, const D: usize> {
    momentum: AdaptiveMomentumState<B, D>,
}

impl<B: Backend> SimpleOptimizer<B> for SelectiveAdamW {
    type State<const D: usize> = SelectiveAdamWState<B, D>;

    fn step<const D: usize>(
        &self,
        learning_rate: f64,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
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

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device<B>) -> Self::State<D> {
        state.momentum = state.momentum.to_device(device);
        state
    }
}

pub fn selective_adamw<B, M>(weight_decay: f32, gradient_clip: f32) -> FlatSelectiveAdamW<B>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    FlatSelectiveAdamW {
        weight_decay,
        beta_1: 0.9,
        beta_2: 0.999,
        epsilon: 1e-8,
        clipping: GradientClippingConfig::Norm(gradient_clip).init(),
        state: None,
    }
}

/// AdamW state packed into decay and no-decay groups.
///
/// Packing eliminates one optimizer launch and one checkpoint readback per parameter while the
/// split keeps Ultralytics' selective weight decay exact.
#[derive(Record, Clone)]
pub struct FlatSelectiveAdamWState<B: Backend> {
    decay_moment_1: Option<Tensor<B, 1>>,
    decay_moment_2: Option<Tensor<B, 1>>,
    no_decay_moment_1: Option<Tensor<B, 1>>,
    no_decay_moment_2: Option<Tensor<B, 1>>,
    time: u64,
}

#[derive(Clone)]
pub struct FlatSelectiveAdamW<B: AutodiffBackend> {
    weight_decay: f32,
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,
    clipping: GradientClipping,
    state: Option<FlatSelectiveAdamWState<B>>,
}

struct FlatGroups<B: Backend> {
    decay_params: Vec<Tensor<B, 1>>,
    decay_grads: Vec<Tensor<B, 1>>,
    no_decay_params: Vec<Tensor<B, 1>>,
    no_decay_grads: Vec<Tensor<B, 1>>,
}

type FlatPair<B> = (Tensor<B, 1>, Tensor<B, 1>);
type UpdatedFlatGroup<B> = (
    Option<Tensor<B, 1>>,
    Option<Tensor<B, 1>>,
    Option<Tensor<B, 1>>,
);

impl<B: Backend> FlatGroups<B> {
    fn new() -> Self {
        Self {
            decay_params: Vec::new(),
            decay_grads: Vec::new(),
            no_decay_params: Vec::new(),
            no_decay_grads: Vec::new(),
        }
    }

    fn concatenate(params: Vec<Tensor<B, 1>>, grads: Vec<Tensor<B, 1>>) -> Option<FlatPair<B>> {
        (!params.is_empty()).then(|| (Tensor::cat(params, 0), Tensor::cat(grads, 0)))
    }
}

struct FlatGroupCollector<'a, B: AutodiffBackend> {
    grads: &'a GradientsParams,
    clipping: &'a GradientClipping,
    groups: FlatGroups<B::InnerBackend>,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for FlatGroupCollector<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let Some(grad) = self.grads.get::<B::InnerBackend, D>(param.id) else {
            return;
        };
        let elements = grad.shape().num_elements();
        let grad = self.clipping.clip_gradient(grad).reshape([elements]);
        let value = param.val().inner().reshape([elements]);
        if D >= 2 {
            self.groups.decay_params.push(value);
            self.groups.decay_grads.push(grad);
        } else {
            self.groups.no_decay_params.push(value);
            self.groups.no_decay_grads.push(grad);
        }
    }
}

struct FlatGroupMapper<'a, B: AutodiffBackend> {
    grads: &'a GradientsParams,
    decay: Option<Tensor<B::InnerBackend, 1>>,
    no_decay: Option<Tensor<B::InnerBackend, 1>>,
    decay_offset: usize,
    no_decay_offset: usize,
}

impl<B: AutodiffBackend> ModuleMapper<B> for FlatGroupMapper<'_, B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        if self.grads.get::<B::InnerBackend, D>(id).is_none() {
            return Param::from_mapped_value(id, tensor, mapper);
        }
        let elements = tensor.shape().num_elements();
        let (flat, offset) = if D >= 2 {
            (self.decay.as_ref().unwrap(), &mut self.decay_offset)
        } else {
            (self.no_decay.as_ref().unwrap(), &mut self.no_decay_offset)
        };
        let updated = flat
            .clone()
            .slice(*offset..*offset + elements)
            .reshape(tensor.shape());
        *offset += elements;
        let requires_grad = tensor.is_require_grad();
        let mut updated = Tensor::from_inner(updated);
        if requires_grad {
            updated = updated.require_grad();
        }
        Param::from_mapped_value(id, updated, mapper)
    }
}

impl<B: AutodiffBackend> FlatSelectiveAdamW<B> {
    fn update_group(
        &self,
        learning_rate: f64,
        params_and_grads: Option<FlatPair<B::InnerBackend>>,
        moment_1: Option<Tensor<B::InnerBackend, 1>>,
        moment_2: Option<Tensor<B::InnerBackend, 1>>,
        time: u64,
        decay: bool,
    ) -> UpdatedFlatGroup<B::InnerBackend> {
        let Some((params, grad)) = params_and_grads else {
            return (None, None, None);
        };
        let moment_1 = moment_1.map_or_else(
            || grad.clone() * (1.0 - self.beta_1),
            |value| value * self.beta_1 + grad.clone() * (1.0 - self.beta_1),
        );
        let moment_2 = moment_2.map_or_else(
            || grad.clone().square() * (1.0 - self.beta_2),
            |value| value * self.beta_2 + grad.clone().square() * (1.0 - self.beta_2),
        );
        let correction_1 = 1.0 - self.beta_1.powi(time as i32);
        let correction_2 = 1.0 - self.beta_2.powi(time as i32);
        let update = (moment_1.clone() / correction_1)
            / ((moment_2.clone() / correction_2).sqrt() + self.epsilon);
        let params = if decay && self.weight_decay != 0.0 {
            params * (1.0 - learning_rate * f64::from(self.weight_decay))
        } else {
            params
        };
        (
            Some(params - update * learning_rate),
            Some(moment_1),
            Some(moment_2),
        )
    }

    fn step_impl<M: AutodiffModule<B>>(
        &mut self,
        learning_rate: f64,
        module: M,
        grads: GradientsParams,
    ) -> M {
        let mut collector = FlatGroupCollector::<B> {
            grads: &grads,
            clipping: &self.clipping,
            groups: FlatGroups::new(),
        };
        module.visit(&mut collector);
        let decay =
            FlatGroups::concatenate(collector.groups.decay_params, collector.groups.decay_grads);
        let no_decay = FlatGroups::concatenate(
            collector.groups.no_decay_params,
            collector.groups.no_decay_grads,
        );
        let previous = self.state.take();
        let time = previous.as_ref().map_or(1, |state| state.time + 1);
        let (decay, decay_moment_1, decay_moment_2) = self.update_group(
            learning_rate,
            decay,
            previous
                .as_ref()
                .and_then(|state| state.decay_moment_1.clone().map(Tensor::inner)),
            previous
                .as_ref()
                .and_then(|state| state.decay_moment_2.clone().map(Tensor::inner)),
            time,
            true,
        );
        let (no_decay, no_decay_moment_1, no_decay_moment_2) = self.update_group(
            learning_rate,
            no_decay,
            previous
                .as_ref()
                .and_then(|state| state.no_decay_moment_1.clone().map(Tensor::inner)),
            previous
                .as_ref()
                .and_then(|state| state.no_decay_moment_2.clone().map(Tensor::inner)),
            time,
            false,
        );
        self.state = Some(FlatSelectiveAdamWState {
            decay_moment_1: decay_moment_1.map(Tensor::from_inner),
            decay_moment_2: decay_moment_2.map(Tensor::from_inner),
            no_decay_moment_1: no_decay_moment_1.map(Tensor::from_inner),
            no_decay_moment_2: no_decay_moment_2.map(Tensor::from_inner),
            time,
        });
        module.map(&mut FlatGroupMapper::<B> {
            grads: &grads,
            decay,
            no_decay,
            decay_offset: 0,
            no_decay_offset: 0,
        })
    }
}

impl<B, M> Optimizer<M, B> for FlatSelectiveAdamW<B>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    type Record = Option<FlatSelectiveAdamWState<B>>;

    fn step(&mut self, lr: f64, module: M, grads: GradientsParams) -> M {
        self.step_impl(lr, module, grads)
    }

    fn step_multi(&mut self, lr: f64, module: M, mut grads: MultiGradientsParams) -> M {
        struct Merge<'a> {
            source: &'a mut MultiGradientsParams,
            merged: GradientsParams,
        }
        impl<B: AutodiffBackend> ModuleVisitor<B> for Merge<'_> {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                if let Some((gradient, _)) = self.source.remove::<B::InnerBackend, D>(param.id) {
                    self.merged.register(param.id, gradient);
                }
            }
        }
        let mut merge = Merge {
            source: &mut grads,
            merged: GradientsParams::new(),
        };
        module.visit(&mut merge);
        self.step_impl(lr, module, merge.merged)
    }

    fn to_record(&self) -> Self::Record {
        self.state.clone()
    }

    fn load_record(mut self, record: Self::Record) -> Self {
        self.state = record;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        backend::Autodiff, module::Initializer, nn::LinearConfig, optim::adaptor::OptimizerAdaptor,
    };
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

    #[test]
    fn selective_adamw_decays_matrices_but_not_vectors() {
        use burn::optim::SimpleOptimizer;

        let device = Default::default();
        let optimizer = SelectiveAdamW {
            weight_decay: 0.5,
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 1e-8,
        };
        let matrix = Tensor::<Flex, 2>::ones([1, 1], &device);
        let matrix_grad = Tensor::<Flex, 2>::ones([1, 1], &device);
        let vector = Tensor::<Flex, 1>::ones([1], &device);
        let vector_grad = Tensor::<Flex, 1>::ones([1], &device);
        let matrix = optimizer.step(0.1, matrix, matrix_grad, None).0;
        let vector = optimizer.step(0.1, vector, vector_grad, None).0;
        let matrix = matrix.into_data().as_slice::<f32>().unwrap()[0];
        let vector = vector.into_data().as_slice::<f32>().unwrap()[0];
        assert!((matrix - 0.85).abs() < 1e-5);
        assert!((vector - 0.9).abs() < 1e-5);
    }

    #[test]
    fn flat_adamw_matches_per_parameter_adamw_and_restores_state() {
        type B = Autodiff<Flex>;
        type Model = burn::nn::Linear<B>;

        fn step<O: Optimizer<Model, B>>(mut optimizer: O, model: Model) -> (O, Model) {
            let device = Default::default();
            let input = Tensor::<B, 2>::from_floats([[0.25, -0.5], [1.0, 0.75]], &device);
            let gradients = model.forward(input).square().sum().backward();
            let gradients = GradientsParams::from_grads(gradients, &model);
            let model = optimizer.step(0.01, model, gradients);
            (optimizer, model)
        }

        let device = Default::default();
        let mut reference: OptimizerAdaptor<SelectiveAdamW, Model, B> =
            OptimizerAdaptor::from(SelectiveAdamW {
                weight_decay: 0.0005,
                beta_1: 0.9,
                beta_2: 0.999,
                epsilon: 1e-8,
            })
            .with_grad_clipping(GradientClippingConfig::Norm(10.0).init());
        let mut flat = selective_adamw::<B, Model>(0.0005, 10.0);
        let mut reference_model = LinearConfig::new(2, 3)
            .with_bias(true)
            .with_initializer(Initializer::Constant { value: 0.25 })
            .init::<B>(&device);
        let mut flat_model = LinearConfig::new(2, 3)
            .with_bias(true)
            .with_initializer(Initializer::Constant { value: 0.25 })
            .init::<B>(&device);
        for _ in 0..2 {
            (reference, reference_model) = step(reference, reference_model);
            (flat, flat_model) = step(flat, flat_model);
        }
        let record = <FlatSelectiveAdamW<B> as Optimizer<Model, B>>::to_record(&flat);
        let restored = <FlatSelectiveAdamW<B> as Optimizer<Model, B>>::load_record(
            selective_adamw::<B, Model>(0.0005, 10.0),
            record,
        );
        (_, reference_model) = step(reference, reference_model);
        (_, flat_model) = step(restored, flat_model);

        let weight_delta = (reference_model.weight.val() - flat_model.weight.val())
            .abs()
            .max()
            .into_data()
            .as_slice::<f32>()
            .unwrap()[0];
        let bias_delta = (reference_model.bias.unwrap().val() - flat_model.bias.unwrap().val())
            .abs()
            .max()
            .into_data()
            .as_slice::<f32>()
            .unwrap()[0];
        assert!(weight_delta < 2e-6, "weight delta {weight_delta}");
        assert!(bias_delta < 2e-6, "bias delta {bias_delta}");
    }
}
