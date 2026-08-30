use serde::{Deserialize, Serialize};

use burn::{
    module::{Module, ModuleMapper, ModuleVisitor, Param, ParamId},
    tensor::{Tensor, TensorData, backend::Backend},
};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmaState {
    pub updates: u64,
    pub base_decay: f64,
}

impl EmaState {
    pub fn new(base_decay: f64) -> Result<Self, &'static str> {
        if !base_decay.is_finite() || !(0.0..1.0).contains(&base_decay) {
            return Err("EMA decay must be finite and in (0, 1)");
        }
        Ok(Self {
            updates: 0,
            base_decay,
        })
    }

    /// Ultralytics/YOLOX warm EMA decay: `decay * (1 - exp(-updates / 2000))`.
    pub fn next_decay(&mut self) -> f64 {
        self.updates += 1;
        self.base_decay * (1.0 - (-(self.updates as f64) / 2000.0).exp())
    }

    pub fn update_slice(&mut self, ema: &mut [f32], current: &[f32]) -> Result<(), &'static str> {
        if ema.len() != current.len() {
            return Err("EMA and current tensor shapes differ");
        }
        if current.iter().any(|value| !value.is_finite()) {
            return Err("current parameter contains a non-finite value");
        }
        let decay = self.next_decay() as f32;
        for (ema, current) in ema.iter_mut().zip(current) {
            *ema = *ema * decay + *current * (1.0 - decay);
        }
        if ema.iter().any(|value| !value.is_finite()) {
            return Err("EMA parameter contains a non-finite value");
        }
        Ok(())
    }
}

/// Update every floating model parameter and running buffer with one warm-decay EMA step.
///
/// Burn's mapper API is dimension-generic, so the current model is first visited into host tensor
/// data keyed by stable parameter IDs and then blended while mapping the EMA clone. This explicit
/// synchronization is slower than a fused device kernel but is deterministic and covers BN
/// running state as well as trainable parameters.
pub fn update_model<B, M>(ema: M, current: &M, state: &mut EmaState) -> Result<M, &'static str>
where
    B: Backend,
    M: Module<B>,
{
    struct Collector {
        values: BTreeMap<ParamId, TensorData>,
    }
    impl<B: Backend> ModuleVisitor<B> for Collector {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            self.values.insert(param.id, param.val().into_data());
        }
    }
    let mut collector = Collector {
        values: BTreeMap::new(),
    };
    current.visit(&mut collector);
    let decay = state.next_decay() as f32;
    struct EmaMapper {
        current: BTreeMap<ParamId, TensorData>,
        decay: f32,
        error: Option<&'static str>,
    }
    impl<B: Backend> ModuleMapper<B> for EmaMapper {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
            let (id, tensor, mapper) = param.consume();
            let Some(current) = self.current.remove(&id) else {
                self.error = Some("EMA model parameter IDs differ from current model");
                return Param::from_mapped_value(id, tensor, mapper);
            };
            let Ok(current_values) = current.as_slice::<f32>() else {
                self.error = Some("EMA current parameter is not f32");
                return Param::from_mapped_value(id, tensor, mapper);
            };
            let ema_data = tensor.clone().into_data();
            let Ok(ema_values) = ema_data.as_slice::<f32>() else {
                self.error = Some("EMA parameter is not f32");
                return Param::from_mapped_value(id, tensor, mapper);
            };
            if current_values.len() != ema_values.len()
                || current_values.iter().any(|value| !value.is_finite())
            {
                self.error = Some("EMA parameter shape differs or current value is non-finite");
                return Param::from_mapped_value(id, tensor, mapper);
            }
            let values = ema_values
                .iter()
                .zip(current_values)
                .map(|(ema, current)| ema * self.decay + current * (1.0 - self.decay))
                .collect::<Vec<_>>();
            let device = tensor.device();
            let tensor = Tensor::from_data(TensorData::new(values, ema_data.shape), &device);
            Param::from_mapped_value(id, tensor, mapper)
        }
    }
    let mut mapper = EmaMapper {
        current: collector.values,
        decay,
        error: None,
    };
    let ema = ema.map(&mut mapper);
    if let Some(error) = mapper.error {
        return Err(error);
    }
    if !mapper.current.is_empty() {
        return Err("current model contains parameters absent from EMA model");
    }
    Ok(ema)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{module::ModuleMapper, nn::LinearConfig};
    use burn_flex::Flex;

    #[test]
    fn updates_once_and_persists_counter() {
        let mut state = EmaState::new(0.9999).unwrap();
        let mut ema = [0.0];
        state.update_slice(&mut ema, &[1.0]).unwrap();
        assert_eq!(state.updates, 1);
        let restored: EmaState =
            serde_json::from_slice(&serde_json::to_vec(&state).unwrap()).unwrap();
        assert_eq!(state, restored);
    }

    #[test]
    fn model_ema_updates_parameters_by_stable_id() {
        struct AddOne;
        impl ModuleMapper<Flex> for AddOne {
            fn map_float<const D: usize>(
                &mut self,
                param: Param<Tensor<Flex, D>>,
            ) -> Param<Tensor<Flex, D>> {
                let (id, value, mapper) = param.consume();
                Param::from_mapped_value(id, value + 1.0, mapper)
            }
        }
        let device = Default::default();
        let initial = LinearConfig::new(2, 2).init::<Flex>(&device);
        let current = initial.clone().map(&mut AddOne);
        let mut state = EmaState::new(0.9999).unwrap();
        let ema = update_model::<Flex, _>(initial.clone(), &current, &mut state).unwrap();
        let input = Tensor::<Flex, 2>::ones([1, 2], &device);
        let before = initial.forward(input.clone()).into_data();
        let after = ema.forward(input.clone()).into_data();
        let target = current.forward(input).into_data();
        assert_ne!(before, after);
        let after = after.as_slice::<f32>().unwrap();
        let target = target.as_slice::<f32>().unwrap();
        assert!(after.iter().zip(target).all(|(a, b)| (a - b).abs() < 5e-3));
        assert_eq!(state.updates, 1);
    }
}
