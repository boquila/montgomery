use serde::{Deserialize, Serialize};

use burn::{
    module::{Module, ModuleMapper, ModuleVisitor, Param},
    optim::GradientsParams,
    tensor::Tensor,
};

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

/// Copy a training model into an EMA model that owns its BatchNorm running statistics.
///
/// `Clone` shares each `RunningState`'s storage, so a cloned EMA would read and overwrite the
/// live model's statistics instead of averaging them. `Module::train` rebuilds every running
/// state with the same parameter ID and separate storage, and keeps parameter values and IDs.
pub fn init_model<M: Module>(model: &M) -> M {
    model.clone().train()
}

/// Update every floating model parameter and running buffer with one warm-decay EMA step.
///
/// Tensors are paired by stable parameter ID through Burn's dimension-erased parameter container,
/// then blended on the backend device. This covers trainable parameters and BN running state
/// without synchronizing every tensor to the host.
pub fn update_model<M>(ema: M, current: &M, state: &mut EmaState) -> Result<M, &'static str>
where
    M: Module,
{
    struct Collector {
        values: GradientsParams,
    }
    impl ModuleVisitor for Collector {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            self.values.register(param.id, param.val());
        }
    }
    let mut collector = Collector {
        values: GradientsParams::new(),
    };
    current.visit(&mut collector);
    let decay = state.next_decay() as f32;
    struct EmaMapper {
        current: GradientsParams,
        decay: f32,
        error: Option<&'static str>,
    }
    impl ModuleMapper for EmaMapper {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
            let (id, tensor, mapper) = param.consume();
            let Some(current) = self.current.remove::<D>(id) else {
                self.error = Some("EMA model parameter IDs differ from current model");
                return Param::from_mapped_value(id, tensor, mapper);
            };
            if current.dims() != tensor.dims() {
                self.error = Some("EMA and current parameter shapes differ");
                return Param::from_mapped_value(id, tensor, mapper);
            }
            let tensor = tensor.detach() * self.decay + current.detach() * (1.0 - self.decay);
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
        impl ModuleMapper for AddOne {
            fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
                let (id, value, mapper) = param.consume();
                Param::from_mapped_value(id, value + 1.0, mapper)
            }
        }
        let device = burn::tensor::Device::flex();
        let initial = LinearConfig::new(2, 2).init(&device);
        let current = initial.clone().map(&mut AddOne);
        let mut state = EmaState::new(0.9999).unwrap();
        let ema = update_model(initial.clone(), &current, &mut state).unwrap();
        let input = Tensor::<2>::ones([1, 2], &device);
        let before = initial.forward(input.clone()).into_data();
        let after = ema.forward(input.clone()).into_data();
        let target = current.forward(input).into_data();
        assert_ne!(before, after);
        let after = after.as_slice::<f32>().unwrap();
        let target = target.as_slice::<f32>().unwrap();
        assert!(after.iter().zip(target).all(|(a, b)| (a - b).abs() < 5e-3));
        assert_eq!(state.updates, 1);
    }

    #[test]
    fn ema_model_owns_and_averages_batch_norm_statistics() {
        use burn::{nn::BatchNormConfig, tensor::Device};

        let device = Device::flex().autodiff();
        let live = BatchNormConfig::new(2).init(&device);
        let ema = init_model(&live);
        let input = Tensor::<4>::from_floats(
            [[[[1.0, 3.0]], [[-2.0, 6.0]]], [[[5.0, -1.0]], [[0.0, 4.0]]]],
            &device,
        );
        let _ = live.forward(input).into_data();
        let mean = |model: &burn::nn::BatchNorm| {
            model
                .running_mean
                .value_sync()
                .into_data()
                .try_into_vec::<f32>()
                .unwrap()
        };
        let live_mean = mean(&live);
        assert_ne!(
            live_mean,
            mean(&ema),
            "EMA shares the live model's running statistics"
        );

        let mut state = EmaState::new(0.9999).unwrap();
        let decay = state.clone().next_decay() as f32;
        let ema = update_model(ema, &live, &mut state).unwrap();
        // The EMA started from zero running means, so one step moves it by `1 - decay`.
        for (ema, live) in mean(&ema).iter().zip(&live_mean) {
            assert!((ema - live * (1.0 - decay)).abs() < 1e-6, "{ema} vs {live}");
        }
    }
}
