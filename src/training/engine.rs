use std::{error::Error, fmt};

use burn::{
    module::AutodiffModule,
    optim::{GradientsAccumulator, GradientsParams, Optimizer},
    tensor::backend::AutodiffBackend,
};

use crate::training::{
    TrainingConfig,
    loss::common::LossOutput,
    report::{RunDirectory, StepEvent},
    scheduler::LrScheduler,
    state::TrainingState,
};

/// Family-specific model adapter used by the explicit native loop.
pub trait TrainableTask<B: AutodiffBackend>: AutodiffModule<B> {
    type Batch;

    fn forward_loss(
        &self,
        batch: &Self::Batch,
        context: LossContext,
    ) -> Result<LossOutput<B>, String>;
}

/// Epoch-dependent criterion switches. Keeping this value outside model modules makes restart
/// semantics explicit and lets the engine persist the exact loss recipe in [`TrainingState`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossContext {
    pub yolox_l1: bool,
    pub one_to_many: f32,
    pub one_to_one: f32,
}

impl LossContext {
    fn from_state(state: &TrainingState) -> Self {
        let (one_to_many, one_to_one) = state.dual_loss.as_ref().map_or((1.0, 1.0), |schedule| {
            (schedule.one_to_many, schedule.one_to_one)
        });
        Self {
            yolox_l1: state.yolox_l1,
            one_to_many,
            one_to_one,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochSummary {
    pub mean_loss: f32,
    pub microbatches: usize,
    pub optimizer_steps: usize,
}

/// Stateful run owner. Concrete model dispatch happens once, then [`train_epoch`] remains fully
/// monomorphized over the task, backend and optimizer.
pub struct Trainer {
    pub config: TrainingConfig,
    pub state: TrainingState,
    pub scheduler: LrScheduler,
    pub run: RunDirectory,
}

impl Trainer {
    pub fn create(
        config: TrainingConfig,
        name: &str,
        batches_per_epoch: usize,
    ) -> Result<Self, EngineError> {
        config
            .validate()
            .map_err(|error| EngineError(error.to_string()))?;
        if batches_per_epoch == 0 {
            return Err(EngineError("training split produces no batches".into()));
        }
        let optimizer_steps_per_epoch = batches_per_epoch.div_ceil(config.accumulation);
        let total_steps = optimizer_steps_per_epoch
            .checked_mul(config.epochs)
            .ok_or_else(|| EngineError("optimizer step count overflow".into()))?
            as u64;
        let scheduler = LrScheduler::new(
            config.schedule,
            config.initial_lr,
            config.final_lr_ratio,
            config.warmup_steps.min(total_steps.saturating_sub(1)),
            total_steps,
        )
        .map_err(|error| EngineError(error.into()))?;
        let mut state = TrainingState::new(config.seed);
        if matches!(
            crate::training::dispatch::recipe_for(config.model.architecture).loss,
            crate::training::dispatch::LossFamily::Yolo26DualDirect
                | crate::training::dispatch::LossFamily::Yolo26DualSegment
        ) {
            state.dual_loss = Some(crate::training::state::DualLossSchedule::yolo26(
                config.epochs,
            ));
        }
        let run = RunDirectory::create(&config, name).map_err(EngineError::io)?;
        Ok(Self {
            config,
            state,
            scheduler,
            run,
        })
    }

    pub fn from_checkpoint(checkpoint: impl AsRef<std::path::Path>) -> Result<Self, EngineError> {
        let manifest = crate::training::checkpoint::load(checkpoint.as_ref())
            .map_err(|error| EngineError(error.to_string()))?;
        let run_root = checkpoint
            .as_ref()
            .parent()
            .and_then(std::path::Path::parent)
            .ok_or_else(|| {
                EngineError("checkpoint is not inside a run checkpoints directory".into())
            })?;
        let run = RunDirectory::open(run_root).map_err(EngineError::io)?;
        Ok(Self {
            config: manifest.config,
            state: manifest.state,
            scheduler: manifest.scheduler,
            run,
        })
    }

    pub fn train_epoch<B, M, O, F>(
        &mut self,
        mut model: M,
        mut optimizer: O,
        batches: &[M::Batch],
        mut after_step: F,
    ) -> Result<(M, O, EpochSummary), EngineError>
    where
        B: AutodiffBackend,
        M: TrainableTask<B>,
        O: Optimizer<M, B>,
        F: FnMut(&M, u64) -> Result<(), String>,
    {
        if batches.is_empty() {
            return Err(EngineError("cannot train an empty epoch".into()));
        }
        let mut accumulator = GradientsAccumulator::<M>::new();
        let mut loss_sum = 0.0_f64;
        let mut optimizer_steps = 0;
        let mut group_start = 0;
        while group_start < batches.len() {
            let group_end = (group_start + self.config.accumulation).min(batches.len());
            let group_len = group_end - group_start;
            for (offset, batch) in batches[group_start..group_end].iter().enumerate() {
                let output = model
                    .forward_loss(batch, LossContext::from_state(&self.state))
                    .map_err(EngineError)?;
                let total_value = output.total_value;
                if !output.finite || !total_value.is_finite() {
                    return Err(EngineError(format!(
                        "non-finite loss at epoch {} batch {}",
                        self.state.epoch,
                        group_start + offset
                    )));
                }
                loss_sum += f64::from(total_value);
                let gradients = GradientsParams::from_grads(
                    (output.total / group_len as f64).backward(),
                    &model,
                );
                if gradients.is_empty() {
                    return Err(EngineError("loss produced no model gradients".into()));
                }
                accumulator.accumulate(&model, gradients);
                self.state.micro_step += 1;
                self.state.next_batch = group_start + offset + 1;
                self.state.accumulation_position = offset + 1;
                self.run
                    .append_event(&StepEvent {
                        epoch: self.state.epoch,
                        micro_step: self.state.micro_step,
                        optimizer_step: self.state.optimizer_step,
                        learning_rate: self.scheduler.current_lr(),
                        total_loss: total_value,
                        components: output.components,
                        targets: output.targets,
                        foreground: output.foreground,
                    })
                    .map_err(EngineError::io)?;
            }
            let lr = self.scheduler.step();
            model = optimizer.step(lr, model, accumulator.grads());
            model = crate::training::optimizer::apply_selective_weight_decay(
                model,
                lr,
                self.config.weight_decay,
            );
            self.state.optimizer_step += 1;
            self.state.accumulation_position = 0;
            after_step(&model, self.state.optimizer_step).map_err(EngineError)?;
            optimizer_steps += 1;
            group_start = group_end;
        }
        let mean_loss = (loss_sum / batches.len() as f64) as f32;
        self.run
            .append_metrics(
                self.state.epoch,
                mean_loss,
                None,
                self.scheduler.current_lr(),
            )
            .map_err(EngineError::io)?;
        self.state.epoch += 1;
        self.state.next_batch = 0;
        if let Some(schedule) = &mut self.state.dual_loss {
            schedule.complete_epoch();
        }
        self.state
            .resolve_augmentation_phase(self.config.epochs, self.config.augmentation.close_mosaic);
        if matches!(
            crate::training::dispatch::recipe_for(self.config.model.architecture).loss,
            crate::training::dispatch::LossFamily::YoloxSimOta
        ) {
            self.state.no_augmentation =
                self.state.augmentation_phase == crate::data::augmentation::PipelinePhase::Closed;
            self.state.yolox_l1 = self.state.no_augmentation;
        }
        Ok((
            model,
            optimizer,
            EpochSummary {
                mean_loss,
                microbatches: batches.len(),
                optimizer_steps,
            },
        ))
    }
}

#[derive(Debug)]
pub struct EngineError(String);

impl EngineError {
    fn io(error: std::io::Error) -> Self {
        Self(error.to_string())
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl Error for EngineError {}
