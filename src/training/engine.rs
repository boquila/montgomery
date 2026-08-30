use std::{error::Error, fmt};

use burn::{
    module::AutodiffModule,
    optim::{GradientsAccumulator, GradientsParams, Optimizer},
    tensor::{Tensor, backend::AutodiffBackend},
};

use crate::training::{
    TrainingConfig,
    loss::common::LossOutput,
    report::{RunDirectory, StepEvent},
    scheduler::LrScheduler,
    state::TrainingState,
};

const DIAGNOSTIC_CHUNK_SIZE: usize = 1024;
type DeferredDiagnostic<B> = (usize, usize, &'static str, Tensor<B, 1>);

fn diagnostic_chunk_full(events: &[StepEvent]) -> bool {
    events.len() >= DIAGNOSTIC_CHUNK_SIZE
}

/// Family-specific model adapter used by the explicit native loop.
pub trait TrainableTask<B: AutodiffBackend>: AutodiffModule<B> {
    type Batch;

    fn forward_loss(
        &self,
        batch: &Self::Batch,
        context: LossContext,
    ) -> Result<LossOutput<B>, String>;
}

/// Bounded, epoch-scoped batch producer. Implementations may decode and prefetch lazily, but must
/// yield exactly `batch_count` batches in deterministic order.
pub trait EpochBatchSource<Batch> {
    fn batch_count(&self) -> usize;
    fn next_batch(&mut self) -> Result<Option<Batch>, String>;
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

    pub fn train_epoch<B, M, O, F, S>(
        &mut self,
        mut model: M,
        mut optimizer: O,
        batches: &mut S,
        external_weight_decay: bool,
        mut after_step: F,
    ) -> Result<(M, O, EpochSummary), EngineError>
    where
        B: AutodiffBackend,
        M: TrainableTask<B>,
        O: Optimizer<M, B>,
        F: FnMut(&M, u64) -> Result<(), String>,
        S: EpochBatchSource<M::Batch>,
    {
        let batch_count = batches.batch_count();
        if batch_count == 0 {
            return Err(EngineError("cannot train an empty epoch".into()));
        }
        let mut accumulator = GradientsAccumulator::<M>::new();
        let mut loss_sum = 0.0_f64;
        let diagnostic_capacity = batch_count.min(DIAGNOSTIC_CHUNK_SIZE);
        let mut events = Vec::with_capacity(diagnostic_capacity);
        let mut deferred = Vec::<DeferredDiagnostic<B>>::with_capacity(diagnostic_capacity);
        let mut optimizer_steps = 0;
        let mut group_start = 0;
        while group_start < batch_count {
            let group_end = (group_start + self.config.accumulation).min(batch_count);
            let group_len = group_end - group_start;
            let mut group = Vec::with_capacity(group_len);
            for batch_index in group_start..group_end {
                group.push(batches.next_batch().map_err(EngineError)?.ok_or_else(|| {
                    EngineError(format!(
                        "epoch batch source ended at batch {batch_index} of {batch_count}"
                    ))
                })?);
            }
            for (offset, batch) in group.iter().enumerate() {
                let output = model
                    .forward_loss(batch, LossContext::from_state(&self.state))
                    .map_err(EngineError)?;
                let total_value = output.total_value;
                if !output.finite
                    || (output.deferred_component.is_none() && !total_value.is_finite())
                {
                    return Err(EngineError(format!(
                        "non-finite loss at epoch {} batch {}",
                        self.state.epoch,
                        group_start + offset
                    )));
                }
                if output.deferred_component.is_none() {
                    loss_sum += f64::from(total_value);
                }
                let deferred_total = output
                    .deferred_component
                    .map(|_| output.total.clone().detach());
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
                let event_index = events.len();
                events.push(StepEvent {
                    epoch: self.state.epoch,
                    micro_step: self.state.micro_step,
                    optimizer_step: self.state.optimizer_step,
                    learning_rate: self.scheduler.current_lr(),
                    total_loss: total_value,
                    components: output.components,
                    targets: output.targets,
                    foreground: output.foreground,
                });
                if let (Some(component), Some(total)) = (output.deferred_component, deferred_total)
                {
                    deferred.push((event_index, group_start + offset, component, total));
                }
            }
            let lr = self.scheduler.step();
            model = optimizer.step(lr, model, accumulator.grads());
            if external_weight_decay {
                model = crate::training::optimizer::apply_selective_weight_decay(
                    model,
                    lr,
                    self.config.weight_decay,
                );
            }
            self.state.optimizer_step += 1;
            self.state.accumulation_position = 0;
            after_step(&model, self.state.optimizer_step).map_err(EngineError)?;
            optimizer_steps += 1;
            group_start = group_end;
            if diagnostic_chunk_full(&events) {
                flush_events(
                    &self.run,
                    self.state.epoch,
                    &mut deferred,
                    &mut events,
                    &mut loss_sum,
                )?;
            }
        }
        flush_events(
            &self.run,
            self.state.epoch,
            &mut deferred,
            &mut events,
            &mut loss_sum,
        )?;
        let mean_loss = (loss_sum / batch_count as f64) as f32;
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
                microbatches: batch_count,
                optimizer_steps,
            },
        ))
    }
}

fn flush_events<B: AutodiffBackend>(
    run: &RunDirectory,
    epoch: usize,
    deferred: &mut Vec<DeferredDiagnostic<B>>,
    events: &mut Vec<StepEvent>,
    loss_sum: &mut f64,
) -> Result<(), EngineError> {
    if !deferred.is_empty() {
        let values = Tensor::cat(
            deferred
                .iter()
                .map(|(_, _, _, value)| value.clone())
                .collect(),
            0,
        )
        .into_data();
        let values = values
            .as_slice::<f32>()
            .expect("loss diagnostics must use f32 storage");
        for ((event_index, batch_index, component, _), value) in
            deferred.drain(..).zip(values.iter().copied())
        {
            if !value.is_finite() {
                return Err(EngineError(format!(
                    "non-finite loss at epoch {epoch} batch {batch_index}"
                )));
            }
            *loss_sum += f64::from(value);
            events[event_index].total_loss = value;
            events[event_index]
                .components
                .insert(component.into(), value);
        }
    }
    run.append_events(events).map_err(EngineError::io)?;
    events.clear();
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::{DIAGNOSTIC_CHUNK_SIZE, diagnostic_chunk_full};

    #[test]
    fn deferred_diagnostics_have_a_bounded_chunk() {
        let event = crate::training::report::StepEvent {
            epoch: 0,
            micro_step: 0,
            optimizer_step: 0,
            learning_rate: 0.0,
            total_loss: 0.0,
            components: Default::default(),
            targets: 0,
            foreground: 0,
        };
        let mut events = vec![event; DIAGNOSTIC_CHUNK_SIZE - 1];
        assert!(!diagnostic_chunk_full(&events));
        events.push(events[0].clone());
        assert!(diagnostic_chunk_full(&events));
    }
}
