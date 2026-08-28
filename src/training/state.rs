use crate::data::augmentation::PipelinePhase;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DualLossSchedule {
    pub one_to_many: f32,
    pub one_to_one: f32,
    pub updates: usize,
    pub total_epochs: usize,
}

impl DualLossSchedule {
    pub fn yolo26(total_epochs: usize) -> Self {
        Self {
            one_to_many: 0.8,
            one_to_one: 0.2,
            updates: 0,
            total_epochs: total_epochs.max(1),
        }
    }

    pub fn complete_epoch(&mut self) {
        self.updates = (self.updates + 1).min(self.total_epochs);
        let progress = self.updates as f32 / self.total_epochs as f32;
        self.one_to_many = 0.8 + (0.1 - 0.8) * progress;
        self.one_to_one = 1.0 - self.one_to_many;
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingState {
    pub epoch: usize,
    pub next_batch: usize,
    pub micro_step: u64,
    pub optimizer_step: u64,
    pub accumulation_position: usize,
    pub ema_updates: u64,
    pub best_fitness: Option<f64>,
    pub best_epoch: Option<usize>,
    pub patience_counter: usize,
    pub global_seed: u64,
    pub no_augmentation: bool,
    pub yolox_l1: bool,
    pub dual_loss: Option<DualLossSchedule>,
    #[serde(default = "default_augmentation_phase")]
    pub augmentation_phase: PipelinePhase,
}

impl TrainingState {
    pub fn new(global_seed: u64) -> Self {
        Self {
            epoch: 0,
            next_batch: 0,
            micro_step: 0,
            optimizer_step: 0,
            accumulation_position: 0,
            ema_updates: 0,
            best_fitness: None,
            best_epoch: None,
            patience_counter: 0,
            global_seed,
            no_augmentation: false,
            yolox_l1: false,
            dual_loss: None,
            augmentation_phase: PipelinePhase::Open,
        }
    }

    pub fn observe_fitness(&mut self, fitness: f64) -> bool {
        if !fitness.is_finite() {
            self.patience_counter += 1;
            return false;
        }
        let improved = self.best_fitness.is_none_or(|best| fitness > best);
        if improved {
            self.best_fitness = Some(fitness);
            self.best_epoch = Some(self.epoch);
            self.patience_counter = 0;
        } else {
            self.patience_counter += 1;
        }
        improved
    }

    pub fn resolve_augmentation_phase(&mut self, epochs: usize, close_mosaic: usize) -> bool {
        let next = PipelinePhase::for_epoch(self.epoch, epochs, close_mosaic);
        let changed = next != self.augmentation_phase;
        self.augmentation_phase = next;
        changed
    }
}

fn default_augmentation_phase() -> PipelinePhase {
    PipelinePhase::Open
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn best_requires_strict_finite_improvement() {
        let mut state = TrainingState::new(7);
        assert!(state.observe_fitness(0.5));
        assert!(!state.observe_fitness(0.5));
        assert!(!state.observe_fitness(f64::NAN));
        assert_eq!(state.best_fitness, Some(0.5));
        assert_eq!(state.patience_counter, 2);
    }

    #[test]
    fn yolo26_weights_decay_and_round_trip() {
        let mut schedule = DualLossSchedule::yolo26(10);
        schedule.complete_epoch();
        let restored: DualLossSchedule =
            serde_json::from_slice(&serde_json::to_vec(&schedule).unwrap()).unwrap();
        assert_eq!(schedule, restored);
        assert!((schedule.one_to_many - 0.73).abs() < 1e-6);
    }

    #[test]
    fn close_mosaic_phase_is_resume_safe_at_boundaries() {
        let mut state = TrainingState::new(7);
        state.epoch = 89;
        assert!(!state.resolve_augmentation_phase(100, 10));
        state.epoch = 90;
        assert!(state.resolve_augmentation_phase(100, 10));
        assert_eq!(state.augmentation_phase, PipelinePhase::Closed);
        let restored: TrainingState =
            serde_json::from_slice(&serde_json::to_vec(&state).unwrap()).unwrap();
        assert_eq!(restored.augmentation_phase, PipelinePhase::Closed);

        state.augmentation_phase = PipelinePhase::Open;
        state.epoch = 0;
        assert!(state.resolve_augmentation_phase(5, 10));
        state.augmentation_phase = PipelinePhase::Open;
        assert!(!state.resolve_augmentation_phase(5, 0));
    }
}
