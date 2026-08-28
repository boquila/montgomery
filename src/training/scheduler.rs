use serde::{Deserialize, Serialize};

use crate::training::config::ScheduleKind;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LrScheduler {
    pub kind: ScheduleKind,
    pub initial_lr: f64,
    pub final_lr_ratio: f64,
    pub warmup_steps: u64,
    pub total_steps: u64,
    pub next_step: u64,
}

impl LrScheduler {
    pub fn new(
        kind: ScheduleKind,
        initial_lr: f64,
        final_lr_ratio: f64,
        warmup_steps: u64,
        total_steps: u64,
    ) -> Result<Self, &'static str> {
        if total_steps == 0 || warmup_steps >= total_steps {
            return Err("total_steps must be positive and greater than warmup_steps");
        }
        if !initial_lr.is_finite() || initial_lr <= 0.0 || !(0.0..=1.0).contains(&final_lr_ratio) {
            return Err("invalid learning-rate configuration");
        }
        Ok(Self {
            kind,
            initial_lr,
            final_lr_ratio,
            warmup_steps,
            total_steps,
            next_step: 0,
        })
    }

    /// LR for the next optimizer update. Calling this does not advance scheduler state.
    pub fn current_lr(&self) -> f64 {
        if self.warmup_steps > 0 && self.next_step < self.warmup_steps {
            return self.initial_lr * self.next_step as f64 / self.warmup_steps as f64;
        }
        let regular_steps = (self.total_steps - self.warmup_steps).max(1);
        let progress = self
            .next_step
            .saturating_sub(self.warmup_steps)
            .min(regular_steps - 1) as f64
            / (regular_steps - 1).max(1) as f64;
        let multiplier = match self.kind {
            ScheduleKind::Linear => 1.0 - progress * (1.0 - self.final_lr_ratio),
            ScheduleKind::Cosine | ScheduleKind::YoloxWarmCosine => {
                self.final_lr_ratio
                    + 0.5
                        * (1.0 - self.final_lr_ratio)
                        * (1.0 + (std::f64::consts::PI * progress).cos())
            }
        };
        self.initial_lr * multiplier
    }

    pub fn step(&mut self) -> f64 {
        let lr = self.current_lr();
        self.next_step = self.next_step.saturating_add(1).min(self.total_steps);
        lr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resume_has_no_off_by_one() {
        let mut scheduler = LrScheduler::new(ScheduleKind::Cosine, 0.1, 0.05, 2, 10).unwrap();
        for _ in 0..5 {
            scheduler.step();
        }
        let encoded = serde_json::to_vec(&scheduler).unwrap();
        let restored: LrScheduler = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(scheduler.current_lr(), restored.current_lr());
        assert_eq!(scheduler.step(), restored.clone().step());
    }

    #[test]
    fn exact_schedule_boundaries() {
        let mut scheduler = LrScheduler::new(ScheduleKind::Linear, 1.0, 0.1, 2, 5).unwrap();
        assert_eq!(scheduler.step(), 0.0);
        assert_eq!(scheduler.step(), 0.5);
        assert_eq!(scheduler.step(), 1.0);
        scheduler.step();
        assert!((scheduler.step() - 0.1).abs() < 1e-12);
    }
}
