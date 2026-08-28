use serde::{Deserialize, Serialize};

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
