use crate::data::augmentation::{AugSample, AugmentationError, PartnerProvider};
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};

/// Sample order is a pure function of global seed and epoch, independent of worker scheduling.
pub fn epoch_permutation(length: usize, seed: u64, epoch: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..length).collect();
    let mut rng = ChaCha20Rng::from_seed(derived_seed(seed, epoch, "permutation", 0, 0));
    order.shuffle(&mut rng);
    order
}

/// Derive an independent deterministic random stream for one sample/augmentation stage.
pub fn derived_seed(
    global_seed: u64,
    epoch: u64,
    stage: &str,
    sample_id: u64,
    draw: u64,
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"boquilens-training-rng-v1");
    hash.update(global_seed.to_le_bytes());
    hash.update(epoch.to_le_bytes());
    hash.update(stage.as_bytes());
    hash.update(sample_id.to_le_bytes());
    hash.update(draw.to_le_bytes());
    hash.finalize().into()
}

/// Deterministic immutable sample pool for mixed-image augmentation.
///
/// The scheduler supplies logical positions, so worker timing and cache hits never influence
/// partner selection. Samples are cloned before transforms, preserving cache immutability.
#[derive(Debug, Clone)]
pub struct DeterministicPartnerPool {
    samples: Vec<AugSample>,
    recent_window: Option<usize>,
}

impl DeterministicPartnerPool {
    pub fn whole_dataset(samples: Vec<AugSample>) -> Self {
        Self {
            samples,
            recent_window: None,
        }
    }
    pub fn recent(samples: Vec<AugSample>, batch_size: usize) -> Self {
        Self {
            samples,
            recent_window: Some((batch_size.saturating_mul(8)).clamp(1, 1000)),
        }
    }
}

impl PartnerProvider for DeterministicPartnerPool {
    fn len(&self) -> usize {
        self.samples.len()
    }
    fn get(&mut self, index: usize) -> Result<AugSample, AugmentationError> {
        self.samples
            .get(index)
            .cloned()
            .ok_or_else(|| AugmentationError::new(format!("partner index {index} out of range")))
    }
    fn candidate_index(&self, logical_position: usize, draw: usize) -> usize {
        if self.samples.is_empty() {
            return 0;
        }
        let window = self
            .recent_window
            .unwrap_or(self.samples.len())
            .min(self.samples.len());
        let start = logical_position.saturating_sub(window - 1) % self.samples.len();
        (start + draw.wrapping_mul(0x9e3779b9usize) % window) % self.samples.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_order_is_reproducible_and_epoch_scoped() {
        assert_eq!(epoch_permutation(100, 7, 3), epoch_permutation(100, 7, 3));
        assert_ne!(epoch_permutation(100, 7, 3), epoch_permutation(100, 7, 4));
    }
}
