use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeedKey<'a> {
    pub run_seed: u64,
    pub epoch: u64,
    pub logical_position: u64,
    pub sample_index: u64,
    pub rank: u32,
    pub path: &'a str,
}

pub struct AugRng(ChaCha12Rng);

impl AugRng {
    pub fn new(key: SeedKey<'_>) -> Self {
        let mut hash = Sha256::new();
        hash.update(b"montgomery-augmentation-seed-v1");
        hash.update(key.run_seed.to_le_bytes());
        hash.update(key.epoch.to_le_bytes());
        hash.update(key.logical_position.to_le_bytes());
        hash.update(key.sample_index.to_le_bytes());
        hash.update(key.rank.to_le_bytes());
        hash.update(key.path.as_bytes());
        Self(ChaCha12Rng::from_seed(hash.finalize().into()))
    }
    pub fn unit(&mut self) -> f32 {
        self.0.random::<f32>()
    }
    pub fn uniform(&mut self, low: f32, high: f32) -> f32 {
        if low == high {
            low
        } else {
            self.0.random_range(low..high)
        }
    }
    pub fn uniform_inclusive_i32(&mut self, low: i32, high: i32) -> i32 {
        self.0.random_range(low..=high)
    }
    pub fn index(&mut self, length: usize) -> usize {
        self.0.random_range(0..length)
    }
    pub fn gate(&mut self, probability: f32) -> bool {
        probability > 0.0 && (probability >= 1.0 || self.unit() < probability)
    }
    pub fn sign(&mut self) -> f32 {
        if self.gate(0.5) { -1.0 } else { 1.0 }
    }
    pub fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let j = self.0.random_range(0..=i);
            values.swap(i, j);
        }
    }
    pub fn normal(&mut self) -> f32 {
        // Box-Muller; clamp away the logarithm singularity.
        let u1 = self.unit().max(f32::MIN_POSITIVE);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
    pub fn beta(&mut self, alpha: f32, beta: f32) -> f32 {
        let x = self.gamma(alpha);
        let y = self.gamma(beta);
        x / (x + y)
    }
    fn gamma(&mut self, shape: f32) -> f32 {
        if shape < 1.0 {
            return self.gamma(shape + 1.0) * self.unit().max(f32::MIN_POSITIVE).powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = (1.0 / (9.0 * d)).sqrt();
        loop {
            let x = self.normal();
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v3 = v * v * v;
            let u = self.unit();
            if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v3 + v3.ln()) {
                return d * v3;
            }
        }
    }
}

/// Python 3's ties-to-even `round`, including negative values.
pub fn python_round(value: f32) -> i64 {
    let floor = value.floor();
    let fraction = value - floor;
    if fraction < 0.5 {
        floor as i64
    } else if fraction > 0.5 {
        floor as i64 + 1
    } else {
        let lower = floor as i64;
        if lower % 2 == 0 { lower } else { lower + 1 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rng() -> AugRng {
        AugRng::new(SeedKey {
            run_seed: 7,
            epoch: 2,
            logical_position: 4,
            sample_index: 9,
            rank: 0,
            path: "mosaic",
        })
    }
    #[test]
    fn fixed_vector_is_stable() {
        let mut a = rng();
        let mut b = rng();
        assert_eq!(
            (0..32).map(|_| a.unit().to_bits()).collect::<Vec<_>>(),
            (0..32).map(|_| b.unit().to_bits()).collect::<Vec<_>>()
        );
    }
    #[test]
    fn bankers_rounding() {
        assert_eq!(python_round(0.5), 0);
        assert_eq!(python_round(1.5), 2);
        assert_eq!(python_round(2.5), 2);
        assert_eq!(python_round(-1.5), -2);
    }
    #[test]
    fn beta_stays_in_range() {
        let mut r = rng();
        for _ in 0..1000 {
            assert!((0.0..=1.0).contains(&r.beta(32.0, 32.0)));
        }
    }
}
