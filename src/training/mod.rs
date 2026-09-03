//! Native-training contracts and parity-tested building blocks.
//!
//! Training is a first-class default capability and uses Burn autodiff on WGPU.

pub mod assign;
#[cfg(test)]
mod capability;
pub mod checkpoint;
pub mod config;
pub mod data;
pub mod dispatch;
pub mod ema;
pub mod engine;
pub mod geometry;
pub mod loss;
pub mod metrics;
pub mod optimizer;
pub mod report;
pub mod runtime;
pub mod scheduler;
pub mod state;

pub use config::{ModelSpec, TaskKind, TrainingConfig, automatic_worker_count};
pub use engine::{TrainableTask, Trainer};
