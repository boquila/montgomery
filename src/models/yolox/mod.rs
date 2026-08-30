//! Native Burn implementation of the YOLOX detector family.
//!
//! Initially adapted from `tracel-ai/models/yolox-burn` under MIT OR Apache-2.0.
//! See the repository `NOTICE` file for provenance.

mod blocks;
mod bottleneck;
mod darknet;
mod head;
pub mod model;
mod pafpn;
pub mod weights;

pub use crate::postprocess::BoundingBox;
pub use head::{FeatureLevelShape, RawPredictions};
pub use model::{Yolox, YoloxConfig};
