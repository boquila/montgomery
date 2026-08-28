//! Minimal data transforms required for correct inference.
//!
//! Rich source ingestion and augmentation intentionally remain outside the MVP.

mod imagenet;
mod letterbox;

pub use imagenet::CLASSES as IMAGENET_CLASSES;
pub(crate) use letterbox::{LetterboxedImage, classify_transform};
