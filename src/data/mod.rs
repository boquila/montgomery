//! Image transforms shared by inference and training.

mod imagenet;
mod letterbox;

#[cfg(feature = "training")]
pub mod augmentation;

pub use imagenet::CLASSES as IMAGENET_CLASSES;
pub(crate) use letterbox::{LetterboxedImage, classify_transform};
