//! Image transforms shared by inference and training.

#[cfg(feature = "pretrained")]
mod imagenet;
mod letterbox;

#[cfg(feature = "training")]
pub mod augmentation;

#[cfg(feature = "pretrained")]
pub use imagenet::CLASSES as IMAGENET_CLASSES;
pub(crate) use letterbox::{LetterboxedImage, classify_transform};
