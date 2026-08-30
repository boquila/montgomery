//! Image transforms shared by inference and training.

use burn::tensor::{Tensor, backend::Backend};

mod imagenet;
mod letterbox;

#[cfg(feature = "training")]
pub mod augmentation;

pub use imagenet::CLASSES as IMAGENET_CLASSES;
pub(crate) use letterbox::{LetterboxedImage, classify_transform};

/// Apply the RGB normalization used by the official YOLOX checkpoints to `[0, 1]` NCHW input.
pub(crate) fn normalize_yolox<B: Backend>(images: Tensor<B, 4>) -> Tensor<B, 4> {
    let device = images.device();
    let mean = Tensor::<B, 1>::from_floats([0.485, 0.456, 0.406], &device).reshape([1, 3, 1, 1]);
    let std = Tensor::<B, 1>::from_floats([0.229, 0.224, 0.225], &device).reshape([1, 3, 1, 1]);
    (images - mean) / std
}
