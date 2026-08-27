//! Minimal data transforms required for correct inference.
//!
//! Rich source ingestion and augmentation intentionally remain outside the MVP.

mod letterbox;

pub(crate) use letterbox::LetterboxedImage;
