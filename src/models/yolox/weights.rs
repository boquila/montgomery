/// Pre-trained weights metadata.
#[allow(dead_code)]
pub struct Weights {
    pub(super) url: &'static str,
    pub(super) num_classes: usize,
    pub(super) sha256: Option<&'static str>,
}

#[cfg(feature = "pretrained")]
mod downloader {
    use super::*;
    use burn::data::network::downloader;
    use std::fs::{File, create_dir_all};
    use std::io::{Read, Write};
    use std::path::{Path, PathBuf};

    use sha2::{Digest, Sha256};

    fn verify_sha256(path: &Path, expected: &str) -> Result<(), std::io::Error> {
        let mut file = File::open(path)?;
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
        }
        let actual = format!("{:x}", hasher.finalize());
        if actual != expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("checkpoint checksum mismatch: expected {expected}, got {actual}"),
            ));
        }
        Ok(())
    }

    impl Weights {
        /// Download the pre-trained weights to the local cache directory.
        pub fn download(&self) -> Result<PathBuf, std::io::Error> {
            // Model cache directory
            let model_dir = dirs::home_dir()
                .expect("Should be able to get home directory")
                .join(".cache")
                .join("boquilens")
                .join("yolox");

            if !model_dir.exists() {
                create_dir_all(&model_dir)?;
            }

            let file_base_name = self.url.rsplit_once('/').unwrap().1;
            let file_name = model_dir.join(file_base_name);
            if !file_name.exists() {
                // Download file content
                let bytes = downloader::download_file_as_bytes(self.url, file_base_name);

                // Write content to file
                let mut output_file = File::create(&file_name)?;
                let bytes_written = output_file.write(&bytes)?;

                if bytes_written != bytes.len() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "Failed to write the whole model weights file.",
                    ));
                }
            }

            if let Some(expected) = self.sha256 {
                verify_sha256(&file_name, expected)?;
            }

            Ok(file_name)
        }
    }
}

pub trait WeightsMeta {
    fn weights(&self) -> Weights;
}

/// YOLOX-Nano pre-trained weights.
pub enum YoloxNano {
    /// These weights were released after the original paper implementation with slightly better results.
    /// mAP (val2017): 25.8
    Coco,
}
impl WeightsMeta for YoloxNano {
    fn weights(&self) -> Weights {
        Weights {
            url: "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth",
            num_classes: 80,
            sha256: Some("cd28f55fbbc1829f99d9ac9b38a16d259a22889739c8728ea877610201feff7b"),
        }
    }
}

/// YOLOX-Tiny pre-trained weights.
pub enum YoloxTiny {
    /// These weights were released after the original paper implementation with slightly better results.
    /// mAP (val2017): 32.8
    Coco,
}
impl WeightsMeta for YoloxTiny {
    fn weights(&self) -> Weights {
        Weights {
            url: "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth",
            num_classes: 80,
            sha256: Some("9de513de589ac98bb92d3bca53b5af7b9acfa9b0bacb831f7999d0f7afaee8f0"),
        }
    }
}

/// YOLOX-S pre-trained weights.
pub enum YoloxS {
    /// These weights were released after the original paper implementation with slightly better results.
    /// mAP (test2017): 40.5
    Coco,
}
impl WeightsMeta for YoloxS {
    fn weights(&self) -> Weights {
        Weights {
            url: "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth",
            num_classes: 80,
            sha256: Some("f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30"),
        }
    }
}

/// YOLOX-M pre-trained weights.
pub enum YoloxM {
    /// These weights were released after the original paper implementation with slightly better results.
    /// mAP (test2017): 47.2
    Coco,
}
impl WeightsMeta for YoloxM {
    fn weights(&self) -> Weights {
        Weights {
            url: "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth",
            num_classes: 80,
            sha256: Some("60076992b32da82951c90cfa7bd6ab70eba9eda243e08b940a396f60ac2d19b6"),
        }
    }
}

/// YOLOX-L pre-trained weights.
pub enum YoloxL {
    /// These weights were released after the original paper implementation with slightly better results.
    /// mAP (test2017): 50.1
    Coco,
}
impl WeightsMeta for YoloxL {
    fn weights(&self) -> Weights {
        Weights {
            url: "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth",
            num_classes: 80,
            sha256: Some("1e6b7fa6240375370b2a8a8eab9066b3cdd43fd1d0bfa8d2027fd3a51def2917"),
        }
    }
}

/// YOLOX-X pre-trained weights.
pub enum YoloxX {
    /// These weights were released after the original paper implementation with slightly better results.
    /// mAP (test2017): 51.5
    Coco,
}
impl WeightsMeta for YoloxX {
    fn weights(&self) -> Weights {
        Weights {
            url: "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth",
            num_classes: 80,
            sha256: Some("5652330b6ae860043f091b8f550a60c10e1129f416edfdb65c259be6caf355cf"),
        }
    }
}
