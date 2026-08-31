//! Official checkpoint provenance and montgomery-native artifact naming for YOLOX.

/// Schema suffix of native YOLOX artifacts written by `montgomery pack-weights`.
const ARTIFACT_SCHEMA: &str = "v1";

/// Native artifact schema string for one scale variant.
pub fn artifact_format(model: &str) -> String {
    format!("{model}-{ARTIFACT_SCHEMA}")
}

/// Recommended filename for a COCO-80 artifact converted from the official YOLOX release.
pub fn coco_artifact_filename(model: &str) -> String {
    format!("{model}-coco-official-v0.1.1rc0-montgomery-v1.bpk")
}

/// Immutable provenance for one official upstream checkpoint.
pub struct OfficialCheckpoint {
    pub model: &'static str,
    pub filename: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
}

/// Official Apache-2.0 checkpoints used as inputs to `pack-weights` and parity tests.
pub const OFFICIAL_CHECKPOINTS: &[OfficialCheckpoint] = &[
    OfficialCheckpoint {
        model: "yolox-nano",
        filename: "yolox_nano.pth",
        url: "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth",
        sha256: "cd28f55fbbc1829f99d9ac9b38a16d259a22889739c8728ea877610201feff7b",
    },
    OfficialCheckpoint {
        model: "yolox-tiny",
        filename: "yolox_tiny.pth",
        url: "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth",
        sha256: "9de513de589ac98bb92d3bca53b5af7b9acfa9b0bacb831f7999d0f7afaee8f0",
    },
    OfficialCheckpoint {
        model: "yolox-s",
        filename: "yolox_s.pth",
        url: "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth",
        sha256: "f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30",
    },
    OfficialCheckpoint {
        model: "yolox-m",
        filename: "yolox_m.pth",
        url: "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth",
        sha256: "60076992b32da82951c90cfa7bd6ab70eba9eda243e08b940a396f60ac2d19b6",
    },
    OfficialCheckpoint {
        model: "yolox-l",
        filename: "yolox_l.pth",
        url: "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth",
        sha256: "1e6b7fa6240375370b2a8a8eab9066b3cdd43fd1d0bfa8d2027fd3a51def2917",
    },
    OfficialCheckpoint {
        model: "yolox-x",
        filename: "yolox_x.pth",
        url: "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth",
        sha256: "5652330b6ae860043f091b8f550a60c10e1129f416edfdb65c259be6caf355cf",
    },
];

pub fn official_checkpoint(model: &str) -> Option<&'static OfficialCheckpoint> {
    OFFICIAL_CHECKPOINTS
        .iter()
        .find(|checkpoint| checkpoint.model == model)
}
