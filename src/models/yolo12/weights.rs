//! Versioned metadata for boquilens-native YOLO12 artifacts.

/// Schema suffix of the native artifact written by `boquilens pack-weights`.
const ARTIFACT_SCHEMA: &str = "v1";

/// Native artifact schema string for one scale variant (e.g. `yolo12n-v1`).
pub fn artifact_format(model: &str) -> String {
    format!("{model}-{ARTIFACT_SCHEMA}")
}

/// Recommended filename for the COCO-80 artifact of one scale variant.
pub fn coco_artifact_filename(model: &str) -> String {
    format!("{model}-coco-ultralytics-v8.4-boquilens-v1.bpk")
}

/// The training dataset tag an official checkpoint was trained on, inferred from its task suffix:
/// detection/segmentation/pose ride COCO, classification is ImageNet-1k, and OBB is DOTA-v1.
pub fn dataset_tag(model: &str) -> &'static str {
    if model.ends_with("-cls") {
        "imagenet1k"
    } else if model.ends_with("-obb") {
        "dotav1"
    } else {
        "coco"
    }
}

/// Recommended filename for the native artifact of any scale/task variant.
pub fn artifact_filename(model: &str) -> String {
    format!(
        "{model}-{}-ultralytics-v8.4-boquilens-v1.bpk",
        dataset_tag(model)
    )
}

/// A verified local release candidate produced during a v1 parity pass.
///
/// Distribution URLs are intentionally absent until the AGPL artifacts are published from a
/// maintained release channel.
pub struct VerifiedArtifact {
    pub model: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
}

/// The verified COCO-80 release candidates, one per scale, in parity-pass order.
pub const COCO_VERIFIED_ARTIFACTS: &[VerifiedArtifact] = &[
    VerifiedArtifact {
        model: "yolo12n",
        bytes: 5_426_592,
        sha256: "65A44ECCF690942511DFEB8BB98173F0FEB45A3BA6C9A2730FCEF8424D4E928C",
    },
    VerifiedArtifact {
        model: "yolo12s",
        bytes: 18_901_920,
        sha256: "60B596F8B8E2ACB5AC93B35773BD7CC05FF751DA52B098DF97E8392EA37D4D96",
    },
    VerifiedArtifact {
        model: "yolo12m",
        bytes: 40_860_064,
        sha256: "DE851D8778A4FB1E7167571ED0F164C99A31757A58C968D91AB3DB6A07A0309E",
    },
    VerifiedArtifact {
        model: "yolo12l",
        bytes: 53_627_808,
        sha256: "654B28CEC86CA060E8011EBE263398C06D1DCEF5D653007EC34C087BBF37C998",
    },
    VerifiedArtifact {
        model: "yolo12x",
        bytes: 119_476_896,
        sha256: "3F64AAE14F3E509B79B4F7C242DC994F3027FEA7E3B0C9D2317B4711C1994CBB",
    },
];
