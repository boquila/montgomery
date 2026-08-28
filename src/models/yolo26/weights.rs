//! Versioned metadata for boquilens-native YOLO26 artifacts.

/// Schema suffix of the native artifact written by `boquilens pack-weights`.
const ARTIFACT_SCHEMA: &str = "v1";

/// Native artifact schema string for one scale variant (e.g. `yolo26n-v1`).
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

/// Recommended filename for the native artifact of any scale/task variant
/// (e.g. `yolo26n-cls-imagenet1k-ultralytics-v8.4-boquilens-v1.bpk`).
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
        model: "yolo26n",
        bytes: 5_016_992,
        sha256: "5FB09D89850E2ECB75C0580893239DEF9BB130E95A228FB319675F267B5B24C6",
    },
    VerifiedArtifact {
        model: "yolo26s",
        bytes: 19_283_872,
        sha256: "DD287F71998783596CBF5204F29D589D215EBF910987623CBCB1DD8F0AD91855",
    },
    VerifiedArtifact {
        model: "yolo26m",
        bytes: 41_216_928,
        sha256: "50A0BE494BA93D5663084161999B3D2B2C9ABB6DABB163D0AED2DB6F37591249",
    },
    VerifiedArtifact {
        model: "yolo26l",
        bytes: 50_140_064,
        sha256: "19D2C802F3266571FC7298DB9C3AB0E912D4DD6004B1D37510124F92A428A171",
    },
    VerifiedArtifact {
        model: "yolo26x",
        bytes: 112_210_080,
        sha256: "D1B1B94FC28423CC4FFD4EA04DEEAE3FE4A352B7E0D8F442D6CE9FA616C813A9",
    },
];

/// The verified ImageNet-1k release candidates, one per classify scale, in parity-pass order.
pub const IMAGENET_VERIFIED_ARTIFACTS: &[VerifiedArtifact] = &[
    VerifiedArtifact {
        model: "yolo26n-cls",
        bytes: 5_712_080,
        sha256: "5A0BC57C4EA137DBB3E52FC2AB7007023474E10401C00BF6B1D857C2E053FB18",
    },
    VerifiedArtifact {
        model: "yolo26s-cls",
        bytes: 13_576_144,
        sha256: "F39B0D7A9FC65495D8D7944BFA7AE9F32C1F6D719AB15043DE40E962FCC811BB",
    },
    VerifiedArtifact {
        model: "yolo26m-cls",
        bytes: 23_434_960,
        sha256: "301F3351F301C5BDEE5A8FC8A54CFF602245BDFA1A64794E61737803FCB684A0",
    },
    VerifiedArtifact {
        model: "yolo26l-cls",
        bytes: 28_472_528,
        sha256: "C759F229D8863D5F78D67FF714A7F1C5AE826417AC7BD906FE08F208DC88AA12",
    },
    VerifiedArtifact {
        model: "yolo26x-cls",
        bytes: 59_609_552,
        sha256: "8468915AA906623DC82E4AEF086C7DC1C236B5E09E30CA849DC03399BF059165",
    },
];

/// The verified instance-segmentation release candidates, one per segment scale, in parity-pass
/// order.
pub const SEG_VERIFIED_ARTIFACTS: &[VerifiedArtifact] = &[
    VerifiedArtifact {
        model: "yolo26n-seg",
        bytes: 5_664_064,
        sha256: "4AB2E714E0684C10E09D3F226BF810C0D8D67D0985440DA3221634A2A7AB4FEE",
    },
    VerifiedArtifact {
        model: "yolo26s-seg",
        bytes: 21_107_520,
        sha256: "C2DD24C137EC530823D3651C23C4CE71E945DDA0B327E50FA668F2144593FDB4",
    },
    VerifiedArtifact {
        model: "yolo26m-seg",
        bytes: 47_565_120,
        sha256: "7EB2FE79189782273B11682C42A479FF4691384DAC1F0A980FA2D1139F32FE55",
    },
    VerifiedArtifact {
        model: "yolo26l-seg",
        bytes: 56_488_000,
        sha256: "188217E220DE61F12BDEF62DF35C24FB5FF836B75420CE2A4617CDACF27233F0",
    },
    VerifiedArtifact {
        model: "yolo26x-seg",
        bytes: 126_443_072,
        sha256: "0575D2EE7AAED7A69566D6A9302D2DAB9CDA9D3C0ED7E36B14484A46B0AF3407",
    },
];
