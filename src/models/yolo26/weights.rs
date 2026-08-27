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
