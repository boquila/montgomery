//! Versioned metadata for boquilens-native YOLOv10 artifacts.

/// Schema suffix of the native artifact written by `boquilens pack-weights`.
const ARTIFACT_SCHEMA: &str = "v1";

/// Native artifact schema string for one scale variant (e.g. `yolov10n-v1`).
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
        model: "yolov10n",
        bytes: 4_779_424,
        sha256: "8A672F4924F52E89F7DF95C689C66CF157A96674CE1ADF3C2CF6A025D5C9C44B",
    },
    VerifiedArtifact {
        model: "yolov10s",
        bytes: 14_822_560,
        sha256: "6E6427357A25CFA6FE96D5BA0130808B16A699B0E282EF37A80366497BAC351F",
    },
    VerifiedArtifact {
        model: "yolov10m",
        bytes: 31_221_920,
        sha256: "65C579B005413714F8E935316EA84FE12B689049621A9B15ED6EFF64B536F84C",
    },
    VerifiedArtifact {
        model: "yolov10b",
        bytes: 38_692_768,
        sha256: "15EB91359D74E3A48D98356CF0410D9566B20AE70705C940A17C29078C73B906",
    },
    VerifiedArtifact {
        model: "yolov10l",
        bytes: 49_446_304,
        sha256: "D0412F77DDE5E9ED53324687551FDF29AEF6F34387B9EBC838322891CA90C260",
    },
    VerifiedArtifact {
        model: "yolov10x",
        bytes: 59_978_400,
        sha256: "B47954C6647A8298C2A0444CA53EEB2E498A4D4B52FDB14A934A4FD2AB6A39C2",
    },
];
