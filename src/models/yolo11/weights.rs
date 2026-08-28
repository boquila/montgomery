//! Versioned metadata for boquilens-native YOLO11 artifacts.

/// Schema suffix of the native artifact written by `boquilens pack-weights`.
const ARTIFACT_SCHEMA: &str = "v1";

/// Native artifact schema string for one scale variant (e.g. `yolo11n-v1`).
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
        model: "yolo11n",
        bytes: 5_399_968,
        sha256: "36ACCB9BCEF72CD1DD3D534F54BE845C9EE4EE1697AD65C731FE028028E68BDF",
    },
    VerifiedArtifact {
        model: "yolo11s",
        bytes: 19_140_768,
        sha256: "4277237339A0975D1E86FBFB7787D861982F9B64B857C458E0D998671AA63DB9",
    },
    VerifiedArtifact {
        model: "yolo11m",
        bytes: 40_561_568,
        sha256: "ACFE957B42A17D81C9988772E2A1576592B3DB293DC8D52AFC91BCECB5595073",
    },
    VerifiedArtifact {
        model: "yolo11l",
        bytes: 51_208_352,
        sha256: "84FE90D17143FB894CEFE6557D3619F000E1602BDC331905FE56E6AC996F953F",
    },
    VerifiedArtifact {
        model: "yolo11x",
        bytes: 114_597_280,
        sha256: "1AC48B4A48165632F7B54A7B2E8471C9FB782CE436DE795C3155BCEF848C156E",
    },
    VerifiedArtifact {
        model: "yolo11n-seg",
        bytes: 5_919_808,
        sha256: "A29FF611095F39E3875A22B03B93DC1FDCD5AE40A1310AA5DF4D3813E17B1FF4",
    },
    VerifiedArtifact {
        model: "yolo11s-seg",
        bytes: 20_465_216,
        sha256: "FD9841F96748BD32A50EF508340F86A161B331D44F3D16678A96BED1A76342BE",
    },
];
