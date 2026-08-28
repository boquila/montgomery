//! Versioned metadata for boquilens-native YOLOv8 artifacts.

/// Schema suffix of the native artifact written by `boquilens pack-weights`.
const ARTIFACT_SCHEMA: &str = "v1";

/// Native artifact schema string for one scale variant (e.g. `yolov8n-v1`).
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
/// (e.g. `yolov8n-cls-imagenet1k-ultralytics-v8.4-boquilens-v1.bpk`).
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
        model: "yolov8n",
        bytes: 6_418_080,
        sha256: "420607A592E014754B1994AD96065E996A87A3F258A0226CE271E35B2A1895C6",
    },
    VerifiedArtifact {
        model: "yolov8s",
        bytes: 22_483_360,
        sha256: "BDFD4C0DF3BB699425E4F7D85AB593088C4FED1E3842835F41F5233BA484E77F",
    },
    VerifiedArtifact {
        model: "yolov8m",
        bytes: 52_056_224,
        sha256: "8457C821CBE154DE426CA91033F8F9913C8C3FA06391525BF30274D80427E036",
    },
    VerifiedArtifact {
        model: "yolov8l",
        bytes: 87_710_112,
        sha256: "C8F8FC496B3EEE137151D71D4ECCFD1C8A376201DCFB09FAE4C2B6B62E82C4BA",
    },
    VerifiedArtifact {
        model: "yolov8x",
        bytes: 136_876_704,
        sha256: "66F8954A2ED7CE6BB5CBD81A2212E4CF902AA616CD5CB4A52C2A8673A32EDE2B",
    },
    VerifiedArtifact {
        model: "yolov8n-seg",
        bytes: 6_937_408,
        sha256: "E5D3A2619A0F6E6E711CFFBFEED54F91B407038369CD53DE84CF6E23D30EB5CA",
    },
    VerifiedArtifact {
        model: "yolov8s-seg",
        bytes: 23_807_808,
        sha256: "A193580E817752E73B45684305E447739B3A647BC2B423C86EC8D29B597771BF",
    },
    VerifiedArtifact {
        model: "yolov8m-seg",
        bytes: 54_839_360,
        sha256: "8B5ED4197DDA3A2AADEE88CDBCF53BB32E0F3F994EB5EB2BFF4C7F5A0FA3A6AB",
    },
    VerifiedArtifact {
        model: "yolov8l-seg",
        bytes: 92_340_288,
        sha256: "969CCBF1F3F1058B2B95ABF78DA089DEC16C2D7A4021AA76CD2E38F3207F3E6E",
    },
    VerifiedArtifact {
        model: "yolov8x-seg",
        bytes: 144_098_368,
        sha256: "69F24863F62769150AA6642671A63F4A6D278372F458FA83ED7E86FFFBCC5503",
    },
];

/// The verified ImageNet-1k release candidates, one per classify scale, in parity-pass order.
pub const IMAGENET_VERIFIED_ARTIFACTS: &[VerifiedArtifact] = &[
    VerifiedArtifact {
        model: "yolov8n-cls",
        bytes: 5_498_064,
        sha256: "9D8729A22CEF3F7BB6CC584D80DC6A0C61758F4AA39376C79E0B06D8ADF56F65",
    },
    VerifiedArtifact {
        model: "yolov8s-cls",
        bytes: 12_804_048,
        sha256: "AC89B1D489E1BFF31D86C6D831875699ED4D86B97F483ABF3C08041EC4B89205",
    },
    VerifiedArtifact {
        model: "yolov8m-cls",
        bytes: 34_248_400,
        sha256: "01FFB857B35FAD528E9CA0F777BC78C690720861277A15541B2B4DB247AADAAB",
    },
    VerifiedArtifact {
        model: "yolov8l-cls",
        bytes: 75_167_952,
        sha256: "DE3D3B45536EE3C85119B0B594D3129B49238809C377582D90891265E4EF14E8",
    },
    VerifiedArtifact {
        model: "yolov8x-cls",
        bytes: 115_106_768,
        sha256: "1B5BF48CF1D710B7E2A967E2A898236BE098475357E4E6C1F686B32D89D5D197",
    },
];
