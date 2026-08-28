use serde::Serialize;

use super::{
    snapshot::TensorAudit,
    spec::{BoxFormat, ExportFamily, ExportTask, ExternalDataPolicy, OnnxPrecision, OnnxProfile},
};

#[derive(Debug, Clone, Serialize)]
pub struct GraphSourceManifest {
    pub kind: String,
    pub expected_revision: String,
    pub resolved_path: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct InputContract {
    pub name: String,
    pub dtype: String,
    pub layout: String,
    pub color: String,
    pub range: [f32; 2],
    pub shape: [usize; 4],
    pub dynamic_batch: bool,
    pub dynamic_spatial: bool,
    pub preprocessing: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct BridgeManifest {
    pub schema: String,
    pub export_spec_version: String,
    pub model_id: String,
    pub family: ExportFamily,
    pub task: ExportTask,
    pub scale: String,
    pub num_classes: usize,
    pub class_names: Vec<String>,
    pub stride: usize,
    pub box_format: Option<BoxFormat>,
    pub nms: bool,
    pub graph_config: String,
    pub graph_source: GraphSourceManifest,
    pub key_map_version: String,
    pub checkpoint_file: String,
    pub checkpoint_sha256: String,
    pub checkpoint_state: String,
    pub weights_file: String,
    pub weights_sha256: String,
    pub tensor_audit: TensorAudit,
    pub input: InputContract,
    pub profile: OnnxProfile,
    pub precision: OnnxPrecision,
    pub opset: u32,
    pub external_data: ExternalDataPolicy,
    pub simplify: bool,
    pub verify: bool,
    pub reproducible: bool,
    pub output_file: String,
    pub sidecar_file: String,
    pub license: String,
    pub notice: String,
    pub boquilens_version: String,
    pub boquilens_git_commit: String,
    pub boquilens_git_dirty: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct PublishedArtifact {
    pub path: std::path::PathBuf,
    pub sidecar: std::path::PathBuf,
    pub sha256: String,
    pub bytes: u64,
}
