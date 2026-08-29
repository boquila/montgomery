use std::{
    collections::BTreeMap,
    error::Error,
    fmt, fs,
    io::Write,
    path::{Path, PathBuf},
};

use burn::{
    record::{BinBytesRecorder, FullPrecisionSettings, Record, Recorder},
    tensor::backend::Backend,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::training::{TrainingConfig, scheduler::LrScheduler, state::TrainingState};

pub const CHECKPOINT_FORMAT: &str = "boquilens-training-v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointManifest {
    pub format: String,
    pub crate_version: String,
    pub config: TrainingConfig,
    pub state: TrainingState,
    pub scheduler: LrScheduler,
    pub payloads: BTreeMap<String, String>,
}

impl CheckpointManifest {
    pub fn new(config: TrainingConfig, state: TrainingState, scheduler: LrScheduler) -> Self {
        Self {
            format: CHECKPOINT_FORMAT.into(),
            crate_version: env!("CARGO_PKG_VERSION").into(),
            config,
            state,
            scheduler,
            payloads: BTreeMap::new(),
        }
    }
}

/// Write payloads and manifest to a sibling temporary directory, then publish with one rename.
pub fn save_atomic(
    destination: impl AsRef<Path>,
    mut manifest: CheckpointManifest,
    payloads: &[(&str, &[u8])],
) -> Result<(), CheckpointError> {
    let destination = destination.as_ref();
    if destination.exists() {
        return Err(CheckpointError::new(format!(
            "checkpoint destination already exists: {}",
            destination.display()
        )));
    }
    let parent = destination
        .parent()
        .ok_or_else(|| CheckpointError::new("checkpoint needs a parent directory"))?;
    fs::create_dir_all(parent).map_err(CheckpointError::io)?;
    let file_name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("checkpoint");
    let temporary = parent.join(format!(".{file_name}.tmp-{}", std::process::id()));
    if temporary.exists() {
        return Err(CheckpointError::new(format!(
            "temporary checkpoint already exists: {}",
            temporary.display()
        )));
    }
    fs::create_dir(&temporary).map_err(CheckpointError::io)?;
    let result = (|| {
        for (name, bytes) in payloads {
            validate_payload_name(name)?;
            let path = temporary.join(name);
            let mut file = fs::File::create(&path).map_err(CheckpointError::io)?;
            file.write_all(bytes).map_err(CheckpointError::io)?;
            file.sync_all().map_err(CheckpointError::io)?;
            manifest.payloads.insert((*name).into(), sha256(bytes));
        }
        let encoded = serde_json::to_vec_pretty(&manifest)
            .map_err(|error| CheckpointError::new(error.to_string()))?;
        let mut file =
            fs::File::create(temporary.join("manifest.json")).map_err(CheckpointError::io)?;
        file.write_all(&encoded).map_err(CheckpointError::io)?;
        file.sync_all().map_err(CheckpointError::io)?;
        drop(file);
        fs::rename(&temporary, destination).map_err(CheckpointError::io)
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&temporary);
    }
    result
}

/// Atomically replace a named rolling checkpoint while retaining the previous directory until the
/// new checkpoint has been fully written and hash-validated.
pub fn replace_atomic(
    destination: impl AsRef<Path>,
    manifest: CheckpointManifest,
    payloads: &[(&str, &[u8])],
) -> Result<(), CheckpointError> {
    let destination = destination.as_ref();
    let parent = destination
        .parent()
        .ok_or_else(|| CheckpointError::new("checkpoint needs a parent directory"))?;
    fs::create_dir_all(parent).map_err(CheckpointError::io)?;
    let name = destination
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("checkpoint");
    let incoming = parent.join(format!(".{name}.incoming-{}", std::process::id()));
    let previous = parent.join(format!(".{name}.previous-{}", std::process::id()));
    if incoming.exists() || previous.exists() {
        return Err(CheckpointError::new(
            "checkpoint rotation temporary path already exists",
        ));
    }
    save_atomic(&incoming, manifest, payloads)?;
    load(&incoming)?;
    if destination.exists() {
        fs::rename(destination, &previous).map_err(CheckpointError::io)?;
    }
    if let Err(error) = fs::rename(&incoming, destination) {
        if previous.exists() {
            let _ = fs::rename(&previous, destination);
        }
        return Err(CheckpointError::io(error));
    }
    if previous.exists() {
        fs::remove_dir_all(previous).map_err(CheckpointError::io)?;
    }
    Ok(())
}

pub fn load(path: impl AsRef<Path>) -> Result<CheckpointManifest, CheckpointError> {
    let path = path.as_ref();
    let bytes = fs::read(path.join("manifest.json")).map_err(CheckpointError::io)?;
    let manifest: CheckpointManifest = serde_json::from_slice(&bytes)
        .map_err(|error| CheckpointError::new(format!("invalid checkpoint manifest: {error}")))?;
    if manifest.format != CHECKPOINT_FORMAT {
        return Err(CheckpointError::new(format!(
            "unsupported checkpoint format {}",
            manifest.format
        )));
    }
    manifest
        .config
        .validate()
        .map_err(|error| CheckpointError::new(error.to_string()))?;
    for (name, expected) in &manifest.payloads {
        validate_payload_name(name)?;
        let payload = fs::read(path.join(name)).map_err(CheckpointError::io)?;
        let actual = sha256(&payload);
        if &actual != expected {
            return Err(CheckpointError::new(format!(
                "checkpoint payload {name} failed SHA-256 validation"
            )));
        }
    }
    Ok(manifest)
}

/// Serialize a Burn model or optimizer record in full precision for a resumable checkpoint.
pub fn encode_record<B: Backend, R: Record<B>>(record: R) -> Result<Vec<u8>, CheckpointError> {
    Recorder::<B>::record(
        &BinBytesRecorder::<FullPrecisionSettings>::default(),
        record,
        (),
    )
    .map_err(|error| CheckpointError::new(error.to_string()))
}

pub fn decode_record<B: Backend, R: Record<B>>(
    bytes: Vec<u8>,
    device: &B::Device,
) -> Result<R, CheckpointError> {
    Recorder::<B>::load(
        &BinBytesRecorder::<FullPrecisionSettings>::default(),
        bytes,
        device,
    )
    .map_err(|error| CheckpointError::new(error.to_string()))
}

fn validate_payload_name(name: &str) -> Result<(), CheckpointError> {
    let path = PathBuf::from(name);
    if name == "manifest.json" || path.components().count() != 1 || path.is_absolute() {
        return Err(CheckpointError::new(format!(
            "invalid checkpoint payload name {name:?}"
        )));
    }
    Ok(())
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Debug)]
pub struct CheckpointError(String);

impl CheckpointError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
    fn io(error: std::io::Error) -> Self {
        Self(error.to_string())
    }
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for CheckpointError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ModelId,
        training::{
            config::{ModelSpec, ScheduleKind},
            scheduler::LrScheduler,
        },
    };

    fn unique_path(label: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "boquilens-{label}-{}-{}",
            std::process::id(),
            nonce
        ))
    }

    #[test]
    fn round_trip_validates_payload_hashes() {
        let path = unique_path("checkpoint");
        let spec = ModelSpec::new(ModelId::YoloxNano, vec!["object".into()], None).unwrap();
        let config = TrainingConfig::yolox(spec, "data.yaml".into(), "runs".into());
        let scheduler = LrScheduler::new(ScheduleKind::Cosine, 0.1, 0.05, 0, 2).unwrap();
        let manifest = CheckpointManifest::new(config, TrainingState::new(7), scheduler);
        save_atomic(&path, manifest, &[("model.bin", b"model")]).unwrap();
        assert_eq!(load(&path).unwrap().state.global_seed, 7);
        fs::write(path.join("model.bin"), b"corrupt").unwrap();
        assert!(load(&path).unwrap_err().to_string().contains("SHA-256"));
        fs::remove_dir_all(&path).unwrap();
    }
}
