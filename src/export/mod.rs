//! Offline ONNX artifact export.
//!
//! Burn 0.21 does not export ONNX graphs. This module therefore loads the exact requested model
//! in Rust, snapshots its parameters to SafeTensors, and launches the pinned repository-owned
//! Python graph adapter. Python, PyTorch, ONNX and ONNX Runtime are build-time export dependencies
//! only; published artifacts have no dependency on them.

mod keymap;
mod manifest;
mod snapshot;
pub mod spec;

use std::{
    env, fs,
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{SystemTime, UNIX_EPOCH},
};

use burn_flex::Flex;
use sha2::{Digest, Sha256};

use crate::{COCO_CLASSES, ModelId, PredictOptions, Predictor, Result, data::IMAGENET_CLASSES};

pub use manifest::PublishedArtifact as OnnxArtifact;
pub use spec::{ExternalDataPolicy, OnnxPrecision, OnnxProfile};

use manifest::{BridgeManifest, BurnReferenceManifest, GraphSourceManifest, InputContract};
use spec::{EXPORT_SPEC_VERSION, ExportFamily, ExportSpec, ExportTask};

const ULTRALYTICS_REVISION: &str = "461196cf09175b64c9b9bd8babebf081c0540520";

#[derive(Debug, Clone)]
pub struct OnnxExportOptions {
    pub output: PathBuf,
    pub input_shape: [usize; 4],
    pub profile: OnnxProfile,
    pub opset: u32,
    pub precision: OnnxPrecision,
    pub dynamic_batch: bool,
    pub dynamic_spatial: bool,
    pub external_data: ExternalDataPolicy,
    pub verify: bool,
    pub python: Option<PathBuf>,
    pub yolox_repo: Option<PathBuf>,
    pub simplify: bool,
    pub force: bool,
    pub keep_intermediate: bool,
    pub reproducible: bool,
    pub checkpoint_state: CheckpointState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum CheckpointState {
    Ema,
    Model,
}

impl OnnxExportOptions {
    pub fn for_model(model_id: ModelId, output: PathBuf) -> Self {
        Self {
            output,
            input_shape: ExportSpec::for_model(model_id).default_input,
            profile: OnnxProfile::Portable,
            opset: 17,
            precision: OnnxPrecision::Fp32,
            dynamic_batch: false,
            dynamic_spatial: false,
            external_data: ExternalDataPolicy::Auto,
            verify: true,
            python: None,
            yolox_repo: None,
            simplify: false,
            force: false,
            keep_intermediate: false,
            reproducible: false,
            checkpoint_state: CheckpointState::Ema,
        }
    }
}

pub fn export_onnx(
    model_id: ModelId,
    weights: &Path,
    options: OnnxExportOptions,
) -> Result<OnnxArtifact> {
    let weights = weights.to_owned();
    std::thread::Builder::new()
        .name("boquilens-onnx-export".into())
        .stack_size(128 * 1024 * 1024)
        .spawn(move || export_onnx_inner(model_id, &weights, options))?
        .join()
        .map_err(|_| -> Box<dyn std::error::Error + Send + Sync> {
            "ONNX export worker panicked".into()
        })?
}

fn export_onnx_inner(
    model_id: ModelId,
    weights: &Path,
    mut options: OnnxExportOptions,
) -> Result<OnnxArtifact> {
    let spec = ExportSpec::for_model(model_id);
    validate_options(spec, weights, &mut options)?;
    let output = absolute_path(&options.output)?;
    let sidecar = sidecar_path(&output);
    let output_data = external_data_path(&output);
    refuse_or_validate_targets(&output, &sidecar, &output_data, options.force)?;

    let python = resolve_python(options.python.as_deref())?;
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let exporter = repository.join("tools/onnx/export.py");
    let ultralytics_repo = repository
        .parent()
        .ok_or("cannot resolve sibling Ultralytics source")?
        .join("ultralytics");
    let yolox_repo = options
        .yolox_repo
        .clone()
        .unwrap_or_else(|| repository.join("target/yolox-ref/YOLOX-0.1.1rc0"));

    run_python(
        &python,
        &exporter,
        &[
            "--preflight".into(),
            "--family".into(),
            family_name(spec.family).into(),
            "--ultralytics-repo".into(),
            ultralytics_repo.as_os_str().to_owned(),
            "--yolox-repo".into(),
            yolox_repo.as_os_str().to_owned(),
        ],
        "Python environment/source preflight",
    )?;

    let intermediate = create_intermediate(output.parent().expect("absolute output has parent"))?;
    let result = export_staged(
        model_id,
        weights,
        &options,
        spec,
        &python,
        &exporter,
        &ultralytics_repo,
        &yolox_repo,
        &intermediate,
        &output,
        &sidecar,
    );
    match result {
        Ok(artifact) => {
            if !options.keep_intermediate {
                fs::remove_dir_all(&intermediate).map_err(|error| {
                    format!(
                        "export succeeded but temporary-directory cleanup failed at {}: {error}",
                        intermediate.display()
                    )
                })?;
            }
            Ok(artifact)
        }
        Err(error) => Err(format!(
            "{error}\nONNX export intermediates were retained at {}",
            intermediate.display()
        )
        .into()),
    }
}

#[allow(clippy::too_many_arguments)]
fn export_staged(
    model_id: ModelId,
    weights: &Path,
    options: &OnnxExportOptions,
    spec: ExportSpec,
    python: &Path,
    exporter: &Path,
    ultralytics_repo: &Path,
    yolox_repo: &Path,
    intermediate: &Path,
    output: &Path,
    sidecar: &Path,
) -> Result<OnnxArtifact> {
    let checkpoint_sha256 = sha256_file(weights)?;
    let snapshot_path = intermediate.join("weights.safetensors");
    let checkpoint = weights.to_owned();
    let snapshot_for_worker = snapshot_path.clone();
    let input_shape = options.input_shape;
    let write_burn_references = options.verify && options.profile == OnnxProfile::Portable;
    let (tensor_audit, burn_reference_files) = std::thread::Builder::new()
        .name("boquilens-onnx-snapshot".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            let predictor =
                Predictor::<Flex>::from_checkpoint(model_id, checkpoint, PredictOptions::default())
                    .map_err(|error| format!("checkpoint loading failed: {error}"))?;
            let audit = snapshot::write_snapshot(&predictor, &snapshot_for_worker, spec)?;
            let references = if write_burn_references {
                snapshot::write_references(
                    &predictor,
                    snapshot_for_worker.parent().unwrap(),
                    input_shape,
                )?
            } else {
                Vec::new()
            };
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>((audit, references))
        })?
        .join()
        .map_err(|_| "snapshot materialization worker panicked")??;
    let weights_file_sha256 = sha256_file(&snapshot_path)?;
    let weights_sha256 = tensor_audit.content_sha256.clone();
    let burn_references = burn_reference_files
        .into_iter()
        .map(|(case, file)| {
            let sha256 = sha256_file(&intermediate.join(&file))?;
            Ok(BurnReferenceManifest {
                input_generator: case.clone(),
                case,
                file,
                sha256,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let output_name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("publication: ONNX output filename is not valid UTF-8")?;
    let sidecar_name = sidecar
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("publication: sidecar filename is not valid UTF-8")?;
    let staged_onnx = intermediate.join(output_name);
    let staged_sidecar = intermediate.join(sidecar_name);
    let staged_data = external_data_path(&staged_onnx);
    let (git_commit, git_dirty) = git_identity();
    let class_names = match spec.task {
        ExportTask::Classify => IMAGENET_CLASSES
            .iter()
            .map(|name| (*name).to_owned())
            .collect(),
        _ => COCO_CLASSES.iter().map(|name| (*name).to_owned()).collect(),
    };
    let preprocessing = match spec.family {
        ExportFamily::Yolox => {
            "top-left fit resize; pad bottom/right with 114; RGB float pixels in [0,255]"
        }
        _ if spec.task == ExportTask::Classify => {
            "anti-aliased shortest-edge resize; centered 224 crop; RGB divided by 255"
        }
        _ => "stride-aligned centered letterbox; fill 114; RGB divided by 255",
    };
    let graph_source = if spec.family == ExportFamily::Yolox {
        GraphSourceManifest {
            kind: "yolox".into(),
            expected_revision: "0.1.1rc0".into(),
            resolved_path: Some(yolox_repo.display().to_string()),
        }
    } else {
        GraphSourceManifest {
            kind: "ultralytics".into(),
            expected_revision: ULTRALYTICS_REVISION.into(),
            resolved_path: Some(ultralytics_repo.display().to_string()),
        }
    };
    let manifest = BridgeManifest {
        schema: "boquilens-onnx-export-input-v1".into(),
        export_spec_version: EXPORT_SPEC_VERSION.into(),
        model_id: model_id.as_str().into(),
        family: spec.family,
        task: spec.task,
        scale: spec.scale.into(),
        num_classes: spec.num_classes,
        class_names,
        stride: spec.stride,
        box_format: spec.box_format,
        nms: spec.nms,
        graph_config: spec.graph_config.into(),
        graph_source,
        key_map_version: spec.key_map_version.into(),
        checkpoint_file: weights
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("checkpoint")
            .into(),
        checkpoint_sha256,
        checkpoint_state: "exact-loaded-state".into(),
        weights_file: "weights.safetensors".into(),
        weights_file_sha256,
        weights_sha256,
        tensor_audit,
        burn_references,
        input: InputContract {
            name: "images".into(),
            dtype: "float32".into(),
            layout: "NCHW".into(),
            color: "RGB".into(),
            range: if spec.family == ExportFamily::Yolox {
                [0.0, 255.0]
            } else {
                [0.0, 1.0]
            },
            shape: options.input_shape,
            dynamic_batch: options.dynamic_batch,
            dynamic_spatial: options.dynamic_spatial,
            preprocessing: preprocessing.into(),
        },
        profile: options.profile,
        precision: options.precision,
        opset: options.opset,
        external_data: options.external_data,
        simplify: options.simplify,
        verify: options.verify,
        reproducible: options.reproducible,
        output_file: output_name.into(),
        sidecar_file: sidecar_name.into(),
        license: spec.license.into(),
        notice: "NOTICE".into(),
        boquilens_version: env!("CARGO_PKG_VERSION").into(),
        boquilens_git_commit: git_commit,
        boquilens_git_dirty: git_dirty,
    };
    let manifest_path = intermediate.join("input-manifest.json");
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)
        .map_err(|error| format!("writing export bridge manifest failed: {error}"))?;

    run_python(
        python,
        exporter,
        &["--manifest".into(), manifest_path.as_os_str().to_owned()],
        "Python ONNX export/validation",
    )?;
    if !staged_onnx.is_file() || !staged_sidecar.is_file() {
        return Err(
            "publication failed: Python exporter did not produce both staged artifacts".into(),
        );
    }
    let sha256 = sha256_file(&staged_onnx)?;
    let bytes = fs::metadata(&staged_onnx)?.len();

    if options.force {
        if output.exists() {
            fs::remove_file(output)?;
        }
        if sidecar.exists() {
            fs::remove_file(sidecar)?;
        }
        let output_data = external_data_path(output);
        if output_data.exists() {
            fs::remove_file(&output_data)?;
        }
    }
    let published_data = if staged_data.is_file() {
        let output_data = external_data_path(output);
        fs::rename(&staged_data, &output_data).map_err(|error| {
            format!(
                "publishing ONNX external data to {} failed: {error}",
                output_data.display()
            )
        })?;
        Some(output_data)
    } else {
        None
    };
    fs::rename(&staged_onnx, output).map_err(|error| {
        format!(
            "publishing ONNX file to {} failed: {error}",
            output.display()
        )
    })?;
    if let Err(error) = fs::rename(&staged_sidecar, sidecar) {
        return Err(format!(
            "ONNX file was published to {}, but publishing sidecar {} failed: {error}; the ONNX file is valid and recoverable",
            output.display(),
            sidecar.display()
        )
        .into());
    }
    Ok(OnnxArtifact {
        path: output.to_owned(),
        sidecar: sidecar.to_owned(),
        external_data: published_data,
        sha256,
        bytes,
    })
}

fn validate_options(
    spec: ExportSpec,
    weights: &Path,
    options: &mut OnnxExportOptions,
) -> Result<()> {
    if !weights.is_file() {
        return Err(format!("checkpoint loading: {} is not a file", weights.display()).into());
    }
    if !spec.supports_profile(options.profile) {
        return Err(format!(
            "argument validation: profile {:?} is not supported for {}",
            options.profile, spec.model_id
        )
        .into());
    }
    if options.precision != OnnxPrecision::Fp32 {
        return Err("argument validation: fp16 publication is disabled until its GPU parity gate is implemented; export fp32".into());
    }
    if options.checkpoint_state != CheckpointState::Ema {
        return Err("argument validation: raw-model selection is disabled until native multi-state training checkpoints are supported; current inputs are already resolved inference states".into());
    }
    if options.dynamic_batch || options.dynamic_spatial {
        return Err("argument validation: dynamic axes are disabled until their multi-shape parity gates are implemented".into());
    }
    if !matches!(options.opset, 17..=19) {
        return Err("argument validation: tested ONNX opsets are 17, 18, and 19".into());
    }
    let [batch, channels, height, width] = options.input_shape;
    if batch == 0 || channels != 3 || height == 0 || width == 0 {
        return Err(
            "argument validation: input shape must be [N,3,H,W] with positive dimensions".into(),
        );
    }
    if spec.task == ExportTask::Classify && (height != 224 || width != 224) {
        return Err("argument validation: classification export is fixed to 224x224".into());
    }
    if spec.task != ExportTask::Classify && (height % spec.stride != 0 || width % spec.stride != 0)
    {
        return Err(format!(
            "argument validation: input height and width must be divisible by stride {}",
            spec.stride
        )
        .into());
    }
    if options.output.extension().and_then(|ext| ext.to_str()) != Some("onnx") {
        options.output.set_extension("onnx");
        eprintln!(
            "ONNX output had no .onnx suffix; using {}",
            options.output.display()
        );
    }
    Ok(())
}

fn resolve_python(explicit: Option<&Path>) -> Result<PathBuf> {
    let candidate = explicit
        .map(Path::to_owned)
        .or_else(|| env::var_os("BOQUILENS_ONNX_PYTHON").map(PathBuf::from))
        .unwrap_or_else(|| {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/.venv");
            if cfg!(windows) {
                root.join("Scripts/python.exe")
            } else {
                root.join("bin/python")
            }
        });
    if !candidate.is_file() {
        return Err(format!(
            "Python environment preflight: {} is missing. Create the export environment with:\n  python -m venv target/.venv\n  {} -m pip install -r tools/onnx/requirements.lock.txt\nNo packages were installed automatically.",
            candidate.display(),
            candidate.display()
        )
        .into());
    }
    Ok(candidate)
}

fn run_python(
    python: &Path,
    exporter: &Path,
    args: &[std::ffi::OsString],
    layer: &str,
) -> Result<()> {
    let status = Command::new(python)
        .arg(exporter)
        .args(args)
        .env("BOQUILENS_ONNX_NO_NETWORK", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|error| format!("{layer}: failed to launch {}: {error}", python.display()))?;
    if !status.success() {
        return Err(format!("{layer} failed with status {status}").into());
    }
    Ok(())
}

fn create_intermediate(parent: &Path) -> Result<PathBuf> {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let path = parent.join(format!(".boquilens-onnx-{}-{nonce}", std::process::id()));
    fs::create_dir(&path).map_err(|error| {
        format!(
            "creating private export directory {} failed: {error}",
            path.display()
        )
    })?;
    Ok(path)
}

fn sidecar_path(output: &Path) -> PathBuf {
    let mut name = output.as_os_str().to_owned();
    name.push(".json");
    PathBuf::from(name)
}

fn external_data_path(output: &Path) -> PathBuf {
    let mut name = output.as_os_str().to_owned();
    name.push(".data");
    PathBuf::from(name)
}

fn refuse_or_validate_targets(
    output: &Path,
    sidecar: &Path,
    output_data: &Path,
    force: bool,
) -> Result<()> {
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    for target in [output, sidecar, output_data] {
        if target.exists() && !force {
            return Err(format!(
                "publication: {} already exists; pass --force to replace all export targets",
                target.display()
            )
            .into());
        }
        if target.exists() && !target.is_file() {
            return Err(format!("publication: {} is not a regular file", target.display()).into());
        }
    }
    Ok(())
}

fn absolute_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_owned())
    } else {
        Ok(env::current_dir()?.join(path))
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn git_identity() -> (String, bool) {
    let root = env!("CARGO_MANIFEST_DIR");
    let commit = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .unwrap_or_else(|| "unknown".into());
    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(root)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .is_some_and(|output| !output.stdout.is_empty());
    (commit, dirty)
}

fn family_name(family: ExportFamily) -> &'static str {
    match family {
        ExportFamily::Yolox => "yolox",
        ExportFamily::Yolov3Tiny => "yolov3-tiny",
        ExportFamily::Yolov10 => "yolov10",
        ExportFamily::Yolo11 => "yolo11",
        ExportFamily::Yolov8 => "yolov8",
        ExportFamily::Yolo12 => "yolo12",
        ExportFamily::Yolo26 => "yolo26",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_follow_task_contract() {
        let detect = OnnxExportOptions::for_model(ModelId::Yolo26N, "x.onnx".into());
        let classify = OnnxExportOptions::for_model(ModelId::Yolo26NCls, "x.onnx".into());
        assert_eq!(detect.input_shape, [1, 3, 640, 640]);
        assert_eq!(classify.input_shape, [1, 3, 224, 224]);
        assert_eq!(detect.opset, 17);
    }

    #[test]
    fn sidecar_uses_onnx_json_suffix() {
        assert_eq!(
            sidecar_path(Path::new("model.onnx")),
            PathBuf::from("model.onnx.json")
        );
    }
}
