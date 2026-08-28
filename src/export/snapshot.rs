use std::path::Path;

use burn_flex::Flex;
use burn_store::{BurnToPyTorchAdapter, ModuleSnapshot, ModuleStore, SafetensorsStore};
use serde::Serialize;

use crate::{Predictor, Result, RuntimeModel};

use super::{keymap::reverse_rules, spec::ExportSpec};

#[derive(Debug, Clone, Serialize)]
pub struct TensorAudit {
    pub tensor_count: usize,
    pub scalar_count: usize,
    pub tensors: Vec<TensorAuditEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TensorAuditEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}

pub(crate) fn write_snapshot(
    predictor: &Predictor<Flex>,
    path: &Path,
    spec: ExportSpec,
) -> Result<TensorAudit> {
    let mut store = SafetensorsStore::from_file(path)
        .with_to_adapter(BurnToPyTorchAdapter)
        .skip_enum_variants(true)
        .metadata("boquilens.schema", "onnx-parameter-snapshot-v1")
        .metadata("boquilens.model_id", spec.model_id.as_str())
        .metadata("boquilens.key_map_version", spec.key_map_version);
    for (from, to) in reverse_rules(spec) {
        store = store.with_key_remapping(from, to);
    }

    macro_rules! save {
        ($model:expr) => {
            $model
                .save_into(&mut store)
                .map_err(|error| format!("snapshot materialization failed: {error}"))?
        };
    }
    match &predictor.model {
        RuntimeModel::Yolox(model) => save!(model.as_ref()),
        RuntimeModel::Yolov3Tiny(model) => save!(model.as_ref()),
        RuntimeModel::Yolov10N(model) => save!(model.as_ref()),
        RuntimeModel::Yolov10S(model) => save!(model.as_ref()),
        RuntimeModel::Yolov10M(model) => save!(model.as_ref()),
        RuntimeModel::Yolov10B(model) => save!(model.as_ref()),
        RuntimeModel::Yolov10L(model) => save!(model.as_ref()),
        RuntimeModel::Yolov10X(model) => save!(model.as_ref()),
        RuntimeModel::Yolo11N(model) => save!(model.as_ref()),
        RuntimeModel::Yolo11S(model) => save!(model.as_ref()),
        RuntimeModel::Yolo11M(model) => save!(model.as_ref()),
        RuntimeModel::Yolo11L(model) => save!(model.as_ref()),
        RuntimeModel::Yolo11X(model) => save!(model.as_ref()),
        RuntimeModel::Yolo11SegN(model) => save!(model.as_ref()),
        RuntimeModel::Yolo11SegS(model) => save!(model.as_ref()),
        RuntimeModel::Yolo11SegM(model) => save!(model.as_ref()),
        RuntimeModel::Yolo11SegL(model) => save!(model.as_ref()),
        RuntimeModel::Yolo11SegX(model) => save!(model.as_ref()),
        RuntimeModel::Yolo11ClsN(model) => save!(model.as_ref()),
        RuntimeModel::Yolo11ClsS(model) => save!(model.as_ref()),
        RuntimeModel::Yolo11ClsM(model) => save!(model.as_ref()),
        RuntimeModel::Yolo11ClsL(model) => save!(model.as_ref()),
        RuntimeModel::Yolo11ClsX(model) => save!(model.as_ref()),
        RuntimeModel::Yolov8N(model) => save!(model.as_ref()),
        RuntimeModel::Yolov8S(model) => save!(model.as_ref()),
        RuntimeModel::Yolov8M(model) => save!(model.as_ref()),
        RuntimeModel::Yolov8L(model) => save!(model.as_ref()),
        RuntimeModel::Yolov8X(model) => save!(model.as_ref()),
        RuntimeModel::Yolov8SegN(model) => save!(model.as_ref()),
        RuntimeModel::Yolov8SegS(model) => save!(model.as_ref()),
        RuntimeModel::Yolov8SegM(model) => save!(model.as_ref()),
        RuntimeModel::Yolov8SegL(model) => save!(model.as_ref()),
        RuntimeModel::Yolov8SegX(model) => save!(model.as_ref()),
        RuntimeModel::Yolov8ClsN(model) => save!(model.as_ref()),
        RuntimeModel::Yolov8ClsS(model) => save!(model.as_ref()),
        RuntimeModel::Yolov8ClsM(model) => save!(model.as_ref()),
        RuntimeModel::Yolov8ClsL(model) => save!(model.as_ref()),
        RuntimeModel::Yolov8ClsX(model) => save!(model.as_ref()),
        RuntimeModel::Yolo12N(model) => save!(model.as_ref()),
        RuntimeModel::Yolo12S(model) => save!(model.as_ref()),
        RuntimeModel::Yolo12M(model) => save!(model.as_ref()),
        RuntimeModel::Yolo12L(model) => save!(model.as_ref()),
        RuntimeModel::Yolo12X(model) => save!(model.as_ref()),
        RuntimeModel::Yolo26N(model) => save!(model.as_ref()),
        RuntimeModel::Yolo26S(model) => save!(model.as_ref()),
        RuntimeModel::Yolo26M(model) => save!(model.as_ref()),
        RuntimeModel::Yolo26L(model) => save!(model.as_ref()),
        RuntimeModel::Yolo26X(model) => save!(model.as_ref()),
        RuntimeModel::Yolo26SegN(model) => save!(model.as_ref()),
        RuntimeModel::Yolo26SegS(model) => save!(model.as_ref()),
        RuntimeModel::Yolo26SegM(model) => save!(model.as_ref()),
        RuntimeModel::Yolo26SegL(model) => save!(model.as_ref()),
        RuntimeModel::Yolo26SegX(model) => save!(model.as_ref()),
        RuntimeModel::Yolo26ClsN(model) => save!(model.as_ref()),
        RuntimeModel::Yolo26ClsS(model) => save!(model.as_ref()),
        RuntimeModel::Yolo26ClsM(model) => save!(model.as_ref()),
        RuntimeModel::Yolo26ClsL(model) => save!(model.as_ref()),
        RuntimeModel::Yolo26ClsX(model) => save!(model.as_ref()),
    }

    let mut read_store = SafetensorsStore::from_file(path);
    let tensors = read_store
        .get_all_snapshots()
        .map_err(|error| format!("snapshot audit failed: {error}"))?;
    let entries = tensors
        .iter()
        .map(|(name, tensor)| TensorAuditEntry {
            name: name.clone(),
            shape: tensor.shape.iter().copied().collect(),
            dtype: format!("{:?}", tensor.dtype).to_lowercase(),
        })
        .collect::<Vec<_>>();
    let scalar_count = entries
        .iter()
        .map(|entry| entry.shape.iter().product::<usize>())
        .sum();
    Ok(TensorAudit {
        tensor_count: entries.len(),
        scalar_count,
        tensors: entries,
    })
}
