use std::path::Path;

use burn::{
    module::{Module, Param},
    tensor::{Tensor, TensorData, backend::Backend},
};
use burn_flex::Flex;
use burn_store::{BurnToPyTorchAdapter, ModuleSnapshot, ModuleStore, SafetensorsStore};
use serde::Serialize;

use crate::{Predictor, Result, RuntimeModel};

use super::{keymap::reverse_rules, spec::ExportSpec};

#[derive(Module, Debug)]
struct YoloxReference<B: Backend> {
    predictions: Param<Tensor<B, 3>>,
}

#[derive(Module, Debug)]
struct DetectReference<B: Backend> {
    boxes: Param<Tensor<B, 3>>,
    scores: Param<Tensor<B, 3>>,
}

#[derive(Module, Debug)]
struct SegmentReference<B: Backend> {
    boxes: Param<Tensor<B, 3>>,
    scores: Param<Tensor<B, 3>>,
    coefficients: Param<Tensor<B, 3>>,
    prototypes: Param<Tensor<B, 4>>,
}

#[derive(Module, Debug)]
struct ClassifyReference<B: Backend> {
    logits: Param<Tensor<B, 2>>,
    probabilities: Param<Tensor<B, 2>>,
}

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

/// Run the loaded Burn model on the same deterministic inputs later used by PyTorch and ORT.
/// Outputs use the portable contract and are stored as SafeTensors so parity compares full tensors,
/// not summaries or a second checkpoint loader.
pub(crate) fn write_references(
    predictor: &Predictor<Flex>,
    directory: &Path,
    input_shape: [usize; 4],
) -> Result<Vec<(String, String)>> {
    let mut written = Vec::new();
    for case in ["zeros", "random-index-hash", "gradient-checkerboard"] {
        let input = reference_input(case, input_shape, &predictor.device);
        let filename = format!("burn-reference-{case}.safetensors");
        let path = directory.join(&filename);
        macro_rules! save_detect {
            ($output:expr) => {{
                let output = $output;
                let module = DetectReference {
                    boxes: Param::from_tensor(output.boxes),
                    scores: Param::from_tensor(output.scores),
                };
                save_reference(&module, &path)?;
            }};
        }
        macro_rules! save_classify {
            ($output:expr) => {{
                let output = $output;
                let module = ClassifyReference {
                    logits: Param::from_tensor(output.logits),
                    probabilities: Param::from_tensor(output.probs),
                };
                save_reference(&module, &path)?;
            }};
        }
        match &predictor.model {
            RuntimeModel::Yolox(model) => {
                let module = YoloxReference {
                    predictions: Param::from_tensor(model.forward(input)),
                };
                save_reference(&module, &path)?;
            }
            RuntimeModel::Yolov3Tiny(model) => save_detect!(model.forward(input)),
            RuntimeModel::Yolov10N(model) => save_detect!(model.forward(input)),
            RuntimeModel::Yolov10S(model) => save_detect!(model.forward(input)),
            RuntimeModel::Yolov10M(model) => save_detect!(model.forward(input)),
            RuntimeModel::Yolov10B(model) => save_detect!(model.forward(input)),
            RuntimeModel::Yolov10L(model) => save_detect!(model.forward(input)),
            RuntimeModel::Yolov10X(model) => save_detect!(model.forward(input)),
            RuntimeModel::Yolo11N(model) => save_detect!(model.forward(input)),
            RuntimeModel::Yolo11S(model) => save_detect!(model.forward(input)),
            RuntimeModel::Yolo11M(model) => save_detect!(model.forward(input)),
            RuntimeModel::Yolo11L(model) => save_detect!(model.forward(input)),
            RuntimeModel::Yolo11X(model) => save_detect!(model.forward(input)),
            RuntimeModel::Yolov8N(model) => save_detect!(model.forward(input)),
            RuntimeModel::Yolov8S(model) => save_detect!(model.forward(input)),
            RuntimeModel::Yolov8M(model) => save_detect!(model.forward(input)),
            RuntimeModel::Yolov8L(model) => save_detect!(model.forward(input)),
            RuntimeModel::Yolov8X(model) => save_detect!(model.forward(input)),
            RuntimeModel::Yolo12N(model) => save_detect!(model.forward(input)),
            RuntimeModel::Yolo12S(model) => save_detect!(model.forward(input)),
            RuntimeModel::Yolo12M(model) => save_detect!(model.forward(input)),
            RuntimeModel::Yolo12L(model) => save_detect!(model.forward(input)),
            RuntimeModel::Yolo12X(model) => save_detect!(model.forward(input)),
            RuntimeModel::Yolo26N(model) => save_detect!(model.forward(input)),
            RuntimeModel::Yolo26S(model) => save_detect!(model.forward(input)),
            RuntimeModel::Yolo26M(model) => save_detect!(model.forward(input)),
            RuntimeModel::Yolo26L(model) => save_detect!(model.forward(input)),
            RuntimeModel::Yolo26X(model) => save_detect!(model.forward(input)),
            RuntimeModel::Yolo11SegN(model) => {
                let output = model.forward(input);
                save_segment(
                    output.boxes,
                    output.scores,
                    output.coefficients,
                    output.prototypes,
                    &path,
                )?;
            }
            RuntimeModel::Yolo11SegS(model) => {
                let output = model.forward(input);
                save_segment(
                    output.boxes,
                    output.scores,
                    output.coefficients,
                    output.prototypes,
                    &path,
                )?;
            }
            RuntimeModel::Yolo11SegM(model) => {
                let output = model.forward(input);
                save_segment(
                    output.boxes,
                    output.scores,
                    output.coefficients,
                    output.prototypes,
                    &path,
                )?;
            }
            RuntimeModel::Yolo11SegL(model) => {
                let output = model.forward(input);
                save_segment(
                    output.boxes,
                    output.scores,
                    output.coefficients,
                    output.prototypes,
                    &path,
                )?;
            }
            RuntimeModel::Yolo11SegX(model) => {
                let output = model.forward(input);
                save_segment(
                    output.boxes,
                    output.scores,
                    output.coefficients,
                    output.prototypes,
                    &path,
                )?;
            }
            RuntimeModel::Yolov8SegN(model) => {
                let output = model.forward(input);
                save_segment(
                    output.boxes,
                    output.scores,
                    output.coefficients,
                    output.prototypes,
                    &path,
                )?;
            }
            RuntimeModel::Yolov8SegS(model) => {
                let output = model.forward(input);
                save_segment(
                    output.boxes,
                    output.scores,
                    output.coefficients,
                    output.prototypes,
                    &path,
                )?;
            }
            RuntimeModel::Yolov8SegM(model) => {
                let output = model.forward(input);
                save_segment(
                    output.boxes,
                    output.scores,
                    output.coefficients,
                    output.prototypes,
                    &path,
                )?;
            }
            RuntimeModel::Yolov8SegL(model) => {
                let output = model.forward(input);
                save_segment(
                    output.boxes,
                    output.scores,
                    output.coefficients,
                    output.prototypes,
                    &path,
                )?;
            }
            RuntimeModel::Yolov8SegX(model) => {
                let output = model.forward(input);
                save_segment(
                    output.boxes,
                    output.scores,
                    output.coefficients,
                    output.prototypes,
                    &path,
                )?;
            }
            RuntimeModel::Yolo26SegN(model) => {
                let output = model.forward(input);
                save_segment(
                    output.decoded.boxes,
                    output.decoded.scores,
                    output.coefficients,
                    output.prototypes,
                    &path,
                )?;
            }
            RuntimeModel::Yolo26SegS(model) => {
                let output = model.forward(input);
                save_segment(
                    output.decoded.boxes,
                    output.decoded.scores,
                    output.coefficients,
                    output.prototypes,
                    &path,
                )?;
            }
            RuntimeModel::Yolo26SegM(model) => {
                let output = model.forward(input);
                save_segment(
                    output.decoded.boxes,
                    output.decoded.scores,
                    output.coefficients,
                    output.prototypes,
                    &path,
                )?;
            }
            RuntimeModel::Yolo26SegL(model) => {
                let output = model.forward(input);
                save_segment(
                    output.decoded.boxes,
                    output.decoded.scores,
                    output.coefficients,
                    output.prototypes,
                    &path,
                )?;
            }
            RuntimeModel::Yolo26SegX(model) => {
                let output = model.forward(input);
                save_segment(
                    output.decoded.boxes,
                    output.decoded.scores,
                    output.coefficients,
                    output.prototypes,
                    &path,
                )?;
            }
            RuntimeModel::Yolo11ClsN(model) => save_classify!(model.forward(input)),
            RuntimeModel::Yolo11ClsS(model) => save_classify!(model.forward(input)),
            RuntimeModel::Yolo11ClsM(model) => save_classify!(model.forward(input)),
            RuntimeModel::Yolo11ClsL(model) => save_classify!(model.forward(input)),
            RuntimeModel::Yolo11ClsX(model) => save_classify!(model.forward(input)),
            RuntimeModel::Yolov8ClsN(model) => save_classify!(model.forward(input)),
            RuntimeModel::Yolov8ClsS(model) => save_classify!(model.forward(input)),
            RuntimeModel::Yolov8ClsM(model) => save_classify!(model.forward(input)),
            RuntimeModel::Yolov8ClsL(model) => save_classify!(model.forward(input)),
            RuntimeModel::Yolov8ClsX(model) => save_classify!(model.forward(input)),
            RuntimeModel::Yolo26ClsN(model) => save_classify!(model.forward(input)),
            RuntimeModel::Yolo26ClsS(model) => save_classify!(model.forward(input)),
            RuntimeModel::Yolo26ClsM(model) => save_classify!(model.forward(input)),
            RuntimeModel::Yolo26ClsL(model) => save_classify!(model.forward(input)),
            RuntimeModel::Yolo26ClsX(model) => save_classify!(model.forward(input)),
        }
        written.push((case.into(), filename));
    }
    Ok(written)
}

fn save_segment(
    boxes: Tensor<Flex, 3>,
    scores: Tensor<Flex, 3>,
    coefficients: Tensor<Flex, 3>,
    prototypes: Tensor<Flex, 4>,
    path: &Path,
) -> Result<()> {
    let module = SegmentReference {
        boxes: Param::from_tensor(boxes),
        scores: Param::from_tensor(scores),
        coefficients: Param::from_tensor(coefficients),
        prototypes: Param::from_tensor(prototypes),
    };
    save_reference(&module, path)
}

fn save_reference<M: Module<Flex> + ModuleSnapshot<Flex>>(module: &M, path: &Path) -> Result<()> {
    let mut store =
        SafetensorsStore::from_file(path).metadata("boquilens.schema", "onnx-burn-reference-v1");
    module
        .save_into(&mut store)
        .map_err(|error| format!("Burn reference serialization failed: {error}").into())
}

fn reference_input(
    case: &str,
    shape: [usize; 4],
    device: &burn::tensor::Device<Flex>,
) -> Tensor<Flex, 4> {
    let [batch, channels, height, width] = shape;
    let count = batch * channels * height * width;
    let mut values = Vec::with_capacity(count);
    for index in 0..count {
        let value = match case {
            "zeros" => 0.0,
            "random-index-hash" => {
                let hashed = (index as u64)
                    .wrapping_mul(1_103_515_245)
                    .wrapping_add(12_345)
                    & 0x00ff_ffff;
                hashed as f32 / 16_777_215.0
            }
            "gradient-checkerboard" => {
                let pixel = index % (height * width);
                let channel = (index / (height * width)) % channels;
                let y = pixel / width;
                let x = pixel % width;
                match channel {
                    0 => x as f32 / width.saturating_sub(1).max(1) as f32,
                    1 => y as f32 / height.saturating_sub(1).max(1) as f32,
                    _ => ((x / 16 + y / 16) % 2) as f32,
                }
            }
            _ => unreachable!(),
        };
        values.push(value);
    }
    Tensor::from_data(TensorData::new(values, shape), device)
}
