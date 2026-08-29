"""ONNX emission, structural validation, ORT execution, metadata and sidecar publication."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, shape_inference
from safetensors.numpy import load_file as load_safetensors

from compare import compare_output

STANDARD_OPERATORS = {
    "Abs", "Add", "AveragePool", "BatchNormalization", "Cast", "Ceil", "Clip", "Concat",
    "Constant", "ConstantOfShape", "Conv", "ConvTranspose", "Div", "Equal", "Erf", "Exp",
    "Expand", "Flatten", "Floor", "Gather", "GatherElements", "GatherND", "Gemm",
    "GlobalAveragePool", "Greater", "Identity", "Less", "MatMul", "Max", "MaxPool", "Min",
    "Mul", "Neg", "NonZero", "Pad", "Pow", "Range", "Reciprocal", "ReduceMax", "ReduceMean",
    "ReduceSum", "Relu", "Reshape", "Resize", "ScatterND", "Shape", "Sigmoid", "Slice",
    "Softmax", "Split", "Sqrt", "Squeeze", "Sub", "Tile", "TopK", "Transpose", "Unsqueeze",
    "Where",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _outputs(value) -> tuple[torch.Tensor, ...]:
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, (tuple, list)) and all(isinstance(item, torch.Tensor) for item in value):
        return tuple(value)
    raise RuntimeError(f"export wrapper returned unsupported output type {type(value)!r}")


def _metadata(manifest: dict) -> dict[str, str]:
    result = {
        "boquilens.schema": "onnx-metadata-v1",
        "boquilens.model_id": manifest["model_id"],
        "boquilens.family": manifest["family"],
        "boquilens.task": manifest["task"],
        "boquilens.profile": manifest["profile"],
        "boquilens.input.layout": manifest["input"]["layout"],
        "boquilens.input.color": manifest["input"]["color"],
        "boquilens.input.range": ",".join(str(value) for value in manifest["input"]["range"]),
        "boquilens.input.shape": ",".join(str(value) for value in manifest["input"]["shape"]),
        "boquilens.stride": str(manifest["stride"]),
        "boquilens.class_names": json.dumps(manifest["class_names"], separators=(",", ":")),
        "boquilens.checkpoint.sha256": manifest["checkpoint_sha256"],
        "boquilens.weights.sha256": manifest["weights_sha256"],
        "boquilens.exporter.version": "bridge-v1",
        "boquilens.git.commit": manifest["boquilens_git_commit"],
        "boquilens.graph_source": (
            f'{manifest["graph_source"]["kind"]}@{manifest["graph_source"]["expected_revision"]}'
        ),
        "boquilens.nms": str(manifest["nms"]).lower(),
        "model_license": manifest["license"],
        "model_notice": manifest["notice"],
    }
    if manifest.get("box_format"):
        result["boquilens.box.format"] = manifest["box_format"]
        result["boquilens.box.space"] = "model-input-pixels"
    if manifest["task"] == "segment":
        result.update(
            {
                "boquilens.mask.coefficients": "32",
                "boquilens.mask.coefficients_raw": "true",
                "boquilens.mask.prototype_stride": "4",
                "boquilens.mask.threshold": "logit>0",
            }
        )
    if not manifest["reproducible"]:
        result["boquilens.exported_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return result


def _set_metadata(model: onnx.ModelProto, metadata: dict[str, str]) -> None:
    del model.metadata_props[:]
    for key, value in sorted(metadata.items()):
        item = model.metadata_props.add()
        item.key = key
        item.value = value


def _expected_ranks(manifest: dict, names: list[str]) -> dict[str, int]:
    if manifest["profile"] == "ultralytics":
        return {name: (4 if name == "output1" else 3 if manifest["task"] != "classify" else 2) for name in names}
    ranks = {"predictions": 3, "boxes": 3, "scores": 3, "coefficients": 3, "prototypes": 4,
             "logits": 2, "probabilities": 2}
    return {name: ranks[name] for name in names}


def validate_model(path: Path, manifest: dict, output_names: list[str]) -> tuple[onnx.ModelProto, list[str]]:
    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    inferred = shape_inference.infer_shapes(model, check_type=True, strict_mode=True, data_prop=True)
    onnx.checker.check_model(inferred, full_check=True)
    if [item.name for item in model.graph.input] != ["images"]:
        raise RuntimeError(f"ONNX input contract mismatch: {[item.name for item in model.graph.input]}")
    actual_names = [item.name for item in model.graph.output]
    if actual_names != output_names:
        raise RuntimeError(f"ONNX output names mismatch: expected {output_names}, got {actual_names}")
    ranks = _expected_ranks(manifest, output_names)
    for item in inferred.graph.output:
        tensor_type = item.type.tensor_type
        if tensor_type.elem_type != TensorProto.FLOAT:
            raise RuntimeError(f"{item.name}: expected float32 output")
        if len(tensor_type.shape.dim) != ranks[item.name]:
            raise RuntimeError(f"{item.name}: expected rank {ranks[item.name]}")
    initializers = [item.name for item in model.graph.initializer]
    if len(initializers) != len(set(initializers)):
        raise RuntimeError("ONNX initializer names are not unique")
    metadata_keys = [item.key for item in model.metadata_props]
    if len(metadata_keys) != len(set(metadata_keys)):
        raise RuntimeError("ONNX metadata keys are not unique")
    operators = sorted({f"{node.domain or 'ai.onnx'}::{node.op_type}" for node in model.graph.node})
    bad_domains = sorted({node.domain for node in model.graph.node if node.domain not in {"", "ai.onnx"}})
    bad_ops = sorted({node.op_type for node in model.graph.node if node.op_type not in STANDARD_OPERATORS})
    if bad_domains or bad_ops:
        raise RuntimeError(f"unexpected ONNX operators/domains: domains={bad_domains}, operators={bad_ops}")
    return model, operators


def _cases(shape: list[int]) -> Iterable[tuple[str, np.ndarray]]:
    yield "zeros", np.zeros(shape, dtype=np.float32)
    indices = np.arange(np.prod(shape), dtype=np.uint64)
    hashed = (indices * np.uint64(1_103_515_245) + np.uint64(12_345)) & np.uint64(0x00FF_FFFF)
    yield "random-index-hash", (hashed.astype(np.float32) / np.float32(16_777_215.0)).reshape(shape)
    _, _, height, width = shape
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, height, dtype=np.float32),
        np.linspace(0.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )
    structured = np.stack((xx, yy, ((np.indices((height, width)).sum(0) // 16) % 2).astype(np.float32)))
    yield "gradient-checkerboard", np.broadcast_to(structured, shape).copy()


def validate_runtime(
    wrapper: torch.nn.Module,
    path: Path,
    manifest: dict,
    output_names: list[str],
) -> list[dict[str, object]]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    reports: list[dict[str, object]] = []
    burn_references = {item["case"]: item for item in manifest.get("burn_references", [])}
    for case_name, input_array in _cases(manifest["input"]["shape"]):
        with torch.no_grad():
            expected = _outputs(wrapper(torch.from_numpy(input_array)))
        actual = session.run(output_names, {"images": input_array})
        if len(expected) != len(actual):
            raise RuntimeError(f"{case_name}: output count mismatch")
        burn = None
        if burn_references:
            reference = burn_references.get(case_name)
            if reference is None:
                raise RuntimeError(f"missing Burn reference for validation case {case_name}")
            reference_path = path.parent / reference["file"]
            if file_sha256(reference_path) != reference["sha256"]:
                raise RuntimeError(f"Burn reference hash mismatch for {case_name}")
            burn = load_safetensors(str(reference_path))
            if sorted(burn) != sorted(output_names):
                raise RuntimeError(
                    f"{case_name}: Burn reference outputs {sorted(burn)} do not match {output_names}"
                )
        for name, torch_value, ort_value in zip(output_names, expected, actual):
            torch_array = torch_value.detach().cpu().numpy()
            if burn is not None:
                burn_report = compare_output(name, burn[name], torch_array)
                burn_report["case"] = case_name
                burn_report["comparison"] = "burn-vs-pytorch"
                reports.append(burn_report)
            report = compare_output(name, torch_array, ort_value)
            report["case"] = case_name
            report["comparison"] = "pytorch-vs-onnxruntime"
            reports.append(report)
    return reports


def export_and_validate(
    wrapper: torch.nn.Module,
    manifest: dict,
    workdir: Path,
    output_names: list[str],
    versions: dict[str, str],
) -> None:
    output = (workdir / manifest["output_file"]).resolve()
    sidecar = (workdir / manifest["sidecar_file"]).resolve()
    if output.parent != workdir.resolve() or sidecar.parent != workdir.resolve():
        raise RuntimeError("manifest output path escapes the private export directory")
    shape = manifest["input"]["shape"]
    sample = torch.zeros(shape, dtype=torch.float32)
    with torch.no_grad():
        dry_outputs = _outputs(wrapper(sample))
    if len(dry_outputs) != len(output_names):
        raise RuntimeError("dry run output count does not match the declared contract")

    torch.onnx.export(
        wrapper,
        (sample,),
        str(output),
        input_names=["images"],
        output_names=output_names,
        opset_version=int(manifest["opset"]),
        export_params=True,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
        dynamic_axes=None,
        dynamo=False,
    )
    model = onnx.load(str(output), load_external_data=True)
    _set_metadata(model, _metadata(manifest))
    onnx.save(model, str(output))

    if manifest["simplify"]:
        try:
            import onnxslim
        except ImportError as error:
            raise RuntimeError("--simplify requires the pinned optional onnxslim package") from error
        simplified = onnxslim.slim(onnx.load(str(output)))
        _set_metadata(simplified, _metadata(manifest))
        onnx.save(simplified, str(output))

    external = manifest["external_data"] == "always" or (
        manifest["external_data"] == "auto" and output.stat().st_size >= 1_500_000_000
    )
    if external:
        model = onnx.load(str(output), load_external_data=True)
        onnx.save_model(
            model,
            str(output),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=output.name + ".data",
            size_threshold=0,
            convert_attribute=False,
        )

    checked, operators = validate_model(output, manifest, output_names)
    parity = validate_runtime(wrapper, output, manifest, output_names) if manifest["verify"] else []
    artifacts = []
    for path in [output, Path(str(output) + ".data")]:
        if not path.is_file():
            continue
        artifacts.append({"filename": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    final = {
        "schema": "boquilens-onnx-artifact-v1",
        "model_id": manifest["model_id"],
        "family": manifest["family"],
        "task": manifest["task"],
        "scale": manifest["scale"],
        "class_names": manifest["class_names"],
        "checkpoint": {
            "filename": manifest["checkpoint_file"],
            "sha256": manifest["checkpoint_sha256"],
            "state": manifest["checkpoint_state"],
        },
        "weights_snapshot_sha256": manifest["weights_sha256"],
        "versions": {
            **versions,
            "boquilens": manifest["boquilens_version"],
            "boquilens_git_commit": manifest["boquilens_git_commit"],
            "boquilens_git_dirty": manifest["boquilens_git_dirty"],
            "graph_source": manifest["graph_source"]["expected_revision"],
        },
        "opset": int(manifest["opset"]),
        "ir_version": checked.ir_version,
        "input": manifest["input"],
        "outputs": [
            {"name": value.name, "dtype": "float32", "shape": [dim.dim_value or dim.dim_param or None for dim in value.type.tensor_type.shape.dim]}
            for value in checked.graph.output
        ],
        "profile": manifest["profile"],
        "precision": manifest["precision"],
        "box_format": manifest.get("box_format"),
        "nms": manifest["nms"],
        "postprocessing": postprocessing(manifest),
        "license": manifest["license"],
        "notice": manifest["notice"],
        "operator_inventory": operators,
        "simplified": bool(manifest["simplify"]),
        "external_data": external,
        "validation": {"onnx_checker": True, "strict_shape_inference": True, "onnxruntime_cpu": bool(manifest["verify"]), "cases": parity},
        "artifacts": artifacts,
        "reproducibility": {"timestamp_omitted": bool(manifest["reproducible"]), "dirty_source": manifest["boquilens_git_dirty"]},
    }
    if not manifest["reproducible"]:
        final["exported_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    sidecar.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def postprocessing(manifest: dict) -> list[str]:
    family = manifest["family"]
    if family == "yolox":
        steps = ["multiply objectness by each class probability", "convert XYWH to XYXY", "confidence filter", "class-aware greedy NMS", "reverse top-left padding geometry"]
    elif family in {"yolov10", "yolo26"}:
        steps = ["best class per anchor", "top-k anchors by best score", "top-k flattened anchor/class pairs", "gather boxes/classes/scores", "confidence filter", "no NMS", "limit to 300", "reverse letterbox geometry"]
    else:
        steps = ["choose class scores", "confidence filter", "class-aware greedy NMS", "convert XYWH to XYXY when declared", "reverse letterbox geometry"]
    if manifest["task"] == "segment":
        steps += ["gather raw coefficients", "coefficients @ prototypes", "bilinear upsample with align_corners=false", "threshold logits > 0", "crop to box", "drop empty masks", "sample canvas mask into source pixels"]
    return steps
