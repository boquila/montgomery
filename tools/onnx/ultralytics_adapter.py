"""Pinned Ultralytics graph construction and strict SafeTensors loading."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

from contracts import UltralyticsCompatible, UltralyticsPortable


def _import_source(root: Path):
    root = root.resolve(strict=True)
    sys.path.insert(0, str(root))
    from ultralytics.nn.tasks import ClassificationModel, DetectionModel, SegmentationModel
    import ultralytics

    resolved = Path(ultralytics.__file__).resolve()
    if root not in resolved.parents:
        raise RuntimeError(f"resolved Ultralytics import {resolved} is outside pinned source {root}")
    return DetectionModel, SegmentationModel, ClassificationModel


def _construct(manifest: dict, root: Path) -> torch.nn.Module:
    DetectionModel, SegmentationModel, ClassificationModel = _import_source(root)
    task = manifest["task"]
    cfg = manifest["graph_config"]
    classes = int(manifest["num_classes"])
    if task == "classify":
        model = ClassificationModel(cfg=cfg, ch=3, nc=classes, verbose=False)
    elif task == "segment":
        model = SegmentationModel(cfg=cfg, ch=3, nc=classes, verbose=False)
    else:
        model = DetectionModel(cfg=cfg, ch=3, nc=classes, verbose=False)
    # Released v8/v11 checkpoints predate the source refactor that changed SPPF.cv1 to act=False.
    # The native Burn graph intentionally follows the pickled checkpoint module, so the adapter
    # must restore that parameter-free SiLU before parity/export.
    if manifest["family"] in {"yolov8", "yolov10", "yolo11"} and task != "classify":
        model.model[9].cv1.act = torch.nn.SiLU(inplace=True)
    return model


def _allowed_missing(key: str, manifest: dict) -> bool:
    if key.endswith("num_batches_tracked") or key.endswith("dfl.conv.weight"):
        return True
    if manifest["family"] in {"yolov10", "yolo26"}:
        head = key.split(".", 2)
        suffix = head[2] if len(head) == 3 and head[0] == "model" else ""
        if suffix.startswith(("cv2.", "cv3.", "cv4.")):
            return True
        if ".proto.semseg." in key:
            return True
    return False


def _load_strict(model: torch.nn.Module, weights: Path, manifest: dict, workdir: Path) -> None:
    state = load_file(str(weights), device="cpu")
    if manifest["family"] == "yolo12":
        modules = dict(model.named_modules())
        for key, value in state.items():
            if key.endswith(".attn.pe.conv.bias"):
                module_name = key.removesuffix(".bias")
                module = modules.get(module_name)
                if not isinstance(module, torch.nn.Conv2d) or module.bias is not None:
                    raise RuntimeError(f"cannot restore checkpoint-era YOLO12 attention bias {key}")
                module.bias = torch.nn.Parameter(torch.zeros_like(value))
    expected = model.state_dict()
    wrong_shape = sorted(
        key
        for key in state.keys() & expected.keys()
        if tuple(state[key].shape) != tuple(expected[key].shape)
    )
    unexpected = sorted(state.keys() - expected.keys())
    missing = sorted(key for key in expected.keys() - state.keys() if not _allowed_missing(key, manifest))
    if wrong_shape or unexpected or missing:
        (workdir / "state-wrong-shape.txt").write_text("\n".join(wrong_shape) + "\n", encoding="utf-8")
        (workdir / "state-unexpected.txt").write_text("\n".join(unexpected) + "\n", encoding="utf-8")
        (workdir / "state-missing.txt").write_text("\n".join(missing) + "\n", encoding="utf-8")
        raise RuntimeError(
            "strict state-dict audit failed: "
            f"{len(missing)} missing, {len(unexpected)} unexpected, {len(wrong_shape)} wrong-shape keys; "
            f"complete lists are under {workdir}"
        )
    result = model.load_state_dict(state, strict=False)
    residual_missing = [key for key in result.missing_keys if not _allowed_missing(key, manifest)]
    if result.unexpected_keys or residual_missing:
        raise RuntimeError(
            f"state-dict loading failed: missing={residual_missing}, unexpected={result.unexpected_keys}"
        )


def build(manifest: dict, workdir: Path) -> tuple[torch.nn.Module, list[str]]:
    source = Path(manifest["graph_source"]["resolved_path"])
    model = _construct(manifest, source).eval().float()
    weights = (workdir / manifest["weights_file"]).resolve(strict=True)
    _load_strict(model, weights, manifest, workdir)
    if manifest["profile"] == "portable":
        wrapper = UltralyticsPortable(
            model,
            manifest["task"],
            manifest["family"],
            int(manifest["num_classes"]),
        )
    else:
        wrapper = UltralyticsCompatible(model, manifest["task"])
    from contracts import output_names

    return wrapper.eval(), output_names(manifest["task"], manifest["profile"], manifest["family"])
