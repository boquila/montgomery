"""Stable task/profile wrappers for repository-owned ONNX output contracts."""

from __future__ import annotations

import torch


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center = boxes[..., :2]
    half = boxes[..., 2:] / 2
    return torch.cat((center - half, center + half), dim=-1)


class UltralyticsPortable(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, task: str, family: str, classes: int):
        super().__init__()
        self.model = model
        self.task = task
        self.family = family
        self.classes = classes

    def forward(self, images: torch.Tensor):
        result = self.model(images)
        if self.task == "classify":
            probabilities, logits = result
            return logits, probabilities

        head = self.model.model[-1]
        if self.task == "segment":
            (decoded, prototypes), raw = result
            if head.end2end:
                decoded = head._inference(raw["one2one"])
            boxes = decoded[:, :4].permute(0, 2, 1)
            scores = decoded[:, 4 : 4 + self.classes].permute(0, 2, 1)
            coefficients = decoded[:, 4 + self.classes :]
            return boxes, scores, coefficients, prototypes

        decoded, raw = result
        if head.end2end:
            decoded = head._inference(raw["one2one"])
        boxes = decoded[:, :4].permute(0, 2, 1)
        if self.family == "yolov3-tiny":
            boxes = _xywh_to_xyxy(boxes)
        scores = decoded[:, 4 : 4 + self.classes].permute(0, 2, 1)
        return boxes, scores


class UltralyticsCompatible(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, task: str):
        super().__init__()
        self.model = model
        self.task = task
        self.model.model[-1].export = True
        self.model.model[-1].format = "onnx"

    def forward(self, images: torch.Tensor):
        result = self.model(images)
        if self.task == "segment":
            return result[0], result[1]
        return result


class YoloxPortable(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        predictions = self.model(images)
        boxes = _xywh_to_xyxy(predictions[..., :4])
        scores = predictions[..., 4:5] * predictions[..., 5:]
        return boxes, scores


def output_names(task: str, profile: str, family: str) -> list[str]:
    if profile == "ultralytics":
        return ["output0", "output1"] if task == "segment" else ["output0"]
    if task == "detect":
        return ["boxes", "scores"]
    if task == "segment":
        return ["boxes", "scores", "coefficients", "prototypes"]
    return ["logits", "probabilities"]
