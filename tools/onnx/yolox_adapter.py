"""Official YOLOX 0.1.1rc0 graph adapter without installing its extension package."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

from contracts import YoloxPortable

SCALES = {
    "yolox-nano": (0.33, 0.25, True),
    "yolox-tiny": (0.33, 0.375, False),
    "yolox-s": (0.33, 0.50, False),
    "yolox-m": (0.67, 0.75, False),
    "yolox-l": (1.00, 1.00, False),
    "yolox-x": (1.33, 1.25, False),
}

BOXES = '''import torch
def bboxes_iou(a, b, xyxy=True):
    tl = torch.max(a[:, None, :2], b[:, :2])
    br = torch.min(a[:, None, 2:], b[:, 2:])
    area_a = torch.prod(a[:, 2:] - a[:, :2], 1)
    area_b = torch.prod(b[:, 2:] - b[:, :2], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en
    return area_i / (area_a[:, None] + area_b - area_i)
'''
LOGURU = '''class _Logger:
    def _noop(self, *args, **kwargs): pass
    error = warning = info = debug = success = _noop
logger = _Logger()
'''


def _shim(source: Path, workdir: Path) -> Path:
    target = workdir / "yolox-source-shim"
    models = target / "yolox/models"
    utils = target / "yolox/utils"
    models.mkdir(parents=True)
    utils.mkdir(parents=True)
    for name in ("network_blocks.py", "darknet.py", "yolo_pafpn.py", "yolo_head.py", "yolox.py", "losses.py"):
        shutil.copyfile(source / "yolox/models" / name, models / name)
    (target / "yolox/__init__.py").write_text("", encoding="utf-8")
    (models / "__init__.py").write_text("", encoding="utf-8")
    (utils / "__init__.py").write_text("from .boxes import bboxes_iou\n", encoding="utf-8")
    (utils / "boxes.py").write_text(BOXES, encoding="utf-8")
    (target / "loguru.py").write_text(LOGURU, encoding="utf-8")
    return target


def build(manifest: dict, workdir: Path) -> tuple[torch.nn.Module, list[str]]:
    source = Path(manifest["graph_source"]["resolved_path"]).resolve(strict=True)
    sys.path.insert(0, str(_shim(source, workdir)))
    from yolox.models.yolo_head import YOLOXHead
    from yolox.models.yolo_pafpn import YOLOPAFPN
    from yolox.models.yolox import YOLOX

    depth, width, depthwise = SCALES[manifest["model_id"]]
    model = YOLOX(
        YOLOPAFPN(depth, width, in_channels=[256, 512, 1024], depthwise=depthwise, act="silu"),
        YOLOXHead(
            int(manifest["num_classes"]),
            width,
            in_channels=[256, 512, 1024],
            depthwise=depthwise,
            act="silu",
        ),
    ).eval().float()
    state = load_file(str((workdir / manifest["weights_file"]).resolve(strict=True)), device="cpu")
    expected = model.state_dict()
    missing = sorted(key for key in expected.keys() - state.keys() if not key.endswith("num_batches_tracked"))
    unexpected = sorted(state.keys() - expected.keys())
    wrong = sorted(key for key in state.keys() & expected.keys() if state[key].shape != expected[key].shape)
    if missing or unexpected or wrong:
        raise RuntimeError(
            f"YOLOX strict state audit failed: {len(missing)} missing, {len(unexpected)} unexpected, {len(wrong)} wrong-shape"
        )
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or [key for key in result.missing_keys if not key.endswith("num_batches_tracked")]:
        raise RuntimeError(f"YOLOX state load failed: {result}")
    return YoloxPortable(model).eval(), ["predictions"]
