"""Strict, offline ONNX export environment checks."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

ULTRALYTICS_REVISION = "461196cf09175b64c9b9bd8babebf081c0540520"


class PreflightError(RuntimeError):
    pass


def _require_modules() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    missing: list[str] = []
    for module_name, version_name in [
        ("torch", "torch"),
        ("onnx", "onnx"),
        ("onnxruntime", "onnxruntime"),
        ("safetensors", "safetensors"),
        ("numpy", "numpy"),
    ]:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            missing.append(module_name)
        else:
            versions[version_name] = str(getattr(module, "__version__", "unknown"))
    if missing:
        joined = ", ".join(sorted(missing))
        raise PreflightError(
            f"missing locked ONNX export packages: {joined}. Install explicitly with:\n"
            f"  {sys.executable} -m pip install -r tools/onnx/requirements.lock.txt\n"
            "No packages were installed and no network request was made by boquilens."
        )
    return versions


def _git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise PreflightError(f"cannot read graph-source revision at {path}: {result.stderr.strip()}")
    return result.stdout.strip()


def verify_ultralytics(path: Path) -> Path:
    root = path.resolve(strict=True)
    revision = _git_revision(root)
    if revision != ULTRALYTICS_REVISION:
        raise PreflightError(
            f"Ultralytics source mismatch at {root}: expected {ULTRALYTICS_REVISION}, got {revision}"
        )
    package = root / "ultralytics" / "__init__.py"
    if not package.is_file():
        raise PreflightError(f"Ultralytics package is missing at {package}")
    return root


def verify_yolox(path: Path) -> Path:
    root = path.resolve(strict=True)
    required = [
        root / "yolox/models/yolox.py",
        root / "yolox/models/yolo_pafpn.py",
        root / "yolox/models/yolo_head.py",
    ]
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise PreflightError(
            "YOLOX 0.1.1rc0 source checkout is incomplete; missing: " + ", ".join(missing)
        )
    return root


def preflight(family: str, ultralytics_repo: Path, yolox_repo: Path) -> dict[str, str]:
    versions = _require_modules()
    if family == "yolox":
        verify_yolox(yolox_repo)
    else:
        verify_ultralytics(ultralytics_repo)
    return versions
