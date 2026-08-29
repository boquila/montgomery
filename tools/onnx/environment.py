"""Strict, offline ONNX export environment checks."""

from __future__ import annotations

import importlib
import hashlib
import subprocess
import sys
from pathlib import Path

ULTRALYTICS_REVISION = "461196cf09175b64c9b9bd8babebf081c0540520"
LOCKED_VERSIONS = {
    "torch": "2.13.0",
    "numpy": "2.4.6",
    "safetensors": "0.6.2",
    "onnx": "1.20.1",
    "onnxruntime": "1.23.2",
}
YOLOX_0_1_1RC0_FILES = {
    "yolox/models/network_blocks.py": "250982431c53a5ed49c3e465266baeabe3e21a41eac941a8e5345c2e71e14a7d",
    "yolox/models/darknet.py": "37fd32ae7f7de29df646f7c457837834fcc0e4e08ce83034a6397092fbae6a62",
    "yolox/models/yolo_pafpn.py": "bfeb522c87fc076659c804927751b503ef0185fe1cf8f9f7153369e9177a9627",
    "yolox/models/yolo_head.py": "1666ece85236b71ae5e13bdc416f04483ececfcbef655f048b0fcc5ffdb99c09",
    "yolox/models/yolox.py": "a4d70f5ef75dfbb1da5394d886dc348c68cdfbb27ab725f36c313e981ba605ee",
    "yolox/models/losses.py": "60e8d8586484c5e211e69e41588340c8160774d6b03d097a96662dd98a641d4e",
}


class PreflightError(RuntimeError):
    pass


def _require_modules() -> dict[str, str]:
    if not ((3, 11) <= sys.version_info[:2] < (3, 15)):
        raise PreflightError(
            f"unsupported Python {sys.version.split()[0]}; requirements.lock.txt requires >=3.11,<3.15"
        )
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
    mismatched = [
        f"{name}={versions[name]} (expected {expected})"
        for name, expected in LOCKED_VERSIONS.items()
        if versions[name].split("+", 1)[0] != expected
    ]
    if mismatched:
        raise PreflightError(
            "ONNX export environment does not match requirements.lock.txt: "
            + ", ".join(mismatched)
            + f". Recreate it explicitly with:\n  {sys.executable} -m pip install -r tools/onnx/requirements.lock.txt"
        )
    return versions


def _git_revision(path: Path) -> str:
    top_level = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=path,
        check=False,
        capture_output=True,
        text=True,
    )
    if top_level.returncode or Path(top_level.stdout.strip()).resolve() != path.resolve():
        raise PreflightError(f"graph source at {path} is not the root of its own Git checkout")
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
    missing = [relative for relative in YOLOX_0_1_1RC0_FILES if not (root / relative).is_file()]
    if missing:
        raise PreflightError(
            "YOLOX 0.1.1rc0 source checkout is incomplete; missing: " + ", ".join(missing)
        )
    mismatched = []
    for relative, expected in YOLOX_0_1_1RC0_FILES.items():
        normalized = (root / relative).read_bytes().replace(b"\r\n", b"\n")
        actual = hashlib.sha256(normalized).hexdigest()
        if actual != expected:
            mismatched.append(f"{relative} ({actual})")
    if mismatched:
        raise PreflightError(
            "YOLOX source does not match the pinned 0.1.1rc0 graph files: " + ", ".join(mismatched)
        )
    return root


def preflight(family: str, ultralytics_repo: Path, yolox_repo: Path) -> dict[str, str]:
    versions = _require_modules()
    if family == "yolox":
        verify_yolox(yolox_repo)
    else:
        verify_ultralytics(ultralytics_repo)
    return versions
