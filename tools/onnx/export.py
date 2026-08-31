"""Repository-owned ONNX bridge entry point.

This tool never downloads models or installs packages. Rust produces its input manifest and exact
SafeTensors snapshot; this process constructs only the pinned reference graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

from environment import PreflightError, preflight


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> tuple[dict, Path]:
    manifest_path = path.resolve(strict=True)
    workdir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "montgomery-onnx-export-input-v1":
        raise RuntimeError(f"unsupported or missing manifest schema: {manifest.get('schema')!r}")
    for field in ("weights_file", "output_file", "sidecar_file"):
        relative = Path(manifest[field])
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise RuntimeError(f"unsafe {field}: {relative}")
    weights = (workdir / manifest["weights_file"]).resolve(strict=True)
    if weights.parent != workdir:
        raise RuntimeError("weights path escapes private export directory")
    actual_hash = sha256(weights)
    if actual_hash != manifest["weights_file_sha256"]:
        raise RuntimeError(
            f"SafeTensors hash mismatch: manifest {manifest['weights_file_sha256']}, actual {actual_hash}"
        )
    return manifest, workdir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--family")
    parser.add_argument("--ultralytics-repo", type=Path)
    parser.add_argument("--yolox-repo", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    if os.environ.get("MONTGOMERY_ONNX_NO_NETWORK") != "1":
        raise RuntimeError("export subprocess must set MONTGOMERY_ONNX_NO_NETWORK=1")
    if args.preflight:
        if not args.family or not args.ultralytics_repo or not args.yolox_repo:
            parser.error("--preflight requires --family, --ultralytics-repo, and --yolox-repo")
        versions = preflight(args.family, args.ultralytics_repo, args.yolox_repo)
        print("ONNX export preflight passed: " + ", ".join(f"{key}={value}" for key, value in sorted(versions.items())))
        return 0

    if not args.manifest:
        parser.error("--manifest is required unless --preflight is used")
    manifest, workdir = load_manifest(args.manifest)
    source = Path(manifest["graph_source"]["resolved_path"])
    ultralytics = source if manifest["family"] != "yolox" else Path(".")
    yolox = source if manifest["family"] == "yolox" else Path(".")
    versions = preflight(manifest["family"], ultralytics, yolox)

    if manifest["family"] == "yolox":
        from yolox_adapter import build
    else:
        from ultralytics_adapter import build
    from common import export_and_validate

    wrapper, names = build(manifest, workdir)
    export_and_validate(wrapper, manifest, workdir, names, versions)
    print(f"validated staged ONNX artifact at {workdir / manifest['output_file']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PreflightError, RuntimeError, OSError, ValueError) as error:
        print(f"ONNX export failed: {error}", file=sys.stderr)
        if os.environ.get("MONTGOMERY_ONNX_TRACEBACK") == "1":
            traceback.print_exc()
        raise SystemExit(1)
