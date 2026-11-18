#!/usr/bin/env python3
"""Download verified official checkpoints and pack the demo's five nano models.

All intermediates, models, and the native packer build stay in examples/web/target.
The CPU-only Python environment is defined beside this script. Runtime inference
uses Rust/Burn; Python is only needed to prepare a fresh set of artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen

DEMO = Path(__file__).resolve().parents[1]
REPO = DEMO.parents[1]
TARGET = DEMO / "target"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(*args: object) -> None:
    subprocess.run([str(arg) for arg in args], cwd=REPO, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packer", type=Path, help="Use an already-built montgomery executable")
    args = parser.parse_args()
    manifest = json.loads((DEMO / "models.json").read_text())
    checkpoints = TARGET / "checkpoints"
    artifacts = TARGET / "models"
    checkpoints.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    if args.packer:
        packer = args.packer.resolve(strict=True)
    else:
        run("cargo", "build", "--release", "--locked", "--manifest-path", REPO / "Cargo.toml",
            "--no-default-features", "--features", "pretrained", "--bin", "montgomery",
            "--target-dir", TARGET / "packer")
        packer = TARGET / "packer/release" / ("montgomery.exe" if os.name == "nt" else "montgomery")

    for model in manifest["models"]:
        name = model["name"]
        checkpoint = checkpoints / f"{name}.pt"
        if not checkpoint.exists():
            temporary = checkpoint.with_suffix(".download")
            try:
                url = f'{manifest["release_url"]}/{name}.pt'
                print(f"Downloading {url}", flush=True)
                with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                if digest(temporary) != model["sha256"]:
                    raise ValueError(f"Checksum mismatch for {name}; refusing to load the checkpoint")
                temporary.replace(checkpoint)
            finally:
                temporary.unlink(missing_ok=True)
        if digest(checkpoint) != model["sha256"]:
            raise ValueError(f"Checksum mismatch for cached {checkpoint}")

        state = checkpoints / f"{name}-state.pt"
        run(sys.executable, REPO / "tools/export_checkpoint_state.py", checkpoint, state)
        # Replace the old artifact only after a complete, successful pack.
        staged = artifacts / f"{name}.staging.bpk"
        staged.unlink(missing_ok=True)
        try:
            run(packer, "pack-weights", "--architecture", name, "--state", state, "--output", staged)
            staged.replace(artifacts / f"{name}.bpk")
        finally:
            staged.unlink(missing_ok=True)
    print(f"Prepared all five models in {artifacts}")


if __name__ == "__main__":
    main()
