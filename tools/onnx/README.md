# ONNX export environment

The Rust `export-onnx` command loads the checkpoint and writes the exact loaded parameters to a
private SafeTensors snapshot. These scripts reconstruct the graph from the pinned sibling source,
export it, run ONNX checker and strict shape inference, execute it with ONNX Runtime CPU, compare
the named outputs, and write the sidecar. They do not download weights, install packages, or use a
floating `ultralytics` wheel.

```powershell
uv venv --python 3.13 target/.venv
uv pip sync --python target/.venv/Scripts/python.exe tools/onnx/requirements.lock.txt
```

Linux/macOS pass `target/.venv/bin/python` to `uv pip sync`. YOLOX additionally requires the official `0.1.1rc0`
source checkout at `target/yolox-ref/YOLOX-0.1.1rc0` or an explicit `--yolox-repo`; the bridge
checks the SHA-256 of every graph source file it copies, so an ancestor Git repository cannot be
mistaken for the pinned checkout.

Portable exports run each deterministic input through the exact loaded Burn graph, the pinned
PyTorch reconstruction, and ONNX Runtime. The sidecar records full-tensor maximum, mean, RMS, and
relative errors. Detection-only numerical exceptions are task-semantic and explicit: YOLOX and
legacy YOLOv3-Tiny boxes are gated at the documented 0.25 confidence threshold,
and NMS-free top-k rows below that threshold may reorder between runtimes. Shapes, dtypes,
finiteness, probability tolerances, all confidence-relevant rows, and the complete ungated error
statistics remain mandatory. `--no-verify` skips the extra Burn comparison, never ONNX Runtime.

`--reproducible` omits timestamps and uses a canonical digest over sorted tensor names, dtypes,
shapes, and values; repeated exports to the same filename are byte-identical. SafeTensors' own
container hash is retained only for private bridge integrity because metadata-map order is not a
stable content identity.
