# ONNX export environment

The Rust `export-onnx` command loads the checkpoint and writes the exact loaded parameters to a
private SafeTensors snapshot. These scripts reconstruct the graph from the pinned sibling source,
export it, run ONNX checker and strict shape inference, execute it with ONNX Runtime CPU, compare
the named outputs, and write the sidecar. They do not download weights, install packages, or use a
floating `ultralytics` wheel.

```powershell
python -m venv target/.venv
target/.venv/Scripts/python.exe -m pip install -r tools/onnx/requirements.lock.txt
```

Linux/macOS use `target/.venv/bin/python`. YOLOX additionally requires the official `0.1.1rc0`
source checkout at `target/yolox-ref/YOLOX-0.1.1rc0` or an explicit `--yolox-repo`.
