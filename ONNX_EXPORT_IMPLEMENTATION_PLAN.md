# ONNX export

Status: implemented. The exporter and validation bridge are shipped; this file records only the
remaining release gates.

Implemented:

- `export-onnx` CLI and Rust API covering every registered model/task variant;
- strict SafeTensors snapshots and reversible key mapping from loaded boquilens weights;
- portable and Ultralytics-compatible detect/segment/classification profiles;
- ONNX checker, shape inference, operator allowlist, metadata/sidecars, hashes, reproducibility,
  external data, overwrite protection, and mandatory ONNX Runtime parity validation.

Remaining release gates:

- dynamic batch/spatial export, FP16 publication, and the separate end-to-end profile remain
  explicit refusal gates until their parity matrices are available;
- full all-variant and cross-platform artifact matrix still requires matching checkpoints and
  runtime environments.

Generated ONNX artifacts and sidecars belong under `target/`.
