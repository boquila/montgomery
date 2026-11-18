// Headless runtime check for the wasm build: instantiates the demo module in Node.js, loads
// the packed Burnpack from the repository root, and runs the full bytes-to-detections path
// inside WebAssembly.
//
// Usage, from examples/web:
//   cargo build --target wasm32-unknown-unknown --release
//   wasm-bindgen --target nodejs --out-dir target/smoke_pkg \
//       target/wasm32-unknown-unknown/release/montgomery-web-demo.wasm
//   node scripts/smoke.cjs
const fs = require("node:fs");
const path = require("node:path");

const pkg = require(
  path.join(__dirname, "..", "target", "smoke_pkg", "montgomery-web-demo.js"),
);
const bytes = fs.readFileSync(
  path.join(__dirname, "..", "..", "..", "yolo26n.bpk"),
);

const summary = pkg.run_smoke(new Uint8Array(bytes));
console.log(summary);

const missing = ["dog", "bicycle", "person"].filter(
  (name) => !summary.includes(name),
);
if (missing.length > 0) {
  console.error(`missing detections: ${missing.join(", ")}`);
  process.exit(1);
}
console.log("wasm smoke OK");
