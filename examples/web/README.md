# Montgomery web demo

A small egui app that runs YOLO26n object detection in the browser through Rust and
WebAssembly, with Montgomery's native Burnpack artifact and the same preprocessing and
postprocessing contract as the CLI. Everything runs client-side; images never leave the
browser.

The same code also builds as a native desktop app for quick iteration.

## Layout

- Left panel: a few bundled sample images. Click one to select it.
- Central panel: the selected image, a **Run inference** button once the model is ready, and
  the detection boxes painted over the image with class names and confidences.

## Prerequisites

```console
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.130   # match examples/web/Cargo.lock
cargo install trunk --locked                      # or grab a prebuilt binary
```

## Pack the model

The demo fetches `yolo26n.bpk` from the repository root at runtime; pack it once:

```console
uv run --project tools tools/export_checkpoint_state.py target/yolo26n.pt target/yolo26n-state.pt
cargo run --bin montgomery -- pack-weights --architecture yolo26n --state target/yolo26n-state.pt
```

`target/yolo26n.pt` is the official Ultralytics checkpoint; `Trunk.toml` copies the packed
artifact into the served bundle.

## Run in the browser

```console
trunk serve
```

Then open <http://127.0.0.1:8080>. The model (~5 MB) is fetched after the page loads; press
**Run inference** to detect objects in the selected photo. Switching images clears the previous
result, and a status banner reports the object count, latency, and active backend. On a first
cold run the graph compiles and inference takes a few seconds; the wasm release build here
measured about 0.3 s per frame after warm-up.

## Share on your tailnet

`Trunk.toml` binds `trunk serve` to `0.0.0.0`, so Tailscale peers can reach
`http://<machine>.<tailnet>.ts.net:8080` if the Windows firewall allows it. For a clean HTTPS
URL without opening a port, serve the built bundle and let Tailscale proxy it:

```console
trunk build --release
node scripts/serve.cjs          # static server for dist on 0.0.0.0:8080
tailscale serve --bg 8080       # prints https://<machine>.<tailnet>.ts.net/
```

## Run natively

```console
cargo run --release
```

Native builds look for `yolo26n.bpk` in the current directory, the repository root
(`../../yolo26n.bpk` when launched from `examples/web`), or the path in
`MONTGOMERY_WEB_DEMO_MODEL`.

## Test

`cargo test` verifies the demo's headline claim headlessly: a bytes-loaded YOLO26n detects
the dog, bicycle, and person on the bundled reference image.

The wasm runtime can be checked without a browser. After packing `yolo26n.bpk` at the
repository root:

```console
cargo build --release --target wasm32-unknown-unknown
wasm-bindgen --target nodejs --out-dir target/smoke_pkg \
    target/wasm32-unknown-unknown/release/montgomery-web-demo.wasm
node scripts/smoke.cjs
```

The script loads the Burnpack, runs the full bytes-to-detections path inside WebAssembly, and
exits non-zero if the expected detections are missing.

## Notes

- Inference currently runs on the main thread in the browser build, so the UI freezes while
  a frame is processed; a web-worker port is the natural next step.
- Montgomery compiles the same Flex CPU backend for `wasm32-unknown-unknown`; burn's flex
  backend falls back to single-threaded execution there.
- Packaging a different detection model means pointing `MODEL_URL` in `src/app.rs` at the
  new artifact and updating the pack step above.
