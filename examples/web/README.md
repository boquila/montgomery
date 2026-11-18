# Montgomery web demo

Rust, egui, and Trunk. All inference runs locally in the browser with Burn's Flex CPU backend;
images never leave the browser. The same app also builds natively.

The selector offers all five experimental YOLO26 nano tasks supported by Montgomery:

| Task | Model | Result |
| --- | --- | --- |
| Detection | `yolo26n` | Labeled boxes |
| Instance segmentation | `yolo26n-seg` | Masks and boxes |
| Classification | `yolo26n-cls` | Top five ImageNet classes |
| Semantic segmentation | `yolo26n-sem` | Cityscapes class overlay |
| Depth estimation | `yolo26n-depth` | Depth heatmap and distances in meters |

Models load on first selection. Results stay cached per task and photo. The confidence slider
filters both boxes and instance masks without rerunning. While loading or running, the image
dims; status stays above it. The wordmark is the original Montgomery SVG, rendered at display
resolution by egui's SVG loader.

## Prepare models

From the repository root:

```console
uv run --locked --project examples/web/scripts python examples/web/scripts/prepare-models.py
```

This builds the native packer, downloads the five official checkpoints, verifies their SHA-256
hashes from `models.json`, exports their tensor states using the repository's conversion tool,
and packs them. The pinned Python environment uses CPU-only PyTorch. Python is needed only for
model preparation, never browser inference.

All generated data stays under `examples/web/target/`: `checkpoints/`, `models/`, and `packer/`.
Nothing is written to root-level Burnpacks. An existing executable can be supplied with
`--packer /path/to/montgomery`. Repacking replaces artifacts only after each pack succeeds.

## Run

```console
rustup target add wasm32-unknown-unknown
cargo install trunk --locked --version 0.22.1
cd examples/web
trunk serve --release
```

Open <http://127.0.0.1:8080>. Trunk copies the prepared artifacts from `target/models/` and
fetches only the selected model (approximately 3-11 MB). Use `cargo run --release` for the native
app; it reads the same artifacts. `MONTGOMERY_WEB_DEMO_MODELS` can override that directory.

`Trunk.toml` listens on `0.0.0.0`, so the same server can be used over your LAN or tailnet.
For an HTTPS tailnet URL, run `tailscale serve --bg 8080` while Trunk is running.

## Deployment

The deployment target is **https://boquila.org/montgomery-lib/**, served by the existing
`boquila.github.io` website. No additional repository, hosting service, DNS record, or
cross-repository secret is needed.

Publishing a Montgomery GitHub release runs `.github/workflows/web-demo.yml`. It checks
the demo, prepares all five models, builds the browser app, and attaches
`montgomery-web-demo.zip` to that release. Pull requests and manual runs perform the same
validation and provide a preview ZIP as a one-day Actions artifact; they do not publish.

The ZIP contains the contents of `examples/web/target/site/`:

```text
index.html          built with /montgomery-lib/ as its public URL
montgomery.svg
source.json         release version and exact source commit
*.js, *.wasm, *.bpk, LICENSE
```

The existing website deployment downloads the latest stable release that has a completed
demo ZIP and extracts it into `dist/montgomery-lib/`. A website text edit never compiles
Rust, installs Python, or rebuilds models. It only downloads and includes the released files.
The release ZIP persists independently of Actions caches and temporary artifacts.

Before including the demo, the website records hashes of every existing output file.
Before deploying, it verifies that all those files are unchanged and that no
`dist/montgomery/` directory was created. The original `montgomery.html` page and its
`/montgomery` URL remain intact.

Website pushes and manual runs deploy the combined site. A lightweight scheduled check
at minutes 17 and 47 compares the available demo release with the published `source.json`;
unchanged versions skip deployment. GitHub may delay scheduled runs. Prereleases and releases
whose demo upload has not completed are skipped, leaving the previous demo available.
If no released demo exists or downloading it fails, deployment stops before changing the live site.

For the initial rollout, merge Montgomery's changes and publish a release containing the
demo ZIP before merging the website workflow change. Keep Pages disabled on Montgomery:
only `boquila.github.io` publishes to Pages. The unused `demo` DNS record can be removed.

Demo source, assets, model manifest, conversion environment, and documentation remain under
`examples/`. Only the GitHub workflow entry lives in `.github/workflows/`, as GitHub requires.
There are no Node/npm build or serving scripts.

To build the deployment locally, from `examples/web`:

```console
trunk build --release --locked --public-url /montgomery-lib/ --dist target/site
```

To preview the combined site, copy `target/site/` into the website's built
`dist/montgomery-lib/` directory and serve the website's `dist/` directory. CI also includes
the repository license and release metadata in the ZIP.

## Verification

From `examples/web`:

```console
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo test --release --locked all_nano_tasks_from_bytes -- --ignored --nocapture --test-threads 1
```

The last check requires prepared models and validates reference-image objects and masks,
ranked class probabilities, source-sized semantic labels, and finite positive depth. Default
tests check source-space overlays and confidence filtering without external weights.

Browser inference currently runs on the main thread. Dense tasks take longer than detection,
and the UI pauses during a run. Model preprocessing is unchanged: 640 px for detection and
instances, a 224 px center crop for classification, 1024 px for semantics, and 768 px for depth.

## Assets and licensing

`assets/montgomery.svg` is a copy of the project's wordmark from `docs/logo.svg`.
The bundled `dog_bike_man`, `bus`, and `zidane` photos come from
[Ultralytics assets](https://github.com/ultralytics/assets) (AGPL-3.0). The other samples are
public-domain and CC0 classics used by [scikit-image](https://scikit-image.org): `camera`
(CC0, Lav Varshney), `astronaut` (NASA, public domain), `chelsea` (CC0, Stefan van der Walt),
and `coffee` (CC0, Rachel Michetti).

Montgomery and the upstream checkpoints/derived Burnpacks are AGPL-3.0. Weights and build
output remain uncommitted. The task catalog is in `src/app.rs`; `src/results.rs` handles
presentation.
