//! Native Burn implementation of the Ultralytics YOLO26-depth monocular depth-estimation
//! family (n/s/m/l/x).
//!
//! Depth estimation is a separate task from detection and segmentation: instead of boxes or
//! class maps it predicts one dense depth map in meters per image. There is deliberately no
//! sharing with the other heads beyond the backbone/neck vocabulary.
//!
//! The depth graph reuses the detect body layers 0-22 verbatim (backbone plus the full
//! feature pyramid, ending in P3/8, P4/16, P5/32) and Ultralytics' `Depth` head (`model.23`):
//! per-level 1x1 projections to a scale-independent 256-channel decoder, top-down fusion
//! with `align_corners = true` bilinear 2x steps, two refinement blocks, and a small tower
//! ending in a transposed convolution (stride-4 output), `exp(clamp(-4, 5))`, and the
//! log-affine calibration `depth^cal_a * exp(cal_b)`.
//!
//! Two checkpoint quirks are load-bearing. First, the released checkpoints carry a third
//! refinement block (`model.23.refine.2`) that the official forward never executes (only the
//! P4- and P3-level refinements run); it is dead weight and therefore absent from the
//! inference graph, like the one-to-many branches of default (non-training) detect builds.
//! Second, the calibration buffers are non-identity in the released checkpoints (e.g.
//! `cal_b = -0.1938` for n) and must be imported, not assumed.
//!
//! Inference returns calibrated positive depth at stride 4 (`[batch, 1, H/4, W/4]`); the
//! runtime upsamples it to the letterboxed canvas, inverts the letterbox geometry, and keeps
//! float meters, mirroring `DepthPredictor.postprocess`.

use burn::{
    module::{Module, Param},
    nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
    tensor::{
        Device, Tensor,
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
    },
};

#[cfg(feature = "pretrained")]
use burn_pack::Error as BurnpackError;
#[cfg(feature = "pretrained")]
use burn_store::{
    BurnpackStore, HalfPrecisionAdapter, ModuleSnapshot, PytorchStore, PytorchStoreError,
};

use super::blocks::{Conv, ConvConfig};
use super::body::{Yolo26BodyLarge, Yolo26BodySmall, Yolo26Features};

#[cfg(feature = "pretrained")]
use super::weights;

/// Intermediate width of the depth fusion decoder. `parse_model` does not width-scale
/// Depth's `c_mid`, so every scale shares this width; only the projection inputs vary.
pub const DECODER_WIDTH: usize = 256;

/// Bilinear 2x upsample with corner-aligned centers, matching the in-graph fusion steps of
/// the official `Depth` forward (whose released weights bake `align_corners = true`).
fn bilinear_upsample_align_corners_2x(input: Tensor<4>) -> Tensor<4> {
    let [_, _, height, width] = input.dims();
    interpolate(
        input,
        InterpolateOptions::new(InterpolateMode::Bilinear)
            .with_align_corners(true)
            .with_output_size([height * 2, width * 2]),
    )
}

/// YOLO26 depth-estimation model output: calibrated per-pixel depth in meters at stride 4.
pub struct DepthOutput {
    /// Positive depth, `[batch, 1, height / 4, width / 4]`.
    pub depth: Tensor<4>,
}

/// Ultralytics `Depth` inference head: per-level projections, top-down fusion with two
/// refinement blocks, and the dense depth tower.
///
/// Field names deliberately match the official `model.23.*` checkpoint keys after remapping
/// (`proj.i`, `refine.i.j`, `head.i`, `cal_a`/`cal_b`). The checkpoint's `refine.2` block is
/// never executed by the official forward and stays unmapped.
#[derive(Module, Debug)]
pub struct DepthHead {
    proj_0: Conv,
    proj_1: Conv,
    proj_2: Conv,
    refine_0_0: Conv,
    refine_0_1: Conv,
    refine_1_0: Conv,
    refine_1_1: Conv,
    head_0: Conv,
    head_1: ConvTranspose2d,
    head_2: Conv,
    head_3: Conv2d,
    cal_a: Param<Tensor<1>>,
    cal_b: Param<Tensor<1>>,
}

impl DepthHead {
    pub fn forward(&self, features: Yolo26Features) -> DepthOutput {
        let [batch, _, _, _] = features.p3.dims();
        let p3 = self.proj_0.forward(features.p3);
        let p4 = self.proj_1.forward(features.p4);
        let p5 = self.proj_2.forward(features.p5);

        // Fuse coarse-to-fine: P5 upsampled onto P4, refined, then upsampled onto P3 and
        // refined again. Only these two refinements execute upstream.
        let fused_p4 = self.refine_1_1.forward(
            self.refine_1_0
                .forward(bilinear_upsample_align_corners_2x(p5) + p4),
        );
        let fused_p3 = self.refine_0_1.forward(
            self.refine_0_0
                .forward(bilinear_upsample_align_corners_2x(fused_p4) + p3),
        );

        let out = self.head_0.forward(fused_p3);
        let out = self.head_1.forward(out);
        let out = self.head_3.forward(self.head_2.forward(out));
        let depth = out.clamp(-4.0, 5.0).exp();
        // Log-affine calibration d' = d^a * exp(b), identity when a = 1 and b = 0. The
        // exp/log formulation broadcasts the scalar buffers without host syncs.
        let cal_a = self.cal_a.val().reshape([1, 1, 1, 1]);
        let cal_b = self.cal_b.val().reshape([1, 1, 1, 1]);
        let calibrated = (cal_a * depth.clone().log()).exp() * cal_b.exp();
        debug_assert_eq!(calibrated.dims()[0], batch);
        debug_assert_eq!(calibrated.dims()[1], 1);
        DepthOutput { depth: calibrated }
    }
}

#[derive(Debug)]
pub struct DepthHeadConfig {
    p3_channels: usize,
    p4_channels: usize,
    p5_channels: usize,
}

impl DepthHeadConfig {
    /// Declare the head for one scale from its P3/P4/P5 input widths. The decoder width is
    /// scale-independent; only the 1x1 projections adapt to the inputs.
    pub fn new(p3_channels: usize, p4_channels: usize, p5_channels: usize) -> Self {
        Self {
            p3_channels,
            p4_channels,
            p5_channels,
        }
    }

    pub fn init(&self, device: &Device) -> DepthHead {
        DepthHead {
            proj_0: ConvConfig::new(self.p3_channels, DECODER_WIDTH, 1, 1).init(device),
            proj_1: ConvConfig::new(self.p4_channels, DECODER_WIDTH, 1, 1).init(device),
            proj_2: ConvConfig::new(self.p5_channels, DECODER_WIDTH, 1, 1).init(device),
            refine_0_0: ConvConfig::new(DECODER_WIDTH, DECODER_WIDTH, 3, 1).init(device),
            refine_0_1: ConvConfig::new(DECODER_WIDTH, DECODER_WIDTH, 3, 1).init(device),
            refine_1_0: ConvConfig::new(DECODER_WIDTH, DECODER_WIDTH, 3, 1).init(device),
            refine_1_1: ConvConfig::new(DECODER_WIDTH, DECODER_WIDTH, 3, 1).init(device),
            head_0: ConvConfig::new(DECODER_WIDTH, DECODER_WIDTH / 2, 3, 1).init(device),
            head_1: ConvTranspose2dConfig::new([DECODER_WIDTH / 2, DECODER_WIDTH / 2], [2, 2])
                .with_stride([2, 2])
                .init(device),
            head_2: ConvConfig::new(DECODER_WIDTH / 2, DECODER_WIDTH / 4, 3, 1).init(device),
            head_3: Conv2dConfig::new([DECODER_WIDTH / 4, 1], [1, 1])
                .with_bias(true)
                .init(device),
            cal_a: Param::from_tensor(Tensor::ones([1], device)),
            cal_b: Param::from_tensor(Tensor::zeros([1], device)),
        }
    }
}

/// Build the PyTorch-state store shared by every YOLO26-depth scale variant.
///
/// The body is layers 0-22 (identical indices to the detect checkpoint) and the head is
/// `model.23`. The checkpoint's dead `refine.2` block is intentionally unmapped and ignored
/// on import.
#[cfg(feature = "pretrained")]
fn pytorch_store(path: impl Into<std::path::PathBuf>) -> PytorchStore {
    PytorchStore::from_file(path)
        .with_top_level_key("model")
        // Body layers retain their Ultralytics graph indices. The head is model.23, so this
        // rule must not match it.
        .with_key_remapping("model\\.([0-9]|1[0-9]|2[0-2])\\.(.+)", "body.model_$1.$2")
        // model.23.proj.{level} are the 1x1 level projections.
        .with_key_remapping("model\\.23\\.proj\\.0\\.conv\\.(.+)", "head.proj_0.conv.$1")
        .with_key_remapping("model\\.23\\.proj\\.0\\.bn\\.(.+)", "head.proj_0.bn.$1")
        .with_key_remapping("model\\.23\\.proj\\.1\\.conv\\.(.+)", "head.proj_1.conv.$1")
        .with_key_remapping("model\\.23\\.proj\\.1\\.bn\\.(.+)", "head.proj_1.bn.$1")
        .with_key_remapping("model\\.23\\.proj\\.2\\.conv\\.(.+)", "head.proj_2.conv.$1")
        .with_key_remapping("model\\.23\\.proj\\.2\\.bn\\.(.+)", "head.proj_2.bn.$1")
        // model.23.refine.{level}.{layer} are the two fusion refinements that execute.
        .with_key_remapping(
            "model\\.23\\.refine\\.0\\.0\\.conv\\.(.+)",
            "head.refine_0_0.conv.$1",
        )
        .with_key_remapping(
            "model\\.23\\.refine\\.0\\.0\\.bn\\.(.+)",
            "head.refine_0_0.bn.$1",
        )
        .with_key_remapping(
            "model\\.23\\.refine\\.0\\.1\\.conv\\.(.+)",
            "head.refine_0_1.conv.$1",
        )
        .with_key_remapping(
            "model\\.23\\.refine\\.0\\.1\\.bn\\.(.+)",
            "head.refine_0_1.bn.$1",
        )
        .with_key_remapping(
            "model\\.23\\.refine\\.1\\.0\\.conv\\.(.+)",
            "head.refine_1_0.conv.$1",
        )
        .with_key_remapping(
            "model\\.23\\.refine\\.1\\.0\\.bn\\.(.+)",
            "head.refine_1_0.bn.$1",
        )
        .with_key_remapping(
            "model\\.23\\.refine\\.1\\.1\\.conv\\.(.+)",
            "head.refine_1_1.conv.$1",
        )
        .with_key_remapping(
            "model\\.23\\.refine\\.1\\.1\\.bn\\.(.+)",
            "head.refine_1_1.bn.$1",
        )
        // model.23.head.0/2 are Convs, head.1 the transposed convolution, head.3 the 1x1
        // depth projection.
        .with_key_remapping("model\\.23\\.head\\.0\\.conv\\.(.+)", "head.head_0.conv.$1")
        .with_key_remapping("model\\.23\\.head\\.0\\.bn\\.(.+)", "head.head_0.bn.$1")
        .with_key_remapping("model\\.23\\.head\\.1\\.(.+)", "head.head_1.$1")
        .with_key_remapping("model\\.23\\.head\\.2\\.conv\\.(.+)", "head.head_2.conv.$1")
        .with_key_remapping("model\\.23\\.head\\.2\\.bn\\.(.+)", "head.head_2.bn.$1")
        .with_key_remapping("model\\.23\\.head\\.3\\.(.+)", "head.head_3.$1")
        // Scalar log-affine calibration buffers.
        .with_key_remapping("model\\.23\\.cal_a", "head.cal_a")
        .with_key_remapping("model\\.23\\.cal_b", "head.cal_b")
}

macro_rules! depth_model {
    ($model:ident, $config:ident, $body_struct:ident, $body_config:path, $p3:expr, $p4:expr, $p5:expr, $id:literal, $doc:expr) => {
        #[doc = $doc]
        #[derive(Module, Debug)]
        pub struct $model {
            body: $body_struct,
            head: DepthHead,
        }

        impl $model {
            pub fn forward(&self, input: Tensor<4>) -> DepthOutput {
                self.head.forward(self.body.forward(input))
            }

            /// Import tensor-only state exported from an official Ultralytics checkpoint.
            #[cfg(feature = "pretrained")]
            pub fn load_pytorch_weights(
                &mut self,
                path: impl Into<std::path::PathBuf>,
            ) -> Result<(), PytorchStoreError> {
                let mut store = pytorch_store(path);
                self.load_from(&mut store).map(|_| ())
            }

            /// Load Montgomery's versioned, half-precision native Burnpack artifact.
            #[cfg(feature = "pretrained")]
            pub fn load_burnpack_weights(
                &mut self,
                path: impl Into<std::path::PathBuf>,
            ) -> Result<(), BurnpackError> {
                let mut store = BurnpackStore::from_file(path.into())
                    .with_from_adapter(HalfPrecisionAdapter::new());
                self.load_from(&mut store).map(|_| ())
            }

            /// Save a versioned native artifact. Existing files are deliberately not overwritten.
            #[cfg(feature = "pretrained")]
            pub fn save_burnpack_weights(
                &self,
                path: impl Into<std::path::PathBuf>,
            ) -> Result<(), BurnpackError> {
                let mut store = BurnpackStore::from_file(path.into())
                    .metadata("montgomery.artifact-format", weights::artifact_format($id))
                    .metadata("montgomery.model", $id)
                    .metadata("montgomery.classes", "depth-1")
                    .metadata("montgomery.precision", "f16")
                    .metadata("montgomery.source", "ultralytics-v8.4")
                    .metadata("montgomery.license", "AGPL-3.0")
                    .with_to_adapter(HalfPrecisionAdapter::new());
                self.save_into(&mut store)
            }
        }

        #[derive(Debug, Default)]
        pub struct $config;

        impl $config {
            pub fn init(&self, device: &Device) -> $model {
                self.init_with_classes(1, device)
            }

            /// Depth has exactly one output channel; the class count only flows through so
            /// the shared artifact loader can construct the graph from the class table.
            pub fn init_with_classes(&self, num_classes: usize, device: &Device) -> $model {
                assert_eq!(
                    num_classes, 1,
                    "depth models predict a single depth channel"
                );
                $model {
                    body: $body_config.init(device),
                    head: DepthHeadConfig::new($p3, $p4, $p5).init(device),
                }
            }
        }
    };
}

depth_model!(
    Yolo26DepthN,
    Yolo26DepthNConfig,
    Yolo26BodySmall,
    super::body::Yolo26BodyNConfig,
    64,
    128,
    256,
    "yolo26n-depth",
    "Native Burn YOLO26n-depth monocular depth-estimation model."
);
depth_model!(
    Yolo26DepthS,
    Yolo26DepthSConfig,
    Yolo26BodySmall,
    super::body::Yolo26BodySConfig,
    128,
    256,
    512,
    "yolo26s-depth",
    "Native Burn YOLO26s-depth monocular depth-estimation model."
);
depth_model!(
    Yolo26DepthM,
    Yolo26DepthMConfig,
    Yolo26BodyLarge,
    super::body::Yolo26BodyMConfig,
    256,
    512,
    512,
    "yolo26m-depth",
    "Native Burn YOLO26m-depth monocular depth-estimation model."
);
depth_model!(
    Yolo26DepthL,
    Yolo26DepthLConfig,
    Yolo26BodyLarge,
    super::body::Yolo26BodyLConfig,
    256,
    512,
    512,
    "yolo26l-depth",
    "Native Burn YOLO26l-depth monocular depth-estimation model."
);
depth_model!(
    Yolo26DepthX,
    Yolo26DepthXConfig,
    Yolo26BodyLarge,
    super::body::Yolo26BodyXConfig,
    384,
    768,
    768,
    "yolo26x-depth",
    "Native Burn YOLO26x-depth monocular depth-estimation model."
);

#[cfg(test)]
mod tests {
    use super::*;

    // Random-initialising N to X takes tens of seconds at opt-level 0, and
    // tests/integration.rs already runs N. CI runs this with `-- --ignored every_scale`.
    #[test]
    #[ignore = "slow: builds every depth scale"]
    fn depth_models_decode_stride_4_depth_for_every_scale() {
        let worker = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let device = Device::flex();
                let input = Tensor::zeros([1, 3, 64, 64], &device);
                // A 64 px input yields a single-channel 16x16 map; depth is positive by
                // construction (exp of the clamped tower output, scaled by calibration).
                for output in [
                    Yolo26DepthNConfig.init(&device).forward(input.clone()),
                    Yolo26DepthSConfig.init(&device).forward(input.clone()),
                    Yolo26DepthMConfig.init(&device).forward(input.clone()),
                    Yolo26DepthLConfig.init(&device).forward(input.clone()),
                    Yolo26DepthXConfig.init(&device).forward(input),
                ] {
                    assert_eq!(output.depth.dims(), [1, 1, 16, 16]);
                }
            })
            .expect("shape-test worker should start");
        worker.join().expect("shape-test worker should not panic");
    }
}

#[cfg(all(test, feature = "pretrained"))]
mod parity_tests {
    use super::*;
    use burn::tensor::{ElementConversion, TensorData};

    use serde::Deserialize;
    use std::collections::BTreeMap;

    #[derive(Deserialize)]
    struct GoldenFixture {
        format: String,
        model: String,
        tensors: BTreeMap<String, GoldenTensor>,
    }

    #[derive(Deserialize)]
    struct GoldenTensor {
        shape: Vec<usize>,
        mean: f64,
        rms: f64,
        min: f64,
        max: f64,
        samples: Vec<(usize, f64)>,
    }

    fn assert_golden<const D: usize>(name: &str, actual: Tensor<D>, expected: &GoldenTensor) {
        assert_eq!(actual.dims().to_vec(), expected.shape, "{name} shape");
        let values: Vec<f64> = actual
            .into_data()
            .iter::<f32>()
            .map(|value| value.elem::<f64>())
            .collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let rms =
            (values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64).sqrt();
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let close =
            |actual: f64, expected: f64| (actual - expected).abs() <= 2e-4 + expected.abs() * 2e-4;

        assert!(
            close(mean, expected.mean),
            "{name} mean: {mean} != {}",
            expected.mean
        );
        assert!(
            close(rms, expected.rms),
            "{name} rms: {rms} != {}",
            expected.rms
        );
        assert!(
            close(min, expected.min),
            "{name} min: {min} != {}",
            expected.min
        );
        assert!(
            close(max, expected.max),
            "{name} max: {max} != {}",
            expected.max
        );
        for &(index, expected_value) in &expected.samples {
            let actual_value = values[index];
            assert!(
                close(actual_value, expected_value),
                "{name}[{index}]: {actual_value} != {expected_value}"
            );
        }
    }

    fn load_reference_image(id: &str, device: &Device) -> Tensor<4> {
        let image = image::open(format!("target/{id}-preprocessed-reference.png"))
            .unwrap()
            .into_rgb8();
        let shape = [image.height() as usize, image.width() as usize, 3];
        Tensor::<3>::from_data(
            TensorData::new(image.into_raw(), shape).convert::<f32>(),
            device,
        )
        .permute([2, 0, 1])
        .unsqueeze::<4>()
            / 255.0
    }

    macro_rules! checkpoint_test {
        ($fn_name:ident, $config:ty, $id:literal) => {
            /// Run manually after converting the official checkpoint. Kept ignored in CI because
            /// the source checkpoint is an external AGPL asset.
            #[test]
            #[ignore]
            fn $fn_name() {
                let checkpoint = std::path::PathBuf::from(concat!("target/", $id, "-state.pt"));
                assert!(
                    checkpoint.exists(),
                    "convert {}.pt with tools/export_checkpoint_state.py first",
                    $id
                );
                let worker = std::thread::Builder::new()
                    .stack_size(64 * 1024 * 1024)
                    .spawn(move || {
                        let device = Device::flex();
                        let mut model = <$config>::default().init(&device);
                        model.load_pytorch_weights(checkpoint).unwrap();
                        let output = model.forward(Tensor::zeros([1, 3, 64, 64], &device));
                        assert_eq!(output.depth.dims(), [1, 1, 16, 16]);
                    })
                    .unwrap();
                worker.join().unwrap();
            }
        };
    }

    macro_rules! golden_test {
        ($fn_name:ident, $config:ty, $id:literal) => {
            #[test]
            #[ignore]
            fn $fn_name() {
                let checkpoint = std::path::PathBuf::from(format!(
                    "target/{}",
                    crate::models::yolo26::weights::artifact_filename($id)
                ));
                let fixture: GoldenFixture = serde_json::from_slice(
                    &std::fs::read(format!("target/{}-golden-v1.json", $id)).unwrap_or_else(|_| {
                        panic!(
                            "generate fixtures with tools/export_yolo26_depth_fixtures.py --model {}",
                            $id
                        )
                    }),
                )
                .unwrap();
                assert_eq!(fixture.format, "montgomery-ultralytics-golden-v1");
                assert_eq!(fixture.model, $id);

                let worker = std::thread::Builder::new()
                    .stack_size(64 * 1024 * 1024)
                    .spawn(move || {
                        let device = Device::flex();
                        let mut model = <$config>::default().init(&device);
                        model.load_burnpack_weights(checkpoint).unwrap();
                        let features = model.body.forward(load_reference_image($id, &device));
                        let p3 = features.p3.clone();
                        let p4 = features.p4.clone();
                        let p5 = features.p5.clone();
                        let output = model.head.forward(features);

                        assert_golden("body_p3", p3, fixture.tensors.get("body_p3").unwrap());
                        assert_golden("body_p4", p4, fixture.tensors.get("body_p4").unwrap());
                        assert_golden("body_p5", p5, fixture.tensors.get("body_p5").unwrap());
                        assert_golden(
                            "depth",
                            output.depth,
                            fixture.tensors.get("depth").unwrap(),
                        );
                    })
                    .unwrap();
                worker.join().unwrap();
            }
        };
    }

    macro_rules! latency_test {
        ($fn_name:ident, $config:ty, $id:literal) => {
            /// Measure single-image batch-1 inference latency with the packed native artifact on
            /// the Flex CPU backend. Run with
            /// `cargo test --release <id> -- --ignored --nocapture --test-threads 1` after the
            /// weight-prep loop.
            #[test]
            #[ignore]
            fn $fn_name() {
                let checkpoint = std::path::PathBuf::from(format!(
                    "target/{}",
                    crate::models::yolo26::weights::artifact_filename($id)
                ));
                assert!(
                    checkpoint.exists(),
                    "pack the {} artifact with pack-weights first",
                    $id
                );
                let worker = std::thread::Builder::new()
                    .stack_size(64 * 1024 * 1024)
                    .spawn(move || {
                        let device = Device::flex();
                        let mut model = <$config>::default().init(&device);
                        model.load_burnpack_weights(checkpoint).unwrap();
                        let input = Tensor::<4>::zeros([1, 3, 768, 768], &device);
                        const WARMUP_RUNS: usize = 3;
                        const TIMED_RUNS: usize = 10;

                        for _ in 0..WARMUP_RUNS {
                            let output = model.forward(input.clone());
                            let _ = output.depth.sum().into_data();
                        }
                        let mut samples = Vec::with_capacity(TIMED_RUNS);
                        for _ in 0..TIMED_RUNS {
                            let started = std::time::Instant::now();
                            let output = model.forward(input.clone());
                            let _ = output.depth.sum().into_data();
                            samples.push(started.elapsed().as_secs_f64() * 1e3);
                        }
                        samples.sort_by(|a, b| a.total_cmp(b));
                        let median = samples[samples.len() / 2];
                        let min = samples[0];
                        println!(
                            "{:>12}: {:>7.1} ms median, {:>7.1} ms min  (single image, batch 1, 768 px, {TIMED_RUNS} runs)",
                            $id, median, min,
                        );
                    })
                    .unwrap();
                worker.join().unwrap();
            }
        };
    }

    #[cfg(feature = "gpu")]
    macro_rules! gpu_latency_test {
        ($fn_name:ident, $config:ty, $id:literal) => {
            /// Measure single-image batch-1 inference latency with the packed native artifact on
            /// the Wgpu GPU backend. Requires the gpu feature and a packed native artifact.
            #[test]
            #[ignore]
            fn $fn_name() {
                let checkpoint = std::path::PathBuf::from(format!(
                    "target/{}",
                    crate::models::yolo26::weights::artifact_filename($id)
                ));
                assert!(
                    checkpoint.exists(),
                    "pack the {} artifact with pack-weights first",
                    $id
                );
                let (device, adapter) = crate::default_wgpu_device();
                println!("GPU adapter: {adapter}");
                let worker = std::thread::Builder::new()
                    .stack_size(64 * 1024 * 1024)
                    .spawn(move || {
                        let mut model = <$config>::default().init(&device);
                        model.load_burnpack_weights(checkpoint).unwrap();
                        let input = Tensor::<4>::zeros([1, 3, 768, 768], &device);
                        const WARMUP_RUNS: usize = 3;
                        const TIMED_RUNS: usize = 10;

                        for _ in 0..WARMUP_RUNS {
                            let output = model.forward(input.clone());
                            let _ = output.depth.sum().into_data();
                        }
                        let mut samples = Vec::with_capacity(TIMED_RUNS);
                        for _ in 0..TIMED_RUNS {
                            let started = std::time::Instant::now();
                            let output = model.forward(input.clone());
                            let _ = output.depth.sum().into_data();
                            samples.push(started.elapsed().as_secs_f64() * 1e3);
                        }
                        samples.sort_by(|a, b| a.total_cmp(b));
                        let median = samples[samples.len() / 2];
                        let min = samples[0];
                        println!(
                            "{:>12}: {:>7.1} ms median, {:>7.1} ms min  (single image, batch 1, 768 px, {TIMED_RUNS} runs, Wgpu GPU)",
                            $id, median, min,
                        );
                    })
                    .unwrap();
                worker.join().unwrap();
            }
        };
    }

    checkpoint_test!(
        yolo26n_depth_imports_official_checkpoint_and_runs_forward,
        Yolo26DepthNConfig,
        "yolo26n-depth"
    );
    checkpoint_test!(
        yolo26s_depth_imports_official_checkpoint_and_runs_forward,
        Yolo26DepthSConfig,
        "yolo26s-depth"
    );
    checkpoint_test!(
        yolo26m_depth_imports_official_checkpoint_and_runs_forward,
        Yolo26DepthMConfig,
        "yolo26m-depth"
    );
    checkpoint_test!(
        yolo26l_depth_imports_official_checkpoint_and_runs_forward,
        Yolo26DepthLConfig,
        "yolo26l-depth"
    );
    checkpoint_test!(
        yolo26x_depth_imports_official_checkpoint_and_runs_forward,
        Yolo26DepthXConfig,
        "yolo26x-depth"
    );

    golden_test!(
        yolo26n_depth_matches_ultralytics_golden_tensors,
        Yolo26DepthNConfig,
        "yolo26n-depth"
    );
    golden_test!(
        yolo26s_depth_matches_ultralytics_golden_tensors,
        Yolo26DepthSConfig,
        "yolo26s-depth"
    );
    golden_test!(
        yolo26m_depth_matches_ultralytics_golden_tensors,
        Yolo26DepthMConfig,
        "yolo26m-depth"
    );
    golden_test!(
        yolo26l_depth_matches_ultralytics_golden_tensors,
        Yolo26DepthLConfig,
        "yolo26l-depth"
    );
    golden_test!(
        yolo26x_depth_matches_ultralytics_golden_tensors,
        Yolo26DepthXConfig,
        "yolo26x-depth"
    );

    latency_test!(
        yolo26n_depth_measures_single_inference_latency,
        Yolo26DepthNConfig,
        "yolo26n-depth"
    );
    latency_test!(
        yolo26s_depth_measures_single_inference_latency,
        Yolo26DepthSConfig,
        "yolo26s-depth"
    );
    latency_test!(
        yolo26m_depth_measures_single_inference_latency,
        Yolo26DepthMConfig,
        "yolo26m-depth"
    );
    latency_test!(
        yolo26l_depth_measures_single_inference_latency,
        Yolo26DepthLConfig,
        "yolo26l-depth"
    );
    latency_test!(
        yolo26x_depth_measures_single_inference_latency,
        Yolo26DepthXConfig,
        "yolo26x-depth"
    );

    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo26n_depth_measures_single_inference_latency_gpu,
        Yolo26DepthNConfig,
        "yolo26n-depth"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo26s_depth_measures_single_inference_latency_gpu,
        Yolo26DepthSConfig,
        "yolo26s-depth"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo26m_depth_measures_single_inference_latency_gpu,
        Yolo26DepthMConfig,
        "yolo26m-depth"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo26l_depth_measures_single_inference_latency_gpu,
        Yolo26DepthLConfig,
        "yolo26l-depth"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo26x_depth_measures_single_inference_latency_gpu,
        Yolo26DepthXConfig,
        "yolo26x-depth"
    );
}
