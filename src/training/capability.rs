use burn::{
    module::Module,
    nn::{
        BatchNorm, BatchNormConfig,
        conv::{Conv2d, Conv2dConfig},
    },
    optim::{GradientsParams, SgdConfig, momentum::MomentumConfig},
    tensor::{Device, Distribution, Tensor},
};

#[derive(Module, Debug)]
struct Spike {
    conv: Conv2d,
    bn: BatchNorm,
}

impl Spike {
    fn init(device: &Device) -> Self {
        Self {
            conv: Conv2dConfig::new([3, 4], [3, 3]).init(device),
            bn: BatchNormConfig::new(4).init(device),
        }
    }

    fn forward(&self, input: Tensor<4>) -> Tensor<1> {
        self.bn
            .forward(self.conv.forward(input))
            .powf_scalar(2.0)
            .mean()
    }
}

/// Hardware-gated phase-0 smoke test. It compiles on every training build and can be run on the
/// selected adapter with `cargo test --features training wgpu_autodiff_capability -- --ignored`.
#[test]
#[ignore = "requires a local WGPU adapter"]
fn wgpu_autodiff_capability() {
    let (device, _) = crate::default_wgpu_device();
    let device = device.autodiff();
    let model = Spike::init(&device);
    let input = Tensor::random([2, 3, 16, 16], Distribution::Default, &device);
    let loss = model.forward(input);
    let grads = GradientsParams::from_grads(loss.backward(), &model);
    // Momentum gives the optimizer per-parameter state, so its record is non-empty.
    let mut optimizer = SgdConfig::new()
        .with_momentum(Some(MomentumConfig::new()))
        .init();
    let model = optimizer.step(1e-3, model, grads);

    // `valid` disables BatchNorm's training flag and gives the copy its own running state, so this
    // forward cannot mutate the trained model's BN statistics.
    let valid = model.valid();
    assert!(!valid.bn.training.is_enabled());
    let input = Tensor::zeros([1, 3, 16, 16], &device.inner());
    let output = valid.forward(input).into_data();
    assert!(output.as_slice::<f32>().unwrap()[0].is_finite());
    assert!(!optimizer.to_record().is_empty());
}

#[test]
fn validation_backend_does_not_mutate_batch_norm_state() {
    let device = Device::flex().autodiff();
    let model = Spike::init(&device);
    let input = Tensor::random([2, 3, 16, 16], Distribution::Default, &device);
    let _ = model.forward(input).into_data();
    let valid = model.valid();
    let valid_device = device.inner();
    let before_mean = valid.bn.running_mean.value().into_data();
    let before_var = valid.bn.running_var.value().into_data();
    let _ = valid
        .forward(Tensor::random(
            [2, 3, 16, 16],
            Distribution::Default,
            &valid_device,
        ))
        .into_data();
    assert_eq!(before_mean, valid.bn.running_mean.value().into_data());
    assert_eq!(before_var, valid.bn.running_var.value().into_data());
}

/// Batch-one depthwise backward exercises the gather path used by detector training.
/// Run on a local adapter with:
/// `cargo test --release --features training wgpu_depthwise_batch_one_backward -- --ignored`.
/// Synthetic inputs keep this regression independent of model weights and training datasets.
#[test]
#[ignore = "requires a local WGPU adapter"]
fn wgpu_depthwise_batch_one_backward() {
    let (device, _) = crate::default_wgpu_device();
    device.seed(42);
    let device = device.autodiff();
    let model = Conv2dConfig::new([64, 64], [3, 3])
        .with_groups(64)
        .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1, 1, 1))
        .with_bias(false)
        .init(&device);
    let before = model.weight.val().into_data();
    let input = Tensor::random([1, 64, 80, 80], Distribution::Default, &device).require_grad();
    let loss = model.forward(input.clone()).powf_scalar(2.0).mean();
    assert!(loss.clone().into_data().as_slice::<f32>().unwrap()[0].is_finite());
    let raw_grads = loss.backward();
    let input_grad = input.grad(&raw_grads).expect("input gradient").into_data();
    let input_grad = input_grad.as_slice::<f32>().unwrap();
    assert!(input_grad.iter().all(|value| value.is_finite()));
    assert!(input_grad.iter().any(|value| *value != 0.0));
    let grads = GradientsParams::from_grads(raw_grads, &model);
    assert!(!grads.is_empty());
    let mut optimizer = SgdConfig::new().init();
    let model = optimizer.step(1e-3, model, grads);
    let after = model.weight.val().into_data();
    let before = before.as_slice::<f32>().unwrap();
    let after = after.as_slice::<f32>().unwrap();
    assert!(after.iter().all(|value| value.is_finite()));
    assert!(
        before
            .iter()
            .zip(after)
            .any(|(before, after)| before != after)
    );
}
