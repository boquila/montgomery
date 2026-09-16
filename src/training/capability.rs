use burn::{
    module::{AutodiffModule, Module},
    nn::{
        BatchNorm, BatchNormConfig,
        conv::{Conv2d, Conv2dConfig},
    },
    optim::{GradientsParams, SgdConfig},
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
/// selected adapter with `cargo test wgpu_autodiff_capability -- --ignored`.
#[test]
#[ignore = "requires a local WGPU adapter"]
fn wgpu_autodiff_capability() {
    let (device, _) = crate::default_wgpu_device();
    let device = device.autodiff();
    let model = Spike::init(&device);
    let input = Tensor::random([2, 3, 16, 16], Distribution::Default, &device);
    let loss = model.forward(input);
    let grads = GradientsParams::from_grads(loss.backward(), &model);
    let mut optimizer = SgdConfig::new().init();
    let model = optimizer.step(1e-3, model, grads);

    // `valid` converts to the inner WGPU graph, so this forward cannot mutate BN running state.
    let valid = model.valid();
    let input = Tensor::zeros([1, 3, 16, 16], &device);
    let output = valid.forward(input).into_data();
    assert!(output.as_slice::<f32>().unwrap()[0].is_finite());
    assert!(!optimizer.to_record().is_empty());
}

#[test]
fn validation_backend_does_not_mutate_batch_norm_state() {
    let device = Device::default().autodiff();
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
