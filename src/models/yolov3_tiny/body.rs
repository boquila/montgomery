use burn::{
    module::Module,
    nn::{
        BatchNorm, BatchNormConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
    },
    tensor::{
        Device, Tensor,
        activation::silu,
        backend::Backend,
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions, PadMode},
    },
};

/// Convolution, batch normalization, and SiLU activation used by Ultralytics YOLO models.
#[derive(Module, Debug)]
pub struct Conv<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B>,
}

impl<B: Backend> Conv<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        silu(self.bn.forward(self.conv.forward(input)))
    }
}

pub(super) struct ConvConfig {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
}

impl ConvConfig {
    pub(super) fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
    ) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
        }
    }

    pub(super) fn init<B: Backend>(&self, device: &Device<B>) -> Conv<B> {
        let padding = (self.kernel_size - 1) / 2;
        let conv = Conv2dConfig::new(
            [self.in_channels, self.out_channels],
            [self.kernel_size, self.kernel_size],
        )
        .with_stride([self.stride, self.stride])
        .with_padding(PaddingConfig2d::Explicit(
            padding, padding, padding, padding,
        ))
        .with_bias(false)
        .init(device);
        // Ultralytics' default model configuration overrides PyTorch's BatchNorm defaults.
        // These values are part of the checkpoint's inference graph, even though momentum only
        // affects training-time running statistics.
        let bn = BatchNormConfig::new(self.out_channels)
            .with_epsilon(1e-3)
            .with_momentum(0.03)
            .init(device);
        Conv { conv, bn }
    }
}

/// Feature maps for the two YOLOv3-Tiny detection scales.
pub struct Yolov3TinyFeatures<B: Backend> {
    /// P4/16 feature map with 256 channels.
    pub p4: Tensor<B, 4>,
    /// P5/32 feature map with 512 channels.
    pub p5: Tensor<B, 4>,
}

/// Complete YOLOv3-Tiny-Ultralytics backbone and feature-pyramid body (layers 0–19).
///
/// Field names retain the source graph indices so official checkpoint remapping stays mechanical
/// and parity failures can be localized to a specific declared layer.
#[derive(Module, Debug)]
pub struct Yolov3TinyBody<B: Backend> {
    model_0: Conv<B>,
    model_1: MaxPool2d,
    model_2: Conv<B>,
    model_3: MaxPool2d,
    model_4: Conv<B>,
    model_5: MaxPool2d,
    model_6: Conv<B>,
    model_7: MaxPool2d,
    model_8: Conv<B>,
    model_9: MaxPool2d,
    model_10: Conv<B>,
    model_12: MaxPool2d,
    model_13: Conv<B>,
    model_14: Conv<B>,
    model_15: Conv<B>,
    model_16: Conv<B>,
    model_19: Conv<B>,
}

impl<B: Backend> Yolov3TinyBody<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Yolov3TinyFeatures<B> {
        let x = self.model_0.forward(input);
        let x = self.model_1.forward(x);
        let x = self.model_2.forward(x);
        let x = self.model_3.forward(x);
        let x = self.model_4.forward(x);
        let x = self.model_5.forward(x);
        let x = self.model_6.forward(x);
        let x = self.model_7.forward(x);
        let route_p4 = self.model_8.forward(x);

        let x = self.model_9.forward(route_p4.clone());
        let x = self.model_10.forward(x);
        // Ultralytics layers 11–12: ZeroPad2d([0, 1, 0, 1]) then MaxPool2d(2, 1, 0).
        let x = x.pad((0, 1, 0, 1), PadMode::Constant(0.0));
        let x = self.model_12.forward(x);
        let x = self.model_13.forward(x);
        let route_p5 = self.model_14.forward(x);
        let p5 = self.model_15.forward(route_p5.clone());

        let x = self.model_16.forward(route_p5);
        let [_, _, height, width] = x.dims();
        let x = interpolate(
            x,
            [height * 2, width * 2],
            InterpolateOptions::new(InterpolateMode::Nearest),
        );
        let x = Tensor::cat(vec![x, route_p4], 1);
        let p4 = self.model_19.forward(x);

        Yolov3TinyFeatures { p4, p5 }
    }
}

/// Configuration for the fixed YOLOv3-Tiny-Ultralytics body.
#[derive(Debug, Default)]
pub struct Yolov3TinyBodyConfig;

impl Yolov3TinyBodyConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolov3TinyBody<B> {
        let pool_stride_2 = || MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();
        Yolov3TinyBody {
            model_0: ConvConfig::new(3, 16, 3, 1).init(device),
            model_1: pool_stride_2(),
            model_2: ConvConfig::new(16, 32, 3, 1).init(device),
            model_3: pool_stride_2(),
            model_4: ConvConfig::new(32, 64, 3, 1).init(device),
            model_5: pool_stride_2(),
            model_6: ConvConfig::new(64, 128, 3, 1).init(device),
            model_7: pool_stride_2(),
            model_8: ConvConfig::new(128, 256, 3, 1).init(device),
            model_9: pool_stride_2(),
            model_10: ConvConfig::new(256, 512, 3, 1).init(device),
            model_12: MaxPool2dConfig::new([2, 2]).with_strides([1, 1]).init(),
            model_13: ConvConfig::new(512, 1024, 3, 1).init(device),
            model_14: ConvConfig::new(1024, 256, 1, 1).init(device),
            model_15: ConvConfig::new(256, 512, 3, 1).init(device),
            model_16: ConvConfig::new(256, 128, 1, 1).init(device),
            model_19: ConvConfig::new(384, 256, 3, 1).init(device),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_flex::Flex;

    #[test]
    fn produces_declared_p4_and_p5_shapes() {
        let worker = std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(|| {
                let device = Default::default();
                let body: Yolov3TinyBody<Flex> = Yolov3TinyBodyConfig.init(&device);
                let input = Tensor::zeros([1, 3, 64, 64], &device);
                let output = body.forward(input);
                assert_eq!(output.p4.dims(), [1, 256, 4, 4]);
                assert_eq!(output.p5.dims(), [1, 512, 2, 2]);
            })
            .expect("shape-test worker should start");
        worker.join().expect("shape-test worker should not panic");
    }
}
