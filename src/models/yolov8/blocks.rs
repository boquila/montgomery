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
        ops::{InterpolateMode, InterpolateOptions},
    },
};

/// Ultralytics `Conv` module: convolution, batch normalization, and optional SiLU activation.
///
/// Field names deliberately match the official checkpoint (`conv`, `bn`) so imported keys map
/// onto the native graph without renaming.
#[derive(Module, Debug)]
pub struct Conv<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B>,
    act: bool,
}

impl<B: Backend> Conv<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.bn.forward(self.conv.forward(input));
        if self.act { silu(x) } else { x }
    }
}

/// Batch-normalization flavor of a [`Conv`] module.
///
/// Detect/segment checkpoints carry Ultralytics' `initialize_weights` values (eps 1e-3, momentum
/// 0.03) while the official YOLOv8-cls checkpoints were trained through the classification
/// pipeline with plain PyTorch `nn.BatchNorm2d` defaults (eps 1e-5, momentum 0.1); the epsilon is
/// part of the checkpoint's inference graph, so the flavor is a checkpoint attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BnFlavor {
    /// Ultralytics convention: eps 1e-3, momentum 0.03.
    #[default]
    Ultralytics,
    /// PyTorch `nn.BatchNorm2d` defaults: eps 1e-5, momentum 0.1.
    Pytorch,
}

impl BnFlavor {
    fn eps(self) -> f64 {
        match self {
            Self::Ultralytics => 1e-3,
            Self::Pytorch => 1e-5,
        }
    }

    fn momentum(self) -> f64 {
        match self {
            Self::Ultralytics => 0.03,
            Self::Pytorch => 0.1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConvConfig {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    groups: usize,
    act: bool,
    bn: BnFlavor,
}

impl ConvConfig {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups: 1,
            act: true,
            bn: BnFlavor::default(),
        }
    }

    pub fn depthwise(mut self) -> Self {
        self.groups = self.out_channels;
        self
    }

    pub fn without_act(mut self) -> Self {
        self.act = false;
        self
    }

    pub fn with_bn_flavor(mut self, bn: BnFlavor) -> Self {
        self.bn = bn;
        self
    }

    pub fn init<B: Backend>(&self, device: &Device<B>) -> Conv<B> {
        let padding = (self.kernel_size - 1) / 2;
        let conv = Conv2dConfig::new(
            [self.in_channels, self.out_channels],
            [self.kernel_size, self.kernel_size],
        )
        .with_stride([self.stride, self.stride])
        .with_padding(PaddingConfig2d::Explicit(
            padding, padding, padding, padding,
        ))
        .with_groups(self.groups)
        .with_bias(false)
        .init(device);
        let bn = BatchNormConfig::new(self.out_channels)
            .with_epsilon(self.bn.eps())
            .with_momentum(self.bn.momentum())
            .init(device);
        Conv {
            conv,
            bn,
            act: self.act,
        }
    }
}

/// Ultralytics `Bottleneck` with two 3x3 convolutions and an explicit hidden width.
///
/// The C2f chain builds its bottlenecks at full width (`e=1.0`), so the hidden width is declared
/// per instance instead of derived from the channel count.
#[derive(Module, Debug)]
pub struct Bottleneck<B: Backend> {
    cv1: Conv<B>,
    cv2: Conv<B>,
    add: bool,
}

impl<B: Backend> Bottleneck<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.cv2.forward(self.cv1.forward(input.clone()));
        if self.add { input + x } else { x }
    }
}

pub struct BottleneckConfig {
    channels: usize,
    hidden: usize,
    add: bool,
    bn: BnFlavor,
}

impl BottleneckConfig {
    /// Declare a bottleneck whose input and output channel counts are equal.
    ///
    /// Every YOLOv8 use site satisfies `c1 == c2`, which is the only case where the shortcut add
    /// in Ultralytics' `Bottleneck` is active.
    pub fn new(channels: usize, hidden: usize, shortcut: bool) -> Self {
        Self {
            channels,
            hidden,
            add: shortcut,
            bn: BnFlavor::default(),
        }
    }

    pub fn with_bn_flavor(mut self, bn: BnFlavor) -> Self {
        self.bn = bn;
        self
    }

    pub fn init<B: Backend>(&self, device: &Device<B>) -> Bottleneck<B> {
        Bottleneck {
            cv1: ConvConfig::new(self.channels, self.hidden, 3, 1)
                .with_bn_flavor(self.bn)
                .init(device),
            cv2: ConvConfig::new(self.hidden, self.channels, 3, 1)
                .with_bn_flavor(self.bn)
                .init(device),
            add: self.add,
        }
    }
}

/// Ultralytics `C2f`: the split-accumulate CSP block of the YOLOv8 backbone and neck.
///
/// `cv1` projects to twice the hidden width, the two halves and every chained full-width
/// bottleneck output are concatenated, and `cv2` projects back. The YAML passes `shortcut=True`
/// for the backbone stages and keeps the default `False` for the neck stages.
#[derive(Module, Debug)]
pub struct C2f<B: Backend> {
    cv1: Conv<B>,
    cv2: Conv<B>,
    m: Vec<Bottleneck<B>>,
}

impl<B: Backend> C2f<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let y = self.cv1.forward(input);
        let [batch, channels, height, width] = y.dims();
        let half = channels / 2;
        let skip = y.clone().slice([0..batch, 0..half, 0..height, 0..width]);
        let mut hidden = y.slice([0..batch, half..channels, 0..height, 0..width]);
        // The bottleneck chain consumes the second chunk, mirroring torch's `m(y[-1])`.
        let mut outputs = vec![skip, hidden.clone()];
        for bottleneck in &self.m {
            hidden = bottleneck.forward(hidden);
            outputs.push(hidden.clone());
        }
        self.cv2.forward(Tensor::cat(outputs, 1))
    }
}

#[derive(Debug, Clone)]
pub struct C2fConfig {
    in_channels: usize,
    out_channels: usize,
    repeats: usize,
    shortcut: bool,
    bn: BnFlavor,
}

impl C2fConfig {
    /// `expansion` is C2f's `e` argument; Ultralytics' default of 0.5 applies at every YOLOv8 use
    /// site, and the chained bottlenecks keep the full hidden width (`e=1.0`).
    pub fn new(in_channels: usize, out_channels: usize, repeats: usize, shortcut: bool) -> Self {
        Self {
            in_channels,
            out_channels,
            repeats,
            shortcut,
            bn: BnFlavor::default(),
        }
    }

    pub fn with_bn_flavor(mut self, bn: BnFlavor) -> Self {
        self.bn = bn;
        self
    }

    pub fn init<B: Backend>(&self, device: &Device<B>) -> C2f<B> {
        let hidden = self.out_channels / 2;
        C2f {
            cv1: ConvConfig::new(self.in_channels, 2 * hidden, 1, 1)
                .with_bn_flavor(self.bn)
                .init(device),
            cv2: ConvConfig::new((2 + self.repeats) * hidden, self.out_channels, 1, 1)
                .with_bn_flavor(self.bn)
                .init(device),
            m: (0..self.repeats)
                .map(|_| {
                    BottleneckConfig::new(hidden, hidden, self.shortcut)
                        .with_bn_flavor(self.bn)
                        .init(device)
                })
                .collect(),
        }
    }
}

/// Ultralytics `SPPF` as built for the YOLOv8 checkpoints: an input projection that keeps its
/// SiLU activation, a 5x5 pooling chain, and no residual add.
///
/// Official YOLOv8 checkpoints predate the SPPF `act=False` refactor and carry neither the `n`
/// repeat count nor the `add` shortcut attribute, exactly like YOLO11's; the pickled module —
/// not the current source — defines the inference graph.
#[derive(Module, Debug)]
pub struct Sppf<B: Backend> {
    cv1: Conv<B>,
    cv2: Conv<B>,
    pool: MaxPool2d,
}

impl<B: Backend> Sppf<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.cv1.forward(input);
        let y0 = x.clone();
        let y1 = self.pool.forward(y0.clone());
        let y2 = self.pool.forward(y1.clone());
        let y3 = self.pool.forward(y2.clone());
        self.cv2.forward(Tensor::cat(vec![y0, y1, y2, y3], 1))
    }
}

pub struct SppfConfig {
    channels: usize,
    bn: BnFlavor,
}

impl SppfConfig {
    pub fn new(channels: usize) -> Self {
        Self {
            channels,
            bn: BnFlavor::default(),
        }
    }

    pub fn with_bn_flavor(mut self, bn: BnFlavor) -> Self {
        self.bn = bn;
        self
    }

    pub fn init<B: Backend>(&self, device: &Device<B>) -> Sppf<B> {
        let hidden = self.channels / 2;
        Sppf {
            cv1: ConvConfig::new(self.channels, hidden, 1, 1)
                .with_bn_flavor(self.bn)
                .init(device),
            cv2: ConvConfig::new(hidden * 4, self.channels, 1, 1)
                .with_bn_flavor(self.bn)
                .init(device),
            pool: MaxPool2dConfig::new([5, 5])
                .with_strides([1, 1])
                .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
                .init(),
        }
    }
}

/// Nearest-neighbor 2x upsample used by the neck.
pub fn upsample_nearest_2x<B: Backend>(input: Tensor<B, 4>) -> Tensor<B, 4> {
    let [_, _, height, width] = input.dims();
    interpolate(
        input,
        [height * 2, width * 2],
        InterpolateOptions::new(InterpolateMode::Nearest),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_flex::Flex;

    #[test]
    fn produces_declared_shapes_for_yolov8n_blocks() {
        let worker = std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(|| {
                let device = Default::default();
                let c2f: C2f<Flex> = C2fConfig::new(32, 64, 1, true).init(&device);
                let out = c2f.forward(Tensor::zeros([1, 32, 40, 40], &device));
                assert_eq!(out.dims(), [1, 64, 40, 40]);

                let c2f_neck: C2f<Flex> = C2fConfig::new(96, 64, 1, false).init(&device);
                let out = c2f_neck.forward(Tensor::zeros([1, 96, 20, 20], &device));
                assert_eq!(out.dims(), [1, 64, 20, 20]);

                let sppf: Sppf<Flex> = SppfConfig::new(256).init(&device);
                let out = sppf.forward(Tensor::zeros([1, 256, 10, 10], &device));
                assert_eq!(out.dims(), [1, 256, 10, 10]);
            })
            .expect("shape-test worker should start");
        worker.join().expect("shape-test worker should not panic");
    }
}
