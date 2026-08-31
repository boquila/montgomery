use burn::{
    module::Module,
    nn::{
        BatchNorm, BatchNormConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
    },
    tensor::{
        Device, Tensor,
        activation::{silu, softmax},
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
    #[cfg(feature = "training")]
    depthwise_training_stencil: bool,
}

impl<B: Backend> Conv<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        #[cfg(feature = "training")]
        let x = if self.depthwise_training_stencil && B::ad_enabled(&input.device()) {
            crate::models::training_ops::depthwise_3x3_stride_1(input, self.conv.weight.val())
        } else {
            self.conv.forward(input)
        };
        #[cfg(not(feature = "training"))]
        let x = self.conv.forward(input);
        let x = self.bn.forward(x);
        if self.act { silu(x) } else { x }
    }
}

#[derive(Debug, Clone)]
pub(super) struct ConvConfig {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    groups: usize,
    act: bool,
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
            groups: 1,
            act: true,
        }
    }

    pub(super) fn depthwise(mut self) -> Self {
        self.groups = self.out_channels;
        self
    }

    pub(super) fn without_act(mut self) -> Self {
        self.act = false;
        self
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
        .with_groups(self.groups)
        .with_bias(false)
        .init(device);
        // Ultralytics' initialize_weights overrides PyTorch's BatchNorm defaults. Epsilon is part
        // of the checkpoint's inference graph even though momentum only affects training.
        let bn = BatchNormConfig::new(self.out_channels)
            .with_epsilon(1e-3)
            .with_momentum(0.03)
            .init(device);
        Conv {
            conv,
            bn,
            act: self.act,
            #[cfg(feature = "training")]
            depthwise_training_stencil: self.groups == self.in_channels
                && self.groups == self.out_channels
                && self.kernel_size == 3
                && self.stride == 1,
        }
    }
}

/// Ultralytics `RepVGGDW`: parallel depth-wise 7x7 and 3x3 convolutions summed and passed through
/// SiLU. Field names match the official checkpoint.
#[derive(Module, Debug)]
pub struct RepVggDw<B: Backend> {
    conv: Conv<B>,
    conv1: Conv<B>,
}

impl<B: Backend> RepVggDw<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        silu(self.conv.forward(input.clone()) + self.conv1.forward(input))
    }
}

pub(super) struct RepVggDwConfig {
    channels: usize,
}

impl RepVggDwConfig {
    pub(super) fn new(channels: usize) -> Self {
        Self { channels }
    }

    pub(super) fn init<B: Backend>(&self, device: &Device<B>) -> RepVggDw<B> {
        RepVggDw {
            conv: ConvConfig::new(self.channels, self.channels, 7, 1)
                .depthwise()
                .without_act()
                .init(device),
            conv1: ConvConfig::new(self.channels, self.channels, 3, 1)
                .depthwise()
                .without_act()
                .init(device),
        }
    }
}

/// Ultralytics `Bottleneck` with equal input/output channels and two 3x3 convolutions.
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

pub(super) struct BottleneckConfig {
    channels: usize,
    shortcut: bool,
}

impl BottleneckConfig {
    pub(super) fn new(channels: usize, shortcut: bool) -> Self {
        Self { channels, shortcut }
    }

    pub(super) fn init<B: Backend>(&self, device: &Device<B>) -> Bottleneck<B> {
        Bottleneck {
            cv1: ConvConfig::new(self.channels, self.channels, 3, 1).init(device),
            cv2: ConvConfig::new(self.channels, self.channels, 3, 1).init(device),
            add: self.shortcut,
        }
    }
}

/// The CIB convolution tower; the tuple preserves the indexed weight paths (`cv1.0.*`,
/// `cv1.1.*`, ...) that the official checkpoint uses for its `nn.Sequential`.
type CibTower<B> = (Conv<B>, Conv<B>, RepVggDw<B>, Conv<B>, Conv<B>);

/// Ultralytics `CIB` (Compact Inverted Block) with the large-kernel depth-wise center (`lk=True`).
#[derive(Module, Debug)]
pub struct Cib<B: Backend> {
    cv1: CibTower<B>,
    add: bool,
}

impl<B: Backend> Cib<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let (dw_0, pw_0, dw_large, pw_1, dw_1) = &self.cv1;
        let x = dw_0.forward(input.clone());
        let x = pw_0.forward(x);
        let x = dw_large.forward(x);
        let x = pw_1.forward(x);
        let x = dw_1.forward(x);
        if self.add { input + x } else { x }
    }
}

pub(super) struct CibConfig {
    channels: usize,
    shortcut: bool,
}

impl CibConfig {
    pub(super) fn new(channels: usize, shortcut: bool) -> Self {
        Self { channels, shortcut }
    }

    pub(super) fn init<B: Backend>(&self, device: &Device<B>) -> Cib<B> {
        let c = self.channels;
        Cib {
            cv1: (
                ConvConfig::new(c, c, 3, 1).depthwise().init(device),
                ConvConfig::new(c, 2 * c, 1, 1).init(device),
                RepVggDwConfig::new(2 * c).init(device),
                ConvConfig::new(2 * c, c, 1, 1).init(device),
                ConvConfig::new(c, c, 3, 1).depthwise().init(device),
            ),
            add: self.shortcut,
        }
    }
}

/// The `lk=False` CIB tower: the center is a plain depth-wise convolution instead of the fused
/// `RepVGGDW`, which also changes the checkpoint key paths (`cv1.2.conv.*` with no `conv1`).
type CibDwTower<B> = (Conv<B>, Conv<B>, Conv<B>, Conv<B>, Conv<B>);

/// Ultralytics `CIB` (Compact Inverted Block) with the plain depth-wise center (`lk=False`).
///
/// This is the variant the s/m/b/l/x checkpoints build; only YOLOv10n/s pass `lk=True`.
#[derive(Module, Debug)]
pub struct CibDw<B: Backend> {
    cv1: CibDwTower<B>,
    add: bool,
}

impl<B: Backend> CibDw<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let (dw_0, pw_0, dw_center, pw_1, dw_1) = &self.cv1;
        let x = dw_0.forward(input.clone());
        let x = pw_0.forward(x);
        let x = dw_center.forward(x);
        let x = pw_1.forward(x);
        let x = dw_1.forward(x);
        if self.add { input + x } else { x }
    }
}

pub(super) struct CibDwConfig {
    channels: usize,
    shortcut: bool,
}

impl CibDwConfig {
    pub(super) fn new(channels: usize, shortcut: bool) -> Self {
        Self { channels, shortcut }
    }

    pub(super) fn init<B: Backend>(&self, device: &Device<B>) -> CibDw<B> {
        let c = self.channels;
        CibDw {
            cv1: (
                ConvConfig::new(c, c, 3, 1).depthwise().init(device),
                ConvConfig::new(c, 2 * c, 1, 1).init(device),
                ConvConfig::new(2 * c, 2 * c, 3, 1).depthwise().init(device),
                ConvConfig::new(2 * c, c, 1, 1).init(device),
                ConvConfig::new(c, c, 3, 1).depthwise().init(device),
            ),
            add: self.shortcut,
        }
    }
}

/// Ultralytics `C2f`: CSP bottleneck with a fed-forward bottleneck chain.
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

/// Ultralytics `C2fCIB`: C2f whose bottleneck chain uses CIB blocks.
#[derive(Module, Debug)]
pub struct C2fCib<B: Backend> {
    cv1: Conv<B>,
    cv2: Conv<B>,
    m: Vec<Cib<B>>,
}

impl<B: Backend> C2fCib<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let y = self.cv1.forward(input);
        let [batch, channels, height, width] = y.dims();
        let half = channels / 2;
        let skip = y.clone().slice([0..batch, 0..half, 0..height, 0..width]);
        let mut hidden = y.slice([0..batch, half..channels, 0..height, 0..width]);
        // The bottleneck chain consumes the second chunk, mirroring torch's `m(y[-1])`.
        let mut outputs = vec![skip, hidden.clone()];
        for cib in &self.m {
            hidden = cib.forward(hidden);
            outputs.push(hidden.clone());
        }
        self.cv2.forward(Tensor::cat(outputs, 1))
    }
}

/// Shared construction logic for the C2f-family blocks.
struct C2fCommon {
    in_channels: usize,
    out_channels: usize,
    hidden: usize,
    repeats: usize,
    shortcut: bool,
}

impl C2fCommon {
    fn new(in_channels: usize, out_channels: usize, repeats: usize, shortcut: bool) -> Self {
        Self {
            in_channels,
            out_channels,
            hidden: out_channels / 2,
            repeats,
            shortcut,
        }
    }

    fn init_conv<B: Backend>(&self, device: &Device<B>) -> (Conv<B>, Conv<B>) {
        (
            ConvConfig::new(self.in_channels, 2 * self.hidden, 1, 1).init(device),
            ConvConfig::new((2 + self.repeats) * self.hidden, self.out_channels, 1, 1).init(device),
        )
    }
}

pub(super) struct C2fConfig {
    common: C2fCommon,
}

impl C2fConfig {
    pub(super) fn new(
        in_channels: usize,
        out_channels: usize,
        repeats: usize,
        shortcut: bool,
    ) -> Self {
        Self {
            common: C2fCommon::new(in_channels, out_channels, repeats, shortcut),
        }
    }

    pub(super) fn init<B: Backend>(&self, device: &Device<B>) -> C2f<B> {
        let (cv1, cv2) = self.common.init_conv(device);
        let m = (0..self.common.repeats)
            .map(|_| BottleneckConfig::new(self.common.hidden, self.common.shortcut).init(device))
            .collect();
        C2f { cv1, cv2, m }
    }
}

pub(super) struct C2fCibConfig {
    common: C2fCommon,
}

impl C2fCibConfig {
    pub(super) fn new(
        in_channels: usize,
        out_channels: usize,
        repeats: usize,
        shortcut: bool,
    ) -> Self {
        Self {
            common: C2fCommon::new(in_channels, out_channels, repeats, shortcut),
        }
    }

    pub(super) fn init<B: Backend>(&self, device: &Device<B>) -> C2fCib<B> {
        let (cv1, cv2) = self.common.init_conv(device);
        let m = (0..self.common.repeats)
            .map(|_| CibConfig::new(self.common.hidden, self.common.shortcut).init(device))
            .collect();
        C2fCib { cv1, cv2, m }
    }
}

/// Ultralytics `C2fCIB` with the plain depth-wise CIB chain (`lk=False`), used by the
/// s/m/b/l/x-scale bodies.
#[derive(Module, Debug)]
pub struct C2fCibDw<B: Backend> {
    cv1: Conv<B>,
    cv2: Conv<B>,
    m: Vec<CibDw<B>>,
}

impl<B: Backend> C2fCibDw<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let y = self.cv1.forward(input);
        let [batch, channels, height, width] = y.dims();
        let half = channels / 2;
        let skip = y.clone().slice([0..batch, 0..half, 0..height, 0..width]);
        let mut hidden = y.slice([0..batch, half..channels, 0..height, 0..width]);
        // The bottleneck chain consumes the second chunk, mirroring torch's `m(y[-1])`.
        let mut outputs = vec![skip, hidden.clone()];
        for cib in &self.m {
            hidden = cib.forward(hidden);
            outputs.push(hidden.clone());
        }
        self.cv2.forward(Tensor::cat(outputs, 1))
    }
}

pub(super) struct C2fCibDwConfig {
    common: C2fCommon,
}

impl C2fCibDwConfig {
    pub(super) fn new(
        in_channels: usize,
        out_channels: usize,
        repeats: usize,
        shortcut: bool,
    ) -> Self {
        Self {
            common: C2fCommon::new(in_channels, out_channels, repeats, shortcut),
        }
    }

    pub(super) fn init<B: Backend>(&self, device: &Device<B>) -> C2fCibDw<B> {
        let (cv1, cv2) = self.common.init_conv(device);
        let m = (0..self.common.repeats)
            .map(|_| CibDwConfig::new(self.common.hidden, self.common.shortcut).init(device))
            .collect();
        C2fCibDw { cv1, cv2, m }
    }
}

/// Ultralytics `SCDown`: pointwise channel reduction followed by strided depth-wise convolution.
#[derive(Module, Debug)]
pub struct ScDown<B: Backend> {
    cv1: Conv<B>,
    cv2: Conv<B>,
}

impl<B: Backend> ScDown<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        self.cv2.forward(self.cv1.forward(input))
    }
}

pub(super) struct ScDownConfig {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
}

impl ScDownConfig {
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

    pub(super) fn init<B: Backend>(&self, device: &Device<B>) -> ScDown<B> {
        ScDown {
            cv1: ConvConfig::new(self.in_channels, self.out_channels, 1, 1).init(device),
            cv2: ConvConfig::new(
                self.out_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
            )
            .depthwise()
            .without_act()
            .init(device),
        }
    }
}

/// Ultralytics `SPPF`: three chained 5x5 max pools concatenated with the projected input.
#[derive(Module, Debug)]
pub struct Sppf<B: Backend> {
    cv1: Conv<B>,
    cv2: Conv<B>,
    pool: MaxPool2d,
}

impl<B: Backend> Sppf<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.cv1.forward(input);
        let first = self.pool.forward(x.clone());
        let second = self.pool.forward(first.clone());
        let third = self.pool.forward(second.clone());
        self.cv2
            .forward(Tensor::cat(vec![x, first, second, third], 1))
    }
}

pub(super) struct SppfConfig {
    channels: usize,
}

impl SppfConfig {
    pub(super) fn new(channels: usize) -> Self {
        Self { channels }
    }

    pub(super) fn init<B: Backend>(&self, device: &Device<B>) -> Sppf<B> {
        let hidden = self.channels / 2;
        Sppf {
            // Official YOLOv10 checkpoints keep the SiLU activation on cv1 even though the
            // current Ultralytics source builds SPPF with act=False; the checkpoint wins.
            cv1: ConvConfig::new(self.channels, hidden, 1, 1).init(device),
            cv2: ConvConfig::new(hidden * 4, self.channels, 1, 1).init(device),
            pool: MaxPool2dConfig::new([5, 5])
                .with_strides([1, 1])
                .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
                .init(),
        }
    }
}

/// Ultralytics `Attention`: multi-head self-attention with a depth-wise positional encoding.
#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    qkv: Conv<B>,
    proj: Conv<B>,
    pe: Conv<B>,
    num_heads: usize,
    head_dim: usize,
    key_dim: usize,
    scale: f32,
}

impl<B: Backend> Attention<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = input.dims();
        let tokens = height * width;
        let kd = self.key_dim;
        let hd = self.head_dim;
        let qkv = self
            .qkv
            .forward(input)
            .reshape([batch, self.num_heads, kd * 2 + hd, tokens]);
        let q = qkv
            .clone()
            .slice([0..batch, 0..self.num_heads, 0..kd, 0..tokens]);
        let k = qkv
            .clone()
            .slice([0..batch, 0..self.num_heads, kd..kd * 2, 0..tokens]);
        let v = qkv.slice([0..batch, 0..self.num_heads, kd * 2..kd * 2 + hd, 0..tokens]);

        let attn = (q * self.scale).swap_dims(2, 3).matmul(k);
        let attn = softmax(attn, 3);
        let attended = v
            .clone()
            .matmul(attn.swap_dims(2, 3))
            .reshape([batch, channels, height, width]);
        let pe = self.pe.forward(v.reshape([batch, channels, height, width]));
        self.proj.forward(attended + pe)
    }
}

pub(super) struct AttentionConfig {
    dim: usize,
    num_heads: usize,
    attn_ratio: f32,
}

impl AttentionConfig {
    pub(super) fn new(dim: usize, num_heads: usize) -> Self {
        Self {
            dim,
            num_heads,
            attn_ratio: 0.5,
        }
    }

    pub(super) fn init<B: Backend>(&self, device: &Device<B>) -> Attention<B> {
        let head_dim = self.dim / self.num_heads;
        let key_dim = (head_dim as f32 * self.attn_ratio) as usize;
        let qkv_channels = self.dim + key_dim * self.num_heads * 2;
        Attention {
            qkv: ConvConfig::new(self.dim, qkv_channels, 1, 1)
                .without_act()
                .init(device),
            proj: ConvConfig::new(self.dim, self.dim, 1, 1)
                .without_act()
                .init(device),
            pe: ConvConfig::new(self.dim, self.dim, 3, 1)
                .depthwise()
                .without_act()
                .init(device),
            num_heads: self.num_heads,
            head_dim,
            key_dim,
            scale: (key_dim as f32).powf(-0.5),
        }
    }
}

/// Ultralytics `PSA`: position-sensitive attention with a parallel skip branch and feed-forward
/// refinement on half the channels.
#[derive(Module, Debug)]
pub struct Psa<B: Backend> {
    cv1: Conv<B>,
    cv2: Conv<B>,
    attn: Attention<B>,
    ffn: (Conv<B>, Conv<B>),
    hidden: usize,
}

impl<B: Backend> Psa<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.cv1.forward(input);
        let [batch, channels, height, width] = x.dims();
        let a = x
            .clone()
            .slice([0..batch, 0..self.hidden, 0..height, 0..width]);
        let b = x.slice([0..batch, self.hidden..channels, 0..height, 0..width]);
        let b = b.clone() + self.attn.forward(b.clone());
        let ffn = self.ffn.1.forward(self.ffn.0.forward(b.clone()));
        let b = b + ffn;
        self.cv2.forward(Tensor::cat(vec![a, b], 1))
    }
}

pub(super) struct PsaConfig {
    channels: usize,
}

impl PsaConfig {
    pub(super) fn new(channels: usize) -> Self {
        Self { channels }
    }

    pub(super) fn init<B: Backend>(&self, device: &Device<B>) -> Psa<B> {
        let hidden = self.channels / 2;
        Psa {
            cv1: ConvConfig::new(self.channels, 2 * hidden, 1, 1).init(device),
            cv2: ConvConfig::new(2 * hidden, self.channels, 1, 1).init(device),
            attn: AttentionConfig::new(hidden, (hidden / 64).max(1)).init(device),
            ffn: (
                ConvConfig::new(hidden, hidden * 2, 1, 1).init(device),
                ConvConfig::new(hidden * 2, hidden, 1, 1)
                    .without_act()
                    .init(device),
            ),
            hidden,
        }
    }
}

/// Nearest-neighbor 2x upsample used by the neck.
pub(super) fn upsample_nearest_2x<B: Backend>(input: Tensor<B, 4>) -> Tensor<B, 4> {
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
    fn produces_declared_shapes_for_yolov10n_blocks() {
        let worker = std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(|| {
                let device = Default::default();
                let c2f: C2f<Flex> = C2fConfig::new(64, 64, 2, true).init(&device);
                let out = c2f.forward(Tensor::zeros([1, 64, 40, 40], &device));
                assert_eq!(out.dims(), [1, 64, 40, 40]);

                let cib_block: C2fCib<Flex> = C2fCibConfig::new(384, 256, 1, true).init(&device);
                let out = cib_block.forward(Tensor::zeros([1, 384, 20, 20], &device));
                assert_eq!(out.dims(), [1, 256, 20, 20]);

                let cib_dw_block: C2fCibDw<Flex> =
                    C2fCibDwConfig::new(384, 256, 1, true).init(&device);
                let out = cib_dw_block.forward(Tensor::zeros([1, 384, 20, 20], &device));
                assert_eq!(out.dims(), [1, 256, 20, 20]);

                let sc_down: ScDown<Flex> = ScDownConfig::new(128, 128, 3, 2).init(&device);
                let out = sc_down.forward(Tensor::zeros([1, 128, 40, 40], &device));
                assert_eq!(out.dims(), [1, 128, 20, 20]);

                let sppf: Sppf<Flex> = SppfConfig::new(256).init(&device);
                let out = sppf.forward(Tensor::zeros([1, 256, 20, 20], &device));
                assert_eq!(out.dims(), [1, 256, 20, 20]);

                let psa: Psa<Flex> = PsaConfig::new(256).init(&device);
                let out = psa.forward(Tensor::zeros([1, 256, 20, 20], &device));
                assert_eq!(out.dims(), [1, 256, 20, 20]);
            })
            .expect("shape-test worker should start");
        worker.join().expect("shape-test worker should not panic");
    }
}
