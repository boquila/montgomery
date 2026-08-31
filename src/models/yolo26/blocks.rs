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

/// Batch-normalization flavor of a `Conv` module.
///
/// Ultralytics' `initialize_weights` overrides PyTorch's BatchNorm defaults (eps 1e-3, momentum
/// 0.03) and the detect/seg/pose/obb checkpoints were trained that way, but the official
/// YOLO26-cls checkpoints carry plain PyTorch BatchNorm defaults (eps 1e-5, momentum 0.1) — the
/// epsilon is part of the checkpoint's inference graph, so the flavor is a checkpoint attribute,
/// not a global constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum BnFlavor {
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
pub(super) struct ConvConfig {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    groups: usize,
    act: bool,
    bn: BnFlavor,
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
            bn: BnFlavor::default(),
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

    pub(super) fn with_bn_flavor(mut self, bn: BnFlavor) -> Self {
        self.bn = bn;
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
        let bn = BatchNormConfig::new(self.out_channels)
            .with_epsilon(self.bn.eps())
            .with_momentum(self.bn.momentum())
            .init(device);
        Conv {
            conv,
            bn,
            act: self.act,
            #[cfg(feature = "training")]
            depthwise_training_stencil: self.groups == self.in_channels
                && self.in_channels == self.out_channels
                && self.kernel_size == 3
                && self.stride == 1,
        }
    }
}

/// Ultralytics `Bottleneck` with two 3x3 convolutions and an explicit hidden width.
///
/// YOLO26 builds this block with two different expansions: C3k2 chains keep Ultralytics' default
/// half-width hidden convolution while C3k's inner chain passes `e=1.0`, so the hidden width is
/// declared per instance instead of derived from the channel count.
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
    hidden: usize,
    add: bool,
    bn: BnFlavor,
}

impl BottleneckConfig {
    /// Declare a bottleneck whose input and output channel counts are equal.
    ///
    /// Every YOLO26 use site satisfies `c1 == c2`, which is the only case where the shortcut add
    /// in Ultralytics' `Bottleneck` is active.
    pub(super) fn new(channels: usize, hidden: usize, shortcut: bool) -> Self {
        Self {
            channels,
            hidden,
            add: shortcut,
            bn: BnFlavor::default(),
        }
    }

    pub(super) fn with_bn_flavor(mut self, bn: BnFlavor) -> Self {
        self.bn = bn;
        self
    }

    pub(super) fn init<B: Backend>(&self, device: &Device<B>) -> Bottleneck<B> {
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

/// Ultralytics `C3k`: a C3 block whose full-width bottleneck chain runs on half-width branches.
#[derive(Module, Debug)]
pub struct C3k<B: Backend> {
    cv1: Conv<B>,
    cv2: Conv<B>,
    cv3: Conv<B>,
    m: Vec<Bottleneck<B>>,
}

impl<B: Backend> C3k<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let branch = self
            .m
            .iter()
            .fold(self.cv1.forward(input.clone()), |x, bottleneck| {
                bottleneck.forward(x)
            });
        let skip = self.cv2.forward(input);
        self.cv3.forward(Tensor::cat(vec![branch, skip], 1))
    }
}

pub(super) struct C3kConfig {
    channels: usize,
    repeats: usize,
    shortcut: bool,
    bn: BnFlavor,
}

impl C3kConfig {
    /// Ultralytics instantiates C3k inside C3k2 as `C3k(c, c, 2, shortcut)` with its default
    /// expansion of 0.5 and default kernel of 3, so those values are fixed here.
    pub(super) fn new(channels: usize, repeats: usize, shortcut: bool) -> Self {
        Self {
            channels,
            repeats,
            shortcut,
            bn: BnFlavor::default(),
        }
    }

    pub(super) fn with_bn_flavor(mut self, bn: BnFlavor) -> Self {
        self.bn = bn;
        self
    }

    pub(super) fn init<B: Backend>(&self, device: &Device<B>) -> C3k<B> {
        let hidden = self.channels / 2;
        C3k {
            cv1: ConvConfig::new(self.channels, hidden, 1, 1)
                .with_bn_flavor(self.bn)
                .init(device),
            cv2: ConvConfig::new(self.channels, hidden, 1, 1)
                .with_bn_flavor(self.bn)
                .init(device),
            cv3: ConvConfig::new(hidden * 2, self.channels, 1, 1)
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

/// Ultralytics `C3k2` with the plain bottleneck chain used by the early backbone stages.
#[derive(Module, Debug)]
pub struct C3k2<B: Backend> {
    cv1: Conv<B>,
    cv2: Conv<B>,
    m: Vec<Bottleneck<B>>,
}

impl<B: Backend> C3k2<B> {
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

/// Shared construction logic for the C3k2 variants: a split convolution, an output projection
/// sized for the skip plus every chained block, and Ultralytics' default expansion of 0.5.
pub(super) struct C3k2Shell {
    in_channels: usize,
    out_channels: usize,
    hidden: usize,
    repeats: usize,
    bn: BnFlavor,
}

impl C3k2Shell {
    /// `expansion` is C3k2's `e` argument; the backbone's small stages pass 0.25 and every other
    /// stage uses the default 0.5.
    pub(super) fn new(
        in_channels: usize,
        out_channels: usize,
        repeats: usize,
        expansion: f32,
    ) -> Self {
        Self {
            in_channels,
            out_channels,
            hidden: (out_channels as f32 * expansion) as usize,
            repeats,
            bn: BnFlavor::default(),
        }
    }

    pub(super) fn with_bn_flavor(mut self, bn: BnFlavor) -> Self {
        self.bn = bn;
        self
    }

    fn init_conv<B: Backend>(&self, device: &Device<B>) -> (Conv<B>, Conv<B>) {
        (
            ConvConfig::new(self.in_channels, 2 * self.hidden, 1, 1)
                .with_bn_flavor(self.bn)
                .init(device),
            ConvConfig::new((2 + self.repeats) * self.hidden, self.out_channels, 1, 1)
                .with_bn_flavor(self.bn)
                .init(device),
        )
    }
}

pub(super) struct C3k2Config {
    shell: C3k2Shell,
    shortcut: bool,
}

impl C3k2Config {
    pub(super) fn new(
        in_channels: usize,
        out_channels: usize,
        repeats: usize,
        expansion: f32,
        shortcut: bool,
    ) -> Self {
        Self {
            shell: C3k2Shell::new(in_channels, out_channels, repeats, expansion),
            shortcut,
        }
    }

    pub(super) fn with_bn_flavor(mut self, bn: BnFlavor) -> Self {
        self.shell = self.shell.with_bn_flavor(bn);
        self
    }

    pub(super) fn init<B: Backend>(&self, device: &Device<B>) -> C3k2<B> {
        let (cv1, cv2) = self.shell.init_conv(device);
        // Ultralytics' plain C3k2 bottleneck keeps the default half-width expansion.
        let hidden = self.shell.hidden;
        let m = (0..self.shell.repeats)
            .map(|_| {
                BottleneckConfig::new(hidden, hidden / 2, self.shortcut)
                    .with_bn_flavor(self.shell.bn)
                    .init(device)
            })
            .collect();
        C3k2 { cv1, cv2, m }
    }
}

/// Ultralytics `C3k2` with a C3k chain, used by the P3/P4 stages.
#[derive(Module, Debug)]
pub struct C3k2C3k<B: Backend> {
    cv1: Conv<B>,
    cv2: Conv<B>,
    m: Vec<C3k<B>>,
}

impl<B: Backend> C3k2C3k<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let y = self.cv1.forward(input);
        let [batch, channels, height, width] = y.dims();
        let half = channels / 2;
        let skip = y.clone().slice([0..batch, 0..half, 0..height, 0..width]);
        let mut hidden = y.slice([0..batch, half..channels, 0..height, 0..width]);
        let mut outputs = vec![skip, hidden.clone()];
        for c3k in &self.m {
            hidden = c3k.forward(hidden);
            outputs.push(hidden.clone());
        }
        self.cv2.forward(Tensor::cat(outputs, 1))
    }
}

pub(super) struct C3k2C3kConfig {
    shell: C3k2Shell,
    shortcut: bool,
}

impl C3k2C3kConfig {
    /// `expansion` is C3k2's `e` argument, which sizes the shell's hidden width. The neck and
    /// P4/P5 backbone stages keep Ultralytics' default 0.5 while the early backbone stages of the
    /// m/l/x scales force `c3k` on with the YAML's 0.25 expansion.
    pub(super) fn new(
        in_channels: usize,
        out_channels: usize,
        repeats: usize,
        shortcut: bool,
        expansion: f32,
    ) -> Self {
        Self {
            shell: C3k2Shell::new(in_channels, out_channels, repeats, expansion),
            shortcut,
        }
    }

    pub(super) fn with_bn_flavor(mut self, bn: BnFlavor) -> Self {
        self.shell = self.shell.with_bn_flavor(bn);
        self
    }

    pub(super) fn init<B: Backend>(&self, device: &Device<B>) -> C3k2C3k<B> {
        let (cv1, cv2) = self.shell.init_conv(device);
        // Ultralytics builds `C3k(self.c, self.c, 2, shortcut)` for every C3k chain block.
        let m = (0..self.shell.repeats)
            .map(|_| {
                C3kConfig::new(self.shell.hidden, 2, self.shortcut)
                    .with_bn_flavor(self.shell.bn)
                    .init(device)
            })
            .collect();
        C3k2C3k { cv1, cv2, m }
    }
}

/// Ultralytics `C3k2` with a bottleneck-plus-attention chain, used by the P5 stage.
#[derive(Module, Debug)]
pub struct C3k2Attn<B: Backend> {
    cv1: Conv<B>,
    cv2: Conv<B>,
    m: Vec<(Bottleneck<B>, PsaBlock<B>)>,
}

impl<B: Backend> C3k2Attn<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let y = self.cv1.forward(input);
        let [batch, channels, height, width] = y.dims();
        let half = channels / 2;
        let skip = y.clone().slice([0..batch, 0..half, 0..height, 0..width]);
        let mut hidden = y.slice([0..batch, half..channels, 0..height, 0..width]);
        let mut outputs = vec![skip, hidden.clone()];
        for (bottleneck, psa_block) in &self.m {
            hidden = psa_block.forward(bottleneck.forward(hidden));
            outputs.push(hidden.clone());
        }
        self.cv2.forward(Tensor::cat(outputs, 1))
    }
}

pub(super) struct C3k2AttnConfig {
    shell: C3k2Shell,
    shortcut: bool,
}

impl C3k2AttnConfig {
    pub(super) fn new(
        in_channels: usize,
        out_channels: usize,
        repeats: usize,
        shortcut: bool,
    ) -> Self {
        Self {
            shell: C3k2Shell::new(in_channels, out_channels, repeats, 0.5),
            shortcut,
        }
    }

    pub(super) fn init<B: Backend>(&self, device: &Device<B>) -> C3k2Attn<B> {
        let (cv1, cv2) = self.shell.init_conv(device);
        // Ultralytics builds `Sequential(Bottleneck(c, c, shortcut), PSABlock(c, 0.5, max(c // 64, 1)))`
        // for every attention chain block; the bottleneck keeps the half-width expansion.
        let hidden = self.shell.hidden;
        let m = (0..self.shell.repeats)
            .map(|_| {
                (
                    BottleneckConfig::new(hidden, hidden / 2, self.shortcut)
                        .with_bn_flavor(self.shell.bn)
                        .init(device),
                    PsaBlockConfig::new(hidden)
                        .with_bn_flavor(self.shell.bn)
                        .init(device),
                )
            })
            .collect();
        C3k2Attn { cv1, cv2, m }
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
    bn: BnFlavor,
}

impl AttentionConfig {
    pub(super) fn new(dim: usize, num_heads: usize) -> Self {
        Self {
            dim,
            num_heads,
            attn_ratio: 0.5,
            bn: BnFlavor::default(),
        }
    }

    pub(super) fn with_bn_flavor(mut self, bn: BnFlavor) -> Self {
        self.bn = bn;
        self
    }

    pub(super) fn init<B: Backend>(&self, device: &Device<B>) -> Attention<B> {
        let head_dim = self.dim / self.num_heads;
        let key_dim = (head_dim as f32 * self.attn_ratio) as usize;
        let qkv_channels = self.dim + key_dim * self.num_heads * 2;
        Attention {
            qkv: ConvConfig::new(self.dim, qkv_channels, 1, 1)
                .with_bn_flavor(self.bn)
                .without_act()
                .init(device),
            proj: ConvConfig::new(self.dim, self.dim, 1, 1)
                .with_bn_flavor(self.bn)
                .without_act()
                .init(device),
            pe: ConvConfig::new(self.dim, self.dim, 3, 1)
                .depthwise()
                .with_bn_flavor(self.bn)
                .without_act()
                .init(device),
            num_heads: self.num_heads,
            head_dim,
            key_dim,
            scale: (key_dim as f32).powf(-0.5),
        }
    }
}

/// Ultralytics `PSABlock`: attention plus a feed-forward refinement, each with a residual add.
///
/// Every YOLO26 use site keeps Ultralytics' default `shortcut=True`, so the residual adds are
/// unconditional.
#[derive(Module, Debug)]
pub struct PsaBlock<B: Backend> {
    attn: Attention<B>,
    ffn: (Conv<B>, Conv<B>),
}

impl<B: Backend> PsaBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = input.clone() + self.attn.forward(input);
        x.clone() + self.ffn.1.forward(self.ffn.0.forward(x))
    }
}

pub(super) struct PsaBlockConfig {
    dim: usize,
    bn: BnFlavor,
}

impl PsaBlockConfig {
    pub(super) fn new(dim: usize) -> Self {
        Self {
            dim,
            bn: BnFlavor::default(),
        }
    }

    pub(super) fn with_bn_flavor(mut self, bn: BnFlavor) -> Self {
        self.bn = bn;
        self
    }

    pub(super) fn init<B: Backend>(&self, device: &Device<B>) -> PsaBlock<B> {
        PsaBlock {
            attn: AttentionConfig::new(self.dim, (self.dim / 64).max(1))
                .with_bn_flavor(self.bn)
                .init(device),
            ffn: (
                ConvConfig::new(self.dim, self.dim * 2, 1, 1)
                    .with_bn_flavor(self.bn)
                    .init(device),
                ConvConfig::new(self.dim * 2, self.dim, 1, 1)
                    .with_bn_flavor(self.bn)
                    .without_act()
                    .init(device),
            ),
        }
    }
}

/// Ultralytics `C2PSA`: split the projected input and refine one half with PSABlocks.
#[derive(Module, Debug)]
pub struct C2Psa<B: Backend> {
    cv1: Conv<B>,
    cv2: Conv<B>,
    m: Vec<PsaBlock<B>>,
}

impl<B: Backend> C2Psa<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let y = self.cv1.forward(input);
        let [batch, channels, height, width] = y.dims();
        let half = channels / 2;
        let a = y.clone().slice([0..batch, 0..half, 0..height, 0..width]);
        let mut b = y.slice([0..batch, half..channels, 0..height, 0..width]);
        for block in &self.m {
            b = block.forward(b);
        }
        self.cv2.forward(Tensor::cat(vec![a, b], 1))
    }
}

pub(super) struct C2PsaConfig {
    channels: usize,
    repeats: usize,
    bn: BnFlavor,
}

impl C2PsaConfig {
    pub(super) fn new(channels: usize, repeats: usize) -> Self {
        Self {
            channels,
            repeats,
            bn: BnFlavor::default(),
        }
    }

    pub(super) fn with_bn_flavor(mut self, bn: BnFlavor) -> Self {
        self.bn = bn;
        self
    }

    pub(super) fn init<B: Backend>(&self, device: &Device<B>) -> C2Psa<B> {
        let hidden = self.channels / 2;
        C2Psa {
            cv1: ConvConfig::new(self.channels, hidden * 2, 1, 1)
                .with_bn_flavor(self.bn)
                .init(device),
            cv2: ConvConfig::new(hidden * 2, self.channels, 1, 1)
                .with_bn_flavor(self.bn)
                .init(device),
            m: (0..self.repeats)
                .map(|_| {
                    PsaBlockConfig::new(hidden)
                        .with_bn_flavor(self.bn)
                        .init(device)
                })
                .collect(),
        }
    }
}

/// Ultralytics `SPPF` as built for YOLO26: an activation-free input projection, a configurable
/// pooling chain, and an optional residual add around the whole module.
#[derive(Module, Debug)]
pub struct Sppf<B: Backend> {
    cv1: Conv<B>,
    cv2: Conv<B>,
    pool: MaxPool2d,
    pools: usize,
    add: bool,
}

impl<B: Backend> Sppf<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.cv1.forward(input.clone());
        let mut pooled = vec![x.clone()];
        for _ in 0..self.pools {
            let next = self.pool.forward(pooled[pooled.len() - 1].clone());
            pooled.push(next);
        }
        let y = self.cv2.forward(Tensor::cat(pooled, 1));
        if self.add { y + input } else { y }
    }
}

pub(super) struct SppfConfig {
    channels: usize,
    pools: usize,
    add: bool,
}

impl SppfConfig {
    /// `add` mirrors Ultralytics' `shortcut and c1 == c2`; the YOLO26n SPPF has equal input and
    /// output channels, so the shortcut add is active.
    pub(super) fn new(channels: usize, pools: usize, add: bool) -> Self {
        Self {
            channels,
            pools,
            add,
        }
    }

    pub(super) fn init<B: Backend>(&self, device: &Device<B>) -> Sppf<B> {
        let hidden = self.channels / 2;
        Sppf {
            cv1: ConvConfig::new(self.channels, hidden, 1, 1)
                .without_act()
                .init(device),
            cv2: ConvConfig::new(hidden * (self.pools + 1), self.channels, 1, 1).init(device),
            pool: MaxPool2dConfig::new([5, 5])
                .with_strides([1, 1])
                .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
                .init(),
            pools: self.pools,
            add: self.add,
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

    #[cfg(feature = "training")]
    #[test]
    fn training_depthwise_stencil_matches_grouped_convolution_and_weight_gradient() {
        use burn::backend::Autodiff;

        type B = Autodiff<Flex>;
        let device = Default::default();
        let conv: Conv<B> = ConvConfig::new(4, 4, 3, 1).depthwise().init(&device);
        let values = (0..2 * 4 * 5 * 5)
            .map(|index| (index as f32 - 50.0) / 37.0)
            .collect::<Vec<_>>();
        let input = Tensor::from_data(burn::tensor::TensorData::new(values, [2, 4, 5, 5]), &device);

        let expected = conv.conv.forward(input.clone());
        let actual = crate::models::training_ops::depthwise_3x3_stride_1(
            input.clone(),
            conv.conv.weight.val(),
        );
        let max_delta = (expected - actual)
            .abs()
            .max()
            .into_data()
            .as_slice::<f32>()
            .unwrap()[0];
        assert!(max_delta < 2e-5, "forward delta {max_delta}");

        let expected_gradients = conv.conv.forward(input.clone()).sum().backward();
        let actual_gradients =
            crate::models::training_ops::depthwise_3x3_stride_1(input, conv.conv.weight.val())
                .sum()
                .backward();
        let expected_weight = conv.conv.weight.grad(&expected_gradients).unwrap();
        let actual_weight = conv.conv.weight.grad(&actual_gradients).unwrap();
        let max_gradient_delta = (expected_weight - actual_weight)
            .abs()
            .max()
            .into_data()
            .as_slice::<f32>()
            .unwrap()[0];
        assert!(
            max_gradient_delta < 2e-4,
            "weight-gradient delta {max_gradient_delta}"
        );
    }

    #[test]
    fn produces_declared_shapes_for_yolo26n_blocks() {
        let worker = std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(|| {
                let device = Default::default();
                let c3k2: C3k2<Flex> = C3k2Config::new(32, 64, 1, 0.25, true).init(&device);
                let out = c3k2.forward(Tensor::zeros([1, 32, 40, 40], &device));
                assert_eq!(out.dims(), [1, 64, 40, 40]);

                let c3k2_c3k: C3k2C3k<Flex> =
                    C3k2C3kConfig::new(128, 128, 1, true, 0.5).init(&device);
                let out = c3k2_c3k.forward(Tensor::zeros([1, 128, 20, 20], &device));
                assert_eq!(out.dims(), [1, 128, 20, 20]);

                let c3k2_attn: C3k2Attn<Flex> =
                    C3k2AttnConfig::new(384, 256, 1, true).init(&device);
                let out = c3k2_attn.forward(Tensor::zeros([1, 384, 10, 10], &device));
                assert_eq!(out.dims(), [1, 256, 10, 10]);

                let sppf: Sppf<Flex> = SppfConfig::new(256, 3, true).init(&device);
                let out = sppf.forward(Tensor::zeros([1, 256, 10, 10], &device));
                assert_eq!(out.dims(), [1, 256, 10, 10]);

                let c2_psa: C2Psa<Flex> = C2PsaConfig::new(256, 1).init(&device);
                let out = c2_psa.forward(Tensor::zeros([1, 256, 10, 10], &device));
                assert_eq!(out.dims(), [1, 256, 10, 10]);
            })
            .expect("shape-test worker should start");
        worker.join().expect("shape-test worker should not panic");
    }
}
