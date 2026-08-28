use burn::{
    module::{Module, Param},
    nn::{
        BatchNorm, BatchNormConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
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
}

impl<B: Backend> Conv<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.bn.forward(self.conv.forward(input));
        if self.act { silu(x) } else { x }
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
    bias: bool,
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
            bias: false,
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

    /// The YOLOv8-era `AAttn` positional-encoding convolution ships a conv bias in the official
    /// checkpoints even though Ultralytics' `Conv` wrapper is bias-free today; this option keeps
    /// the checkpoint's inference graph.
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
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
        .with_bias(self.bias)
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
        }
    }
}

/// Ultralytics `Bottleneck` with two 3x3 convolutions and an explicit hidden width.
///
/// The C3k inner chain builds its bottlenecks at full width (`e=1.0`) while the plain C3k2 chain
/// keeps the default half width, so the hidden width is declared per instance.
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
}

impl BottleneckConfig {
    /// Declare a bottleneck whose input and output channel counts are equal.
    pub fn new(channels: usize, hidden: usize, shortcut: bool) -> Self {
        Self {
            channels,
            hidden,
            add: shortcut,
        }
    }

    pub fn init<B: Backend>(&self, device: &Device<B>) -> Bottleneck<B> {
        Bottleneck {
            cv1: ConvConfig::new(self.channels, self.hidden, 3, 1).init(device),
            cv2: ConvConfig::new(self.hidden, self.channels, 3, 1).init(device),
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

pub struct C3kConfig {
    channels: usize,
    repeats: usize,
    shortcut: bool,
}

impl C3kConfig {
    /// Ultralytics instantiates C3k with its default expansion of 0.5 and default kernel of 3,
    /// so those values are fixed here.
    pub fn new(channels: usize, repeats: usize, shortcut: bool) -> Self {
        Self {
            channels,
            repeats,
            shortcut,
        }
    }

    pub fn init<B: Backend>(&self, device: &Device<B>) -> C3k<B> {
        let hidden = self.channels / 2;
        C3k {
            cv1: ConvConfig::new(self.channels, hidden, 1, 1).init(device),
            cv2: ConvConfig::new(self.channels, hidden, 1, 1).init(device),
            cv3: ConvConfig::new(hidden * 2, self.channels, 1, 1).init(device),
            m: (0..self.repeats)
                .map(|_| BottleneckConfig::new(hidden, hidden, self.shortcut).init(device))
                .collect(),
        }
    }
}

/// Ultralytics `C3k2` with the plain bottleneck chain.
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
pub struct C3k2Shell {
    in_channels: usize,
    out_channels: usize,
    hidden: usize,
    repeats: usize,
}

impl C3k2Shell {
    /// `expansion` is C3k2's `e` argument; the early backbone stages pass 0.25 and every other
    /// stage uses the default 0.5.
    pub fn new(in_channels: usize, out_channels: usize, repeats: usize, expansion: f32) -> Self {
        Self {
            in_channels,
            out_channels,
            hidden: (out_channels as f32 * expansion) as usize,
            repeats,
        }
    }

    fn init_conv<B: Backend>(&self, device: &Device<B>) -> (Conv<B>, Conv<B>) {
        (
            ConvConfig::new(self.in_channels, 2 * self.hidden, 1, 1).init(device),
            ConvConfig::new((2 + self.repeats) * self.hidden, self.out_channels, 1, 1).init(device),
        )
    }
}

pub struct C3k2Config {
    shell: C3k2Shell,
    shortcut: bool,
}

impl C3k2Config {
    pub fn new(
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

    pub fn init<B: Backend>(&self, device: &Device<B>) -> C3k2<B> {
        let (cv1, cv2) = self.shell.init_conv(device);
        // Ultralytics' plain C3k2 bottleneck keeps the default half-width expansion.
        let hidden = self.shell.hidden;
        let m = (0..self.shell.repeats)
            .map(|_| BottleneckConfig::new(hidden, hidden / 2, self.shortcut).init(device))
            .collect();
        C3k2 { cv1, cv2, m }
    }
}

/// Ultralytics `C3k2` with a C3k chain.
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

pub struct C3k2C3kConfig {
    shell: C3k2Shell,
    shortcut: bool,
}

impl C3k2C3kConfig {
    /// `expansion` sizes the shell's hidden width; Ultralytics builds
    /// `C3k(self.c, self.c, 2, shortcut)` for every C3k chain block.
    pub fn new(
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

    pub fn init<B: Backend>(&self, device: &Device<B>) -> C3k2C3k<B> {
        let (cv1, cv2) = self.shell.init_conv(device);
        let m = (0..self.shell.repeats)
            .map(|_| C3kConfig::new(self.shell.hidden, 2, self.shortcut).init(device))
            .collect();
        C3k2C3k { cv1, cv2, m }
    }
}

/// Ultralytics `AAttn`: area-attention with a depth-wise positional encoding.
///
/// The feature map is split into `area` horizontal strips; attention runs inside each strip and
/// the outputs are stitched back together. The official YOLOv8-era checkpoints keep a conv bias
/// on the 7x7 positional-encoding convolution even though the current source constructs it
/// bias-free â€” the checkpoint's inference graph wins.
#[derive(Module, Debug)]
pub struct AAttn<B: Backend> {
    qkv: Conv<B>,
    proj: Conv<B>,
    pe: Conv<B>,
    area: usize,
    num_heads: usize,
    head_dim: usize,
    all_head_dim: usize,
    scale: f32,
}

impl<B: Backend> AAttn<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, _, height, width] = input.dims();
        let tokens = height * width;
        let hd = self.head_dim;
        let all = self.all_head_dim;

        let mut qkv = self
            .qkv
            .forward(input)
            .reshape([batch, all * 3, tokens])
            .swap_dims(1, 2);
        let mut batch_eff = batch;
        let mut tokens_eff = tokens;
        if self.area > 1 {
            qkv = qkv.reshape([batch * self.area, tokens / self.area, all * 3]);
            batch_eff = batch * self.area;
            tokens_eff = tokens / self.area;
        }
        let qkv = qkv.reshape([batch_eff, tokens_eff, self.num_heads, hd * 3]);
        let qkv = qkv.permute([0, 2, 3, 1]);
        let q = qkv
            .clone()
            .slice([0..batch_eff, 0..self.num_heads, 0..hd, 0..tokens_eff]);
        let k = qkv
            .clone()
            .slice([0..batch_eff, 0..self.num_heads, hd..hd * 2, 0..tokens_eff]);
        let v = qkv.slice([
            0..batch_eff,
            0..self.num_heads,
            hd * 2..hd * 3,
            0..tokens_eff,
        ]);

        let attn = (q * self.scale).swap_dims(2, 3).matmul(k);
        let attn = softmax(attn, 3);
        let mut x = v.clone().matmul(attn.swap_dims(2, 3)).permute([0, 3, 1, 2]);
        let mut v = v.permute([0, 3, 1, 2]);

        if self.area > 1 {
            // The strip-parallel layout merges back into the full token sequence; both tensors
            // keep `[batch, tokens, heads, head_dim]` element order at this point.
            x = x.reshape([batch, tokens, self.num_heads, hd]);
            v = v.reshape([batch, tokens, self.num_heads, hd]);
        }
        let x = x.reshape([batch, height, width, all]).permute([0, 3, 1, 2]);
        let v = v.reshape([batch, height, width, all]).permute([0, 3, 1, 2]);

        let x = x + self.pe.forward(v);
        self.proj.forward(x)
    }
}

pub struct AAttnConfig {
    dim: usize,
    area: usize,
}

impl AAttnConfig {
    pub fn new(dim: usize, area: usize) -> Self {
        Self { dim, area }
    }

    pub fn init<B: Backend>(&self, device: &Device<B>) -> AAttn<B> {
        let num_heads = self.dim / 32;
        let head_dim = self.dim / num_heads;
        AAttn {
            qkv: ConvConfig::new(self.dim, self.dim * 3, 1, 1)
                .without_act()
                .init(device),
            proj: ConvConfig::new(self.dim, self.dim, 1, 1)
                .without_act()
                .init(device),
            pe: ConvConfig::new(self.dim, self.dim, 7, 1)
                .depthwise()
                .without_act()
                .with_bias(true)
                .init(device),
            area: self.area,
            num_heads,
            head_dim,
            all_head_dim: self.dim,
            scale: (head_dim as f32).powf(-0.5),
        }
    }
}

/// Ultralytics `ABlock`: area attention plus a feed-forward refinement, each with a residual add.
#[derive(Module, Debug)]
pub struct ABlock<B: Backend> {
    attn: AAttn<B>,
    mlp: (Conv<B>, Conv<B>),
}

impl<B: Backend> ABlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = input.clone() + self.attn.forward(input);
        x.clone() + self.mlp.1.forward(self.mlp.0.forward(x))
    }
}

pub struct ABlockConfig {
    dim: usize,
    mlp_ratio: f32,
    area: usize,
}

impl ABlockConfig {
    pub fn new(dim: usize, mlp_ratio: f32, area: usize) -> Self {
        Self {
            dim,
            mlp_ratio,
            area,
        }
    }

    pub fn init<B: Backend>(&self, device: &Device<B>) -> ABlock<B> {
        let hidden = (self.dim as f32 * self.mlp_ratio) as usize;
        ABlock {
            attn: AAttnConfig::new(self.dim, self.area).init(device),
            mlp: (
                ConvConfig::new(self.dim, hidden, 1, 1).init(device),
                ConvConfig::new(hidden, self.dim, 1, 1)
                    .without_act()
                    .init(device),
            ),
        }
    }
}

/// Ultralytics `A2C2f` with the area-attention chain (YAML `a2=True`): the C2f-style split shell
/// whose chain blocks are pairs of ABlocks, plus the optional learnable gamma residual that the
/// l/x scales carry (`parse_model` extends the YAML args with `residual=True` there).
///
/// Field names follow the checkpoint (`cv1`, `cv2`, `gamma`, `m.<item>.<block>...`).
#[derive(Module, Debug)]
pub struct A2C2fAttn<B: Backend> {
    cv1: Conv<B>,
    cv2: Conv<B>,
    gamma: Option<Param<Tensor<B, 1>>>,
    m: Vec<(ABlock<B>, ABlock<B>)>,
}

impl<B: Backend> A2C2fAttn<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let y0 = self.cv1.forward(input.clone());
        let mut outputs = vec![y0.clone()];
        let mut hidden = y0;
        for (first, second) in &self.m {
            hidden = second.forward(first.forward(hidden));
            outputs.push(hidden.clone());
        }
        let y = self.cv2.forward(Tensor::cat(outputs, 1));
        match &self.gamma {
            Some(gamma) => {
                let channels = gamma.val().dims()[0];
                input + gamma.val().reshape([1, channels, 1, 1]) * y
            }
            None => y,
        }
    }
}

pub struct A2C2fAttnConfig {
    in_channels: usize,
    out_channels: usize,
    repeats: usize,
    area: usize,
    mlp_ratio: f32,
    residual: bool,
}

impl A2C2fAttnConfig {
    /// Declare an area-attention A2C2f stage. The YAML always keeps Ultralytics' default
    /// expansion `e=0.5`, so the hidden width is half the output width (a multiple of 32, which
    /// ABlock's head-count rule `dim // 32` requires).
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        repeats: usize,
        area: usize,
        mlp_ratio: f32,
        residual: bool,
    ) -> Self {
        Self {
            in_channels,
            out_channels,
            repeats,
            area,
            mlp_ratio,
            residual,
        }
    }

    pub fn init<B: Backend>(&self, device: &Device<B>) -> A2C2fAttn<B> {
        let hidden = self.out_channels / 2;
        A2C2fAttn {
            cv1: ConvConfig::new(self.in_channels, hidden, 1, 1).init(device),
            cv2: ConvConfig::new((1 + self.repeats) * hidden, self.out_channels, 1, 1).init(device),
            gamma: self
                .residual
                .then(|| Param::from_tensor(Tensor::ones([self.out_channels], device) * 0.01)),
            m: (0..self.repeats)
                .map(|_| {
                    (
                        ABlockConfig::new(hidden, self.mlp_ratio, self.area).init(device),
                        ABlockConfig::new(hidden, self.mlp_ratio, self.area).init(device),
                    )
                })
                .collect(),
        }
    }
}

/// Ultralytics `A2C2f` with the C3k chain (YAML `a2=False`, the neck stages): the same C2f-style
/// split shell whose chain blocks are C3k modules with the default shortcut.
///
/// Field names follow the checkpoint (`cv1`, `cv2`, `m.<item>...`).
#[derive(Module, Debug)]
pub struct A2C2fC3k<B: Backend> {
    cv1: Conv<B>,
    cv2: Conv<B>,
    m: Vec<C3k<B>>,
}

impl<B: Backend> A2C2fC3k<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let y0 = self.cv1.forward(input);
        let mut outputs = vec![y0.clone()];
        let mut hidden = y0;
        for c3k in &self.m {
            hidden = c3k.forward(hidden);
            outputs.push(hidden.clone());
        }
        self.cv2.forward(Tensor::cat(outputs, 1))
    }
}

pub struct A2C2fC3kConfig {
    in_channels: usize,
    out_channels: usize,
    repeats: usize,
}

impl A2C2fC3kConfig {
    /// Declare a C3k-chain A2C2f stage with Ultralytics' defaults (`a2=False`, expansion 0.5,
    /// shortcut on, two bottleneck repeats per C3k).
    pub fn new(in_channels: usize, out_channels: usize, repeats: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            repeats,
        }
    }

    pub fn init<B: Backend>(&self, device: &Device<B>) -> A2C2fC3k<B> {
        let hidden = self.out_channels / 2;
        A2C2fC3k {
            cv1: ConvConfig::new(self.in_channels, hidden, 1, 1).init(device),
            cv2: ConvConfig::new((1 + self.repeats) * hidden, self.out_channels, 1, 1).init(device),
            m: (0..self.repeats)
                .map(|_| C3kConfig::new(hidden, 2, true).init(device))
                .collect(),
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
    fn produces_declared_shapes_for_yolo12n_blocks() {
        let worker = std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(|| {
                let device = Default::default();
                let c3k2: C3k2<Flex> = C3k2Config::new(32, 64, 1, 0.25, true).init(&device);
                let out = c3k2.forward(Tensor::zeros([1, 32, 40, 40], &device));
                assert_eq!(out.dims(), [1, 64, 40, 40]);

                // yolo12n layer 6: A2C2f(128, 128, n=2, area=4), hidden 64.
                let a2: A2C2fAttn<Flex> =
                    A2C2fAttnConfig::new(128, 128, 2, 4, 2.0, false).init(&device);
                let out = a2.forward(Tensor::zeros([1, 128, 10, 10], &device));
                assert_eq!(out.dims(), [1, 128, 10, 10]);

                // yolo12n layer 11: A2C2f C3k path (384, 128, n=1), hidden 64.
                let a2_c3k: A2C2fC3k<Flex> = A2C2fC3kConfig::new(384, 128, 1).init(&device);
                let out = a2_c3k.forward(Tensor::zeros([1, 384, 10, 10], &device));
                assert_eq!(out.dims(), [1, 128, 10, 10]);

                let c3k2_c3k: C3k2C3k<Flex> =
                    C3k2C3kConfig::new(384, 256, 1, true, 0.5).init(&device);
                let out = c3k2_c3k.forward(Tensor::zeros([1, 384, 10, 10], &device));
                assert_eq!(out.dims(), [1, 256, 10, 10]);
            })
            .expect("shape-test worker should start");
        worker.join().expect("shape-test worker should not panic");
    }
}
