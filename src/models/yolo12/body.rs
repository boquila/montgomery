use burn::{
    module::Module,
    tensor::{Device, Tensor, backend::Backend},
};

use super::blocks::{
    A2C2fAttn, A2C2fAttnConfig, A2C2fC3k, A2C2fC3kConfig, C3k2, C3k2C3k, C3k2C3kConfig, C3k2Config,
    Conv, ConvConfig, upsample_nearest_2x,
};

/// Feature maps for the three YOLO12 detection scales.
///
/// The type is shared with the YOLO11 detection head, whose graph YOLO12 reuses byte for byte
/// (light DWConv classification towers, DFL decode).
pub type Yolo12Features<B> = crate::models::yolo11::body::Yolo11Features<B>;

/// Complete YOLO12 backbone and feature-pyramid body (layers 0-20), n and s scales.
///
/// Field names retain the source graph indices so official checkpoint remapping stays mechanical
/// and parity failures can be localized to a specific declared layer. The early backbone stages
/// (layers 2/4) keep the plain C3k2 bottleneck chain at 0.25 expansion, the P4/P5 backbone stages
/// (6/8) are area-attention A2C2f blocks, the neck stages (11/14/17) are the C3k-chain A2C2f
/// flavor, and the P5 stage (20) is a C3k2 with a C3k chain. The m/l/x scales force the C3k chain
/// onto layers 2/4 (`parse_model`'s m/l/x rule) and use [`Yolo12BodyLarge`] instead.
#[derive(Module, Debug)]
pub struct Yolo12BodySmall<B: Backend> {
    model_0: Conv<B>,
    model_1: Conv<B>,
    model_2: C3k2<B>,
    model_3: Conv<B>,
    model_4: C3k2<B>,
    model_5: Conv<B>,
    model_6: A2C2fAttn<B>,
    model_7: Conv<B>,
    model_8: A2C2fAttn<B>,
    model_11: A2C2fC3k<B>,
    model_14: A2C2fC3k<B>,
    model_15: Conv<B>,
    model_17: A2C2fC3k<B>,
    model_18: Conv<B>,
    model_20: C3k2C3k<B>,
}

impl<B: Backend> Yolo12BodySmall<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Yolo12Features<B> {
        let x = self.model_0.forward(input);
        let x = self.model_1.forward(x);
        let x = self.model_2.forward(x);
        let x = self.model_3.forward(x);
        let route_p3 = self.model_4.forward(x);

        let x = self.model_5.forward(route_p3.clone());
        let route_p4 = self.model_6.forward(x);

        let x = self.model_7.forward(route_p4.clone());
        let route_p5 = self.model_8.forward(x);

        let x = upsample_nearest_2x(route_p5.clone());
        let x = Tensor::cat(vec![x, route_p4], 1);
        let neck_p4 = self.model_11.forward(x);

        let x = upsample_nearest_2x(neck_p4.clone());
        let x = Tensor::cat(vec![x, route_p3], 1);
        let p3 = self.model_14.forward(x);

        let x = self.model_15.forward(p3.clone());
        let x = Tensor::cat(vec![x, neck_p4], 1);
        let p4 = self.model_17.forward(x);

        let x = self.model_18.forward(p4.clone());
        let x = Tensor::cat(vec![x, route_p5], 1);
        let p5 = self.model_20.forward(x);

        Yolo12Features { p3, p4, p5 }
    }
}

/// Shared construction table.
///
/// Widths follow the depth/width-scaled YAML: stage 2/3 run at `w2` channels, stage 4 (the
/// backbone P3 tap feeding the neck concat) at `w4`, stages 5/6 at `w5`, stages 7/8 at `w7`, and
/// the neck outputs at `p3_out`/`w5`/`w7`. `a2_mlp_ratio`/`a2_residual` apply to the backbone
/// area-attention stages only; `parse_model` extends their YAML args with `(True, 1.2)` for the
/// l/x scales and keeps the defaults `(False, 2.0)` everywhere else.
#[derive(Debug)]
pub struct Yolo12BodyConfig {
    w0: usize,
    w1: usize,
    w2: usize,
    w4: usize,
    w5: usize,
    w7: usize,
    p3_out: usize,
    early_repeats: usize,
    a2_repeats: [usize; 2],
    neck_repeats: [usize; 3],
    stage20_repeats: usize,
    area: [usize; 2],
    a2_mlp_ratio: f32,
    a2_residual: bool,
}

/// The declared layer set shared by both body graph flavors.
struct Yolo12BodyParts<B: Backend> {
    model_0: Conv<B>,
    model_1: Conv<B>,
    model_2: Option<C3k2<B>>,
    model_2_c3k: Option<C3k2C3k<B>>,
    model_3: Conv<B>,
    model_4: Option<C3k2<B>>,
    model_4_c3k: Option<C3k2C3k<B>>,
    model_5: Conv<B>,
    model_6: A2C2fAttn<B>,
    model_7: Conv<B>,
    model_8: A2C2fAttn<B>,
    model_11: A2C2fC3k<B>,
    model_14: A2C2fC3k<B>,
    model_15: Conv<B>,
    model_17: A2C2fC3k<B>,
    model_18: Conv<B>,
    model_20: C3k2C3k<B>,
}

impl Yolo12BodyConfig {
    fn init<B: Backend>(&self, device: &Device<B>, early_c3k: bool) -> Yolo12BodyParts<B> {
        let [r6, r8] = self.a2_repeats;
        let [r11, r14, r17] = self.neck_repeats;
        let [area4, area1] = self.area;
        Yolo12BodyParts {
            model_0: ConvConfig::new(3, self.w0, 3, 2).init(device),
            model_1: ConvConfig::new(self.w0, self.w1, 3, 2).init(device),
            model_2: (!early_c3k).then(|| {
                C3k2Config::new(self.w1, self.w2, self.early_repeats, 0.25, true).init(device)
            }),
            model_2_c3k: early_c3k.then(|| {
                C3k2C3kConfig::new(self.w1, self.w2, self.early_repeats, true, 0.25).init(device)
            }),
            model_3: ConvConfig::new(self.w2, self.w2, 3, 2).init(device),
            model_4: (!early_c3k).then(|| {
                C3k2Config::new(self.w2, self.w4, self.early_repeats, 0.25, true).init(device)
            }),
            model_4_c3k: early_c3k.then(|| {
                C3k2C3kConfig::new(self.w2, self.w4, self.early_repeats, true, 0.25).init(device)
            }),
            model_5: ConvConfig::new(self.w4, self.w5, 3, 2).init(device),
            model_6: A2C2fAttnConfig::new(
                self.w5,
                self.w5,
                r6,
                area4,
                self.a2_mlp_ratio,
                self.a2_residual,
            )
            .init(device),
            model_7: ConvConfig::new(self.w5, self.w7, 3, 2).init(device),
            model_8: A2C2fAttnConfig::new(
                self.w7,
                self.w7,
                r8,
                area1,
                self.a2_mlp_ratio,
                self.a2_residual,
            )
            .init(device),
            model_11: A2C2fC3kConfig::new(self.w7 + self.w5, self.w5, r11).init(device),
            model_14: A2C2fC3kConfig::new(self.w5 + self.w4, self.p3_out, r14).init(device),
            model_15: ConvConfig::new(self.p3_out, self.p3_out, 3, 2).init(device),
            model_17: A2C2fC3kConfig::new(self.p3_out + self.w5, self.w5, r17).init(device),
            model_18: ConvConfig::new(self.w5, self.w5, 3, 2).init(device),
            model_20: C3k2C3kConfig::new(
                self.w5 + self.w7,
                self.w7,
                self.stage20_repeats,
                true,
                0.5,
            )
            .init(device),
        }
    }
}

/// Configuration for the fixed YOLO12n body (depth 0.50, width 0.25, max channels 1024).
#[derive(Debug, Default)]
pub struct Yolo12BodyNConfig;

impl Yolo12BodyNConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo12BodySmall<B> {
        let parts = Yolo12BodyConfig {
            w0: 16,
            w1: 32,
            w2: 64,
            w4: 128,
            w5: 128,
            w7: 256,
            p3_out: 64,
            early_repeats: 1,
            a2_repeats: [2, 2],
            neck_repeats: [1, 1, 1],
            stage20_repeats: 1,
            area: [4, 1],
            a2_mlp_ratio: 2.0,
            a2_residual: false,
        }
        .init(device, false);
        assemble_small(parts)
    }
}

/// Configuration for the fixed YOLO12s body (depth 0.50, width 0.50, max channels 1024).
#[derive(Debug, Default)]
pub struct Yolo12BodySConfig;

impl Yolo12BodySConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo12BodySmall<B> {
        let parts = Yolo12BodyConfig {
            w0: 32,
            w1: 64,
            w2: 128,
            w4: 256,
            w5: 256,
            w7: 512,
            p3_out: 128,
            early_repeats: 1,
            a2_repeats: [2, 2],
            neck_repeats: [1, 1, 1],
            stage20_repeats: 1,
            area: [4, 1],
            a2_mlp_ratio: 2.0,
            a2_residual: false,
        }
        .init(device, false);
        assemble_small(parts)
    }
}

/// Configuration for the fixed YOLO12m body (depth 0.50, width 1.00, max channels 512).
///
/// The m scale forces the C3k chain onto the early backbone stages but keeps the n/s depth gain,
/// so its repeat counts match n/s while its graph matches l/x.
#[derive(Debug, Default)]
pub struct Yolo12BodyMConfig;

impl Yolo12BodyMConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo12BodyLarge<B> {
        let parts = Yolo12BodyConfig {
            w0: 64,
            w1: 128,
            w2: 256,
            w4: 512,
            w5: 512,
            w7: 512,
            p3_out: 256,
            early_repeats: 1,
            a2_repeats: [2, 2],
            neck_repeats: [1, 1, 1],
            stage20_repeats: 1,
            area: [4, 1],
            a2_mlp_ratio: 2.0,
            a2_residual: false,
        }
        .init(device, true);
        assemble_large(parts)
    }
}

/// Configuration for the fixed YOLO12l body (depth 1.00, width 1.00, max channels 512).
///
/// The l scale additionally extends the YAML args of the area-attention stages with
/// `residual=True, mlp_ratio=1.2`, adding the learnable gamma residual.
#[derive(Debug, Default)]
pub struct Yolo12BodyLConfig;

impl Yolo12BodyLConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo12BodyLarge<B> {
        let parts = Yolo12BodyConfig {
            w0: 64,
            w1: 128,
            w2: 256,
            w4: 512,
            w5: 512,
            w7: 512,
            p3_out: 256,
            early_repeats: 2,
            a2_repeats: [4, 4],
            neck_repeats: [2, 2, 2],
            stage20_repeats: 2,
            area: [4, 1],
            a2_mlp_ratio: 1.2,
            a2_residual: true,
        }
        .init(device, true);
        assemble_large(parts)
    }
}

/// Configuration for the fixed YOLO12x body (depth 1.00, width 1.50, max channels 512).
#[derive(Debug, Default)]
pub struct Yolo12BodyXConfig;

impl Yolo12BodyXConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo12BodyLarge<B> {
        let parts = Yolo12BodyConfig {
            w0: 96,
            w1: 192,
            w2: 384,
            w4: 768,
            w5: 768,
            w7: 768,
            p3_out: 384,
            early_repeats: 2,
            a2_repeats: [4, 4],
            neck_repeats: [2, 2, 2],
            stage20_repeats: 2,
            area: [4, 1],
            a2_mlp_ratio: 1.2,
            a2_residual: true,
        }
        .init(device, true);
        assemble_large(parts)
    }
}

fn assemble_small<B: Backend>(parts: Yolo12BodyParts<B>) -> Yolo12BodySmall<B> {
    Yolo12BodySmall {
        model_0: parts.model_0,
        model_1: parts.model_1,
        model_2: parts
            .model_2
            .expect("n/s bodies build the plain C3k2 chain"),
        model_3: parts.model_3,
        model_4: parts
            .model_4
            .expect("n/s bodies build the plain C3k2 chain"),
        model_5: parts.model_5,
        model_6: parts.model_6,
        model_7: parts.model_7,
        model_8: parts.model_8,
        model_11: parts.model_11,
        model_14: parts.model_14,
        model_15: parts.model_15,
        model_17: parts.model_17,
        model_18: parts.model_18,
        model_20: parts.model_20,
    }
}

fn assemble_large<B: Backend>(parts: Yolo12BodyParts<B>) -> Yolo12BodyLarge<B> {
    Yolo12BodyLarge {
        model_0: parts.model_0,
        model_1: parts.model_1,
        model_2: parts.model_2_c3k.expect("m/l/x bodies build the C3k chain"),
        model_3: parts.model_3,
        model_4: parts.model_4_c3k.expect("m/l/x bodies build the C3k chain"),
        model_5: parts.model_5,
        model_6: parts.model_6,
        model_7: parts.model_7,
        model_8: parts.model_8,
        model_11: parts.model_11,
        model_14: parts.model_14,
        model_15: parts.model_15,
        model_17: parts.model_17,
        model_18: parts.model_18,
        model_20: parts.model_20,
    }
}

/// Complete YOLO12 backbone and feature-pyramid body (layers 0-20), m/l/x scales.
///
/// `parse_model` forces `c3k=True` on every C3k2 stage for the m/l/x scales, so the early backbone
/// stages (layers 2 and 4) build C3k chains at the YAML's 0.25 expansion and this graph differs
/// structurally from the n/s body. The l/x scales carry the learnable gamma residual on the
/// area-attention stages.
#[derive(Module, Debug)]
pub struct Yolo12BodyLarge<B: Backend> {
    model_0: Conv<B>,
    model_1: Conv<B>,
    model_2: C3k2C3k<B>,
    model_3: Conv<B>,
    model_4: C3k2C3k<B>,
    model_5: Conv<B>,
    model_6: A2C2fAttn<B>,
    model_7: Conv<B>,
    model_8: A2C2fAttn<B>,
    model_11: A2C2fC3k<B>,
    model_14: A2C2fC3k<B>,
    model_15: Conv<B>,
    model_17: A2C2fC3k<B>,
    model_18: Conv<B>,
    model_20: C3k2C3k<B>,
}

impl<B: Backend> Yolo12BodyLarge<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Yolo12Features<B> {
        let x = self.model_0.forward(input);
        let x = self.model_1.forward(x);
        let x = self.model_2.forward(x);
        let x = self.model_3.forward(x);
        let route_p3 = self.model_4.forward(x);

        let x = self.model_5.forward(route_p3.clone());
        let route_p4 = self.model_6.forward(x);

        let x = self.model_7.forward(route_p4.clone());
        let route_p5 = self.model_8.forward(x);

        let x = upsample_nearest_2x(route_p5.clone());
        let x = Tensor::cat(vec![x, route_p4], 1);
        let neck_p4 = self.model_11.forward(x);

        let x = upsample_nearest_2x(neck_p4.clone());
        let x = Tensor::cat(vec![x, route_p3], 1);
        let p3 = self.model_14.forward(x);

        let x = self.model_15.forward(p3.clone());
        let x = Tensor::cat(vec![x, neck_p4], 1);
        let p4 = self.model_17.forward(x);

        let x = self.model_18.forward(p4.clone());
        let x = Tensor::cat(vec![x, route_p5], 1);
        let p5 = self.model_20.forward(x);

        Yolo12Features { p3, p4, p5 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_flex::Flex;

    #[test]
    fn produces_declared_p3_p4_p5_shapes() {
        let worker = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let device = Default::default();
                let input = Tensor::zeros([1, 3, 64, 64], &device);

                let body: Yolo12BodySmall<Flex> = Yolo12BodyNConfig.init(&device);
                let output = body.forward(input.clone());
                assert_eq!(output.p3.dims(), [1, 64, 8, 8]);
                assert_eq!(output.p4.dims(), [1, 128, 4, 4]);
                assert_eq!(output.p5.dims(), [1, 256, 2, 2]);

                let body: Yolo12BodySmall<Flex> = Yolo12BodySConfig.init(&device);
                let output = body.forward(input.clone());
                assert_eq!(output.p3.dims(), [1, 128, 8, 8]);
                assert_eq!(output.p4.dims(), [1, 256, 4, 4]);
                assert_eq!(output.p5.dims(), [1, 512, 2, 2]);

                let body: Yolo12BodyLarge<Flex> = Yolo12BodyMConfig.init(&device);
                let output = body.forward(input.clone());
                assert_eq!(output.p3.dims(), [1, 256, 8, 8]);
                assert_eq!(output.p4.dims(), [1, 512, 4, 4]);
                assert_eq!(output.p5.dims(), [1, 512, 2, 2]);

                let body: Yolo12BodyLarge<Flex> = Yolo12BodyLConfig.init(&device);
                let output = body.forward(input.clone());
                assert_eq!(output.p3.dims(), [1, 256, 8, 8]);
                assert_eq!(output.p4.dims(), [1, 512, 4, 4]);
                assert_eq!(output.p5.dims(), [1, 512, 2, 2]);

                let body: Yolo12BodyLarge<Flex> = Yolo12BodyXConfig.init(&device);
                let output = body.forward(input);
                assert_eq!(output.p3.dims(), [1, 384, 8, 8]);
                assert_eq!(output.p4.dims(), [1, 768, 4, 4]);
                assert_eq!(output.p5.dims(), [1, 768, 2, 2]);
            })
            .expect("shape-test worker should start");
        worker.join().expect("shape-test worker should not panic");
    }
}
