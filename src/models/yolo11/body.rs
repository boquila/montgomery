use burn::{
    module::Module,
    tensor::{Device, Tensor, backend::Backend},
};

use super::blocks::{
    C2Psa, C2PsaConfig, C3k2, C3k2C3k, C3k2C3kConfig, C3k2Config, Conv, ConvConfig, Sppf,
    SppfConfig, upsample_nearest_2x,
};

/// Feature maps for the three YOLO11 detection scales.
pub struct Yolo11Features<B: Backend> {
    /// P3/8 feature map.
    pub p3: Tensor<B, 4>,
    /// P4/16 feature map.
    pub p4: Tensor<B, 4>,
    /// P5/32 feature map.
    pub p5: Tensor<B, 4>,
}

/// Complete YOLO11 backbone and feature-pyramid body (layers 0-22), n and s scales.
///
/// Field names retain the source graph indices so official checkpoint remapping stays mechanical
/// and parity failures can be localized to a specific declared layer. At n/s width the early
/// backbone stages (layers 2 and 4) and the neck stages (layers 13/16/19) keep the plain C3k2
/// bottleneck chain; the m/l/x scales force `c3k=True` on every C3k2 stage and use
/// [`Yolo11BodyLarge`] instead. Two structural notes relative to the sibling families: layer 9's
/// SPPF keeps Ultralytics' plain form without a residual add (the YAML passes only the kernel
/// size), and the P5 stage at layer 22 is a plain C3k2 with a C3k chain — YOLO11 has no attention
/// P5 stage.
#[derive(Module, Debug)]
pub struct Yolo11BodySmall<B: Backend> {
    model_0: Conv<B>,
    model_1: Conv<B>,
    model_2: C3k2<B>,
    model_3: Conv<B>,
    model_4: C3k2<B>,
    model_5: Conv<B>,
    model_6: C3k2C3k<B>,
    model_7: Conv<B>,
    model_8: C3k2C3k<B>,
    model_9: Sppf<B>,
    model_10: C2Psa<B>,
    model_13: C3k2<B>,
    model_16: C3k2<B>,
    model_17: Conv<B>,
    model_19: C3k2<B>,
    model_20: Conv<B>,
    model_22: C3k2C3k<B>,
}

impl<B: Backend> Yolo11BodySmall<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Yolo11Features<B> {
        let x = self.model_0.forward(input);
        let x = self.model_1.forward(x);
        let x = self.model_2.forward(x);
        let x = self.model_3.forward(x);
        let route_p3 = self.model_4.forward(x);

        let x = self.model_5.forward(route_p3.clone());
        let route_p4 = self.model_6.forward(x);

        let x = self.model_7.forward(route_p4.clone());
        let x = self.model_8.forward(x);
        let x = self.model_9.forward(x);
        let route_p5 = self.model_10.forward(x);

        let x = upsample_nearest_2x(route_p5.clone());
        let x = Tensor::cat(vec![x, route_p4], 1);
        let neck_p4 = self.model_13.forward(x);

        let x = upsample_nearest_2x(neck_p4.clone());
        let x = Tensor::cat(vec![x, route_p3], 1);
        let p3 = self.model_16.forward(x);

        let x = self.model_17.forward(p3.clone());
        let x = Tensor::cat(vec![x, neck_p4], 1);
        let p4 = self.model_19.forward(x);

        let x = self.model_20.forward(p4.clone());
        let x = Tensor::cat(vec![x, route_p5], 1);
        let p5 = self.model_22.forward(x);

        Yolo11Features { p3, p4, p5 }
    }
}

/// Configuration for the fixed YOLO11n body (depth 0.50, width 0.25, max channels 1024).
///
/// The declared repeats of 2 on the C3k2 and C2PSA stages become one block per stage at the
/// n-scale depth gain of 0.50.
#[derive(Debug, Default)]
pub struct Yolo11BodyNConfig;

impl Yolo11BodyNConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo11BodySmall<B> {
        Yolo11BodySmall {
            model_0: ConvConfig::new(3, 16, 3, 2).init(device),
            model_1: ConvConfig::new(16, 32, 3, 2).init(device),
            model_2: C3k2Config::new(32, 64, 1, 0.25, true).init(device),
            model_3: ConvConfig::new(64, 64, 3, 2).init(device),
            model_4: C3k2Config::new(64, 128, 1, 0.25, true).init(device),
            model_5: ConvConfig::new(128, 128, 3, 2).init(device),
            model_6: C3k2C3kConfig::new(128, 128, 1, true, 0.5).init(device),
            model_7: ConvConfig::new(128, 256, 3, 2).init(device),
            model_8: C3k2C3kConfig::new(256, 256, 1, true, 0.5).init(device),
            model_9: SppfConfig::new(256, 3).init(device),
            model_10: C2PsaConfig::new(256, 1).init(device),
            model_13: C3k2Config::new(384, 128, 1, 0.5, true).init(device),
            model_16: C3k2Config::new(256, 64, 1, 0.5, true).init(device),
            model_17: ConvConfig::new(64, 64, 3, 2).init(device),
            model_19: C3k2Config::new(192, 128, 1, 0.5, true).init(device),
            model_20: ConvConfig::new(128, 128, 3, 2).init(device),
            model_22: C3k2C3kConfig::new(384, 256, 1, true, 0.5).init(device),
        }
    }
}

/// Configuration for the fixed YOLO11s body (depth 0.50, width 0.50, max channels 1024).
#[derive(Debug, Default)]
pub struct Yolo11BodySConfig;

impl Yolo11BodySConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo11BodySmall<B> {
        Yolo11BodySmall {
            model_0: ConvConfig::new(3, 32, 3, 2).init(device),
            model_1: ConvConfig::new(32, 64, 3, 2).init(device),
            model_2: C3k2Config::new(64, 128, 1, 0.25, true).init(device),
            model_3: ConvConfig::new(128, 128, 3, 2).init(device),
            model_4: C3k2Config::new(128, 256, 1, 0.25, true).init(device),
            model_5: ConvConfig::new(256, 256, 3, 2).init(device),
            model_6: C3k2C3kConfig::new(256, 256, 1, true, 0.5).init(device),
            model_7: ConvConfig::new(256, 512, 3, 2).init(device),
            model_8: C3k2C3kConfig::new(512, 512, 1, true, 0.5).init(device),
            model_9: SppfConfig::new(512, 3).init(device),
            model_10: C2PsaConfig::new(512, 1).init(device),
            model_13: C3k2Config::new(768, 256, 1, 0.5, true).init(device),
            model_16: C3k2Config::new(512, 128, 1, 0.5, true).init(device),
            model_17: ConvConfig::new(128, 128, 3, 2).init(device),
            model_19: C3k2Config::new(384, 256, 1, 0.5, true).init(device),
            model_20: ConvConfig::new(256, 256, 3, 2).init(device),
            model_22: C3k2C3kConfig::new(768, 512, 1, true, 0.5).init(device),
        }
    }
}

/// Complete YOLO11 backbone and feature-pyramid body (layers 0-22), m/l/x scales.
///
/// `parse_model` forces `c3k=True` on every C3k2 stage for the m/l/x scales, so the early backbone
/// stages (layers 2 and 4) build C3k chains at the YAML's 0.25 expansion and this graph differs
/// structurally from the n/s body.
#[derive(Module, Debug)]
pub struct Yolo11BodyLarge<B: Backend> {
    model_0: Conv<B>,
    model_1: Conv<B>,
    model_2: C3k2C3k<B>,
    model_3: Conv<B>,
    model_4: C3k2C3k<B>,
    model_5: Conv<B>,
    model_6: C3k2C3k<B>,
    model_7: Conv<B>,
    model_8: C3k2C3k<B>,
    model_9: Sppf<B>,
    model_10: C2Psa<B>,
    model_13: C3k2C3k<B>,
    model_16: C3k2C3k<B>,
    model_17: Conv<B>,
    model_19: C3k2C3k<B>,
    model_20: Conv<B>,
    model_22: C3k2C3k<B>,
}

impl<B: Backend> Yolo11BodyLarge<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Yolo11Features<B> {
        let x = self.model_0.forward(input);
        let x = self.model_1.forward(x);
        let x = self.model_2.forward(x);
        let x = self.model_3.forward(x);
        let route_p3 = self.model_4.forward(x);

        let x = self.model_5.forward(route_p3.clone());
        let route_p4 = self.model_6.forward(x);

        let x = self.model_7.forward(route_p4.clone());
        let x = self.model_8.forward(x);
        let x = self.model_9.forward(x);
        let route_p5 = self.model_10.forward(x);

        let x = upsample_nearest_2x(route_p5.clone());
        let x = Tensor::cat(vec![x, route_p4], 1);
        let neck_p4 = self.model_13.forward(x);

        let x = upsample_nearest_2x(neck_p4.clone());
        let x = Tensor::cat(vec![x, route_p3], 1);
        let p3 = self.model_16.forward(x);

        let x = self.model_17.forward(p3.clone());
        let x = Tensor::cat(vec![x, neck_p4], 1);
        let p4 = self.model_19.forward(x);

        let x = self.model_20.forward(p4.clone());
        let x = Tensor::cat(vec![x, route_p5], 1);
        let p5 = self.model_22.forward(x);

        Yolo11Features { p3, p4, p5 }
    }
}

/// Configuration for the fixed YOLO11m body (depth 0.50, width 1.00, max channels 512).
#[derive(Debug, Default)]
pub struct Yolo11BodyMConfig;

impl Yolo11BodyMConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo11BodyLarge<B> {
        Yolo11BodyLarge {
            model_0: ConvConfig::new(3, 64, 3, 2).init(device),
            model_1: ConvConfig::new(64, 128, 3, 2).init(device),
            model_2: C3k2C3kConfig::new(128, 256, 1, true, 0.25).init(device),
            model_3: ConvConfig::new(256, 256, 3, 2).init(device),
            model_4: C3k2C3kConfig::new(256, 512, 1, true, 0.25).init(device),
            model_5: ConvConfig::new(512, 512, 3, 2).init(device),
            model_6: C3k2C3kConfig::new(512, 512, 1, true, 0.5).init(device),
            model_7: ConvConfig::new(512, 512, 3, 2).init(device),
            model_8: C3k2C3kConfig::new(512, 512, 1, true, 0.5).init(device),
            model_9: SppfConfig::new(512, 3).init(device),
            model_10: C2PsaConfig::new(512, 1).init(device),
            model_13: C3k2C3kConfig::new(1024, 512, 1, true, 0.5).init(device),
            model_16: C3k2C3kConfig::new(1024, 256, 1, true, 0.5).init(device),
            model_17: ConvConfig::new(256, 256, 3, 2).init(device),
            model_19: C3k2C3kConfig::new(768, 512, 1, true, 0.5).init(device),
            model_20: ConvConfig::new(512, 512, 3, 2).init(device),
            model_22: C3k2C3kConfig::new(1024, 512, 1, true, 0.5).init(device),
        }
    }
}

/// Configuration for the fixed YOLO11l body (depth 1.00, width 1.00, max channels 512).
#[derive(Debug, Default)]
pub struct Yolo11BodyLConfig;

impl Yolo11BodyLConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo11BodyLarge<B> {
        Yolo11BodyLarge {
            model_0: ConvConfig::new(3, 64, 3, 2).init(device),
            model_1: ConvConfig::new(64, 128, 3, 2).init(device),
            model_2: C3k2C3kConfig::new(128, 256, 2, true, 0.25).init(device),
            model_3: ConvConfig::new(256, 256, 3, 2).init(device),
            model_4: C3k2C3kConfig::new(256, 512, 2, true, 0.25).init(device),
            model_5: ConvConfig::new(512, 512, 3, 2).init(device),
            model_6: C3k2C3kConfig::new(512, 512, 2, true, 0.5).init(device),
            model_7: ConvConfig::new(512, 512, 3, 2).init(device),
            model_8: C3k2C3kConfig::new(512, 512, 2, true, 0.5).init(device),
            model_9: SppfConfig::new(512, 3).init(device),
            model_10: C2PsaConfig::new(512, 2).init(device),
            model_13: C3k2C3kConfig::new(1024, 512, 2, true, 0.5).init(device),
            model_16: C3k2C3kConfig::new(1024, 256, 2, true, 0.5).init(device),
            model_17: ConvConfig::new(256, 256, 3, 2).init(device),
            model_19: C3k2C3kConfig::new(768, 512, 2, true, 0.5).init(device),
            model_20: ConvConfig::new(512, 512, 3, 2).init(device),
            model_22: C3k2C3kConfig::new(1024, 512, 2, true, 0.5).init(device),
        }
    }
}

/// Configuration for the fixed YOLO11x body (depth 1.00, width 1.50, max channels 512).
#[derive(Debug, Default)]
pub struct Yolo11BodyXConfig;

impl Yolo11BodyXConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo11BodyLarge<B> {
        Yolo11BodyLarge {
            model_0: ConvConfig::new(3, 96, 3, 2).init(device),
            model_1: ConvConfig::new(96, 192, 3, 2).init(device),
            model_2: C3k2C3kConfig::new(192, 384, 2, true, 0.25).init(device),
            model_3: ConvConfig::new(384, 384, 3, 2).init(device),
            model_4: C3k2C3kConfig::new(384, 768, 2, true, 0.25).init(device),
            model_5: ConvConfig::new(768, 768, 3, 2).init(device),
            model_6: C3k2C3kConfig::new(768, 768, 2, true, 0.5).init(device),
            model_7: ConvConfig::new(768, 768, 3, 2).init(device),
            model_8: C3k2C3kConfig::new(768, 768, 2, true, 0.5).init(device),
            model_9: SppfConfig::new(768, 3).init(device),
            model_10: C2PsaConfig::new(768, 2).init(device),
            model_13: C3k2C3kConfig::new(1536, 768, 2, true, 0.5).init(device),
            model_16: C3k2C3kConfig::new(1536, 384, 2, true, 0.5).init(device),
            model_17: ConvConfig::new(384, 384, 3, 2).init(device),
            model_19: C3k2C3kConfig::new(1152, 768, 2, true, 0.5).init(device),
            model_20: ConvConfig::new(768, 768, 3, 2).init(device),
            model_22: C3k2C3kConfig::new(1536, 768, 2, true, 0.5).init(device),
        }
    }
}
