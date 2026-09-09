use burn::{
    module::Module,
    tensor::{Device, Tensor},
};

use super::blocks::{
    C2f, C2fCib, C2fCibConfig, C2fCibDw, C2fCibDwConfig, C2fConfig, Conv, ConvConfig, Psa,
    PsaConfig, ScDown, ScDownConfig, Sppf, SppfConfig, upsample_nearest_2x,
};

/// Feature maps for the three YOLOv10 detection scales.
pub struct Yolov10Features {
    /// P3/8 feature map.
    pub p3: Tensor<4>,
    /// P4/16 feature map.
    pub p4: Tensor<4>,
    /// P5/32 feature map.
    pub p5: Tensor<4>,
}

/// Complete YOLOv10 backbone and feature-pyramid body (layers 0-22).
///
/// The scale variants differ in per-layer channel widths, depth-scaled repeats, and which stages
/// use the C2fCIB flavor, so each variant declares its own body struct. Field names retain the
/// source graph indices so official checkpoint remapping stays mechanical and parity failures can
/// be localized to a specific declared layer. Layer types per variant follow the official
/// per-scale YAMLs (which are not mere scale-row swaps): YOLOv10n keeps a plain C2f at layer 8,
/// s uses large-kernel C2fCIB towers, and m/b/l/x use the plain depth-wise C2fCIB flavor.
#[derive(Module, Debug)]
pub struct Yolov10BodyN {
    model_0: Conv,
    model_1: Conv,
    model_2: C2f,
    model_3: Conv,
    model_4: C2f,
    model_5: ScDown,
    model_6: C2f,
    model_7: ScDown,
    model_8: C2f,
    model_9: Sppf,
    model_10: Psa,
    model_13: C2f,
    model_16: C2f,
    model_17: Conv,
    model_19: C2f,
    model_20: ScDown,
    model_22: C2fCib,
}

impl Yolov10BodyN {
    pub fn forward(&self, input: Tensor<4>) -> Yolov10Features {
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

        Yolov10Features { p3, p4, p5 }
    }
}

/// Configuration for the fixed YOLOv10n body (depth 0.33, width 0.25, max channels 1024).
#[derive(Debug, Default)]
pub struct Yolov10BodyNConfig;

impl Yolov10BodyNConfig {
    pub fn init(&self, device: &Device) -> Yolov10BodyN {
        Yolov10BodyN {
            model_0: ConvConfig::new(3, 16, 3, 2).init(device),
            model_1: ConvConfig::new(16, 32, 3, 2).init(device),
            model_2: C2fConfig::new(32, 32, 1, true).init(device),
            model_3: ConvConfig::new(32, 64, 3, 2).init(device),
            model_4: C2fConfig::new(64, 64, 2, true).init(device),
            model_5: ScDownConfig::new(64, 128, 3, 2).init(device),
            model_6: C2fConfig::new(128, 128, 2, true).init(device),
            model_7: ScDownConfig::new(128, 256, 3, 2).init(device),
            model_8: C2fConfig::new(256, 256, 1, true).init(device),
            model_9: SppfConfig::new(256).init(device),
            model_10: PsaConfig::new(256).init(device),
            model_13: C2fConfig::new(384, 128, 1, false).init(device),
            model_16: C2fConfig::new(192, 64, 1, false).init(device),
            model_17: ConvConfig::new(64, 64, 3, 2).init(device),
            model_19: C2fConfig::new(192, 128, 1, false).init(device),
            model_20: ScDownConfig::new(128, 128, 3, 2).init(device),
            model_22: C2fCibConfig::new(384, 256, 1, true).init(device),
        }
    }
}

/// YOLOv10s body (depth 0.33, width 0.50, max channels 1024). Layers 8 and 22 are large-kernel
/// C2fCIB towers (`lk=True`), unlike n's plain C2f at layer 8.
#[derive(Module, Debug)]
pub struct Yolov10BodyS {
    model_0: Conv,
    model_1: Conv,
    model_2: C2f,
    model_3: Conv,
    model_4: C2f,
    model_5: ScDown,
    model_6: C2f,
    model_7: ScDown,
    model_8: C2fCib,
    model_9: Sppf,
    model_10: Psa,
    model_13: C2f,
    model_16: C2f,
    model_17: Conv,
    model_19: C2f,
    model_20: ScDown,
    model_22: C2fCib,
}

impl Yolov10BodyS {
    pub fn forward(&self, input: Tensor<4>) -> Yolov10Features {
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

        Yolov10Features { p3, p4, p5 }
    }
}

/// Configuration for the fixed YOLOv10s body (depth 0.33, width 0.50, max channels 1024).
#[derive(Debug, Default)]
pub struct Yolov10BodySConfig;

impl Yolov10BodySConfig {
    pub fn init(&self, device: &Device) -> Yolov10BodyS {
        Yolov10BodyS {
            model_0: ConvConfig::new(3, 32, 3, 2).init(device),
            model_1: ConvConfig::new(32, 64, 3, 2).init(device),
            model_2: C2fConfig::new(64, 64, 1, true).init(device),
            model_3: ConvConfig::new(64, 128, 3, 2).init(device),
            model_4: C2fConfig::new(128, 128, 2, true).init(device),
            model_5: ScDownConfig::new(128, 256, 3, 2).init(device),
            model_6: C2fConfig::new(256, 256, 2, true).init(device),
            model_7: ScDownConfig::new(256, 512, 3, 2).init(device),
            model_8: C2fCibConfig::new(512, 512, 1, true).init(device),
            model_9: SppfConfig::new(512).init(device),
            model_10: PsaConfig::new(512).init(device),
            model_13: C2fConfig::new(768, 256, 1, false).init(device),
            model_16: C2fConfig::new(384, 128, 1, false).init(device),
            model_17: ConvConfig::new(128, 128, 3, 2).init(device),
            model_19: C2fConfig::new(384, 256, 1, false).init(device),
            model_20: ScDownConfig::new(256, 256, 3, 2).init(device),
            model_22: C2fCibConfig::new(768, 512, 1, true).init(device),
        }
    }
}

/// YOLOv10m body (depth 0.67, width 0.75, max channels 768). Every C2fCIB stage uses the plain
/// depth-wise flavor (`lk=False`), including neck layer 19.
#[derive(Module, Debug)]
pub struct Yolov10BodyM {
    model_0: Conv,
    model_1: Conv,
    model_2: C2f,
    model_3: Conv,
    model_4: C2f,
    model_5: ScDown,
    model_6: C2f,
    model_7: ScDown,
    model_8: C2fCibDw,
    model_9: Sppf,
    model_10: Psa,
    model_13: C2f,
    model_16: C2f,
    model_17: Conv,
    model_19: C2fCibDw,
    model_20: ScDown,
    model_22: C2fCibDw,
}

impl Yolov10BodyM {
    pub fn forward(&self, input: Tensor<4>) -> Yolov10Features {
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

        Yolov10Features { p3, p4, p5 }
    }
}

/// Configuration for the fixed YOLOv10m body (depth 0.67, width 0.75, max channels 768).
#[derive(Debug, Default)]
pub struct Yolov10BodyMConfig;

impl Yolov10BodyMConfig {
    pub fn init(&self, device: &Device) -> Yolov10BodyM {
        Yolov10BodyM {
            model_0: ConvConfig::new(3, 48, 3, 2).init(device),
            model_1: ConvConfig::new(48, 96, 3, 2).init(device),
            model_2: C2fConfig::new(96, 96, 2, true).init(device),
            model_3: ConvConfig::new(96, 192, 3, 2).init(device),
            model_4: C2fConfig::new(192, 192, 4, true).init(device),
            model_5: ScDownConfig::new(192, 384, 3, 2).init(device),
            model_6: C2fConfig::new(384, 384, 4, true).init(device),
            model_7: ScDownConfig::new(384, 576, 3, 2).init(device),
            model_8: C2fCibDwConfig::new(576, 576, 2, true).init(device),
            model_9: SppfConfig::new(576).init(device),
            model_10: PsaConfig::new(576).init(device),
            model_13: C2fConfig::new(960, 384, 2, false).init(device),
            model_16: C2fConfig::new(576, 192, 2, false).init(device),
            model_17: ConvConfig::new(192, 192, 3, 2).init(device),
            model_19: C2fCibDwConfig::new(576, 384, 2, true).init(device),
            model_20: ScDownConfig::new(384, 384, 3, 2).init(device),
            model_22: C2fCibDwConfig::new(960, 576, 2, true).init(device),
        }
    }
}

/// YOLOv10b and YOLOv10l body (width 1.00, max channels 512). The two scales share the same
/// module types and channel table; only the depth-scaled repeats differ (b: 2/4, l: 3/6).
#[derive(Module, Debug)]
pub struct Yolov10BodyB {
    model_0: Conv,
    model_1: Conv,
    model_2: C2f,
    model_3: Conv,
    model_4: C2f,
    model_5: ScDown,
    model_6: C2f,
    model_7: ScDown,
    model_8: C2fCibDw,
    model_9: Sppf,
    model_10: Psa,
    model_13: C2fCibDw,
    model_16: C2f,
    model_17: Conv,
    model_19: C2fCibDw,
    model_20: ScDown,
    model_22: C2fCibDw,
}

impl Yolov10BodyB {
    pub fn forward(&self, input: Tensor<4>) -> Yolov10Features {
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

        Yolov10Features { p3, p4, p5 }
    }
}

/// Configuration for the fixed YOLOv10b body (depth 0.67, width 1.00, max channels 512).
#[derive(Debug, Default)]
pub struct Yolov10BodyBConfig;

impl Yolov10BodyBConfig {
    pub fn init(&self, device: &Device) -> Yolov10BodyB {
        Yolov10BodyB {
            model_0: ConvConfig::new(3, 64, 3, 2).init(device),
            model_1: ConvConfig::new(64, 128, 3, 2).init(device),
            model_2: C2fConfig::new(128, 128, 2, true).init(device),
            model_3: ConvConfig::new(128, 256, 3, 2).init(device),
            model_4: C2fConfig::new(256, 256, 4, true).init(device),
            model_5: ScDownConfig::new(256, 512, 3, 2).init(device),
            model_6: C2fConfig::new(512, 512, 4, true).init(device),
            model_7: ScDownConfig::new(512, 512, 3, 2).init(device),
            model_8: C2fCibDwConfig::new(512, 512, 2, true).init(device),
            model_9: SppfConfig::new(512).init(device),
            model_10: PsaConfig::new(512).init(device),
            model_13: C2fCibDwConfig::new(1024, 512, 2, true).init(device),
            model_16: C2fConfig::new(768, 256, 2, false).init(device),
            model_17: ConvConfig::new(256, 256, 3, 2).init(device),
            model_19: C2fCibDwConfig::new(768, 512, 2, true).init(device),
            model_20: ScDownConfig::new(512, 512, 3, 2).init(device),
            model_22: C2fCibDwConfig::new(1024, 512, 2, true).init(device),
        }
    }
}

/// Configuration for the fixed YOLOv10l body (depth 1.00, width 1.00, max channels 512).
#[derive(Debug, Default)]
pub struct Yolov10BodyLConfig;

impl Yolov10BodyLConfig {
    pub fn init(&self, device: &Device) -> Yolov10BodyB {
        Yolov10BodyB {
            model_0: ConvConfig::new(3, 64, 3, 2).init(device),
            model_1: ConvConfig::new(64, 128, 3, 2).init(device),
            model_2: C2fConfig::new(128, 128, 3, true).init(device),
            model_3: ConvConfig::new(128, 256, 3, 2).init(device),
            model_4: C2fConfig::new(256, 256, 6, true).init(device),
            model_5: ScDownConfig::new(256, 512, 3, 2).init(device),
            model_6: C2fConfig::new(512, 512, 6, true).init(device),
            model_7: ScDownConfig::new(512, 512, 3, 2).init(device),
            model_8: C2fCibDwConfig::new(512, 512, 3, true).init(device),
            model_9: SppfConfig::new(512).init(device),
            model_10: PsaConfig::new(512).init(device),
            model_13: C2fCibDwConfig::new(1024, 512, 3, true).init(device),
            model_16: C2fConfig::new(768, 256, 3, false).init(device),
            model_17: ConvConfig::new(256, 256, 3, 2).init(device),
            model_19: C2fCibDwConfig::new(768, 512, 3, true).init(device),
            model_20: ScDownConfig::new(512, 512, 3, 2).init(device),
            model_22: C2fCibDwConfig::new(1024, 512, 3, true).init(device),
        }
    }
}

/// YOLOv10x body (depth 1.00, width 1.25, max channels 512). Layer 6 also becomes a C2fCIB stage.
#[derive(Module, Debug)]
pub struct Yolov10BodyX {
    model_0: Conv,
    model_1: Conv,
    model_2: C2f,
    model_3: Conv,
    model_4: C2f,
    model_5: ScDown,
    model_6: C2fCibDw,
    model_7: ScDown,
    model_8: C2fCibDw,
    model_9: Sppf,
    model_10: Psa,
    model_13: C2fCibDw,
    model_16: C2f,
    model_17: Conv,
    model_19: C2fCibDw,
    model_20: ScDown,
    model_22: C2fCibDw,
}

impl Yolov10BodyX {
    pub fn forward(&self, input: Tensor<4>) -> Yolov10Features {
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

        Yolov10Features { p3, p4, p5 }
    }
}

/// Configuration for the fixed YOLOv10x body (depth 1.00, width 1.25, max channels 512).
#[derive(Debug, Default)]
pub struct Yolov10BodyXConfig;

impl Yolov10BodyXConfig {
    pub fn init(&self, device: &Device) -> Yolov10BodyX {
        Yolov10BodyX {
            model_0: ConvConfig::new(3, 80, 3, 2).init(device),
            model_1: ConvConfig::new(80, 160, 3, 2).init(device),
            model_2: C2fConfig::new(160, 160, 3, true).init(device),
            model_3: ConvConfig::new(160, 320, 3, 2).init(device),
            model_4: C2fConfig::new(320, 320, 6, true).init(device),
            model_5: ScDownConfig::new(320, 640, 3, 2).init(device),
            model_6: C2fCibDwConfig::new(640, 640, 6, true).init(device),
            model_7: ScDownConfig::new(640, 640, 3, 2).init(device),
            model_8: C2fCibDwConfig::new(640, 640, 3, true).init(device),
            model_9: SppfConfig::new(640).init(device),
            model_10: PsaConfig::new(640).init(device),
            model_13: C2fCibDwConfig::new(1280, 640, 3, true).init(device),
            model_16: C2fConfig::new(960, 320, 3, false).init(device),
            model_17: ConvConfig::new(320, 320, 3, 2).init(device),
            model_19: C2fCibDwConfig::new(960, 640, 3, true).init(device),
            model_20: ScDownConfig::new(640, 640, 3, 2).init(device),
            model_22: C2fCibDwConfig::new(1280, 640, 3, true).init(device),
        }
    }
}
