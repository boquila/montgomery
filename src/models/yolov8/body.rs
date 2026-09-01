use burn::{
    module::Module,
    tensor::{Device, Tensor, backend::Backend},
};

use super::blocks::{C2f, C2fConfig, Conv, ConvConfig, Sppf, SppfConfig, upsample_nearest_2x};

/// Feature maps for the three YOLOv8 detection scales.
pub struct Yolov8Features<B: Backend> {
    /// P3/8 feature map.
    pub p3: Tensor<B, 4>,
    /// P4/16 feature map.
    pub p4: Tensor<B, 4>,
    /// P5/32 feature map.
    pub p5: Tensor<B, 4>,
}

/// Complete YOLOv8 backbone and feature-pyramid body (layers 0-21).
///
/// Field names retain the source graph indices so official checkpoint remapping stays mechanical
/// and parity failures can be localized to a specific declared layer. YOLOv8 was verified to be a
/// pure width/depth rescaling of one graph: every scale builds the same Conv/C2f/SPPF layers with
/// the backbone C2f stages carrying shortcuts and the neck C2f stages running without them; only
/// the depth-scaled repeat counts and the width-scaled channel counts change per scale.
#[derive(Module, Debug)]
pub struct Yolov8Body<B: Backend> {
    model_0: Conv<B>,
    model_1: Conv<B>,
    model_2: C2f<B>,
    model_3: Conv<B>,
    model_4: C2f<B>,
    model_5: Conv<B>,
    model_6: C2f<B>,
    model_7: Conv<B>,
    model_8: C2f<B>,
    model_9: Sppf<B>,
    model_12: C2f<B>,
    model_15: C2f<B>,
    model_16: Conv<B>,
    model_18: C2f<B>,
    model_19: Conv<B>,
    model_21: C2f<B>,
}

impl<B: Backend> Yolov8Body<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Yolov8Features<B> {
        let x = self.model_0.forward(input);
        let x = self.model_1.forward(x);
        let x = self.model_2.forward(x);
        let x = self.model_3.forward(x);
        let route_p3 = self.model_4.forward(x);

        let x = self.model_5.forward(route_p3.clone());
        let route_p4 = self.model_6.forward(x);

        let x = self.model_7.forward(route_p4.clone());
        let x = self.model_8.forward(x);
        let route_p5 = self.model_9.forward(x);

        let x = upsample_nearest_2x(route_p5.clone());
        let x = Tensor::cat(vec![x, route_p4], 1);
        let neck_p4 = self.model_12.forward(x);

        let x = upsample_nearest_2x(neck_p4.clone());
        let x = Tensor::cat(vec![x, route_p3], 1);
        let p3 = self.model_15.forward(x);

        let x = self.model_16.forward(p3.clone());
        let x = Tensor::cat(vec![x, neck_p4], 1);
        let p4 = self.model_18.forward(x);

        let x = self.model_19.forward(p4.clone());
        let x = Tensor::cat(vec![x, route_p5], 1);
        let p5 = self.model_21.forward(x);

        Yolov8Features { p3, p4, p5 }
    }
}

/// Shared construction logic: one declared layer table per scale.
///
/// `backbone_repeats` covers the C2f stages at layers 2/4/6/8 and `neck_repeats` the C2f stages
/// at layers 12/15/18/21; both are the depth-scaled repeat counts (round(n * depth)).
#[derive(Debug)]
pub struct Yolov8BodyConfig {
    widths: [usize; 10],
    backbone_repeats: [usize; 4],
    neck_repeats: [usize; 4],
}

impl Yolov8BodyConfig {
    fn init<B: Backend>(&self, device: &Device<B>) -> Yolov8Body<B> {
        // Channel table per layer: out_0..out_8 are the outputs of backbone layers 0-8 (the SPPF
        // width equals out_8); the neck reuses those widths symmetrically: 12 out = out_5, 15 out
        // = out_4 (the backbone P3 tap), 18 out = out_5, 21 out = out_8.
        let [w0, w1, w2, w3, w4, w5, w6, w7, w8, _] = self.widths;
        let [r2, r4, r6, r8] = self.backbone_repeats;
        let [r12, r15, r18, r21] = self.neck_repeats;
        Yolov8Body {
            model_0: ConvConfig::new(3, w0, 3, 2).init(device),
            model_1: ConvConfig::new(w0, w1, 3, 2).init(device),
            model_2: C2fConfig::new(w1, w2, r2, true).init(device),
            model_3: ConvConfig::new(w2, w3, 3, 2).init(device),
            model_4: C2fConfig::new(w3, w4, r4, true).init(device),
            model_5: ConvConfig::new(w4, w5, 3, 2).init(device),
            model_6: C2fConfig::new(w5, w6, r6, true).init(device),
            model_7: ConvConfig::new(w6, w7, 3, 2).init(device),
            model_8: C2fConfig::new(w7, w8, r8, true).init(device),
            model_9: SppfConfig::new(w8).init(device),
            model_12: C2fConfig::new(w8 + w6, w5, r12, false).init(device),
            model_15: C2fConfig::new(w5 + w4, w4, r15, false).init(device),
            model_16: ConvConfig::new(w4, w4, 3, 2).init(device),
            model_18: C2fConfig::new(w4 + w5, w5, r18, false).init(device),
            model_19: ConvConfig::new(w5, w5, 3, 2).init(device),
            model_21: C2fConfig::new(w5 + w8, w8, r21, false).init(device),
        }
    }
}

/// Configuration for the fixed YOLOv8n body (depth 0.33, width 0.25, max channels 1024).
///
/// The declared repeats of 3/6 become 1/2 at the n-scale depth gain, identical to s.
#[derive(Debug, Default)]
pub struct Yolov8BodyNConfig;

impl Yolov8BodyNConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolov8Body<B> {
        Yolov8BodyConfig {
            widths: [16, 32, 32, 64, 64, 128, 128, 256, 256, 256],
            backbone_repeats: [1, 2, 2, 1],
            neck_repeats: [1, 1, 1, 1],
        }
        .init(device)
    }
}

/// Configuration for the fixed YOLOv8s body (depth 0.33, width 0.50, max channels 1024).
#[derive(Debug, Default)]
pub struct Yolov8BodySConfig;

impl Yolov8BodySConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolov8Body<B> {
        Yolov8BodyConfig {
            widths: [32, 64, 64, 128, 128, 256, 256, 512, 512, 512],
            backbone_repeats: [1, 2, 2, 1],
            neck_repeats: [1, 1, 1, 1],
        }
        .init(device)
    }
}

/// Configuration for the fixed YOLOv8m body (depth 0.67, width 0.75, max channels 768).
#[derive(Debug, Default)]
pub struct Yolov8BodyMConfig;

impl Yolov8BodyMConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolov8Body<B> {
        Yolov8BodyConfig {
            widths: [48, 96, 96, 192, 192, 384, 384, 576, 576, 576],
            backbone_repeats: [2, 4, 4, 2],
            neck_repeats: [2, 2, 2, 2],
        }
        .init(device)
    }
}

/// Configuration for the fixed YOLOv8l body (depth 1.00, width 1.00, max channels 512).
#[derive(Debug, Default)]
pub struct Yolov8BodyLConfig;

impl Yolov8BodyLConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolov8Body<B> {
        Yolov8BodyConfig {
            widths: [64, 128, 128, 256, 256, 512, 512, 512, 512, 512],
            backbone_repeats: [3, 6, 6, 3],
            neck_repeats: [3, 3, 3, 3],
        }
        .init(device)
    }
}

/// Configuration for the fixed YOLOv8x body (depth 1.00, width 1.25, max channels 512).
#[derive(Debug, Default)]
pub struct Yolov8BodyXConfig;

impl Yolov8BodyXConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolov8Body<B> {
        Yolov8BodyConfig {
            widths: [80, 160, 160, 320, 320, 640, 640, 640, 640, 640],
            backbone_repeats: [3, 6, 6, 3],
            neck_repeats: [3, 3, 3, 3],
        }
        .init(device)
    }
}
