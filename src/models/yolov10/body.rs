use burn::{
    module::Module,
    tensor::{Device, Tensor, backend::Backend},
};

use super::blocks::{
    C2f, C2fCib, C2fCibConfig, C2fConfig, Conv, ConvConfig, Psa, PsaConfig, ScDown, ScDownConfig,
    Sppf, SppfConfig, upsample_nearest_2x,
};

/// Feature maps for the three YOLOv10 detection scales.
pub struct Yolov10Features<B: Backend> {
    /// P3/8 feature map with 64 channels.
    pub p3: Tensor<B, 4>,
    /// P4/16 feature map with 128 channels.
    pub p4: Tensor<B, 4>,
    /// P5/32 feature map with 256 channels.
    pub p5: Tensor<B, 4>,
}

/// Complete YOLOv10n backbone and feature-pyramid body (layers 0-22).
///
/// Field names retain the source graph indices so official checkpoint remapping stays mechanical
/// and parity failures can be localized to a specific declared layer.
#[derive(Module, Debug)]
pub struct Yolov10Body<B: Backend> {
    model_0: Conv<B>,
    model_1: Conv<B>,
    model_2: C2f<B>,
    model_3: Conv<B>,
    model_4: C2f<B>,
    model_5: ScDown<B>,
    model_6: C2f<B>,
    model_7: ScDown<B>,
    model_8: C2f<B>,
    model_9: Sppf<B>,
    model_10: Psa<B>,
    model_13: C2f<B>,
    model_16: C2f<B>,
    model_17: Conv<B>,
    model_19: C2f<B>,
    model_20: ScDown<B>,
    model_22: C2fCib<B>,
}

impl<B: Backend> Yolov10Body<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Yolov10Features<B> {
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
pub struct Yolov10BodyConfig;

impl Yolov10BodyConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolov10Body<B> {
        Yolov10Body {
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
                let body: Yolov10Body<Flex> = Yolov10BodyConfig.init(&device);
                let input = Tensor::zeros([1, 3, 64, 64], &device);
                let output = body.forward(input);
                assert_eq!(output.p3.dims(), [1, 64, 8, 8]);
                assert_eq!(output.p4.dims(), [1, 128, 4, 4]);
                assert_eq!(output.p5.dims(), [1, 256, 2, 2]);
            })
            .expect("shape-test worker should start");
        worker.join().expect("shape-test worker should not panic");
    }
}
