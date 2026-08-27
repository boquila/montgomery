use burn::{
    module::Module,
    tensor::{Device, Tensor, backend::Backend},
};

use super::blocks::{
    C2Psa, C2PsaConfig, C3k2, C3k2Attn, C3k2AttnConfig, C3k2C3k, C3k2C3kConfig, C3k2Config, Conv,
    ConvConfig, Sppf, SppfConfig, upsample_nearest_2x,
};

/// Feature maps for the three YOLO26 detection scales.
pub struct Yolo26Features<B: Backend> {
    /// P3/8 feature map with 64 channels.
    pub p3: Tensor<B, 4>,
    /// P4/16 feature map with 128 channels.
    pub p4: Tensor<B, 4>,
    /// P5/32 feature map with 256 channels.
    pub p5: Tensor<B, 4>,
}

/// Complete YOLO26n backbone and feature-pyramid body (layers 0-22).
///
/// Field names retain the source graph indices so official checkpoint remapping stays mechanical
/// and parity failures can be localized to a specific declared layer.
#[derive(Module, Debug)]
pub struct Yolo26Body<B: Backend> {
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
    model_13: C3k2C3k<B>,
    model_16: C3k2C3k<B>,
    model_17: Conv<B>,
    model_19: C3k2C3k<B>,
    model_20: Conv<B>,
    model_22: C3k2Attn<B>,
}

impl<B: Backend> Yolo26Body<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Yolo26Features<B> {
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

        Yolo26Features { p3, p4, p5 }
    }
}

/// Configuration for the fixed YOLO26n body (depth 0.50, width 0.25, max channels 1024).
///
/// The declared repeats of 2 on the C3k2 and C2PSA stages become one block per stage at the
/// n-scale depth gain of 0.50; the SPPF pooling count and shortcut come from its own YAML args.
#[derive(Debug, Default)]
pub struct Yolo26BodyConfig;

impl Yolo26BodyConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo26Body<B> {
        Yolo26Body {
            model_0: ConvConfig::new(3, 16, 3, 2).init(device),
            model_1: ConvConfig::new(16, 32, 3, 2).init(device),
            model_2: C3k2Config::new(32, 64, 1, 0.25, true).init(device),
            model_3: ConvConfig::new(64, 64, 3, 2).init(device),
            model_4: C3k2Config::new(64, 128, 1, 0.25, true).init(device),
            model_5: ConvConfig::new(128, 128, 3, 2).init(device),
            model_6: C3k2C3kConfig::new(128, 128, 1, true).init(device),
            model_7: ConvConfig::new(128, 256, 3, 2).init(device),
            model_8: C3k2C3kConfig::new(256, 256, 1, true).init(device),
            model_9: SppfConfig::new(256, 3, true).init(device),
            model_10: C2PsaConfig::new(256, 1).init(device),
            model_13: C3k2C3kConfig::new(384, 128, 1, true).init(device),
            model_16: C3k2C3kConfig::new(256, 64, 1, true).init(device),
            model_17: ConvConfig::new(64, 64, 3, 2).init(device),
            model_19: C3k2C3kConfig::new(192, 128, 1, true).init(device),
            model_20: ConvConfig::new(128, 128, 3, 2).init(device),
            model_22: C3k2AttnConfig::new(384, 256, 1, true).init(device),
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
                let body: Yolo26Body<Flex> = Yolo26BodyConfig.init(&device);
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
