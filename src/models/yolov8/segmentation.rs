use burn::{
    module::Module,
    nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
    tensor::{Device, Tensor, backend::Backend},
};

use super::blocks::{Conv, ConvConfig};
use super::body::Yolov8Features;
use super::head::{Yolov8Head, Yolov8HeadConfig};

/// Number of mask prototypes and per-detection mask coefficients (`nm`).
pub const NUM_MASKS: usize = 32;

/// Mask prototypes for one instance segmentation channel.
///
/// Mirrors Ultralytics' `Proto` module on the P3 feature map: two 3x3 convolutions around a
/// 2x transposed convolution produce one 32-channel prototype map at stride 4 (160x160 at 640 px
/// input). Field names deliberately match the official `model.22.proto.*` checkpoint keys after
/// remapping.
#[derive(Module, Debug)]
pub struct Proto<B: Backend> {
    cv1: Conv<B>,
    upsample: ConvTranspose2d<B>,
    cv2: Conv<B>,
    cv3: Conv<B>,
}

impl<B: Backend> Proto<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.cv1.forward(input);
        let x = self.upsample.forward(x);
        self.cv3.forward(self.cv2.forward(x))
    }
}

struct ProtoConfig {
    input_channels: usize,
    hidden_channels: usize,
}

impl ProtoConfig {
    fn init<B: Backend>(&self, device: &Device<B>) -> Proto<B> {
        Proto {
            cv1: ConvConfig::new(self.input_channels, self.hidden_channels, 3, 1).init(device),
            upsample: ConvTranspose2dConfig::new(
                [self.hidden_channels, self.hidden_channels],
                [2, 2],
            )
            .with_stride([2, 2])
            .init(device),
            cv2: ConvConfig::new(self.hidden_channels, self.hidden_channels, 3, 1).init(device),
            cv3: ConvConfig::new(self.hidden_channels, NUM_MASKS, 1, 1).init(device),
        }
    }
}

/// One mask-coefficient scale of the YOLOv8 segment head (`cv4`).
///
/// Ultralytics builds the mask tower from full 3x3 `Conv` layers regardless of the head's cv3
/// flavor, exactly like the YOLO11-seg mask towers. Field names match the official `cv4`
/// checkpoint keys after remapping.
#[derive(Module, Debug)]
pub struct MaskBranch<B: Backend> {
    mask_0: Conv<B>,
    mask_1: Conv<B>,
    mask_out: Conv2d<B>,
}

impl<B: Backend> MaskBranch<B> {
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, _, height, width] = input.dims();
        let x = self.mask_1.forward(self.mask_0.forward(input));
        self.mask_out
            .forward(x)
            .reshape([batch, NUM_MASKS, height * width])
    }
}

struct MaskBranchConfig {
    input_channels: usize,
    mask_channels: usize,
}

impl MaskBranchConfig {
    fn init<B: Backend>(&self, device: &Device<B>) -> MaskBranch<B> {
        MaskBranch {
            mask_0: ConvConfig::new(self.input_channels, self.mask_channels, 3, 1).init(device),
            mask_1: ConvConfig::new(self.mask_channels, self.mask_channels, 3, 1).init(device),
            mask_out: Conv2dConfig::new([self.mask_channels, NUM_MASKS], [1, 1])
                .with_bias(true)
                .init(device),
        }
    }
}

/// Ultralytics YOLOv8 `Segment` head: the classic DFL detection head (legacy 3x3 cv3 towers) plus
/// the Proto module and one mask-coefficient branch per scale.
///
/// The head is not end-to-end; the runtime applies the same class-aware NMS as the detect path
/// with the 32 raw mask coefficients carried along per surviving anchor, mirroring Ultralytics'
/// `non_max_suppression` on `[boxes, scores, mask_coefficients]` rows. The output type is shared
/// with the YOLO11-seg runtime path.
#[derive(Module, Debug)]
pub struct Yolov8SegHead<B: Backend> {
    pub(crate) detect: Yolov8Head<B>,
    proto: Proto<B>,
    p3_mask: MaskBranch<B>,
    p4_mask: MaskBranch<B>,
    p5_mask: MaskBranch<B>,
}

impl<B: Backend> Yolov8SegHead<B> {
    pub fn forward(&self, features: Yolov8Features<B>) -> crate::models::yolo11::SegmentOutput<B> {
        let Yolov8Features { p3, p4, p5 } = features;
        let decoded = self.detect.forward(Yolov8Features {
            p3: p3.clone(),
            p4: p4.clone(),
            p5: p5.clone(),
        });
        let prototypes = self.proto.forward(p3.clone());
        let coefficients = Tensor::cat(
            vec![
                self.p3_mask.forward(p3),
                self.p4_mask.forward(p4),
                self.p5_mask.forward(p5),
            ],
            2,
        );
        crate::models::yolo11::SegmentOutput {
            boxes: decoded.boxes,
            scores: decoded.scores,
            coefficients,
            prototypes,
        }
    }
}

#[derive(Debug)]
pub struct Yolov8SegHeadConfig {
    detect: Yolov8HeadConfig,
    proto_input_channels: usize,
    proto_hidden_channels: usize,
    mask_input_channels: [usize; 3],
    mask_channels: usize,
}

impl Yolov8SegHeadConfig {
    /// Declare the segment head for one scale from its P3/P4/P5 input widths and the
    /// width-scaled prototype channel count (`npr`).
    ///
    /// `parse_model` scales the YAML's 256 prototype channels as
    /// `make_divisible(min(256, max_channels) * width, 8)` (64 at n, 128 at s, 192 at m, 256 at
    /// l, 320 at x), and the mask tower width is `max(ch[0] / 4, nm)` — full 3x3 convolutions at
    /// every scale.
    pub fn new(
        p3_channels: usize,
        p4_channels: usize,
        p5_channels: usize,
        proto_hidden_channels: usize,
    ) -> Self {
        Self {
            detect: Yolov8HeadConfig::new(p3_channels, p4_channels, p5_channels),
            proto_input_channels: p3_channels,
            proto_hidden_channels,
            mask_input_channels: [p3_channels, p4_channels, p5_channels],
            mask_channels: (p3_channels / 4).max(NUM_MASKS),
        }
    }

    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolov8SegHead<B> {
        let proto = ProtoConfig {
            input_channels: self.proto_input_channels,
            hidden_channels: self.proto_hidden_channels,
        }
        .init(device);
        let mask = |input_channels: usize| {
            MaskBranchConfig {
                input_channels,
                mask_channels: self.mask_channels,
            }
            .init(device)
        };
        Yolov8SegHead {
            detect: self.detect.init(device),
            proto,
            p3_mask: mask(self.mask_input_channels[0]),
            p4_mask: mask(self.mask_input_channels[1]),
            p5_mask: mask(self.mask_input_channels[2]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_flex::Flex;

    #[test]
    fn decodes_segment_tensors_for_three_feature_levels() {
        let worker = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let device = Default::default();
                let head = Yolov8SegHeadConfig::new(64, 128, 256, 64).init::<Flex>(&device);
                let body_features = Yolov8Features {
                    p3: Tensor::zeros([1, 64, 8, 8], &device),
                    p4: Tensor::zeros([1, 128, 4, 4], &device),
                    p5: Tensor::zeros([1, 256, 2, 2], &device),
                };
                let output = head.forward(body_features);
                assert_eq!(output.boxes.dims(), [1, 84, 4]);
                assert_eq!(output.scores.dims(), [1, 84, 80]);
                assert_eq!(output.coefficients.dims(), [1, NUM_MASKS, 84]);
                // P3 at 8x8 upsampled by the Proto module to stride 4.
                assert_eq!(output.prototypes.dims(), [1, NUM_MASKS, 16, 16]);
            })
            .expect("shape-test worker should start");
        worker.join().expect("shape-test worker should not panic");
    }
}
