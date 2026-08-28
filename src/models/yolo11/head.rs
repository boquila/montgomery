use burn::{
    module::Module,
    nn::conv::{Conv2d, Conv2dConfig},
    tensor::{Device, Tensor, TensorData, activation, backend::Backend},
};

use super::blocks::{Conv, ConvConfig};
use super::body::Yolo11Features;

const NUM_CLASSES: usize = 80;

/// YOLO11 keeps Ultralytics' classic DFL head: each box side is a 16-bin distribution projected
/// onto the integers 0..15.
const REG_MAX: usize = 16;

/// Raw one2many predictions before DFL projection and anchor-grid decoding.
pub struct RawPredictions<B: Backend> {
    /// `[batch, 4 * reg_max, anchors]`.
    pub boxes: Tensor<B, 3>,
    /// `[batch, classes, anchors]`.
    pub scores: Tensor<B, 3>,
}

/// Decoded predictions in model-input space, matching Ultralytics' classic head output layout.
pub struct DecodedPredictions<B: Backend> {
    /// Center-size `XYWH` model-input pixels, `[batch, anchors, 4]`. The classic head emits
    /// center/width/height boxes (unlike the end-to-end families' XYXY); the crate's NMS helper
    /// consumes this layout directly.
    pub boxes: Tensor<B, 3>,
    /// Per-class sigmoid probabilities, `[batch, anchors, classes]`.
    pub scores: Tensor<B, 3>,
}

/// One detection scale of the YOLO11 head.
///
/// The box tower matches Ultralytics' shared `cv2` layout; the light classification tower matches
/// `Detect.cv3` with `legacy = False` (C3k2 bodies switch the head to the light DWConv/Conv
/// tower): depth-wise/pointwise pairs followed by a biased 1x1 projection. Field names
/// deliberately match the official `cv2`/`cv3` checkpoint keys after remapping.
#[derive(Module, Debug)]
struct DetectionBranch<B: Backend> {
    box_0: Conv<B>,
    box_1: Conv<B>,
    box_out: Conv2d<B>,
    cls_dw_0: Conv<B>,
    cls_pw_0: Conv<B>,
    cls_dw_1: Conv<B>,
    cls_pw_1: Conv<B>,
    cls_out: Conv2d<B>,
}

impl<B: Backend> DetectionBranch<B> {
    fn forward(&self, input: Tensor<B, 4>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [batch, _, height, width] = input.dims();
        let boxes = self
            .box_out
            .forward(self.box_1.forward(self.box_0.forward(input.clone())))
            .reshape([batch, 4 * REG_MAX, height * width]);
        let cls = self.cls_pw_0.forward(self.cls_dw_0.forward(input.clone()));
        let cls = self.cls_pw_1.forward(self.cls_dw_1.forward(cls));
        let scores = self
            .cls_out
            .forward(cls)
            .reshape([batch, NUM_CLASSES, height * width]);
        (boxes, scores)
    }
}

struct DetectionBranchConfig {
    input_channels: usize,
    box_channels: usize,
    cls_channels: usize,
}

impl DetectionBranchConfig {
    fn init<B: Backend>(&self, device: &Device<B>) -> DetectionBranch<B> {
        DetectionBranch {
            box_0: ConvConfig::new(self.input_channels, self.box_channels, 3, 1).init(device),
            box_1: ConvConfig::new(self.box_channels, self.box_channels, 3, 1).init(device),
            box_out: Conv2dConfig::new([self.box_channels, 4 * REG_MAX], [1, 1])
                .with_bias(true)
                .init(device),
            cls_dw_0: ConvConfig::new(self.input_channels, self.input_channels, 3, 1)
                .depthwise()
                .init(device),
            cls_pw_0: ConvConfig::new(self.input_channels, self.cls_channels, 1, 1).init(device),
            cls_dw_1: ConvConfig::new(self.cls_channels, self.cls_channels, 3, 1)
                .depthwise()
                .init(device),
            cls_pw_1: ConvConfig::new(self.cls_channels, self.cls_channels, 1, 1).init(device),
            cls_out: Conv2dConfig::new([self.cls_channels, NUM_CLASSES], [1, 1])
                .with_bias(true)
                .init(device),
        }
    }
}

/// Ultralytics YOLO11 `Detect` head, classic one2many inference branch.
///
/// Unlike YOLOv10/YOLO26 this head is not end-to-end: its predictions are plain per-anchor
/// distributions that require DFL projection, anchor-grid decoding, and external class-aware
/// non-maximum suppression (applied by the runtime, mirroring Ultralytics' postprocess).
#[derive(Module, Debug)]
pub struct Yolo11Head<B: Backend> {
    p3: DetectionBranch<B>,
    p4: DetectionBranch<B>,
    p5: DetectionBranch<B>,
}

impl<B: Backend> Yolo11Head<B> {
    pub fn forward_raw(&self, features: Yolo11Features<B>) -> RawPredictions<B> {
        let (boxes_p3, scores_p3) = self.p3.forward(features.p3);
        let (boxes_p4, scores_p4) = self.p4.forward(features.p4);
        let (boxes_p5, scores_p5) = self.p5.forward(features.p5);
        RawPredictions {
            boxes: Tensor::cat(vec![boxes_p3, boxes_p4, boxes_p5], 2),
            scores: Tensor::cat(vec![scores_p3, scores_p4, scores_p5], 2),
        }
    }

    pub fn forward(&self, features: Yolo11Features<B>) -> DecodedPredictions<B> {
        let p3_shape = features.p3.dims();
        let p4_shape = features.p4.dims();
        let p5_shape = features.p5.dims();
        let device = features.p3.device();
        let raw = self.forward_raw(features);
        let [batch, _, anchors_count] = raw.boxes.dims();

        // DFL integral: softmax each 16-bin side distribution, then project onto [0, 15].
        let distribution =
            activation::softmax(raw.boxes.reshape([batch, 4, REG_MAX, anchors_count]), 2);
        let projection = Tensor::<B, 4>::from_data(
            TensorData::new(
                (0..REG_MAX).map(|value| value as f32).collect(),
                [1, 1, REG_MAX, 1],
            ),
            &device,
        );
        let distances = (distribution * projection)
            .sum_dim(2)
            .squeeze_dim::<3>(2)
            .swap_dims(1, 2);

        let (anchors, strides) = make_anchors::<B>(
            [
                (p3_shape[2], p3_shape[3], 8.0),
                (p4_shape[2], p4_shape[3], 16.0),
                (p5_shape[2], p5_shape[3], 32.0),
            ],
            &device,
        );
        let anchors = anchors.unsqueeze::<3>();
        let strides = strides.unsqueeze::<3>();

        // Classic Ultralytics decode: the first two distances are the top-left offsets from the
        // anchor, the last two the bottom-right offsets; the head emits center/width/height
        // boxes, all scaled to model-input pixels (`dist2bbox(..., xywh=True) * strides`).
        let left_top = distances.clone().slice([0..batch, 0..anchors_count, 0..2]);
        let right_bottom = distances.slice([0..batch, 0..anchors_count, 2..4]);
        let center = anchors + (right_bottom.clone() - left_top.clone()) / 2.0;
        let size = left_top + right_bottom;
        let boxes = Tensor::cat(vec![center, size], 2) * strides;

        DecodedPredictions {
            boxes,
            scores: activation::sigmoid(raw.scores).swap_dims(1, 2),
        }
    }
}

#[derive(Debug)]
pub struct Yolo11HeadConfig {
    p3_channels: usize,
    p4_channels: usize,
    p5_channels: usize,
    box_channels: usize,
    cls_channels: usize,
}

impl Yolo11HeadConfig {
    /// Declare the head for one scale from its P3/P4/P5 input widths.
    ///
    /// `Detect` derives the box tower width as `max(16, ch[0] / 4, reg_max * 4)` and the light
    /// classification tower width as `max(ch[0], min(nc, 100))`; with `reg_max = 16` and
    /// `nc = 80` the box width is 64 except at x scale (ch[0] = 384) where it is 96, and the
    /// classification width is 80 at n scale (ch[0] = 64) and ch[0] elsewhere.
    pub fn new(p3_channels: usize, p4_channels: usize, p5_channels: usize) -> Self {
        let box_channels = (16).max(p3_channels / 4).max(4 * REG_MAX);
        let cls_channels = p3_channels.max(NUM_CLASSES.min(100));
        Self {
            p3_channels,
            p4_channels,
            p5_channels,
            box_channels,
            cls_channels,
        }
    }

    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo11Head<B> {
        let config = |input_channels: usize| DetectionBranchConfig {
            input_channels,
            box_channels: self.box_channels,
            cls_channels: self.cls_channels,
        };
        Yolo11Head {
            p3: config(self.p3_channels).init(device),
            p4: config(self.p4_channels).init(device),
            p5: config(self.p5_channels).init(device),
        }
    }
}

fn make_anchors<B: Backend>(
    levels: [(usize, usize, f32); 3],
    device: &Device<B>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let total: usize = levels.iter().map(|(height, width, _)| height * width).sum();
    let mut anchors = Vec::with_capacity(total * 2);
    let mut strides = Vec::with_capacity(total);
    for (height, width, stride) in levels {
        for y in 0..height {
            for x in 0..width {
                anchors.extend([x as f32 + 0.5, y as f32 + 0.5]);
                strides.push(stride);
            }
        }
    }
    (
        Tensor::from_data(TensorData::new(anchors, [total, 2]), device),
        Tensor::from_data(TensorData::new(strides, [total, 1]), device),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::yolo11::body::Yolo11BodyNConfig;
    use burn_flex::Flex;

    #[test]
    fn decodes_three_feature_levels_to_xywh_and_scores() {
        let worker = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let device = Default::default();
                let body = Yolo11BodyNConfig.init::<Flex>(&device);
                let head = Yolo11HeadConfig::new(64, 128, 256).init::<Flex>(&device);
                let input = Tensor::zeros([1, 3, 64, 64], &device);
                let output = head.forward(body.forward(input));
                assert_eq!(output.boxes.dims(), [1, 84, 4]);
                assert_eq!(output.scores.dims(), [1, 84, NUM_CLASSES]);
            })
            .expect("shape-test worker should start");
        worker.join().expect("shape-test worker should not panic");
    }
}
