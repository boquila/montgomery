use burn::{
    module::Module,
    nn::conv::{Conv2d, Conv2dConfig},
    tensor::{Device, Tensor, TensorData, activation, backend::Backend},
};

use super::blocks::{Conv, ConvConfig};
use super::body::Yolov10Features;

const NUM_CLASSES: usize = 80;
const REG_MAX: usize = 16;

/// Ultralytics v10Detect inference parameters.
pub const MAX_DETECTIONS: usize = 300;

/// Raw one2one predictions before DFL projection and anchor-grid decoding.
pub struct RawPredictions<B: Backend> {
    /// `[batch, 4 * reg_max, anchors]`.
    pub boxes: Tensor<B, 3>,
    /// `[batch, classes, anchors]`.
    pub scores: Tensor<B, 3>,
}

/// Decoded predictions in model-input space.
pub struct DecodedPredictions<B: Backend> {
    /// Unnormalized `XYXY` model-input pixels, `[batch, anchors, 4]`.
    pub boxes: Tensor<B, 3>,
    /// Per-class sigmoid probabilities, `[batch, anchors, classes]`.
    pub scores: Tensor<B, 3>,
}

/// One detection scale of the YOLOv10 one2one head.
///
/// The box tower matches Ultralytics' shared `cv2` layout; the light classification tower matches
/// `v10Detect.cv3`: depth-wise/pointwise pairs followed by a biased 1x1 projection. Field names
/// deliberately match the official `one2one_cv2`/`one2one_cv3` checkpoint keys after remapping.
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

/// Ultralytics YOLOv10 `v10Detect` head, one2one (NMS-free) inference branch.
///
/// The training-only one2many branch is intentionally not implemented: official inference decodes
/// the one2one predictions and selects the top-scoring detections without non-maximum suppression.
#[derive(Module, Debug)]
pub struct Yolov10Head<B: Backend> {
    p3: DetectionBranch<B>,
    p4: DetectionBranch<B>,
    p5: DetectionBranch<B>,
}

impl<B: Backend> Yolov10Head<B> {
    pub fn forward_raw(&self, features: Yolov10Features<B>) -> RawPredictions<B> {
        let (boxes_p3, scores_p3) = self.p3.forward(features.p3);
        let (boxes_p4, scores_p4) = self.p4.forward(features.p4);
        let (boxes_p5, scores_p5) = self.p5.forward(features.p5);
        RawPredictions {
            boxes: Tensor::cat(vec![boxes_p3, boxes_p4, boxes_p5], 2),
            scores: Tensor::cat(vec![scores_p3, scores_p4, scores_p5], 2),
        }
    }

    pub fn forward(&self, features: Yolov10Features<B>) -> DecodedPredictions<B> {
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
        let left_top = distances.clone().slice([0..batch, 0..anchors_count, 0..2]);
        let right_bottom = distances.slice([0..batch, 0..anchors_count, 2..4]);
        let boxes =
            Tensor::cat(vec![anchors.clone() - left_top, anchors + right_bottom], 2) * strides;

        DecodedPredictions {
            boxes,
            scores: activation::sigmoid(raw.scores).swap_dims(1, 2),
        }
    }
}

#[derive(Debug, Default)]
pub struct Yolov10HeadConfig;

impl Yolov10HeadConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolov10Head<B> {
        // Detect's box width is max(16, ch[0] / 4, reg_max * 4) and the light classification
        // width is max(ch[0], min(nc, 100)); with ch[0] = 64 and nc = 80 both resolve to 64/80.
        let config = |input_channels: usize| DetectionBranchConfig {
            input_channels,
            box_channels: 4 * REG_MAX,
            cls_channels: NUM_CLASSES,
        };
        Yolov10Head {
            p3: config(64).init(device),
            p4: config(128).init(device),
            p5: config(256).init(device),
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
    use crate::models::yolov10::body::Yolov10BodyConfig;
    use burn_flex::Flex;

    #[test]
    fn decodes_three_feature_levels_to_xyxy_and_scores() {
        let worker = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let device = Default::default();
                let body = Yolov10BodyConfig.init::<Flex>(&device);
                let head = Yolov10HeadConfig.init::<Flex>(&device);
                let input = Tensor::zeros([1, 3, 64, 64], &device);
                let output = head.forward(body.forward(input));
                assert_eq!(output.boxes.dims(), [1, 84, 4]);
                assert_eq!(output.scores.dims(), [1, 84, NUM_CLASSES]);
            })
            .expect("shape-test worker should start");
        worker.join().expect("shape-test worker should not panic");
    }
}
