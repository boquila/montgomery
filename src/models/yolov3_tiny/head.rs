use burn::{
    module::Module,
    nn::conv::{Conv2d, Conv2dConfig},
    tensor::{Device, Tensor, TensorData, activation, backend::Backend},
};

use super::body::{Conv, ConvConfig, Yolov3TinyFeatures};

const DEFAULT_NUM_CLASSES: usize = 80;
const REG_MAX: usize = 16;

/// Raw predictions before DFL projection and anchor-grid decoding.
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

#[derive(Module, Debug)]
struct DetectionBranch<B: Backend> {
    box_0: Conv<B>,
    box_1: Conv<B>,
    box_2: Conv2d<B>,
    cls_0: Conv<B>,
    cls_1: Conv<B>,
    cls_2: Conv2d<B>,
    num_classes: usize,
}

impl<B: Backend> DetectionBranch<B> {
    fn forward(&self, input: Tensor<B, 4>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [batch, _, height, width] = input.dims();
        let boxes = self
            .box_2
            .forward(self.box_1.forward(self.box_0.forward(input.clone())))
            .reshape([batch, 4 * REG_MAX, height * width]);
        let scores = self
            .cls_2
            .forward(self.cls_1.forward(self.cls_0.forward(input)))
            .reshape([batch, self.num_classes, height * width]);
        (boxes, scores)
    }
}

struct DetectionBranchConfig {
    input_channels: usize,
    num_classes: usize,
}

impl DetectionBranchConfig {
    fn init<B: Backend>(&self, device: &Device<B>) -> DetectionBranch<B> {
        // Detect.legacy=True for v3/v5/v8/v9: both towers use ordinary Conv blocks.
        let box_channels = 64;
        let class_channels = 256;
        DetectionBranch {
            box_0: ConvConfig::new(self.input_channels, box_channels, 3, 1).init(device),
            box_1: ConvConfig::new(box_channels, box_channels, 3, 1).init(device),
            box_2: Conv2dConfig::new([box_channels, 4 * REG_MAX], [1, 1])
                .with_bias(true)
                .init(device),
            cls_0: ConvConfig::new(self.input_channels, class_channels, 3, 1).init(device),
            cls_1: ConvConfig::new(class_channels, class_channels, 3, 1).init(device),
            cls_2: Conv2dConfig::new([class_channels, self.num_classes], [1, 1])
                .with_bias(true)
                .init(device),
            num_classes: self.num_classes,
        }
    }
}

/// Ultralytics anchor-free, objectness-free split detection head used by YOLOv3-Tiny-U.
#[derive(Module, Debug)]
pub struct DetectHead<B: Backend> {
    p4: DetectionBranch<B>,
    p5: DetectionBranch<B>,
}

impl<B: Backend> DetectHead<B> {
    pub fn forward_raw(&self, features: Yolov3TinyFeatures<B>) -> RawPredictions<B> {
        let (boxes_p4, scores_p4) = self.p4.forward(features.p4);
        let (boxes_p5, scores_p5) = self.p5.forward(features.p5);
        RawPredictions {
            boxes: Tensor::cat(vec![boxes_p4, boxes_p5], 2),
            scores: Tensor::cat(vec![scores_p4, scores_p5], 2),
        }
    }

    pub fn forward(&self, features: Yolov3TinyFeatures<B>) -> DecodedPredictions<B> {
        let p4_shape = features.p4.dims();
        let p5_shape = features.p5.dims();
        let device = features.p4.device();
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

#[derive(Debug)]
pub struct DetectHeadConfig {
    num_classes: usize,
}

impl Default for DetectHeadConfig {
    fn default() -> Self {
        Self::new(DEFAULT_NUM_CLASSES)
    }
}

impl DetectHeadConfig {
    pub fn new(num_classes: usize) -> Self {
        assert!(num_classes > 0, "class count must be positive");
        Self { num_classes }
    }

    pub fn init<B: Backend>(&self, device: &Device<B>) -> DetectHead<B> {
        DetectHead {
            p4: DetectionBranchConfig {
                input_channels: 256,
                num_classes: self.num_classes,
            }
            .init(device),
            p5: DetectionBranchConfig {
                input_channels: 512,
                num_classes: self.num_classes,
            }
            .init(device),
        }
    }
}

fn make_anchors<B: Backend>(
    levels: [(usize, usize, f32); 2],
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
    use crate::models::yolov3_tiny::body::Yolov3TinyBodyConfig;
    use burn_flex::Flex;

    #[test]
    fn decodes_two_feature_levels_to_xyxy_and_scores() {
        let worker = std::thread::Builder::new()
            .stack_size(48 * 1024 * 1024)
            .spawn(|| {
                let device = Default::default();
                let body = Yolov3TinyBodyConfig.init::<Flex>(&device);
                let head = DetectHeadConfig::default().init::<Flex>(&device);
                let input = Tensor::zeros([1, 3, 64, 64], &device);
                let output = head.forward(body.forward(input));
                assert_eq!(output.boxes.dims(), [1, 20, 4]);
                assert_eq!(output.scores.dims(), [1, 20, DEFAULT_NUM_CLASSES]);
            })
            .expect("shape-test worker should start");
        worker.join().expect("shape-test worker should not panic");
    }
}
