use burn::{
    module::Module,
    nn::conv::{Conv2d, Conv2dConfig},
    tensor::{Device, Tensor, TensorData, activation},
};

use super::body::{Conv, ConvConfig, Yolov3TinyFeatures};

const DEFAULT_NUM_CLASSES: usize = 80;
const REG_MAX: usize = 16;

/// Raw predictions before DFL projection and anchor-grid decoding.
pub struct RawPredictions {
    /// `[batch, 4 * reg_max, anchors]`.
    pub boxes: Tensor<3>,
    /// `[batch, classes, anchors]`.
    pub scores: Tensor<3>,
}

/// Decoded predictions in model-input space.
pub struct DecodedPredictions {
    /// Unnormalized `XYXY` model-input pixels, `[batch, anchors, 4]`.
    pub boxes: Tensor<3>,
    /// Per-class sigmoid probabilities, `[batch, anchors, classes]`.
    pub scores: Tensor<3>,
}

#[derive(Module, Debug)]
struct DetectionBranch {
    box_0: Conv,
    box_1: Conv,
    box_2: Conv2d,
    cls_0: Conv,
    cls_1: Conv,
    cls_2: Conv2d,
    num_classes: usize,
}

impl DetectionBranch {
    fn forward(&self, input: Tensor<4>) -> (Tensor<3>, Tensor<3>) {
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
    fn init(&self, device: &Device) -> DetectionBranch {
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
pub struct DetectHead {
    p4: DetectionBranch,
    p5: DetectionBranch,
}

impl DetectHead {
    pub fn forward_raw(&self, features: Yolov3TinyFeatures) -> RawPredictions {
        let (boxes_p4, scores_p4) = self.p4.forward(features.p4);
        let (boxes_p5, scores_p5) = self.p5.forward(features.p5);
        RawPredictions {
            boxes: Tensor::cat(vec![boxes_p4, boxes_p5], 2),
            scores: Tensor::cat(vec![scores_p4, scores_p5], 2),
        }
    }

    pub fn forward(&self, features: Yolov3TinyFeatures) -> DecodedPredictions {
        let p4_shape = features.p4.dims();
        let p5_shape = features.p5.dims();
        let device = features.p4.device();
        let raw = self.forward_raw(features);
        let [batch, _, anchors_count] = raw.boxes.dims();

        // DFL integral: softmax each 16-bin side distribution, then project onto [0, 15].
        let distribution =
            activation::softmax(raw.boxes.reshape([batch, 4, REG_MAX, anchors_count]), 2);
        let projection = Tensor::<4>::from_data(
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

        let (anchors, strides) = make_anchors(
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

    pub fn init(&self, device: &Device) -> DetectHead {
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

fn make_anchors(levels: [(usize, usize, f32); 2], device: &Device) -> (Tensor<2>, Tensor<2>) {
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

    #[test]
    fn decodes_two_feature_levels_to_xyxy_and_scores() {
        let worker = std::thread::Builder::new()
            .stack_size(48 * 1024 * 1024)
            .spawn(|| {
                let device = Default::default();
                let body = Yolov3TinyBodyConfig.init(&device);
                let head = DetectHeadConfig::default().init(&device);
                let input = Tensor::zeros([1, 3, 64, 64], &device);
                let output = head.forward(body.forward(input));
                assert_eq!(output.boxes.dims(), [1, 20, 4]);
                assert_eq!(output.scores.dims(), [1, 20, DEFAULT_NUM_CLASSES]);
            })
            .expect("shape-test worker should start");
        worker.join().expect("shape-test worker should not panic");
    }
}
