use alloc::{vec, vec::Vec};
use burn::{
    module::Module,
    nn::{
        Initializer, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{Device, Int, Shape, Tensor, activation::sigmoid, backend::Backend},
};
use itertools::{izip, multiunzip};

use super::{
    blocks::{BaseConv, BaseConvConfig, ConvBlock, ConvBlockConfig, expand},
    pafpn::FpnFeatures,
};

const STRIDES: [usize; 3] = [8, 16, 32];
const IN_CHANNELS: [usize; 3] = [256, 512, 1024];
const PRIOR_PROB: f64 = 1e-2;

/// Create a 2D coordinate grid for the specified dimensions.
/// Similar to [`numpy.indices`](https://numpy.org/doc/stable/reference/generated/numpy.indices.html)
/// but specific to two dimensions.
fn create_2d_grid<B: Backend>(x: usize, y: usize, device: &Device<B>) -> Tensor<B, 3, Int> {
    let y_idx = Tensor::arange(0..y as i64, device)
        .reshape::<2, _>(Shape::new([y, 1]))
        .repeat_dim(1, x)
        .reshape::<2, _>(Shape::new([y, x]));
    let x_idx = Tensor::arange(0..x as i64, device)
        .reshape::<2, _>(Shape::new([1, x])) // can only repeat with dim=1
        .repeat_dim(0, y)
        .reshape(Shape::new([y, x]));

    Tensor::stack(vec![x_idx, y_idx], 2)
}

/// YOLOX head.
#[derive(Module, Debug)]
pub struct Head<B: Backend> {
    stems: Vec<BaseConv<B>>,
    cls_convs: Vec<ConvBlock<B>>,
    reg_convs: Vec<ConvBlock<B>>,
    cls_preds: Vec<Conv2d<B>>,
    reg_preds: Vec<Conv2d<B>>,
    obj_preds: Vec<Conv2d<B>>,
    num_classes: usize,
}

/// One feature level's geometry in a YOLOX raw training output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeatureLevelShape {
    pub height: usize,
    pub width: usize,
    pub stride: usize,
}

/// Raw YOLOX predictions consumed by SimOTA and the native loss.
pub struct RawPredictions<B: Backend> {
    /// Raw center offsets and log-width/log-height, `[batch, anchors, 4]`.
    pub regression: Tensor<B, 3>,
    /// Raw objectness logits, `[batch, anchors, 1]`.
    pub objectness_logits: Tensor<B, 3>,
    /// Raw class logits, `[batch, anchors, classes]`.
    pub class_logits: Tensor<B, 3>,
    /// Differentiably decoded center-size boxes in canvas pixels, `[batch, anchors, 4]`.
    pub decoded_boxes: Tensor<B, 3>,
    pub levels: [FeatureLevelShape; 3],
}

impl<B: Backend> Head<B> {
    pub fn forward(&self, x: FpnFeatures<B>) -> Tensor<B, 3> {
        let raw = self.forward_train(x);
        Tensor::cat(
            vec![
                raw.decoded_boxes,
                sigmoid(raw.objectness_logits),
                sigmoid(raw.class_logits),
            ],
            2,
        )
    }

    /// Run the shared towers without sigmoid or post-processing.
    pub fn forward_train(&self, x: FpnFeatures<B>) -> RawPredictions<B> {
        let features: [Tensor<B, 4>; 3] = [x.0, x.1, x.2];

        // Outputs for each feature map
        let (outputs, shapes): (Vec<Tensor<B, 3>>, Vec<(usize, usize)>) = izip!(
            features,
            &self.stems,
            &self.cls_convs,
            &self.cls_preds,
            &self.reg_convs,
            &self.reg_preds,
            &self.obj_preds,
            &STRIDES
        )
        .map(
            |(feat, stem, cls_conv, cls_pred, reg_conv, reg_pred, obj_pred, _stride)| {
                let feat = stem.forward(feat);

                let cls_feat = cls_conv.forward(feat.clone());
                let cls_out = cls_pred.forward(cls_feat);

                let reg_feat = reg_conv.forward(feat);
                let reg_out = reg_pred.forward(reg_feat.clone());

                let obj_out = obj_pred.forward(reg_feat);

                // Output [B, 5 + num_classes, num_anchors]. Sigmoid belongs to inference only.
                let out = Tensor::cat(vec![reg_out, obj_out, cls_out], 1);
                let [_, _, h, w] = out.dims();
                (out.flatten(2, 3), (h, w))
            },
        )
        .unzip();

        let outputs = Tensor::cat(outputs, 2).swap_dims(2, 1);
        let [batch, anchors, _] = outputs.dims();
        let regression = outputs.clone().slice([0..batch, 0..anchors, 0..4]);
        let objectness_logits = outputs.clone().slice([0..batch, 0..anchors, 4..5]);
        let class_logits = outputs.slice([0..batch, 0..anchors, 5..5 + self.num_classes]);
        let decoded_boxes = self.decode_boxes(regression.clone(), shapes.as_ref());
        RawPredictions {
            regression,
            objectness_logits,
            class_logits,
            decoded_boxes,
            levels: core::array::from_fn(|index| FeatureLevelShape {
                height: shapes[index].0,
                width: shapes[index].1,
                stride: STRIDES[index],
            }),
        }
    }

    /// Decode bounding box absolute values from regression output offsets.
    fn decode_boxes(&self, outputs: Tensor<B, 3>, shapes: &[(usize, usize)]) -> Tensor<B, 3> {
        let device = outputs.device();
        let [b, num_anchors, _] = outputs.dims();

        let (grids, strides) = shapes
            .iter()
            .zip(STRIDES)
            .map(|((h, w), stride)| {
                // Grid (x, y) coordinates
                let num_anchors = w * h;
                let grid =
                    create_2d_grid::<B>(*w, *h, &device).reshape(Shape::new([1, num_anchors, 2]));
                let strides: Tensor<B, 3, Int> =
                    Tensor::full(Shape::new([1, num_anchors, 1]), stride as i64, &device);

                (grid, strides)
            })
            .unzip();

        let grids = Tensor::cat(grids, 1).float();
        let strides = Tensor::cat(strides, 1).float();

        Tensor::cat(
            vec![
                // Add grid offset to center coordinates and scale to image dimensions
                (outputs.clone().slice([0..b, 0..num_anchors, 0..2]) + grids) * strides.clone(),
                // Decode `log` encoded boxes with `exp`and scale to image dimensions
                outputs.clone().slice([0..b, 0..num_anchors, 2..4]).exp() * strides,
            ],
            2,
        )
    }
}

/// [YOLOX head](Head) configuration.
pub struct HeadConfig {
    stems: Vec<BaseConvConfig>,
    cls_convs: Vec<ConvBlockConfig>,
    reg_convs: Vec<ConvBlockConfig>,
    cls_preds: Vec<Conv2dConfig>,
    reg_preds: Vec<Conv2dConfig>,
    obj_preds: Vec<Conv2dConfig>,
    num_classes: usize,
}

impl HeadConfig {
    /// Create a new instance of the YOLOX head [config](HeadConfig).
    pub fn new(num_classes: usize, width: f64, depthwise: bool) -> Self {
        let hidden_channels: usize = 256;
        // Initialize conv2d biases for classification and objectness heads
        let bias = -f64::ln((1.0 - PRIOR_PROB) / PRIOR_PROB);

        let (stems, cls_convs, reg_convs, cls_preds, reg_preds, obj_preds) =
            multiunzip(IN_CHANNELS.into_iter().map(|in_channels| {
                let stem = BaseConvConfig::new(
                    expand(in_channels, width),
                    expand(hidden_channels, width),
                    1,
                    1,
                    1,
                );

                let cls_conv =
                    ConvBlockConfig::new(expand(hidden_channels, width), 3, 1, depthwise);
                let reg_conv =
                    ConvBlockConfig::new(expand(hidden_channels, width), 3, 1, depthwise);

                let cls_pred =
                    Conv2dConfig::new([expand(hidden_channels, width), num_classes], [1, 1])
                        .with_padding(PaddingConfig2d::Explicit(0, 0, 0, 0))
                        .with_initializer(Initializer::Constant { value: bias });
                let reg_pred = Conv2dConfig::new([expand(hidden_channels, width), 4], [1, 1])
                    .with_padding(PaddingConfig2d::Explicit(0, 0, 0, 0));
                let obj_pred = Conv2dConfig::new([expand(hidden_channels, width), 1], [1, 1])
                    .with_padding(PaddingConfig2d::Explicit(0, 0, 0, 0))
                    .with_initializer(Initializer::Constant { value: bias });

                (stem, cls_conv, reg_conv, cls_pred, reg_pred, obj_pred)
            }));

        Self {
            stems,
            cls_convs,
            reg_convs,
            cls_preds,
            reg_preds,
            obj_preds,
            num_classes,
        }
    }

    /// Initialize a new [YOLOX head](Head) module.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Head<B> {
        Head {
            stems: self.stems.iter().map(|m| m.init(device)).collect(),
            cls_convs: self.cls_convs.iter().map(|m| m.init(device)).collect(),
            reg_convs: self.reg_convs.iter().map(|m| m.init(device)).collect(),
            cls_preds: self.cls_preds.iter().map(|m| m.init(device)).collect(),
            reg_preds: self.reg_preds.iter().map(|m| m.init(device)).collect(),
            obj_preds: self.obj_preds.iter().map(|m| m.init(device)).collect(),
            num_classes: self.num_classes,
        }
    }
}
