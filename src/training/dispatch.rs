use serde::{Deserialize, Serialize};

use burn::tensor::{
    backend::AutodiffBackend,
    module::interpolate,
    ops::{InterpolateMode, InterpolateOptions},
};

use crate::{
    ModelId,
    models::{
        yolo11::{
            Yolo11ClsL, Yolo11ClsM, Yolo11ClsN, Yolo11ClsS, Yolo11ClsX, Yolo11L, Yolo11M, Yolo11N,
            Yolo11S, Yolo11SegL, Yolo11SegM, Yolo11SegN, Yolo11SegS, Yolo11SegX, Yolo11X,
        },
        yolo26::{
            Yolo26ClsL, Yolo26ClsM, Yolo26ClsN, Yolo26ClsS, Yolo26ClsX, Yolo26L, Yolo26M, Yolo26N,
            Yolo26S, Yolo26SegL, Yolo26SegM, Yolo26SegN, Yolo26SegS, Yolo26SegX, Yolo26X,
        },
        yolov3_tiny::Yolov3Tiny,
        yolov8::{Yolov8ClsL, Yolov8ClsM, Yolov8ClsN, Yolov8ClsS, Yolov8ClsX},
        yolov10::{Yolov10B, Yolov10L, Yolov10M, Yolov10N, Yolov10S, Yolov10X},
        yolox::Yolox,
    },
    training::{
        assign::{simota::GroundTruth, tal::TalGroundTruth},
        data::batch::{ClassificationBatch, DetectionBatch, SegmentationBatch},
        engine::{LossContext, TrainableTask},
        geometry::{BoxXyxy, FeatureLevelLayout},
        loss::{classification, segmentation, ultralytics_detect, yolox},
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LossFamily {
    YoloxSimOta,
    TalDfl,
    TalDflSegment,
    Yolov10DualDfl,
    Yolo26DualDirect,
    Yolo26DualSegment,
    Classification,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainingRecipe {
    pub loss: LossFamily,
    pub levels: usize,
    pub reg_max: usize,
    pub end_to_end: bool,
}

/// Resolve architecture semantics once before entering a monomorphized family-specific loop.
pub const fn recipe_for(model: ModelId) -> TrainingRecipe {
    use ModelId::*;
    match model {
        YoloxNano | YoloxTiny | YoloxS | YoloxM | YoloxL | YoloxX => TrainingRecipe {
            loss: LossFamily::YoloxSimOta,
            levels: 3,
            reg_max: 1,
            end_to_end: false,
        },
        Yolov3TinyU => TrainingRecipe {
            loss: LossFamily::TalDfl,
            levels: 2,
            reg_max: 16,
            end_to_end: false,
        },
        Yolov10N | Yolov10S | Yolov10M | Yolov10B | Yolov10L | Yolov10X => TrainingRecipe {
            loss: LossFamily::Yolov10DualDfl,
            levels: 3,
            reg_max: 16,
            end_to_end: true,
        },
        Yolo11N | Yolo11S | Yolo11M | Yolo11L | Yolo11X | Yolov8N | Yolov8S | Yolov8M | Yolov8L
        | Yolov8X | Yolo12N | Yolo12S | Yolo12M | Yolo12L | Yolo12X => TrainingRecipe {
            loss: LossFamily::TalDfl,
            levels: 3,
            reg_max: 16,
            end_to_end: false,
        },
        Yolo11NSeg | Yolo11SSeg | Yolo11MSeg | Yolo11LSeg | Yolo11XSeg | Yolov8NSeg
        | Yolov8SSeg | Yolov8MSeg | Yolov8LSeg | Yolov8XSeg => TrainingRecipe {
            loss: LossFamily::TalDflSegment,
            levels: 3,
            reg_max: 16,
            end_to_end: false,
        },
        Yolo26N | Yolo26S | Yolo26M | Yolo26L | Yolo26X => TrainingRecipe {
            loss: LossFamily::Yolo26DualDirect,
            levels: 3,
            reg_max: 1,
            end_to_end: true,
        },
        Yolo26NSeg | Yolo26SSeg | Yolo26MSeg | Yolo26LSeg | Yolo26XSeg => TrainingRecipe {
            loss: LossFamily::Yolo26DualSegment,
            levels: 3,
            reg_max: 1,
            end_to_end: true,
        },
        Yolo11NCls | Yolo11SCls | Yolo11MCls | Yolo11LCls | Yolo11XCls | Yolov8NCls
        | Yolov8SCls | Yolov8MCls | Yolov8LCls | Yolov8XCls | Yolo26NCls | Yolo26SCls
        | Yolo26MCls | Yolo26LCls | Yolo26XCls => TrainingRecipe {
            loss: LossFamily::Classification,
            levels: 0,
            reg_max: 0,
            end_to_end: false,
        },
    }
}

fn detection_targets<B: burn::tensor::backend::Backend>(
    batch: &DetectionBatch<B>,
) -> Result<Vec<Vec<TalGroundTruth>>, String> {
    let [images, max_targets] = batch.classes.dims();
    let classes_data = batch.classes.clone().into_data();
    let boxes_data = batch.boxes_xyxy.clone().into_data();
    let valid_data = batch.valid.clone().into_data();
    let classes = classes_data
        .as_slice::<i64>()
        .map_err(|_| "target classes are not i64")?;
    let boxes = boxes_data
        .as_slice::<f32>()
        .map_err(|_| "target boxes are not f32")?;
    let valid = valid_data
        .as_slice::<bool>()
        .map_err(|_| "target validity is not bool")?;
    let mut output = vec![Vec::new(); images];
    for (image, image_output) in output.iter_mut().enumerate().take(images) {
        for target in 0..max_targets {
            let flat = image * max_targets + target;
            if valid[flat] {
                let class_id =
                    usize::try_from(classes[flat]).map_err(|_| "negative target class")?;
                image_output.push(TalGroundTruth {
                    class_id,
                    bbox: BoxXyxy::new(boxes[flat * 4..flat * 4 + 4].try_into().unwrap())
                        .map_err(str::to_string)?,
                });
            }
        }
    }
    Ok(output)
}

impl<B: AutodiffBackend> TrainableTask<B> for Yolox<B> {
    type Batch = DetectionBatch<B>;

    fn forward_loss(
        &self,
        batch: &Self::Batch,
        context: LossContext,
    ) -> Result<crate::training::loss::common::LossOutput<B>, String> {
        let tal = detection_targets(batch)?;
        let targets = tal
            .into_iter()
            .map(|items| {
                items
                    .into_iter()
                    .map(|item| GroundTruth {
                        class_id: item.class_id,
                        bbox: item.bbox,
                    })
                    .collect()
            })
            .collect::<Vec<_>>();
        yolox::tensor_loss(
            self.forward_train(batch.images.clone() * 255.0),
            &targets,
            context.yolox_l1,
        )
        .map_err(str::to_string)
    }
}

macro_rules! classification_task {
    ($($model:ty),+ $(,)?) => {$ (
        impl<B: AutodiffBackend> TrainableTask<B> for $model {
            type Batch = ClassificationBatch<B>;
            fn forward_loss(
                &self,
                batch: &Self::Batch,
                _context: LossContext,
            ) -> Result<crate::training::loss::common::LossOutput<B>, String> {
                classification::tensor_loss(
                    self.forward_train(batch.images.clone()),
                    batch.classes.clone(),
                )
                .map_err(str::to_string)
            }
        }
    )+ };
}

classification_task!(
    Yolo11ClsN<B>,
    Yolo11ClsS<B>,
    Yolo11ClsM<B>,
    Yolo11ClsL<B>,
    Yolo11ClsX<B>,
    Yolo26ClsN<B>,
    Yolo26ClsS<B>,
    Yolo26ClsM<B>,
    Yolo26ClsL<B>,
    Yolo26ClsX<B>,
    Yolov8ClsN<B>,
    Yolov8ClsS<B>,
    Yolov8ClsM<B>,
    Yolov8ClsL<B>,
    Yolov8ClsX<B>,
);

impl<B: AutodiffBackend> TrainableTask<B> for Yolov3Tiny<B> {
    type Batch = DetectionBatch<B>;

    fn forward_loss(
        &self,
        batch: &Self::Batch,
        _context: LossContext,
    ) -> Result<crate::training::loss::common::LossOutput<B>, String> {
        let [_, _, height, width] = batch.images.dims();
        let output = self.forward_train(batch.images.clone());
        ultralytics_detect::tensor_loss(
            output.boxes,
            output.scores,
            &[
                FeatureLevelLayout {
                    height: height / 16,
                    width: width / 16,
                    stride: 16,
                },
                FeatureLevelLayout {
                    height: height / 32,
                    width: width / 32,
                    stride: 32,
                },
            ],
            &detection_targets(batch)?,
            ultralytics_detect::DetectionLossConfig::dfl([height, width], 10),
        )
        .map_err(str::to_string)
    }
}

macro_rules! yolo11_detect_task {
    ($($model:ty),+ $(,)?) => {$ (
        impl<B: AutodiffBackend> TrainableTask<B> for $model {
            type Batch = DetectionBatch<B>;
            fn forward_loss(
                &self,
                batch: &Self::Batch,
                _context: LossContext,
            ) -> Result<crate::training::loss::common::LossOutput<B>, String> {
                let [_, _, height, width] = batch.images.dims();
                let output = self.forward_train(batch.images.clone());
                ultralytics_detect::tensor_loss(
                    output.boxes,
                    output.scores,
                    &[
                        FeatureLevelLayout { height: height / 8, width: width / 8, stride: 8 },
                        FeatureLevelLayout { height: height / 16, width: width / 16, stride: 16 },
                        FeatureLevelLayout { height: height / 32, width: width / 32, stride: 32 },
                    ],
                    &detection_targets(batch)?,
                    ultralytics_detect::DetectionLossConfig::dfl([height, width], 10),
                )
                .map_err(str::to_string)
            }
        }
    )+ };
}

yolo11_detect_task!(Yolo11N<B>, Yolo11S<B>, Yolo11M<B>, Yolo11L<B>, Yolo11X<B>);

fn combine_dual<B: burn::tensor::backend::Backend>(
    one_to_many: crate::training::loss::common::LossOutput<B>,
    one_to_one: crate::training::loss::common::LossOutput<B>,
    weights: [f32; 2],
) -> crate::training::loss::common::LossOutput<B> {
    let mut components = std::collections::BTreeMap::new();
    for (name, value) in one_to_many.components {
        components.insert(format!("one_to_many_{name}"), value);
    }
    for (name, value) in one_to_one.components {
        components.insert(format!("one_to_one_{name}"), value);
    }
    let total = one_to_many.total * weights[0] as f64 + one_to_one.total * weights[1] as f64;
    let finite = one_to_many.finite
        && one_to_one.finite
        && crate::training::loss::common::scalar_value(total.clone()).is_finite();
    crate::training::loss::common::LossOutput {
        total,
        components,
        targets: one_to_many.targets.max(one_to_one.targets),
        foreground: one_to_many.foreground + one_to_one.foreground,
        finite,
    }
}

macro_rules! dual_detect_task {
    ($config:ident, $forward:literal; $($model:ty),+ $(,)?) => {$ (
        impl<B: AutodiffBackend> TrainableTask<B> for $model {
            type Batch = DetectionBatch<B>;

            fn forward_loss(
                &self,
                batch: &Self::Batch,
                context: LossContext,
            ) -> Result<crate::training::loss::common::LossOutput<B>, String> {
                let [_, _, height, width] = batch.images.dims();
                let output = self.forward_train_dual(batch.images.clone());
                let levels = [
                    FeatureLevelLayout { height: height / 8, width: width / 8, stride: 8 },
                    FeatureLevelLayout { height: height / 16, width: width / 16, stride: 16 },
                    FeatureLevelLayout { height: height / 32, width: width / 32, stride: 32 },
                ];
                let targets = detection_targets(batch)?;
                let one_to_many = ultralytics_detect::tensor_loss(
                    output.one_to_many.boxes,
                    output.one_to_many.scores,
                    &levels,
                    &targets,
                    ultralytics_detect::DetectionLossConfig::$config([height, width], 10),
                ).map_err(str::to_string)?;
                let one_to_one = ultralytics_detect::tensor_loss(
                    output.one_to_one.boxes,
                    output.one_to_one.scores,
                    &levels,
                    &targets,
                    ultralytics_detect::DetectionLossConfig::$config([height, width], $forward),
                ).map_err(str::to_string)?;
                Ok(combine_dual(
                    one_to_many,
                    one_to_one,
                    [context.one_to_many, context.one_to_one],
                ))
            }
        }
    )+ };
}

// YOLOv10 uses the historical equal-weight dual DFL loss with one-to-one top-k 1.
dual_detect_task!(dfl, 1; Yolov10N<B>, Yolov10S<B>, Yolov10M<B>, Yolov10B<B>, Yolov10L<B>, Yolov10X<B>);
// YOLO26 is DFL-free and follows the persisted epoch-decaying E2E weighting schedule.
dual_detect_task!(direct, 7; Yolo26N<B>, Yolo26S<B>, Yolo26M<B>, Yolo26L<B>, Yolo26X<B>);

macro_rules! yolo11_segment_task {
    ($($model:ty),+ $(,)?) => {$ (
        impl<B: AutodiffBackend> TrainableTask<B> for $model {
            type Batch = SegmentationBatch<B>;

            fn forward_loss(
                &self,
                batch: &Self::Batch,
                _context: LossContext,
            ) -> Result<crate::training::loss::common::LossOutput<B>, String> {
                let [_, _, height, width] = batch.detection.images.dims();
                let output = self.forward_train(batch.detection.images.clone());
                let (mut detection, matches) = ultralytics_detect::tensor_loss_with_matches(
                    output.detection.boxes,
                    output.detection.scores,
                    &[
                        FeatureLevelLayout { height: height / 8, width: width / 8, stride: 8 },
                        FeatureLevelLayout { height: height / 16, width: width / 16, stride: 16 },
                        FeatureLevelLayout { height: height / 32, width: width / 32, stride: 32 },
                    ],
                    &detection_targets(&batch.detection)?,
                    ultralytics_detect::DetectionLossConfig::dfl([height, width], 10),
                ).map_err(str::to_string)?;
                let mask = segmentation::instance_mask_loss(
                    output.coefficients,
                    output.prototypes,
                    batch.masks.clone(),
                    &matches,
                ).map_err(str::to_string)?;
                let mask_value = crate::training::loss::common::scalar_value(mask.clone());
                detection.total = detection.total + mask * 7.5;
                detection.components.insert("mask_loss".into(), mask_value);
                detection.finite &= mask_value.is_finite();
                Ok(detection)
            }
        }
    )+ };
}

yolo11_segment_task!(
    Yolo11SegN<B>,
    Yolo11SegS<B>,
    Yolo11SegM<B>,
    Yolo11SegL<B>,
    Yolo11SegX<B>,
);

macro_rules! yolo26_segment_task {
    ($($model:ty),+ $(,)?) => {$ (
        impl<B: AutodiffBackend> TrainableTask<B> for $model {
            type Batch = SegmentationBatch<B>;

            fn forward_loss(
                &self,
                batch: &Self::Batch,
                context: LossContext,
            ) -> Result<crate::training::loss::common::LossOutput<B>, String> {
                let [_, _, height, width] = batch.detection.images.dims();
                let output = self.forward_train(batch.detection.images.clone());
                let levels = [
                    FeatureLevelLayout { height: height / 8, width: width / 8, stride: 8 },
                    FeatureLevelLayout { height: height / 16, width: width / 16, stride: 16 },
                    FeatureLevelLayout { height: height / 32, width: width / 32, stride: 32 },
                ];
                let targets = detection_targets(&batch.detection)?;
                let (mut many, many_matches) = ultralytics_detect::tensor_loss_with_matches(
                    output.detection.one_to_many.boxes,
                    output.detection.one_to_many.scores,
                    &levels,
                    &targets,
                    ultralytics_detect::DetectionLossConfig::direct([height, width], 10),
                ).map_err(str::to_string)?;
                let (mut one, one_matches) = ultralytics_detect::tensor_loss_with_matches(
                    output.detection.one_to_one.boxes,
                    output.detection.one_to_one.scores,
                    &levels,
                    &targets,
                    ultralytics_detect::DetectionLossConfig::direct([height, width], 7),
                ).map_err(str::to_string)?;
                let many_mask = segmentation::instance_mask_loss(
                    output.one_to_many_coefficients,
                    output.one_to_many_prototypes,
                    batch.masks.clone(),
                    &many_matches,
                ).map_err(str::to_string)?;
                let one_mask = segmentation::instance_mask_loss(
                    output.one_to_one_coefficients,
                    output.one_to_one_prototypes,
                    batch.masks.clone(),
                    &one_matches,
                ).map_err(str::to_string)?;
                let [_, target_height, target_width] = batch.semantic_class_map.dims();
                let many_semantic = segmentation::semantic_bce_dice_loss(
                    interpolate(
                        output.one_to_many_semantic,
                        [target_height, target_width],
                        InterpolateOptions::new(InterpolateMode::Bilinear),
                    ),
                    batch.semantic_class_map.clone(),
                    batch.semantic_coverage.clone(),
                ).map_err(str::to_string)?;
                let one_semantic = segmentation::semantic_bce_dice_loss(
                    interpolate(
                        output.one_to_one_semantic,
                        [target_height, target_width],
                        InterpolateOptions::new(InterpolateMode::Bilinear),
                    ),
                    batch.semantic_class_map.clone(),
                    batch.semantic_coverage.clone(),
                ).map_err(str::to_string)?;
                let many_mask_value = crate::training::loss::common::scalar_value(many_mask.clone());
                let one_mask_value = crate::training::loss::common::scalar_value(one_mask.clone());
                let many_semantic_value = crate::training::loss::common::scalar_value(many_semantic.clone());
                let one_semantic_value = crate::training::loss::common::scalar_value(one_semantic.clone());
                many.total = many.total + many_mask * 7.5 + many_semantic;
                many.components.insert("mask_loss".into(), many_mask_value);
                many.components.insert("semantic_loss".into(), many_semantic_value);
                many.finite &= many_mask_value.is_finite() && many_semantic_value.is_finite();
                one.total = one.total + one_mask * 7.5 + one_semantic;
                one.components.insert("mask_loss".into(), one_mask_value);
                one.components.insert("semantic_loss".into(), one_semantic_value);
                one.finite &= one_mask_value.is_finite() && one_semantic_value.is_finite();
                Ok(combine_dual(
                    many,
                    one,
                    [context.one_to_many, context.one_to_one],
                ))
            }
        }
    )+ };
}

yolo26_segment_task!(
    Yolo26SegN<B>,
    Yolo26SegS<B>,
    Yolo26SegM<B>,
    Yolo26SegL<B>,
    Yolo26SegX<B>,
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_level_and_dfl_free_invariants_are_explicit() {
        assert_eq!(recipe_for(ModelId::Yolov3TinyU).levels, 2);
        assert_eq!(recipe_for(ModelId::Yolo26N).reg_max, 1);
        assert!(recipe_for(ModelId::Yolov10N).end_to_end);
        assert_eq!(recipe_for(ModelId::Yolo11NCls).levels, 0);
    }

    #[test]
    fn yolo26_one_to_one_features_are_detached_from_body() {
        use burn::{
            backend::Autodiff,
            module::{Module, ModuleVisitor, Param, ParamId},
            optim::GradientsParams,
            tensor::Tensor,
        };
        use burn_flex::Flex;

        struct Paths {
            body: Vec<ParamId>,
            stack: Vec<String>,
        }
        impl<B: burn::tensor::backend::Backend> ModuleVisitor<B> for Paths {
            fn enter_module(&mut self, name: &str, _container_type: &str) {
                self.stack.push(name.to_owned());
            }
            fn exit_module(&mut self, _name: &str, _container_type: &str) {
                self.stack.pop();
            }
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                if self.stack.iter().any(|part| part == "body") {
                    self.body.push(param.id);
                }
            }
        }

        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                type B = Autodiff<Flex>;
                let device = Default::default();
                let model = crate::models::yolo26::Yolo26NConfig.init::<B>(&device);
                let mut paths = Paths {
                    body: Vec::new(),
                    stack: Vec::new(),
                };
                model.visit(&mut paths);
                assert!(!paths.body.is_empty());

                let input = Tensor::ones([1, 3, 64, 64], &device);
                let output = model.forward_train_dual(input.clone());
                let mut gradients =
                    (output.one_to_one.boxes.mean() + output.one_to_one.scores.mean()).backward();
                let detached = GradientsParams::from_params(&mut gradients, &model, &paths.body);
                assert!(
                    detached.is_empty(),
                    "one-to-one loss reached body parameters"
                );

                let output = model.forward_train_dual(input);
                let mut gradients =
                    (output.one_to_many.boxes.mean() + output.one_to_many.scores.mean()).backward();
                let connected = GradientsParams::from_params(&mut gradients, &model, &paths.body);
                assert!(!connected.is_empty(), "one-to-many loss did not reach body");
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
