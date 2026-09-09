use crate::{
    ModelId,
    models::{
        yolo11::{
            Yolo11ClsL, Yolo11ClsM, Yolo11ClsN, Yolo11ClsS, Yolo11ClsX, Yolo11L, Yolo11M, Yolo11N,
            Yolo11S, Yolo11SegL, Yolo11SegM, Yolo11SegN, Yolo11SegS, Yolo11SegX, Yolo11X,
        },
        yolo12::{Yolo12L, Yolo12M, Yolo12N, Yolo12S, Yolo12X},
        yolo26::{
            Yolo26ClsL, Yolo26ClsM, Yolo26ClsN, Yolo26ClsS, Yolo26ClsX, Yolo26L, Yolo26M, Yolo26N,
            Yolo26S, Yolo26SegL, Yolo26SegM, Yolo26SegN, Yolo26SegS, Yolo26SegX, Yolo26X,
        },
        yolov3_tiny::Yolov3Tiny,
        yolov8::{
            Yolov8ClsL, Yolov8ClsM, Yolov8ClsN, Yolov8ClsS, Yolov8ClsX, Yolov8L, Yolov8M, Yolov8N,
            Yolov8S, Yolov8SegL, Yolov8SegM, Yolov8SegN, Yolov8SegS, Yolov8SegX, Yolov8X,
        },
        yolov10::{Yolov10B, Yolov10L, Yolov10M, Yolov10N, Yolov10S, Yolov10X},
        yolox::Yolox,
    },
    training::{
        assign::{simota::GroundTruth, tal::TalGroundTruth},
        data::batch::{ClassificationBatch, DetectionBatch, SegmentationBatch},
        engine::{LossContext, TrainableTask},
        geometry::FeatureLevelLayout,
        loss::{classification, segmentation, ultralytics_detect, yolox},
    },
};
use serde::{Deserialize, Serialize};

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

fn detection_targets(batch: &DetectionBatch) -> Result<&[Vec<TalGroundTruth>], String> {
    let images = batch.images.dims()[0];
    if batch.targets.len() != images {
        return Err("host target batch size differs from image batch".into());
    }
    Ok(&batch.targets)
}

impl TrainableTask for Yolox {
    type Batch = DetectionBatch;

    fn forward_loss(
        &self,
        batch: &Self::Batch,
        context: LossContext,
    ) -> Result<crate::training::loss::common::LossOutput, String> {
        let tal = detection_targets(batch)?;
        let targets = tal
            .iter()
            .map(|items| {
                items
                    .iter()
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
        impl TrainableTask for $model {
            type Batch = ClassificationBatch;
            fn forward_loss(
                &self,
                batch: &Self::Batch,
                _context: LossContext,
            ) -> Result<crate::training::loss::common::LossOutput, String> {
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
    Yolo11ClsN, Yolo11ClsS, Yolo11ClsM, Yolo11ClsL, Yolo11ClsX, Yolo26ClsN, Yolo26ClsS, Yolo26ClsM,
    Yolo26ClsL, Yolo26ClsX, Yolov8ClsN, Yolov8ClsS, Yolov8ClsM, Yolov8ClsL, Yolov8ClsX,
);

impl TrainableTask for Yolov3Tiny {
    type Batch = DetectionBatch;

    fn forward_loss(
        &self,
        batch: &Self::Batch,
        _context: LossContext,
    ) -> Result<crate::training::loss::common::LossOutput, String> {
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
            detection_targets(batch)?,
            ultralytics_detect::DetectionLossConfig::dfl([height, width], 10),
        )
        .map_err(str::to_string)
    }
}

macro_rules! yolo11_detect_task {
    ($($model:ty),+ $(,)?) => {$ (
        impl TrainableTask for $model {
            type Batch = DetectionBatch;
            fn forward_loss(
                &self,
                batch: &Self::Batch,
                _context: LossContext,
            ) -> Result<crate::training::loss::common::LossOutput, String> {
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
                    detection_targets(batch)?,
                    ultralytics_detect::DetectionLossConfig::dfl([height, width], 10),
                )
                .map_err(str::to_string)
            }
        }
    )+ };
}

yolo11_detect_task!(
    Yolo11N, Yolo11S, Yolo11M, Yolo11L, Yolo11X, Yolov8N, Yolov8S, Yolov8M, Yolov8L, Yolov8X,
    Yolo12N, Yolo12S, Yolo12M, Yolo12L, Yolo12X,
);

fn combine_dual(
    mut one_to_many: crate::training::loss::common::LossOutput,
    mut one_to_one: crate::training::loss::common::LossOutput,
    weights: [f32; 2],
) -> crate::training::loss::common::LossOutput {
    let has_deferred_total = one_to_many.has_deferred_total() || one_to_one.has_deferred_total();
    let mut components = std::collections::BTreeMap::new();
    for (name, value) in one_to_many.components {
        components.insert(format!("one_to_many_{name}"), value);
    }
    for (name, value) in one_to_one.components {
        components.insert(format!("one_to_one_{name}"), value);
    }
    let total_value = one_to_many.total_value * weights[0] + one_to_one.total_value * weights[1];
    let total = one_to_many.total * weights[0] as f64 + one_to_one.total * weights[1] as f64;
    let mut deferred = Vec::new();
    for mut value in one_to_many.deferred.drain(..) {
        if !value.total {
            value.component = value.component.map(|name| format!("one_to_many_{name}"));
            deferred.push(value);
        }
    }
    for mut value in one_to_one.deferred.drain(..) {
        if !value.total {
            value.component = value.component.map(|name| format!("one_to_one_{name}"));
            deferred.push(value);
        }
    }
    if has_deferred_total {
        deferred.push(crate::training::loss::common::DeferredScalar::total(
            total.clone(),
        ));
    }
    let finite =
        one_to_many.finite && one_to_one.finite && (has_deferred_total || total_value.is_finite());
    crate::training::loss::common::LossOutput {
        total,
        total_value,
        deferred,
        components,
        targets: one_to_many.targets.max(one_to_one.targets),
        foreground: one_to_many.foreground + one_to_one.foreground,
        finite,
    }
}

macro_rules! dual_detect_task {
    ($config:ident, $forward:literal; $($model:ty),+ $(,)?) => {$ (
        impl TrainableTask for $model {
            type Batch = DetectionBatch;

            fn forward_loss(
                &self,
                batch: &Self::Batch,
                context: LossContext,
            ) -> Result<crate::training::loss::common::LossOutput, String> {
                let [_, _, height, width] = batch.images.dims();
                let output = self.forward_train_dual(batch.images.clone());
                let levels = [
                    FeatureLevelLayout { height: height / 8, width: width / 8, stride: 8 },
                    FeatureLevelLayout { height: height / 16, width: width / 16, stride: 16 },
                    FeatureLevelLayout { height: height / 32, width: width / 32, stride: 32 },
                ];
                let targets = detection_targets(batch)?;
                let (one_to_many, one_to_one) = ultralytics_detect::tensor_dual_loss(
                    (
                        output.one_to_many.boxes,
                        output.one_to_many.scores,
                        ultralytics_detect::DetectionLossConfig::$config([height, width], 10),
                    ),
                    (
                        output.one_to_one.boxes,
                        output.one_to_one.scores,
                        ultralytics_detect::DetectionLossConfig::$config([height, width], $forward),
                    ),
                    &levels,
                    &targets,
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

// YOLOv10 uses equal-weight dual DFL loss with one-to-one top-k 1.
dual_detect_task!(dfl, 1; Yolov10N, Yolov10S, Yolov10M, Yolov10B, Yolov10L, Yolov10X);
// YOLO26 is DFL-free and follows the persisted epoch-decaying E2E weighting schedule.
dual_detect_task!(direct, 7; Yolo26N, Yolo26S, Yolo26M, Yolo26L, Yolo26X);

macro_rules! yolo11_segment_task {
    ($($model:ty),+ $(,)?) => {$ (
        impl TrainableTask for $model {
            type Batch = SegmentationBatch;

            fn forward_loss(
                &self,
                batch: &Self::Batch,
                _context: LossContext,
            ) -> Result<crate::training::loss::common::LossOutput, String> {
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
                    detection_targets(&batch.detection)?,
                    ultralytics_detect::DetectionLossConfig::dfl([height, width], 10),
                ).map_err(str::to_string)?;
                let mask = segmentation::instance_mask_loss(
                    output.coefficients,
                    output.prototypes,
                    batch.masks.clone(),
                    &matches,
                ).map_err(str::to_string)?;
                detection.total =
                    detection.total + mask.clone() * segmentation::SEGMENTATION_GAIN;
                detection.defer_component("mask_loss", mask);
                detection.replace_deferred_total();
                Ok(detection)
            }
        }
    )+ };
}

yolo11_segment_task!(
    Yolo11SegN, Yolo11SegS, Yolo11SegM, Yolo11SegL, Yolo11SegX, Yolov8SegN, Yolov8SegS, Yolov8SegM,
    Yolov8SegL, Yolov8SegX,
);

macro_rules! yolo26_segment_task {
    ($($model:ty),+ $(,)?) => {$ (
        impl TrainableTask for $model {
            type Batch = SegmentationBatch;

            fn forward_loss(
                &self,
                batch: &Self::Batch,
                context: LossContext,
            ) -> Result<crate::training::loss::common::LossOutput, String> {
                let [_, _, height, width] = batch.detection.images.dims();
                let output = self.forward_train(batch.detection.images.clone());
                let levels = [
                    FeatureLevelLayout { height: height / 8, width: width / 8, stride: 8 },
                    FeatureLevelLayout { height: height / 16, width: width / 16, stride: 16 },
                    FeatureLevelLayout { height: height / 32, width: width / 32, stride: 32 },
                ];
                let targets = detection_targets(&batch.detection)?;
                let ((mut many, many_matches), (mut one, one_matches)) =
                    ultralytics_detect::tensor_dual_loss_with_matches(
                    (
                        output.detection.one_to_many.boxes,
                        output.detection.one_to_many.scores,
                        ultralytics_detect::DetectionLossConfig::direct([height, width], 10),
                    ),
                    (
                        output.detection.one_to_one.boxes,
                        output.detection.one_to_one.scores,
                        ultralytics_detect::DetectionLossConfig::direct([height, width], 7),
                    ),
                    &levels,
                    &targets,
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
                let (many_semantic, one_semantic) = segmentation::dual_semantic_bce_dice_loss(
                    segmentation::bilinear_upsample_2x(output.one_to_many_semantic),
                    batch.semantic_class_map.clone(),
                    batch.semantic_coverage.clone(),
                ).map_err(str::to_string)?;
                many.total = many.total
                    + (many_mask.clone() + many_semantic.clone())
                        * segmentation::SEGMENTATION_GAIN;
                many.defer_component("mask_loss", many_mask);
                many.defer_component("semantic_loss", many_semantic);
                many.replace_deferred_total();
                one.total = one.total
                    + (one_mask.clone() + one_semantic.clone())
                        * segmentation::SEGMENTATION_GAIN;
                one.defer_component("mask_loss", one_mask);
                one.defer_component("semantic_loss", one_semantic);
                one.replace_deferred_total();
                Ok(combine_dual(
                    many,
                    one,
                    [context.one_to_many, context.one_to_one],
                ))
            }
        }
    )+ };
}

yolo26_segment_task!(Yolo26SegN, Yolo26SegS, Yolo26SegM, Yolo26SegL, Yolo26SegX,);

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
    fn detached_targets_use_the_host_batch() {
        use crate::training::geometry::BoxXyxy;
        use burn::tensor::Tensor;

        let device = Default::default();
        let batch = DetectionBatch {
            images: Tensor::zeros([1, 3, 8, 8], &device),
            targets: vec![vec![TalGroundTruth {
                class_id: 7,
                bbox: BoxXyxy::new([1.0, 1.0, 7.0, 7.0]).unwrap(),
            }]],
            metadata: Vec::new(),
        };

        let targets = detection_targets(&batch).unwrap();
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].len(), 1);
        assert_eq!(targets[0][0].class_id, 7);
    }

    #[test]
    fn yolo26_one_to_one_features_are_detached_from_body() {
        use burn::{
            module::{Module, ModuleVisitor, Param, ParamId},
            optim::GradientsParams,
            tensor::Tensor,
        };

        struct Paths {
            body: Vec<ParamId>,
            stack: Vec<String>,
        }
        impl ModuleVisitor for Paths {
            fn enter_module(&mut self, name: &str, _container_type: &str) {
                self.stack.push(name.to_owned());
            }
            fn exit_module(&mut self, _name: &str, _container_type: &str) {
                self.stack.pop();
            }
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
                if self.stack.iter().any(|part| part == "body") {
                    self.body.push(param.id);
                }
            }
        }

        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let device = burn::tensor::Device::default().autodiff();
                let model = crate::models::yolo26::Yolo26NConfig.init(&device);
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
