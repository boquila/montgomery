use serde::{Deserialize, Serialize};

use burn::tensor::backend::AutodiffBackend;

use crate::{
    ModelId,
    models::{
        yolo11::{
            Yolo11ClsL, Yolo11ClsM, Yolo11ClsN, Yolo11ClsS, Yolo11ClsX, Yolo11L, Yolo11M, Yolo11N,
            Yolo11S, Yolo11X,
        },
        yolo26::{Yolo26ClsL, Yolo26ClsM, Yolo26ClsN, Yolo26ClsS, Yolo26ClsX},
        yolov3_tiny::Yolov3Tiny,
        yolov8::{Yolov8ClsL, Yolov8ClsM, Yolov8ClsN, Yolov8ClsS, Yolov8ClsX},
        yolox::Yolox,
    },
    training::{
        assign::{simota::GroundTruth, tal::TalGroundTruth},
        data::batch::{ClassificationBatch, DetectionBatch},
        engine::TrainableTask,
        geometry::{BoxXyxy, FeatureLevelLayout},
        loss::{classification, ultralytics_detect, yolox},
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
    for image in 0..images {
        for target in 0..max_targets {
            let flat = image * max_targets + target;
            if valid[flat] {
                let class_id =
                    usize::try_from(classes[flat]).map_err(|_| "negative target class")?;
                output[image].push(TalGroundTruth {
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
            false,
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
}
