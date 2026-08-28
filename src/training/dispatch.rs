use serde::{Deserialize, Serialize};

use crate::ModelId;

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
