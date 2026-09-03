//! Native Burn implementation of the Ultralytics YOLO26-seg instance-segmentation family
//! (n/s/m/l/x).
//!
//! The segmentation graph shares the complete detection body (layers 0-22) and one2one detection
//! head with the detect family; Ultralytics' `Segment26` head adds mask-coefficient towers
//! (`cv4`/`one2one_cv4`, full 3x3 Conv towers like the classic Segment head) and `Proto26`.
//! Unlike YOLO11-seg's P3-only Proto, `Proto26` fuses all three feature levels: P4/P5 are 1x1
//! refined, nearest-upsampled by 2x/4x, and summed onto P3 before the classic
//! conv/upsample/conv/proto projection at stride 4.
//!
//! Because YOLO26 is end-to-end (`end2end = True`), the head output rows are already top-300
//! selected with the raw mask coefficients gathered along — the runtime applies the same
//! score filter and no NMS, then assembles masks exactly like `ops.process_mask(upsample=True)`.

use burn::{
    module::Module,
    nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
    tensor::{Device, Tensor, backend::Backend},
};

#[cfg(feature = "pretrained")]
use burn_store::{
    BurnpackError, BurnpackStore, HalfPrecisionAdapter, ModuleSnapshot, PytorchStore,
    PytorchStoreError,
};

use super::blocks::{Conv, ConvConfig};
use super::body::{Yolo26BodyLarge, Yolo26BodySmall};
#[cfg(feature = "training")]
use super::head::DualRawPredictions;
use super::head::{DecodedPredictions, Yolo26Head, Yolo26HeadConfig};

#[cfg(feature = "pretrained")]
use super::weights;

/// Number of mask prototypes and per-detection mask coefficients (`nm`).
pub const NUM_MASKS: usize = 32;

/// `Proto26`: multi-scale prototype generator.
///
/// Field names deliberately match the official `model.23.proto.*` checkpoint keys after
/// remapping. The semantic tower exists only in training builds and is therefore absent from the
/// default inference graph.
#[derive(Module, Debug)]
pub struct Proto26<B: Backend> {
    cv1: Conv<B>,
    upsample: ConvTranspose2d<B>,
    cv2: Conv<B>,
    cv3: Conv<B>,
    feat_refine_0: Conv<B>,
    feat_refine_1: Conv<B>,
    feat_fuse: Conv<B>,
    #[cfg(feature = "training")]
    sem_0: Conv<B>,
    #[cfg(feature = "training")]
    sem_1: Conv<B>,
    #[cfg(feature = "training")]
    sem_out: Conv2d<B>,
}

impl<B: Backend> Proto26<B> {
    /// Fuse the P4/P5 features onto P3 and project to the prototype maps at stride 4.
    pub fn forward(&self, p3: Tensor<B, 4>, p4: Tensor<B, 4>, p5: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward_fused(p3, p4, p5).0
    }

    fn forward_fused(
        &self,
        p3: Tensor<B, 4>,
        p4: Tensor<B, 4>,
        p5: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let feat = p3 + super::blocks::upsample_nearest_2x(self.feat_refine_0.forward(p4));
        let feat = feat
            + super::blocks::upsample_nearest_2x(super::blocks::upsample_nearest_2x(
                self.feat_refine_1.forward(p5),
            ));
        let x = self.cv1.forward(self.feat_fuse.forward(feat.clone()));
        let x = self.upsample.forward(x);
        let prototypes = self.cv3.forward(self.cv2.forward(x));
        #[cfg(feature = "training")]
        let semantic = self
            .sem_out
            .forward(self.sem_1.forward(self.sem_0.forward(feat)));
        #[cfg(not(feature = "training"))]
        let semantic = prototypes.clone().slice([
            0..prototypes.dims()[0],
            0..1,
            0..prototypes.dims()[2],
            0..prototypes.dims()[3],
        ]) * 0.0;
        (prototypes, semantic)
    }
}

#[derive(Debug)]
struct Proto26Config {
    p3_channels: usize,
    p4_channels: usize,
    p5_channels: usize,
    hidden_channels: usize,
    #[cfg(feature = "training")]
    num_classes: usize,
}

impl Proto26Config {
    fn init<B: Backend>(&self, device: &Device<B>) -> Proto26<B> {
        Proto26 {
            cv1: ConvConfig::new(self.hidden_channels, self.hidden_channels, 3, 1).init(device),
            upsample: ConvTranspose2dConfig::new(
                [self.hidden_channels, self.hidden_channels],
                [2, 2],
            )
            .with_stride([2, 2])
            .init(device),
            cv2: ConvConfig::new(self.hidden_channels, self.hidden_channels, 3, 1).init(device),
            cv3: ConvConfig::new(self.hidden_channels, NUM_MASKS, 1, 1).init(device),
            feat_refine_0: ConvConfig::new(self.p4_channels, self.p3_channels, 1, 1).init(device),
            feat_refine_1: ConvConfig::new(self.p5_channels, self.p3_channels, 1, 1).init(device),
            feat_fuse: ConvConfig::new(self.p3_channels, self.hidden_channels, 3, 1).init(device),
            #[cfg(feature = "training")]
            sem_0: ConvConfig::new(self.p3_channels, self.hidden_channels, 3, 1).init(device),
            #[cfg(feature = "training")]
            sem_1: ConvConfig::new(self.hidden_channels, self.hidden_channels, 3, 1).init(device),
            #[cfg(feature = "training")]
            sem_out: Conv2dConfig::new([self.hidden_channels, self.num_classes], [1, 1])
                .with_bias(true)
                .init(device),
        }
    }
}

/// One mask-coefficient scale of the YOLO26 `Segment26` head (`one2one_cv4`).
///
/// Like the classic Segment head, the mask tower is built from full 3x3 `Conv` layers (not the
/// light DWConv classification flavor). Field names match the official `one2one_cv4` checkpoint
/// keys after remapping.
#[derive(Module, Debug)]
struct MaskBranch<B: Backend> {
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

/// YOLO26-seg model output: the shared end-to-end detection decode plus the mask tensors.
pub struct SegmentOutput<B: Backend> {
    /// Decoded end-to-end detection predictions in model-input space.
    pub decoded: DecodedPredictions<B>,
    /// Raw (unnormalized) mask coefficients, `[batch, num_masks, anchors]`.
    pub coefficients: Tensor<B, 3>,
    /// Mask prototypes, `[batch, num_masks, proto_height, proto_width]` at stride 4.
    pub prototypes: Tensor<B, 4>,
}

#[cfg(feature = "training")]
pub struct DualSegmentTrainOutput<B: Backend> {
    pub detection: DualRawPredictions<B>,
    pub one_to_many_coefficients: Tensor<B, 3>,
    pub one_to_one_coefficients: Tensor<B, 3>,
    pub one_to_many_prototypes: Tensor<B, 4>,
    pub one_to_one_prototypes: Tensor<B, 4>,
    pub one_to_many_semantic: Tensor<B, 4>,
    pub one_to_one_semantic: Tensor<B, 4>,
}

/// Ultralytics YOLO26 `Segment26` head: the shared end-to-end detect head plus the Proto26 module
/// and one one2one mask-coefficient branch per scale.
#[derive(Module, Debug)]
pub struct Yolo26SegHead<B: Backend> {
    detect: Yolo26Head<B>,
    proto: Proto26<B>,
    p3_mask: MaskBranch<B>,
    p4_mask: MaskBranch<B>,
    p5_mask: MaskBranch<B>,
    #[cfg(feature = "training")]
    o2m_p3_mask: MaskBranch<B>,
    #[cfg(feature = "training")]
    o2m_p4_mask: MaskBranch<B>,
    #[cfg(feature = "training")]
    o2m_p5_mask: MaskBranch<B>,
}

impl<B: Backend> Yolo26SegHead<B> {
    #[cfg(feature = "training")]
    pub fn forward_train(
        &self,
        features: super::body::Yolo26Features<B>,
    ) -> DualSegmentTrainOutput<B> {
        let super::body::Yolo26Features { p3, p4, p5 } = features;
        let detection = self.detect.forward_dual(super::body::Yolo26Features {
            p3: p3.clone(),
            p4: p4.clone(),
            p5: p5.clone(),
        });
        let one_to_many_coefficients = Tensor::cat(
            vec![
                self.o2m_p3_mask.forward(p3.clone()),
                self.o2m_p4_mask.forward(p4.clone()),
                self.o2m_p5_mask.forward(p5.clone()),
            ],
            2,
        );
        let one_to_one_coefficients = Tensor::cat(
            vec![
                self.p3_mask.forward(p3.clone().detach()),
                self.p4_mask.forward(p4.clone().detach()),
                self.p5_mask.forward(p5.clone().detach()),
            ],
            2,
        );
        let (prototypes, semantic) = self.proto.forward_fused(p3, p4, p5);
        DualSegmentTrainOutput {
            detection,
            one_to_many_coefficients,
            one_to_one_coefficients,
            one_to_many_prototypes: prototypes.clone(),
            one_to_one_prototypes: prototypes.detach(),
            one_to_many_semantic: semantic.clone(),
            one_to_one_semantic: semantic.detach(),
        }
    }

    pub fn forward(&self, features: super::body::Yolo26Features<B>) -> SegmentOutput<B> {
        let coefficients = Tensor::cat(
            vec![
                self.p3_mask.forward(features.p3.clone()),
                self.p4_mask.forward(features.p4.clone()),
                self.p5_mask.forward(features.p5.clone()),
            ],
            2,
        );
        let prototypes = self.proto.forward(
            features.p3.clone(),
            features.p4.clone(),
            features.p5.clone(),
        );
        let decoded = self.detect.forward(super::body::Yolo26Features {
            p3: features.p3,
            p4: features.p4,
            p5: features.p5,
        });
        SegmentOutput {
            decoded,
            coefficients,
            prototypes,
        }
    }
}

#[derive(Debug)]
pub struct Yolo26SegHeadConfig {
    p3_channels: usize,
    p4_channels: usize,
    p5_channels: usize,
    proto_channels: usize,
    mask_channels: usize,
    num_classes: usize,
}

impl Yolo26SegHeadConfig {
    /// Declare the segment head for one scale.
    ///
    /// `parse_model` width-scales the prototype channels as
    /// `make_divisible(min(256, max_channels) * width, 8)` (64 at n, 128 at s, 256 at m/l, 384 at
    /// x) and builds the mask tower with hidden width `max(ch[0] / 4, nm)` (32 at n/s, 64 at m/l,
    /// 96 at x).
    pub fn new(
        p3_channels: usize,
        p4_channels: usize,
        p5_channels: usize,
        proto_channels: usize,
    ) -> Self {
        Self {
            p3_channels,
            p4_channels,
            p5_channels,
            proto_channels,
            mask_channels: (p3_channels / 4).max(NUM_MASKS),
            num_classes: 80,
        }
    }

    pub fn with_num_classes(mut self, num_classes: usize) -> Self {
        assert!(num_classes > 0, "class count must be positive");
        self.num_classes = num_classes;
        self
    }

    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo26SegHead<B> {
        Yolo26SegHead {
            detect: Yolo26HeadConfig::new(self.p3_channels, self.p4_channels, self.p5_channels)
                .with_num_classes(self.num_classes)
                .init(device),
            proto: Proto26Config {
                p3_channels: self.p3_channels,
                p4_channels: self.p4_channels,
                p5_channels: self.p5_channels,
                hidden_channels: self.proto_channels,
                #[cfg(feature = "training")]
                num_classes: self.num_classes,
            }
            .init(device),
            p3_mask: MaskBranchConfig {
                input_channels: self.p3_channels,
                mask_channels: self.mask_channels,
            }
            .init(device),
            p4_mask: MaskBranchConfig {
                input_channels: self.p4_channels,
                mask_channels: self.mask_channels,
            }
            .init(device),
            p5_mask: MaskBranchConfig {
                input_channels: self.p5_channels,
                mask_channels: self.mask_channels,
            }
            .init(device),
            #[cfg(feature = "training")]
            o2m_p3_mask: MaskBranchConfig {
                input_channels: self.p3_channels,
                mask_channels: self.mask_channels,
            }
            .init(device),
            #[cfg(feature = "training")]
            o2m_p4_mask: MaskBranchConfig {
                input_channels: self.p4_channels,
                mask_channels: self.mask_channels,
            }
            .init(device),
            #[cfg(feature = "training")]
            o2m_p5_mask: MaskBranchConfig {
                input_channels: self.p5_channels,
                mask_channels: self.mask_channels,
            }
            .init(device),
        }
    }
}

/// Build the PyTorch-state store shared by every YOLO26-seg scale variant.
///
/// The body is layers 0-22 (identical to the detect checkpoint), the head is model.23, and the
/// Training builds additionally remap the one-to-many detection/mask towers and `proto.semseg`.
#[cfg(feature = "pretrained")]
fn pytorch_store(path: impl Into<std::path::PathBuf>) -> PytorchStore {
    #[allow(unused_mut)]
    let mut store = PytorchStore::from_file(path)
        .with_top_level_key("model")
        .with_key_remapping("model\\.([0-9]|1[0-9]|2[0-2])\\.(.+)", "body.model_$1.$2")
        .with_key_remapping(
            "model\\.23\\.one2one_cv2\\.0\\.0\\.(.+)",
            "head.detect.p3.box_0.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv2\\.0\\.1\\.(.+)",
            "head.detect.p3.box_1.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv2\\.0\\.2\\.(.+)",
            "head.detect.p3.box_out.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv2\\.1\\.0\\.(.+)",
            "head.detect.p4.box_0.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv2\\.1\\.1\\.(.+)",
            "head.detect.p4.box_1.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv2\\.1\\.2\\.(.+)",
            "head.detect.p4.box_out.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv2\\.2\\.0\\.(.+)",
            "head.detect.p5.box_0.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv2\\.2\\.1\\.(.+)",
            "head.detect.p5.box_1.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv2\\.2\\.2\\.(.+)",
            "head.detect.p5.box_out.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv3\\.0\\.0\\.0\\.(.+)",
            "head.detect.p3.cls_dw_0.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv3\\.0\\.0\\.1\\.(.+)",
            "head.detect.p3.cls_pw_0.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv3\\.0\\.1\\.0\\.(.+)",
            "head.detect.p3.cls_dw_1.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv3\\.0\\.1\\.1\\.(.+)",
            "head.detect.p3.cls_pw_1.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv3\\.0\\.2\\.(.+)",
            "head.detect.p3.cls_out.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv3\\.1\\.0\\.0\\.(.+)",
            "head.detect.p4.cls_dw_0.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv3\\.1\\.0\\.1\\.(.+)",
            "head.detect.p4.cls_pw_0.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv3\\.1\\.1\\.0\\.(.+)",
            "head.detect.p4.cls_dw_1.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv3\\.1\\.1\\.1\\.(.+)",
            "head.detect.p4.cls_pw_1.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv3\\.1\\.2\\.(.+)",
            "head.detect.p4.cls_out.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv3\\.2\\.0\\.0\\.(.+)",
            "head.detect.p5.cls_dw_0.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv3\\.2\\.0\\.1\\.(.+)",
            "head.detect.p5.cls_pw_0.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv3\\.2\\.1\\.0\\.(.+)",
            "head.detect.p5.cls_dw_1.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv3\\.2\\.1\\.1\\.(.+)",
            "head.detect.p5.cls_pw_1.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv3\\.2\\.2\\.(.+)",
            "head.detect.p5.cls_out.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv4\\.0\\.0\\.(.+)",
            "head.p3_mask.mask_0.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv4\\.0\\.1\\.(.+)",
            "head.p3_mask.mask_1.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv4\\.0\\.2\\.(.+)",
            "head.p3_mask.mask_out.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv4\\.1\\.0\\.(.+)",
            "head.p4_mask.mask_0.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv4\\.1\\.1\\.(.+)",
            "head.p4_mask.mask_1.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv4\\.1\\.2\\.(.+)",
            "head.p4_mask.mask_out.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv4\\.2\\.0\\.(.+)",
            "head.p5_mask.mask_0.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv4\\.2\\.1\\.(.+)",
            "head.p5_mask.mask_1.$1",
        )
        .with_key_remapping(
            "model\\.23\\.one2one_cv4\\.2\\.2\\.(.+)",
            "head.p5_mask.mask_out.$1",
        )
        .with_key_remapping("model\\.23\\.proto\\.cv1\\.(.+)", "head.proto.cv1.$1")
        .with_key_remapping("model\\.23\\.proto\\.cv2\\.(.+)", "head.proto.cv2.$1")
        .with_key_remapping("model\\.23\\.proto\\.cv3\\.(.+)", "head.proto.cv3.$1")
        .with_key_remapping(
            "model\\.23\\.proto\\.upsample\\.(.+)",
            "head.proto.upsample.$1",
        )
        .with_key_remapping(
            "model\\.23\\.proto\\.feat_refine\\.0\\.(.+)",
            "head.proto.feat_refine_0.$1",
        )
        .with_key_remapping(
            "model\\.23\\.proto\\.feat_refine\\.1\\.(.+)",
            "head.proto.feat_refine_1.$1",
        )
        .with_key_remapping(
            "model\\.23\\.proto\\.feat_fuse\\.(.+)",
            "head.proto.feat_fuse.$1",
        );
    #[cfg(feature = "training")]
    {
        for (scale, branch) in [(0, "p3"), (1, "p4"), (2, "p5")] {
            for (layer, name) in [(0, "box_0"), (1, "box_1"), (2, "box_out")] {
                store = store.with_key_remapping(
                    format!(r"model\.23\.cv2\.{scale}\.{layer}\.(.+)"),
                    format!("head.detect.o2m_{branch}.{name}.$1"),
                );
            }
            for (path, name) in [
                ("0\\.0", "cls_dw_0"),
                ("0\\.1", "cls_pw_0"),
                ("1\\.0", "cls_dw_1"),
                ("1\\.1", "cls_pw_1"),
                ("2", "cls_out"),
            ] {
                store = store.with_key_remapping(
                    format!(r"model\.23\.cv3\.{scale}\.{path}\.(.+)"),
                    format!("head.detect.o2m_{branch}.{name}.$1"),
                );
            }
            for (layer, name) in [(0, "mask_0"), (1, "mask_1"), (2, "mask_out")] {
                store = store.with_key_remapping(
                    format!(r"model\.23\.cv4\.{scale}\.{layer}\.(.+)"),
                    format!("head.o2m_{branch}_mask.{name}.$1"),
                );
            }
        }
        store = store
            .with_key_remapping(
                "model\\.23\\.proto\\.semseg\\.0\\.(.+)",
                "head.proto.sem_0.$1",
            )
            .with_key_remapping(
                "model\\.23\\.proto\\.semseg\\.1\\.(.+)",
                "head.proto.sem_1.$1",
            )
            .with_key_remapping(
                "model\\.23\\.proto\\.semseg\\.2\\.(.+)",
                "head.proto.sem_out.$1",
            );
    }
    store
}

macro_rules! seg_model {
    ($model:ident, $config:ident, $body_struct:ident, $id:literal, $doc:expr) => {
        #[doc = $doc]
        #[derive(Module, Debug)]
        pub struct $model<B: Backend> {
            body: $body_struct<B>,
            head: Yolo26SegHead<B>,
        }

        impl<B: Backend> $model<B> {
            pub fn forward(&self, input: Tensor<B, 4>) -> SegmentOutput<B> {
                self.head.forward(self.body.forward(input))
            }

            #[cfg(feature = "training")]
            pub fn forward_train(&self, input: Tensor<B, 4>) -> DualSegmentTrainOutput<B> {
                self.head.forward_train(self.body.forward(input))
            }

            /// Import tensor-only state exported from an official Ultralytics checkpoint.
            #[cfg(feature = "pretrained")]
            pub fn load_pytorch_weights(
                &mut self,
                path: impl Into<std::path::PathBuf>,
            ) -> Result<(), PytorchStoreError> {
                let mut store = pytorch_store(path);
                self.load_from(&mut store).map(|_| ())
            }

            /// Load Montgomery's versioned, half-precision native Burnpack artifact.
            #[cfg(feature = "pretrained")]
            pub fn load_burnpack_weights(
                &mut self,
                path: impl Into<std::path::PathBuf>,
            ) -> Result<(), BurnpackError> {
                let mut store = BurnpackStore::from_file(path.into())
                    .with_from_adapter(HalfPrecisionAdapter::new())
                    .zero_copy(true);
                self.load_from(&mut store).map(|_| ())
            }

            /// Save a versioned native artifact. Existing files are deliberately not overwritten.
            #[cfg(feature = "pretrained")]
            pub fn save_burnpack_weights(
                &self,
                path: impl Into<std::path::PathBuf>,
            ) -> Result<(), BurnpackError> {
                let mut store = BurnpackStore::from_file(path.into())
                    .metadata("montgomery.artifact-format", weights::artifact_format($id))
                    .metadata("montgomery.model", $id)
                    .metadata("montgomery.classes", "coco-80")
                    .metadata("montgomery.precision", "f16")
                    .metadata("montgomery.source", "ultralytics-v8.4")
                    .metadata("montgomery.license", "AGPL-3.0")
                    .with_to_adapter(HalfPrecisionAdapter::new());
                self.save_into(&mut store)
            }
        }

        #[derive(Debug, Default)]
        pub struct $config;
    };
}

seg_model!(
    Yolo26SegN,
    Yolo26SegNConfig,
    Yolo26BodySmall,
    "yolo26n-seg",
    "Native Burn YOLO26n-seg model."
);
seg_model!(
    Yolo26SegS,
    Yolo26SegSConfig,
    Yolo26BodySmall,
    "yolo26s-seg",
    "Native Burn YOLO26s-seg model."
);
seg_model!(
    Yolo26SegM,
    Yolo26SegMConfig,
    Yolo26BodyLarge,
    "yolo26m-seg",
    "Native Burn YOLO26m-seg model."
);
seg_model!(
    Yolo26SegL,
    Yolo26SegLConfig,
    Yolo26BodyLarge,
    "yolo26l-seg",
    "Native Burn YOLO26l-seg model."
);
seg_model!(
    Yolo26SegX,
    Yolo26SegXConfig,
    Yolo26BodyLarge,
    "yolo26x-seg",
    "Native Burn YOLO26x-seg model."
);

impl Yolo26SegNConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo26SegN<B> {
        self.init_with_classes(80, device)
    }
    pub fn init_with_classes<B: Backend>(
        &self,
        classes: usize,
        device: &Device<B>,
    ) -> Yolo26SegN<B> {
        Yolo26SegN {
            body: super::body::Yolo26BodyNConfig.init(device),
            head: Yolo26SegHeadConfig::new(64, 128, 256, 64)
                .with_num_classes(classes)
                .init(device),
        }
    }
}

impl Yolo26SegSConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo26SegS<B> {
        self.init_with_classes(80, device)
    }
    pub fn init_with_classes<B: Backend>(
        &self,
        classes: usize,
        device: &Device<B>,
    ) -> Yolo26SegS<B> {
        Yolo26SegS {
            body: super::body::Yolo26BodySConfig.init(device),
            head: Yolo26SegHeadConfig::new(128, 256, 512, 128)
                .with_num_classes(classes)
                .init(device),
        }
    }
}

impl Yolo26SegMConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo26SegM<B> {
        self.init_with_classes(80, device)
    }
    pub fn init_with_classes<B: Backend>(
        &self,
        classes: usize,
        device: &Device<B>,
    ) -> Yolo26SegM<B> {
        Yolo26SegM {
            body: super::body::Yolo26BodyMConfig.init(device),
            head: Yolo26SegHeadConfig::new(256, 512, 512, 256)
                .with_num_classes(classes)
                .init(device),
        }
    }
}

impl Yolo26SegLConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo26SegL<B> {
        self.init_with_classes(80, device)
    }
    pub fn init_with_classes<B: Backend>(
        &self,
        classes: usize,
        device: &Device<B>,
    ) -> Yolo26SegL<B> {
        Yolo26SegL {
            body: super::body::Yolo26BodyLConfig.init(device),
            head: Yolo26SegHeadConfig::new(256, 512, 512, 256)
                .with_num_classes(classes)
                .init(device),
        }
    }
}

impl Yolo26SegXConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolo26SegX<B> {
        self.init_with_classes(80, device)
    }
    pub fn init_with_classes<B: Backend>(
        &self,
        classes: usize,
        device: &Device<B>,
    ) -> Yolo26SegX<B> {
        Yolo26SegX {
            body: super::body::Yolo26BodyXConfig.init(device),
            head: Yolo26SegHeadConfig::new(384, 768, 768, 384)
                .with_num_classes(classes)
                .init(device),
        }
    }
}

#[cfg(all(test, feature = "pretrained"))]
mod parity_tests {
    use super::*;
    use crate::models::yolo26::body::Yolo26Features;
    use burn::tensor::{ElementConversion, TensorData};
    use burn_flex::Flex;
    use serde::Deserialize;
    use std::collections::BTreeMap;

    #[derive(Deserialize)]
    struct GoldenFixture {
        format: String,
        model: String,
        tensors: BTreeMap<String, GoldenTensor>,
    }

    #[derive(Deserialize)]
    struct GoldenTensor {
        shape: Vec<usize>,
        mean: f64,
        rms: f64,
        min: f64,
        max: f64,
        samples: Vec<(usize, f64)>,
    }

    fn assert_golden<const D: usize>(name: &str, actual: Tensor<Flex, D>, expected: &GoldenTensor) {
        assert_eq!(actual.dims().to_vec(), expected.shape, "{name} shape");
        let values: Vec<f64> = actual
            .into_data()
            .iter::<f32>()
            .map(|value| value.elem::<f64>())
            .collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let rms =
            (values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64).sqrt();
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let close =
            |actual: f64, expected: f64| (actual - expected).abs() <= 2e-4 + expected.abs() * 2e-4;

        assert!(
            close(mean, expected.mean),
            "{name} mean: {mean} != {}",
            expected.mean
        );
        assert!(
            close(rms, expected.rms),
            "{name} rms: {rms} != {}",
            expected.rms
        );
        assert!(
            close(min, expected.min),
            "{name} min: {min} != {}",
            expected.min
        );
        assert!(
            close(max, expected.max),
            "{name} max: {max} != {}",
            expected.max
        );
        for &(index, expected_value) in &expected.samples {
            let actual_value = values[index];
            assert!(
                close(actual_value, expected_value),
                "{name}[{index}]: {actual_value} != {expected_value}"
            );
        }
    }

    fn load_reference_image(id: &str, device: &Device<Flex>) -> Tensor<Flex, 4> {
        let image = image::open(format!("target/{id}-preprocessed-reference.png"))
            .unwrap()
            .into_rgb8();
        let shape = [image.height() as usize, image.width() as usize, 3];
        Tensor::<Flex, 3>::from_data(
            TensorData::new(image.into_raw(), shape).convert::<f32>(),
            device,
        )
        .permute([2, 0, 1])
        .unsqueeze::<4>()
            / 255.0
    }

    macro_rules! checkpoint_test {
        ($fn_name:ident, $config:ty, $id:literal) => {
            /// Run manually after converting the official checkpoint. Kept ignored in CI because
            /// the source checkpoint is an external AGPL asset.
            #[test]
            #[ignore]
            fn $fn_name() {
                let checkpoint = std::path::PathBuf::from(concat!("target/", $id, "-state.pt"));
                assert!(
                    checkpoint.exists(),
                    "convert {}.pt with tools/export_ultralytics_state.py first",
                    $id
                );
                let worker = std::thread::Builder::new()
                    .stack_size(64 * 1024 * 1024)
                    .spawn(move || {
                        let device = Default::default();
                        let mut model = <$config>::default().init::<Flex>(&device);
                        model.load_pytorch_weights(checkpoint).unwrap();
                        let output = model.forward(Tensor::zeros([1, 3, 64, 64], &device));
                        assert_eq!(output.decoded.boxes.dims(), [1, 84, 4]);
                        assert_eq!(output.decoded.scores.dims(), [1, 84, 80]);
                        assert_eq!(output.coefficients.dims()[0..2], [1, NUM_MASKS]);
                        assert_eq!(output.prototypes.dims()[0..2], [1, NUM_MASKS]);
                    })
                    .unwrap();
                worker.join().unwrap();
            }
        };
    }

    macro_rules! golden_test {
        ($fn_name:ident, $config:ty, $id:literal) => {
            #[test]
            #[ignore]
            fn $fn_name() {
                let checkpoint = std::path::PathBuf::from(format!(
                    "target/{}",
                    crate::models::yolo26::weights::artifact_filename($id)
                ));
                let fixture: GoldenFixture = serde_json::from_slice(
                    &std::fs::read(format!("target/{}-golden-v1.json", $id)).unwrap_or_else(|_| {
                        panic!(
                            "generate fixtures with tools/export_yolo26_seg_fixtures.py --model {}",
                            $id
                        )
                    }),
                )
                .unwrap();
                assert_eq!(fixture.format, "montgomery-ultralytics-golden-v1");
                assert_eq!(fixture.model, $id);

                let worker = std::thread::Builder::new()
                    .stack_size(64 * 1024 * 1024)
                    .spawn(move || {
                        let device = Default::default();
                        let mut model = <$config>::default().init::<Flex>(&device);
                        model.load_burnpack_weights(checkpoint).unwrap();
                        let features = model.body.forward(load_reference_image($id, &device));
                        let raw = model.head.detect.forward_raw(Yolo26Features {
                            p3: features.p3.clone(),
                            p4: features.p4.clone(),
                            p5: features.p5.clone(),
                        });
                        let output = model.head.forward(features);

                        assert_golden(
                            "raw_boxes",
                            raw.boxes,
                            fixture.tensors.get("raw_boxes").unwrap(),
                        );
                        assert_golden(
                            "raw_scores",
                            raw.scores,
                            fixture.tensors.get("raw_scores").unwrap(),
                        );
                        assert_golden(
                            "decoded_boxes_xyxy",
                            output.decoded.boxes,
                            fixture.tensors.get("decoded_boxes_xyxy").unwrap(),
                        );
                        assert_golden(
                            "decoded_scores",
                            output.decoded.scores,
                            fixture.tensors.get("decoded_scores").unwrap(),
                        );
                        assert_golden(
                            "mask_coeffs",
                            output.coefficients,
                            fixture.tensors.get("mask_coeffs").unwrap(),
                        );
                        assert_golden(
                            "protos",
                            output.prototypes,
                            fixture.tensors.get("protos").unwrap(),
                        );
                    })
                    .unwrap();
                worker.join().unwrap();
            }
        };
    }

    macro_rules! latency_test {
        ($fn_name:ident, $config:ty, $id:literal) => {
            /// Measure single-image batch-1 inference latency with the packed native artifact on
            /// the Flex CPU backend. Run with
            /// `cargo test --release <id> -- --ignored --nocapture` after the weight-prep loop.
            #[test]
            #[ignore]
            fn $fn_name() {
                let checkpoint = std::path::PathBuf::from(format!(
                    "target/{}",
                    crate::models::yolo26::weights::artifact_filename($id)
                ));
                assert!(
                    checkpoint.exists(),
                    "pack the {} artifact with pack-weights first",
                    $id
                );
                let worker = std::thread::Builder::new()
                    .stack_size(64 * 1024 * 1024)
                    .spawn(move || {
                        let device = Default::default();
                        let mut model = <$config>::default().init::<Flex>(&device);
                        model.load_burnpack_weights(checkpoint).unwrap();
                        let input = Tensor::<Flex, 4>::zeros([1, 3, 640, 640], &device);
                        const WARMUP_RUNS: usize = 3;
                        const TIMED_RUNS: usize = 10;

                        for _ in 0..WARMUP_RUNS {
                            let output = model.forward(input.clone());
                            let _ = output.decoded.scores.sum().into_data();
                        }
                        let mut samples = Vec::with_capacity(TIMED_RUNS);
                        for _ in 0..TIMED_RUNS {
                            let started = std::time::Instant::now();
                            let output = model.forward(input.clone());
                            let _ = output.decoded.scores.sum().into_data();
                            samples.push(started.elapsed().as_secs_f64() * 1e3);
                        }
                        samples.sort_by(|a, b| a.total_cmp(&b));
                        let median = samples[samples.len() / 2];
                        let min = samples[0];
                        println!(
                            "{:>11}: {:>7.1} ms median, {:>7.1} ms min  (single image, batch 1, 640 px, {TIMED_RUNS} runs)",
                            $id, median, min,
                        );
                    })
                    .unwrap();
                worker.join().unwrap();
            }
        };
    }

    #[cfg(feature = "gpu")]
    macro_rules! gpu_latency_test {
        ($fn_name:ident, $config:ty, $id:literal) => {
            /// Measure single-image batch-1 inference latency with the packed native artifact on
            /// the Wgpu GPU backend. Requires the gpu feature and a packed native artifact.
            #[test]
            #[ignore]
            fn $fn_name() {
                let checkpoint = std::path::PathBuf::from(format!(
                    "target/{}",
                    crate::models::yolo26::weights::artifact_filename($id)
                ));
                assert!(
                    checkpoint.exists(),
                    "pack the {} artifact with pack-weights first",
                    $id
                );
                let (device, adapter) = crate::default_wgpu_device();
                println!("GPU adapter: {adapter}");
                let worker = std::thread::Builder::new()
                    .stack_size(64 * 1024 * 1024)
                    .spawn(move || {
                        let mut model = <$config>::default().init::<burn::backend::Wgpu>(&device);
                        model.load_burnpack_weights(checkpoint).unwrap();
                        let input = Tensor::<burn::backend::Wgpu, 4>::zeros([1, 3, 640, 640], &device);
                        const WARMUP_RUNS: usize = 3;
                        const TIMED_RUNS: usize = 10;

                        for _ in 0..WARMUP_RUNS {
                            let output = model.forward(input.clone());
                            let _ = output.decoded.scores.sum().into_data();
                        }
                        let mut samples = Vec::with_capacity(TIMED_RUNS);
                        for _ in 0..TIMED_RUNS {
                            let started = std::time::Instant::now();
                            let output = model.forward(input.clone());
                            let _ = output.decoded.scores.sum().into_data();
                            samples.push(started.elapsed().as_secs_f64() * 1e3);
                        }
                        samples.sort_by(|a, b| a.total_cmp(&b));
                        let median = samples[samples.len() / 2];
                        let min = samples[0];
                        println!(
                            "{:>11}: {:>7.1} ms median, {:>7.1} ms min  (single image, batch 1, 640 px, {TIMED_RUNS} runs, Wgpu GPU)",
                            $id, median, min,
                        );
                    })
                    .unwrap();
                worker.join().unwrap();
            }
        };
    }

    checkpoint_test!(
        yolo26n_seg_imports_official_checkpoint_and_runs_forward,
        Yolo26SegNConfig,
        "yolo26n-seg"
    );
    checkpoint_test!(
        yolo26s_seg_imports_official_checkpoint_and_runs_forward,
        Yolo26SegSConfig,
        "yolo26s-seg"
    );
    checkpoint_test!(
        yolo26m_seg_imports_official_checkpoint_and_runs_forward,
        Yolo26SegMConfig,
        "yolo26m-seg"
    );
    checkpoint_test!(
        yolo26l_seg_imports_official_checkpoint_and_runs_forward,
        Yolo26SegLConfig,
        "yolo26l-seg"
    );
    checkpoint_test!(
        yolo26x_seg_imports_official_checkpoint_and_runs_forward,
        Yolo26SegXConfig,
        "yolo26x-seg"
    );

    golden_test!(
        yolo26n_seg_matches_ultralytics_golden_tensors,
        Yolo26SegNConfig,
        "yolo26n-seg"
    );
    golden_test!(
        yolo26s_seg_matches_ultralytics_golden_tensors,
        Yolo26SegSConfig,
        "yolo26s-seg"
    );
    golden_test!(
        yolo26m_seg_matches_ultralytics_golden_tensors,
        Yolo26SegMConfig,
        "yolo26m-seg"
    );
    golden_test!(
        yolo26l_seg_matches_ultralytics_golden_tensors,
        Yolo26SegLConfig,
        "yolo26l-seg"
    );
    golden_test!(
        yolo26x_seg_matches_ultralytics_golden_tensors,
        Yolo26SegXConfig,
        "yolo26x-seg"
    );

    latency_test!(
        yolo26n_seg_measures_single_inference_latency,
        Yolo26SegNConfig,
        "yolo26n-seg"
    );
    latency_test!(
        yolo26s_seg_measures_single_inference_latency,
        Yolo26SegSConfig,
        "yolo26s-seg"
    );
    latency_test!(
        yolo26m_seg_measures_single_inference_latency,
        Yolo26SegMConfig,
        "yolo26m-seg"
    );
    latency_test!(
        yolo26l_seg_measures_single_inference_latency,
        Yolo26SegLConfig,
        "yolo26l-seg"
    );
    latency_test!(
        yolo26x_seg_measures_single_inference_latency,
        Yolo26SegXConfig,
        "yolo26x-seg"
    );

    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo26n_seg_measures_single_inference_latency_gpu,
        Yolo26SegNConfig,
        "yolo26n-seg"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo26s_seg_measures_single_inference_latency_gpu,
        Yolo26SegSConfig,
        "yolo26s-seg"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo26m_seg_measures_single_inference_latency_gpu,
        Yolo26SegMConfig,
        "yolo26m-seg"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo26l_seg_measures_single_inference_latency_gpu,
        Yolo26SegLConfig,
        "yolo26l-seg"
    );
    #[cfg(feature = "gpu")]
    gpu_latency_test!(
        yolo26x_seg_measures_single_inference_latency_gpu,
        Yolo26SegXConfig,
        "yolo26x-seg"
    );
}
