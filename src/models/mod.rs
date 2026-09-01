//! Native model implementations owned and versioned by Montgomery.

/// Define one concrete Ultralytics classification scale without repeating the shared wrapper,
/// checkpoint I/O, and head construction. Body configs remain explicit because they encode the
/// checkpoint graph; this macro only captures the behavior that is identical for every scale.
macro_rules! classify_model {
    (
        $model:ident,
        $config:ident,
        $body:ident,
        $body_config:ident,
        $head_channels:expr,
        $id:literal,
        $doc:literal,
        $checkpoint_doc:literal
    ) => {
        #[doc = $doc]
        #[derive(Module, Debug)]
        pub struct $model<B: Backend> {
            body: $body<B>,
            head: ClassifyHead<B>,
        }

        impl<B: Backend> $model<B> {
            pub fn forward(&self, input: Tensor<B, 4>) -> ClassificationOutput<B> {
                self.head.forward(self.body.forward(input))
            }

            pub fn forward_train(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
                self.head.forward_train(self.body.forward(input))
            }

            #[doc = $checkpoint_doc]
            #[cfg(feature = "pretrained")]
            pub fn load_pytorch_weights(
                &mut self,
                path: impl Into<std::path::PathBuf>,
            ) -> Result<(), burn_store::PytorchStoreError> {
                let mut store = pytorch_store(path);
                self.load_from(&mut store).map(|_| ())
            }

            /// Load Montgomery's versioned, half-precision native Burnpack artifact.
            #[cfg(feature = "pretrained")]
            pub fn load_burnpack_weights(
                &mut self,
                path: impl Into<std::path::PathBuf>,
            ) -> Result<(), burn_store::BurnpackError> {
                let mut store = burn_store::BurnpackStore::from_file(path.into())
                    .with_from_adapter(burn_store::HalfPrecisionAdapter::new())
                    .zero_copy(true);
                self.load_from(&mut store).map(|_| ())
            }

            /// Save a versioned native artifact. Existing files are deliberately not overwritten.
            #[cfg(feature = "pretrained")]
            pub fn save_burnpack_weights(
                &self,
                path: impl Into<std::path::PathBuf>,
            ) -> Result<(), burn_store::BurnpackError> {
                let mut store = burn_store::BurnpackStore::from_file(path.into())
                    .metadata(
                        "montgomery.artifact-format",
                        super::weights::artifact_format($id),
                    )
                    .metadata("montgomery.model", $id)
                    .metadata("montgomery.classes", "imagenet-1000")
                    .metadata("montgomery.precision", "f16")
                    .with_to_adapter(burn_store::HalfPrecisionAdapter::new());
                self.save_into(&mut store)
            }
        }

        #[derive(Debug, Default)]
        pub struct $config;

        impl $config {
            pub fn init<B: Backend>(&self, device: &Device<B>) -> $model<B> {
                self.init_with_classes(crate::models::yolo26::classification::NUM_CLASSES, device)
            }

            pub fn init_with_classes<B: Backend>(
                &self,
                num_classes: usize,
                device: &Device<B>,
            ) -> $model<B> {
                $model {
                    body: $body_config.init(device),
                    head: ClassifyHeadConfig::new($head_channels)
                        .with_num_classes(num_classes)
                        .init(device),
                }
            }
        }
    };
}

#[cfg(feature = "training")]
pub(crate) mod training_ops;
pub mod yolo11;
pub mod yolo12;
pub mod yolo26;
pub mod yolov10;
pub mod yolov3_tiny;
pub mod yolov8;
pub mod yolox;
