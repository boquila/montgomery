use std::{error::Error, path::PathBuf};

use burn::{
    backend::{Autodiff, Wgpu},
    module::Module,
    optim::{AdamWConfig, Optimizer, SgdConfig, momentum::MomentumConfig},
    tensor::Tensor,
    tensor::backend::Backend,
};
use serde::Serialize;

use crate::{
    ModelId,
    data::augmentation::{
        AugSample, AugmentationPipeline, ByteImage, ClassificationPipeline, ColorOrder,
        FormattedClassificationSample, FormattedDetectionSample, SeedKey,
    },
    models::{
        yolo11::{
            Yolo11ClsLConfig, Yolo11ClsMConfig, Yolo11ClsNConfig, Yolo11ClsSConfig,
            Yolo11ClsXConfig,
        },
        yolo26::{
            Yolo26ClsLConfig, Yolo26ClsMConfig, Yolo26ClsNConfig, Yolo26ClsSConfig,
            Yolo26ClsXConfig, Yolo26LConfig, Yolo26MConfig, Yolo26NConfig, Yolo26SConfig,
            Yolo26XConfig,
        },
        yolov3_tiny::Yolov3TinyConfig,
        yolov8::{
            Yolov8ClsLConfig, Yolov8ClsMConfig, Yolov8ClsNConfig, Yolov8ClsSConfig,
            Yolov8ClsXConfig,
        },
        yolov10::{
            Yolov10BConfig, Yolov10LConfig, Yolov10MConfig, Yolov10NConfig, Yolov10SConfig,
            Yolov10XConfig,
        },
        yolox::Yolox,
    },
    training::{
        ModelSpec, Trainer,
        checkpoint::{
            CheckpointManifest, decode_record, encode_record, replace_atomic, save_atomic,
        },
        config::{OptimizerKind, ScheduleKind, TrainingConfig},
        data::{
            DatasetFormat, DatasetManifest,
            batch::{
                ClassificationBatch, DetectionBatch, FormattedClassificationBatch,
                FormattedDetectionBatch, SegmentationBatch, segmentation_into_device,
            },
            sample::ImageMeta,
        },
        engine::TrainableTask,
    },
};

type TrainBackend = Autodiff<Wgpu>;

#[derive(Debug, Clone)]
pub struct TrainingRequest {
    pub model: ModelId,
    pub data: PathBuf,
    pub epochs: usize,
    pub batch_size: usize,
    pub accumulation: usize,
    pub image_size: Option<usize>,
    pub seed: u64,
    pub run_root: PathBuf,
    pub name: String,
    pub dry_run: bool,
    pub resume: Option<PathBuf>,
}

pub fn train(request: TrainingRequest) -> Result<PathBuf, Box<dyn Error + Send + Sync>> {
    let resume_manifest = request
        .resume
        .as_ref()
        .map(crate::training::checkpoint::load)
        .transpose()?;
    if let Some(manifest) = &resume_manifest
        && manifest.config.model.architecture != request.model
    {
        return Err("--model conflicts with the immutable resume checkpoint architecture".into());
    }
    let dataset_path = resume_manifest
        .as_ref()
        .map(|manifest| manifest.config.data.clone())
        .unwrap_or_else(|| request.data.clone());
    let dataset = DatasetManifest::load(&dataset_path)?;
    let spec = if let Some(manifest) = &resume_manifest {
        if manifest.config.model.class_names != dataset.class_names {
            return Err("resume checkpoint class table differs from the resolved dataset".into());
        }
        manifest.config.model.clone()
    } else {
        ModelSpec::new(
            request.model,
            dataset.class_names.clone(),
            request.image_size.map(|side| [side, side]),
        )?
    };
    if (spec.task == crate::training::TaskKind::Classify)
        != (dataset.format == DatasetFormat::ClassificationFolders)
    {
        return Err("classification models require classification folders; detector models require YOLO/COCO annotations".into());
    }
    let mut config = resume_manifest
        .as_ref()
        .map(|manifest| manifest.config.clone())
        .unwrap_or_else(|| TrainingConfig::yolox(spec, dataset_path, request.run_root));
    if resume_manifest.is_none() {
        config.epochs = request.epochs;
        config.batch_size = request.batch_size;
        config.accumulation = request.accumulation;
        config.seed = request.seed;
        if matches!(
            crate::training::dispatch::recipe_for(config.model.architecture).loss,
            crate::training::dispatch::LossFamily::YoloxSimOta
        ) {
            config.optimizer = OptimizerKind::Sgd;
            config.schedule = ScheduleKind::YoloxWarmCosine;
            config.initial_lr = 0.01;
        } else {
            config.optimizer = OptimizerKind::AdamW;
            config.schedule = ScheduleKind::Cosine;
            config.initial_lr = 1e-3;
        }
        config.augmentation.imgsz = config.model.input_size[0];
    }
    config.validate()?;
    let (device, adapter) = crate::default_wgpu_device();
    eprintln!("Training adapter: {adapter}");
    let epoch = resume_manifest
        .as_ref()
        .map_or(0, |value| value.state.epoch as u64);
    let classification_batches = if config.model.task == crate::training::TaskKind::Classify {
        Some(build_classification_batches(
            &config,
            &dataset,
            &dataset.train_images,
            &device,
            epoch,
            true,
        )?)
    } else {
        None
    };
    let detection_batches = if config.model.task == crate::training::TaskKind::Detect {
        Some(build_detection_batches(
            &config,
            &dataset,
            &dataset.train_images,
            &device,
            epoch,
            true,
        )?)
    } else {
        None
    };
    let segmentation_batches = if config.model.task == crate::training::TaskKind::Segment {
        Some(build_segmentation_batches(
            &config,
            &dataset,
            &dataset.train_images,
            &device,
            epoch,
            true,
        )?)
    } else {
        None
    };
    let batch_count = classification_batches
        .as_ref()
        .map(Vec::len)
        .or_else(|| detection_batches.as_ref().map(Vec::len))
        .or_else(|| segmentation_batches.as_ref().map(Vec::len))
        .ok_or("task produced no batch collection")?;
    let trainer = if let Some(resume) = &request.resume {
        Trainer::from_checkpoint(resume)?
    } else {
        let trainer = Trainer::create(config.clone(), &request.name, batch_count)?;
        trainer.run.write_dataset(&dataset)?;
        trainer.run.write_environment(&adapter, &dataset)?;
        trainer
    };
    let classes = config.model.num_classes;

    macro_rules! run {
        ($config:expr) => {{
            let model = $config.init_with_classes(classes, &device);
            run_task(
                model,
                trainer,
                classification_batches.expect("classification dispatch has batches"),
                |epoch| {
                    build_classification_batches(
                        &config,
                        &dataset,
                        &dataset.train_images,
                        &device,
                        epoch,
                        true,
                    )
                },
                request.dry_run,
                request.resume.as_ref(),
                &device,
            )
        }};
    }
    macro_rules! run_detect {
        ($model:expr) => {{
            run_task(
                $model,
                trainer,
                detection_batches.expect("detection dispatch has batches"),
                |epoch| {
                    build_detection_batches(
                        &config,
                        &dataset,
                        &dataset.train_images,
                        &device,
                        epoch,
                        true,
                    )
                },
                request.dry_run,
                request.resume.as_ref(),
                &device,
            )
        }};
    }
    macro_rules! run_segment {
        ($model:expr) => {{
            run_task(
                $model,
                trainer,
                segmentation_batches.expect("segmentation dispatch has batches"),
                |epoch| {
                    build_segmentation_batches(
                        &config,
                        &dataset,
                        &dataset.train_images,
                        &device,
                        epoch,
                        true,
                    )
                },
                request.dry_run,
                request.resume.as_ref(),
                &device,
            )
        }};
    }
    let run = match request.model {
        ModelId::Yolo11NCls => run!(Yolo11ClsNConfig),
        ModelId::Yolo11SCls => run!(Yolo11ClsSConfig),
        ModelId::Yolo11MCls => run!(Yolo11ClsMConfig),
        ModelId::Yolo11LCls => run!(Yolo11ClsLConfig),
        ModelId::Yolo11XCls => run!(Yolo11ClsXConfig),
        ModelId::Yolo26NCls => run!(Yolo26ClsNConfig),
        ModelId::Yolo26SCls => run!(Yolo26ClsSConfig),
        ModelId::Yolo26MCls => run!(Yolo26ClsMConfig),
        ModelId::Yolo26LCls => run!(Yolo26ClsLConfig),
        ModelId::Yolo26XCls => run!(Yolo26ClsXConfig),
        ModelId::Yolov8NCls => run!(Yolov8ClsNConfig),
        ModelId::Yolov8SCls => run!(Yolov8ClsSConfig),
        ModelId::Yolov8MCls => run!(Yolov8ClsMConfig),
        ModelId::Yolov8LCls => run!(Yolov8ClsLConfig),
        ModelId::Yolov8XCls => run!(Yolov8ClsXConfig),
        ModelId::YoloxNano => run_detect!(Yolox::yolox_nano(classes, &device)),
        ModelId::YoloxTiny => run_detect!(Yolox::yolox_tiny(classes, &device)),
        ModelId::YoloxS => run_detect!(Yolox::yolox_s(classes, &device)),
        ModelId::YoloxM => run_detect!(Yolox::yolox_m(classes, &device)),
        ModelId::YoloxL => run_detect!(Yolox::yolox_l(classes, &device)),
        ModelId::YoloxX => run_detect!(Yolox::yolox_x(classes, &device)),
        ModelId::Yolov3TinyU => {
            run_detect!(Yolov3TinyConfig.init_with_classes(classes, &device))
        }
        ModelId::Yolov10N => run_detect!(Yolov10NConfig.init_with_classes(classes, &device)),
        ModelId::Yolov10S => run_detect!(Yolov10SConfig.init_with_classes(classes, &device)),
        ModelId::Yolov10M => run_detect!(Yolov10MConfig.init_with_classes(classes, &device)),
        ModelId::Yolov10B => run_detect!(Yolov10BConfig.init_with_classes(classes, &device)),
        ModelId::Yolov10L => run_detect!(Yolov10LConfig.init_with_classes(classes, &device)),
        ModelId::Yolov10X => run_detect!(Yolov10XConfig.init_with_classes(classes, &device)),
        ModelId::Yolo11N => {
            run_detect!(crate::models::yolo11::Yolo11NConfig.init_with_classes(classes, &device))
        }
        ModelId::Yolo11S => {
            run_detect!(crate::models::yolo11::Yolo11SConfig.init_with_classes(classes, &device))
        }
        ModelId::Yolo11M => {
            run_detect!(crate::models::yolo11::Yolo11MConfig.init_with_classes(classes, &device))
        }
        ModelId::Yolo11L => {
            run_detect!(crate::models::yolo11::Yolo11LConfig.init_with_classes(classes, &device))
        }
        ModelId::Yolo11X => {
            run_detect!(crate::models::yolo11::Yolo11XConfig.init_with_classes(classes, &device))
        }
        ModelId::Yolo26N => run_detect!(Yolo26NConfig.init_with_classes(classes, &device)),
        ModelId::Yolo26S => run_detect!(Yolo26SConfig.init_with_classes(classes, &device)),
        ModelId::Yolo26M => run_detect!(Yolo26MConfig.init_with_classes(classes, &device)),
        ModelId::Yolo26L => run_detect!(Yolo26LConfig.init_with_classes(classes, &device)),
        ModelId::Yolo26X => run_detect!(Yolo26XConfig.init_with_classes(classes, &device)),
        ModelId::Yolo11NSeg => run_segment!(
            crate::models::yolo11::Yolo11SegNConfig.init_with_classes(classes, &device)
        ),
        ModelId::Yolo11SSeg => run_segment!(
            crate::models::yolo11::Yolo11SegSConfig.init_with_classes(classes, &device)
        ),
        ModelId::Yolo11MSeg => run_segment!(
            crate::models::yolo11::Yolo11SegMConfig.init_with_classes(classes, &device)
        ),
        ModelId::Yolo11LSeg => run_segment!(
            crate::models::yolo11::Yolo11SegLConfig.init_with_classes(classes, &device)
        ),
        ModelId::Yolo11XSeg => run_segment!(
            crate::models::yolo11::Yolo11SegXConfig.init_with_classes(classes, &device)
        ),
        ModelId::Yolo26NSeg => run_segment!(
            crate::models::yolo26::Yolo26SegNConfig.init_with_classes(classes, &device)
        ),
        ModelId::Yolo26SSeg => run_segment!(
            crate::models::yolo26::Yolo26SegSConfig.init_with_classes(classes, &device)
        ),
        ModelId::Yolo26MSeg => run_segment!(
            crate::models::yolo26::Yolo26SegMConfig.init_with_classes(classes, &device)
        ),
        ModelId::Yolo26LSeg => run_segment!(
            crate::models::yolo26::Yolo26SegLConfig.init_with_classes(classes, &device)
        ),
        ModelId::Yolo26XSeg => run_segment!(
            crate::models::yolo26::Yolo26SegXConfig.init_with_classes(classes, &device)
        ),
        _ => return Err("train CLI dispatch for this task is not complete".into()),
    }?;
    Ok(run)
}

fn run_task<M, F>(
    model: M,
    trainer: Trainer,
    batches: Vec<M::Batch>,
    rebuild_batches: F,
    dry_run: bool,
    resume: Option<&PathBuf>,
    device: &burn::tensor::Device<TrainBackend>,
) -> Result<PathBuf, Box<dyn Error + Send + Sync>>
where
    M: TrainableTask<TrainBackend> + Clone,
    F: FnMut(u64) -> Result<Vec<M::Batch>, Box<dyn Error + Send + Sync>>,
{
    if dry_run {
        let output = model.forward_loss(
            &batches[0],
            crate::training::engine::LossContext {
                yolox_l1: trainer.state.yolox_l1,
                one_to_many: trainer
                    .state
                    .dual_loss
                    .as_ref()
                    .map_or(1.0, |value| value.one_to_many),
                one_to_one: trainer
                    .state
                    .dual_loss
                    .as_ref()
                    .map_or(1.0, |value| value.one_to_one),
            },
        )?;
        if !output.finite {
            return Err("dry-run loss is non-finite".into());
        }
        let gradients = burn::optim::GradientsParams::from_grads(output.total.backward(), &model);
        if gradients.is_empty() {
            return Err("dry-run produced no gradients".into());
        }
        return Ok(trainer.run.root);
    }
    let gradient_clip = trainer.config.gradient_clip as f32;
    let momentum = trainer.config.momentum;
    match trainer.config.optimizer {
        OptimizerKind::AdamW => run_task_with_optimizer(
            model,
            trainer,
            batches,
            rebuild_batches,
            resume,
            device,
            AdamWConfig::new()
                .with_weight_decay(0.0)
                .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Norm(
                    gradient_clip,
                )))
                .init(),
        ),
        OptimizerKind::Sgd => run_task_with_optimizer(
            model,
            trainer,
            batches,
            rebuild_batches,
            resume,
            device,
            SgdConfig::new()
                .with_momentum(Some(
                    MomentumConfig::new()
                        .with_momentum(momentum)
                        .with_dampening(0.0)
                        .with_nesterov(true),
                ))
                .with_gradient_clipping(Some(burn::grad_clipping::GradientClippingConfig::Norm(
                    gradient_clip,
                )))
                .init(),
        ),
    }
}

fn run_task_with_optimizer<M, O, F>(
    mut model: M,
    mut trainer: Trainer,
    mut batches: Vec<M::Batch>,
    mut rebuild_batches: F,
    resume: Option<&PathBuf>,
    device: &burn::tensor::Device<TrainBackend>,
    mut optimizer: O,
) -> Result<PathBuf, Box<dyn Error + Send + Sync>>
where
    M: TrainableTask<TrainBackend> + Clone,
    O: Optimizer<M, TrainBackend>,
    F: FnMut(u64) -> Result<Vec<M::Batch>, Box<dyn Error + Send + Sync>>,
{
    let mut ema_model = model.clone();
    let mut ema_state = crate::training::ema::EmaState::new(0.9999)?;
    ema_state.updates = trainer.state.ema_updates;
    if let Some(path) = resume {
        let model_bytes = std::fs::read(path.join("model.bin"))?;
        let optimizer_bytes = std::fs::read(path.join("optimizer.bin"))?;
        model = model.load_record(decode_record::<TrainBackend, _>(model_bytes, device)?);
        optimizer =
            optimizer.load_record(decode_record::<TrainBackend, _>(optimizer_bytes, device)?);
        let ema_bytes = std::fs::read(path.join("ema.bin"))?;
        ema_model = ema_model.load_record(decode_record::<TrainBackend, _>(ema_bytes, device)?);
    }
    while trainer.state.epoch < trainer.config.epochs {
        let result = trainer.train_epoch::<TrainBackend, _, _, _>(
            model,
            optimizer,
            &batches,
            |current, _step| {
                ema_model =
                    crate::training::ema::update_model(ema_model.clone(), current, &mut ema_state)?;
                Ok(())
            },
        )?;
        model = result.0;
        optimizer = result.1;
        trainer.state.ema_updates = ema_state.updates;
        let improved = trainer
            .state
            .observe_fitness(-f64::from(result.2.mean_loss));
        let model_bytes = encode_record::<TrainBackend, _>(model.clone().into_record())?;
        let ema_bytes = encode_record::<TrainBackend, _>(ema_model.clone().into_record())?;
        let optimizer_bytes = encode_record::<TrainBackend, _>(optimizer.to_record())?;
        let manifest = CheckpointManifest::new(
            trainer.config.clone(),
            trainer.state.clone(),
            trainer.scheduler.clone(),
        );
        let epoch_path = trainer
            .run
            .checkpoints
            .join(format!("epoch-{:04}", trainer.state.epoch));
        save_atomic(
            &epoch_path,
            manifest.clone(),
            &[
                ("model.bin", &model_bytes),
                ("ema.bin", &ema_bytes),
                ("optimizer.bin", &optimizer_bytes),
            ],
        )?;
        replace_atomic(
            trainer.run.checkpoints.join("last"),
            manifest.clone(),
            &[
                ("model.bin", &model_bytes),
                ("ema.bin", &ema_bytes),
                ("optimizer.bin", &optimizer_bytes),
            ],
        )?;
        if improved {
            replace_atomic(
                trainer.run.checkpoints.join("best"),
                manifest,
                &[
                    ("model.bin", &model_bytes),
                    ("ema.bin", &ema_bytes),
                    ("optimizer.bin", &optimizer_bytes),
                ],
            )?;
        }
        eprintln!(
            "epoch {}: loss {:.6}",
            trainer.state.epoch, result.2.mean_loss
        );
        if trainer.state.epoch < trainer.config.epochs {
            batches = rebuild_batches(trainer.state.epoch as u64)?;
            if batches.is_empty() {
                return Err("rebuilt epoch contains no batches".into());
            }
        }
    }
    Ok(trainer.run.root)
}

fn build_classification_batches<B: Backend>(
    config: &TrainingConfig,
    dataset: &crate::training::data::ResolvedDataset,
    images: &[PathBuf],
    device: &burn::tensor::Device<B>,
    epoch: u64,
    training: bool,
) -> Result<Vec<ClassificationBatch<B>>, Box<dyn Error + Send + Sync>> {
    let pipeline = ClassificationPipeline::new(
        config
            .augmentation
            .resolve(crate::training::TaskKind::Classify, training)?,
    )?;
    let order = crate::training::data::loader::epoch_permutation(images.len(), config.seed, epoch);
    let mut formatted = Vec::<FormattedClassificationSample>::with_capacity(order.len());
    let mut metadata = Vec::<ImageMeta>::with_capacity(order.len());
    for (logical_position, index) in order.into_iter().enumerate() {
        let path = &images[index];
        let class_name = path
            .parent()
            .and_then(std::path::Path::file_name)
            .and_then(|value| value.to_str())
            .ok_or("classification image has no class parent directory")?;
        let class_id = dataset
            .class_names
            .iter()
            .position(|name| name == class_name)
            .ok_or("classification directory is absent from class table")?;
        let rgb = image::open(path)?.into_rgb8();
        let source_size = [rgb.width(), rgb.height()];
        let image = ByteImage::new(
            rgb.width() as usize,
            rgb.height() as usize,
            3,
            ColorOrder::Rgb,
            rgb.into_raw(),
        )?;
        formatted.push(pipeline.apply(
            image,
            class_id as u32,
            SeedKey {
                run_seed: config.seed,
                epoch,
                logical_position: logical_position as u64,
                sample_index: index as u64,
                rank: 0,
                path: &path.to_string_lossy(),
            },
        )?);
        metadata.push(ImageMeta {
            image_id: path.to_string_lossy().into_owned(),
            source_size,
            canvas_size: [
                config.model.input_size[1] as u32,
                config.model.input_size[0] as u32,
            ],
            scale: [1.0, 1.0],
            pad: [0.0, 0.0],
        });
    }
    let mut batches = Vec::new();
    for start in (0..formatted.len()).step_by(config.batch_size) {
        let end = (start + config.batch_size).min(formatted.len());
        batches.push(
            FormattedClassificationBatch::collate(&formatted[start..end])?
                .into_device(metadata[start..end].to_vec(), device)?,
        );
    }
    Ok(batches)
}

fn build_detection_batches<B: Backend>(
    config: &TrainingConfig,
    dataset: &crate::training::data::ResolvedDataset,
    images: &[PathBuf],
    device: &burn::tensor::Device<B>,
    epoch: u64,
    training: bool,
) -> Result<Vec<DetectionBatch<B>>, Box<dyn Error + Send + Sync>> {
    if dataset.format != DatasetFormat::Yolo {
        return Err("the streaming detector loader currently requires YOLO labels; convert COCO JSON through the native COCO reader before training".into());
    }
    let pipeline = AugmentationPipeline::for_epoch(
        config
            .augmentation
            .resolve(crate::training::TaskKind::Detect, training)?,
        epoch as usize,
        config.epochs,
    )?;
    let mut samples = Vec::with_capacity(images.len());
    for (index, path) in images.iter().enumerate() {
        let sample = crate::training::data::loader::load_yolo_sample(dataset, path)?;
        samples.push(AugSample::from_vision(
            sample,
            index,
            config.augmentation.imgsz,
            false,
        )?);
    }
    let order = crate::training::data::loader::epoch_permutation(images.len(), config.seed, epoch);
    let mut provider =
        crate::training::data::loader::DeterministicPartnerPool::whole_dataset(samples.clone());
    let mut formatted = Vec::<FormattedDetectionSample>::with_capacity(order.len());
    let mut metadata = Vec::<ImageMeta>::with_capacity(order.len());
    for (logical_position, index) in order.into_iter().enumerate() {
        let path_text = images[index].to_string_lossy().into_owned();
        let (sample, _) = pipeline.apply(
            samples[index].clone(),
            &mut provider,
            SeedKey {
                run_seed: config.seed,
                epoch,
                logical_position: logical_position as u64,
                sample_index: index as u64,
                rank: 0,
                path: &path_text,
            },
        )?;
        let [_, canvas_height, canvas_width] = sample.image_shape;
        let geometry = &sample.geometry;
        metadata.push(ImageMeta {
            image_id: path_text,
            source_size: [
                geometry.original_shape[1] as u32,
                geometry.original_shape[0] as u32,
            ],
            canvas_size: [canvas_width as u32, canvas_height as u32],
            scale: geometry.ratio,
            pad: geometry.pad,
        });
        formatted.push(sample);
    }
    let mut batches = Vec::new();
    for start in (0..formatted.len()).step_by(config.batch_size) {
        let end = (start + config.batch_size).min(formatted.len());
        batches.push(
            FormattedDetectionBatch::collate(&formatted[start..end])?
                .into_device(metadata[start..end].to_vec(), device)?,
        );
    }
    Ok(batches)
}

fn build_yolox_validation_batches<B: Backend>(
    config: &TrainingConfig,
    dataset: &crate::training::data::ResolvedDataset,
    images: &[PathBuf],
    device: &burn::tensor::Device<B>,
) -> Result<Vec<DetectionBatch<B>>, Box<dyn Error + Send + Sync>> {
    if dataset.format != DatasetFormat::Yolo {
        return Err("the streaming detector loader currently requires YOLO labels".into());
    }
    let [height, width] = config.model.input_size;
    if height != width {
        return Err("YOLOX validation currently requires a square input".into());
    }
    let mut formatted = Vec::with_capacity(images.len());
    let mut metadata = Vec::with_capacity(images.len());
    for path in images {
        let sample = crate::training::data::loader::load_yolo_sample(dataset, path)?;
        let prepared = crate::data::LetterboxedImage::yolox(&sample.image, width);
        let (scale, pad_x, pad_y) = prepared.letterbox_geometry();
        let rgb = prepared.image().to_rgb8();
        let mut chw = vec![0_u8; 3 * height * width];
        for y in 0..height {
            for x in 0..width {
                let pixel = rgb.get_pixel(x as u32, y as u32).0;
                for channel in 0..3 {
                    chw[(channel * height + y) * width + x] = pixel[channel];
                }
            }
        }
        let mut classes = Vec::with_capacity(sample.targets.len());
        let mut boxes = Vec::with_capacity(sample.targets.len());
        for target in sample.targets {
            let xmin = target.bbox.xmin * scale + pad_x;
            let ymin = target.bbox.ymin * scale + pad_y;
            let xmax = target.bbox.xmax * scale + pad_x;
            let ymax = target.bbox.ymax * scale + pad_y;
            classes.push(u32::try_from(target.class_id)?);
            boxes.push([
                (xmin + xmax) * 0.5 / width as f32,
                (ymin + ymax) * 0.5 / height as f32,
                (xmax - xmin) / width as f32,
                (ymax - ymin) / height as f32,
            ]);
        }
        formatted.push(FormattedDetectionSample {
            image_chw_u8: chw,
            image_shape: [3, height, width],
            classes,
            boxes_xywh_normalized: boxes,
            masks: None,
            geometry: crate::data::augmentation::GeometryMetadata {
                original_shape: [
                    sample.source_size[1] as usize,
                    sample.source_size[0] as usize,
                ],
                current_shape: [height, width],
                ratio: [scale, scale],
                pad: [pad_x, pad_y],
                reversible: true,
            },
        });
        metadata.push(ImageMeta {
            image_id: sample.image_id,
            source_size: sample.source_size,
            canvas_size: [width as u32, height as u32],
            scale: [scale, scale],
            pad: [pad_x, pad_y],
        });
    }
    let mut batches = Vec::new();
    for start in (0..formatted.len()).step_by(config.batch_size) {
        let end = (start + config.batch_size).min(formatted.len());
        batches.push(
            FormattedDetectionBatch::collate(&formatted[start..end])?
                .into_device(metadata[start..end].to_vec(), device)?,
        );
    }
    Ok(batches)
}

fn build_segmentation_batches<B: Backend>(
    config: &TrainingConfig,
    dataset: &crate::training::data::ResolvedDataset,
    images: &[PathBuf],
    device: &burn::tensor::Device<B>,
    epoch: u64,
    training: bool,
) -> Result<Vec<SegmentationBatch<B>>, Box<dyn Error + Send + Sync>> {
    if dataset.format != DatasetFormat::Yolo {
        return Err(
            "the streaming segmentation loader currently requires YOLO polygon labels".into(),
        );
    }
    let pipeline = AugmentationPipeline::for_epoch(
        config
            .augmentation
            .resolve(crate::training::TaskKind::Segment, training)?,
        epoch as usize,
        config.epochs,
    )?;
    let mut samples = Vec::with_capacity(images.len());
    for (index, path) in images.iter().enumerate() {
        let sample = crate::training::data::loader::load_yolo_sample(dataset, path)?;
        samples.push(AugSample::from_vision(
            sample,
            index,
            config.augmentation.imgsz,
            false,
        )?);
    }
    let order = crate::training::data::loader::epoch_permutation(images.len(), config.seed, epoch);
    let mut provider =
        crate::training::data::loader::DeterministicPartnerPool::whole_dataset(samples.clone());
    let mut formatted = Vec::<FormattedDetectionSample>::with_capacity(order.len());
    let mut metadata = Vec::<ImageMeta>::with_capacity(order.len());
    for (logical_position, index) in order.into_iter().enumerate() {
        let path_text = images[index].to_string_lossy().into_owned();
        let (sample, _) = pipeline.apply(
            samples[index].clone(),
            &mut provider,
            SeedKey {
                run_seed: config.seed,
                epoch,
                logical_position: logical_position as u64,
                sample_index: index as u64,
                rank: 0,
                path: &path_text,
            },
        )?;
        let [_, canvas_height, canvas_width] = sample.image_shape;
        let geometry = &sample.geometry;
        metadata.push(ImageMeta {
            image_id: path_text,
            source_size: [
                geometry.original_shape[1] as u32,
                geometry.original_shape[0] as u32,
            ],
            canvas_size: [canvas_width as u32, canvas_height as u32],
            scale: geometry.ratio,
            pad: geometry.pad,
        });
        formatted.push(sample);
    }
    let mut batches = Vec::new();
    for start in (0..formatted.len()).step_by(config.batch_size) {
        let end = (start + config.batch_size).min(formatted.len());
        batches.push(segmentation_into_device(
            &formatted[start..end],
            metadata[start..end].to_vec(),
            device,
        )?);
    }
    Ok(batches)
}

pub fn inspect_checkpoint(
    path: PathBuf,
) -> Result<CheckpointManifest, Box<dyn Error + Send + Sync>> {
    Ok(crate::training::checkpoint::load(path)?)
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationSummary {
    pub model: ModelId,
    pub split: String,
    pub images: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_loss: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top1_accuracy: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top5_accuracy: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub box_metrics: Option<crate::training::metrics::detection::DetectionMetrics>,
}

pub fn validate(checkpoint: PathBuf) -> Result<ValidationSummary, Box<dyn Error + Send + Sync>> {
    let manifest = crate::training::checkpoint::load(&checkpoint)?;
    let dataset = DatasetManifest::load(&manifest.config.data)?;
    if dataset.class_names != manifest.config.model.class_names {
        return Err("validation dataset class table differs from checkpoint metadata".into());
    }
    if dataset.val_images.is_empty() {
        return Err("dataset manifest has no validation images".into());
    }
    let (device, adapter) = crate::default_wgpu_device();
    eprintln!("Validation adapter: {adapter}");
    let bytes = std::fs::read(checkpoint.join("ema.bin"))
        .or_else(|_| std::fs::read(checkpoint.join("model.bin")))?;
    let classes = manifest.config.model.num_classes;
    if manifest.config.model.task == crate::training::TaskKind::Detect {
        let batches = if matches!(
            manifest.config.model.architecture,
            ModelId::YoloxNano
                | ModelId::YoloxTiny
                | ModelId::YoloxS
                | ModelId::YoloxM
                | ModelId::YoloxL
                | ModelId::YoloxX
        ) {
            build_yolox_validation_batches(
                &manifest.config,
                &dataset,
                &dataset.val_images,
                &device,
            )?
        } else {
            build_detection_batches(
                &manifest.config,
                &dataset,
                &dataset.val_images,
                &device,
                manifest.state.epoch as u64,
                false,
            )?
        };
        macro_rules! run_detect {
            ($model:expr) => {{ validate_detection_model($model, bytes, batches, manifest.config.model.architecture) }};
        }
        return match manifest.config.model.architecture {
            ModelId::YoloxNano => run_detect!(Yolox::yolox_nano(classes, &device)),
            ModelId::YoloxTiny => run_detect!(Yolox::yolox_tiny(classes, &device)),
            ModelId::YoloxS => run_detect!(Yolox::yolox_s(classes, &device)),
            ModelId::YoloxM => run_detect!(Yolox::yolox_m(classes, &device)),
            ModelId::YoloxL => run_detect!(Yolox::yolox_l(classes, &device)),
            ModelId::YoloxX => run_detect!(Yolox::yolox_x(classes, &device)),
            ModelId::Yolov3TinyU => {
                run_detect!(Yolov3TinyConfig.init_with_classes(classes, &device))
            }
            ModelId::Yolov10N => {
                run_detect!(Yolov10NConfig.init_with_classes(classes, &device))
            }
            ModelId::Yolov10S => {
                run_detect!(Yolov10SConfig.init_with_classes(classes, &device))
            }
            ModelId::Yolov10M => {
                run_detect!(Yolov10MConfig.init_with_classes(classes, &device))
            }
            ModelId::Yolov10B => {
                run_detect!(Yolov10BConfig.init_with_classes(classes, &device))
            }
            ModelId::Yolov10L => {
                run_detect!(Yolov10LConfig.init_with_classes(classes, &device))
            }
            ModelId::Yolov10X => {
                run_detect!(Yolov10XConfig.init_with_classes(classes, &device))
            }
            ModelId::Yolo11N => run_detect!(
                crate::models::yolo11::Yolo11NConfig.init_with_classes(classes, &device)
            ),
            ModelId::Yolo11S => run_detect!(
                crate::models::yolo11::Yolo11SConfig.init_with_classes(classes, &device)
            ),
            ModelId::Yolo11M => run_detect!(
                crate::models::yolo11::Yolo11MConfig.init_with_classes(classes, &device)
            ),
            ModelId::Yolo11L => run_detect!(
                crate::models::yolo11::Yolo11LConfig.init_with_classes(classes, &device)
            ),
            ModelId::Yolo11X => run_detect!(
                crate::models::yolo11::Yolo11XConfig.init_with_classes(classes, &device)
            ),
            ModelId::Yolo26N => run_detect!(Yolo26NConfig.init_with_classes(classes, &device)),
            ModelId::Yolo26S => run_detect!(Yolo26SConfig.init_with_classes(classes, &device)),
            ModelId::Yolo26M => run_detect!(Yolo26MConfig.init_with_classes(classes, &device)),
            ModelId::Yolo26L => run_detect!(Yolo26LConfig.init_with_classes(classes, &device)),
            ModelId::Yolo26X => run_detect!(Yolo26XConfig.init_with_classes(classes, &device)),
            _ => Err("checkpoint model is not a supported detector".into()),
        };
    }
    if manifest.config.model.task != crate::training::TaskKind::Classify {
        return Err("native CLI mask validation is not complete; detection and classification checkpoints are supported".into());
    }
    let batches = build_classification_batches(
        &manifest.config,
        &dataset,
        &dataset.val_images,
        &device,
        manifest.state.epoch as u64,
        false,
    )?;
    macro_rules! run {
        ($config:expr) => {{
            let model = $config.init_with_classes(classes, &device);
            validate_classification_model(model, bytes, batches, manifest.config.model.architecture)
        }};
    }
    match manifest.config.model.architecture {
        ModelId::Yolo11NCls => run!(Yolo11ClsNConfig),
        ModelId::Yolo11SCls => run!(Yolo11ClsSConfig),
        ModelId::Yolo11MCls => run!(Yolo11ClsMConfig),
        ModelId::Yolo11LCls => run!(Yolo11ClsLConfig),
        ModelId::Yolo11XCls => run!(Yolo11ClsXConfig),
        ModelId::Yolo26NCls => run!(Yolo26ClsNConfig),
        ModelId::Yolo26SCls => run!(Yolo26ClsSConfig),
        ModelId::Yolo26MCls => run!(Yolo26ClsMConfig),
        ModelId::Yolo26LCls => run!(Yolo26ClsLConfig),
        ModelId::Yolo26XCls => run!(Yolo26ClsXConfig),
        ModelId::Yolov8NCls => run!(Yolov8ClsNConfig),
        ModelId::Yolov8SCls => run!(Yolov8ClsSConfig),
        ModelId::Yolov8MCls => run!(Yolov8ClsMConfig),
        ModelId::Yolov8LCls => run!(Yolov8ClsLConfig),
        ModelId::Yolov8XCls => run!(Yolov8ClsXConfig),
        _ => Err("checkpoint model is not a supported classifier".into()),
    }
}

fn validate_classification_model<M>(
    model: M,
    bytes: Vec<u8>,
    batches: Vec<ClassificationBatch<Wgpu>>,
    model_id: ModelId,
) -> Result<ValidationSummary, Box<dyn Error + Send + Sync>>
where
    M: burn::module::Module<Wgpu> + ClassificationForward,
{
    let device = batches[0].images.device();
    let model = model.load_record(decode_record::<Wgpu, _>(bytes, &device)?);
    let mut loss = 0.0_f64;
    let mut top1 = 0;
    let mut top5 = 0;
    let mut count = 0;
    for batch in batches {
        let logits_data = model.classification_logits(batch.images).into_data();
        let classes_data = batch.classes.into_data();
        let [batch_size, class_count] = logits_data.shape.dims::<2>();
        let values = logits_data.as_slice::<f32>()?;
        let labels = classes_data.as_slice::<i64>()?;
        let rows = values
            .chunks_exact(class_count)
            .map(Vec::from)
            .collect::<Vec<_>>();
        let labels = labels
            .iter()
            .map(|value| usize::try_from(*value))
            .collect::<Result<Vec<_>, _>>()?;
        let metrics = crate::training::loss::classification::classification_loss(&rows, &labels)?;
        loss += f64::from(metrics.mean_loss) * batch_size as f64;
        top1 += metrics.top1_correct;
        top5 += metrics.top5_correct;
        count += batch_size;
    }
    Ok(ValidationSummary {
        model: model_id,
        split: "val".into(),
        images: count,
        mean_loss: Some((loss / count as f64) as f32),
        top1_accuracy: Some(top1 as f32 / count as f32),
        top5_accuracy: Some(top5 as f32 / count as f32),
        box_metrics: None,
    })
}

trait ClassificationForward {
    fn classification_logits(&self, images: Tensor<Wgpu, 4>) -> Tensor<Wgpu, 2>;
}

macro_rules! classification_forward {
    ($($model:ty),+ $(,)?) => {$ (
        impl ClassificationForward for $model {
            fn classification_logits(&self, images: Tensor<Wgpu, 4>) -> Tensor<Wgpu, 2> {
                self.forward_train(images)
            }
        }
    )+ };
}

classification_forward!(
    crate::models::yolo11::Yolo11ClsN<Wgpu>,
    crate::models::yolo11::Yolo11ClsS<Wgpu>,
    crate::models::yolo11::Yolo11ClsM<Wgpu>,
    crate::models::yolo11::Yolo11ClsL<Wgpu>,
    crate::models::yolo11::Yolo11ClsX<Wgpu>,
    crate::models::yolo26::Yolo26ClsN<Wgpu>,
    crate::models::yolo26::Yolo26ClsS<Wgpu>,
    crate::models::yolo26::Yolo26ClsM<Wgpu>,
    crate::models::yolo26::Yolo26ClsL<Wgpu>,
    crate::models::yolo26::Yolo26ClsX<Wgpu>,
    crate::models::yolov8::Yolov8ClsN<Wgpu>,
    crate::models::yolov8::Yolov8ClsS<Wgpu>,
    crate::models::yolov8::Yolov8ClsM<Wgpu>,
    crate::models::yolov8::Yolov8ClsL<Wgpu>,
    crate::models::yolov8::Yolov8ClsX<Wgpu>,
);

const VALIDATION_CONFIDENCE: f32 = 0.001;
const VALIDATION_IOU: f32 = 0.7;
const VALIDATION_MAX_DETECTIONS: usize = 300;

trait DetectionForward {
    fn validation_detections(
        &self,
        images: Tensor<Wgpu, 4>,
    ) -> Vec<Vec<Vec<crate::models::yolox::BoundingBox>>>;
}

impl DetectionForward for Yolox<Wgpu> {
    fn validation_detections(
        &self,
        images: Tensor<Wgpu, 4>,
    ) -> Vec<Vec<Vec<crate::models::yolox::BoundingBox>>> {
        let output = self.forward(images * 255.0);
        let [batch, anchors, outputs] = output.dims();
        let boxes = output.clone().slice([0..batch, 0..anchors, 0..4]);
        let objectness = output.clone().slice([0..batch, 0..anchors, 4..5]);
        let scores = output.slice([0..batch, 0..anchors, 5..outputs]) * objectness;
        crate::models::yolox::boxes::nms(boxes, scores, VALIDATION_IOU, VALIDATION_CONFIDENCE)
    }
}

impl DetectionForward for crate::models::yolov3_tiny::Yolov3Tiny<Wgpu> {
    fn validation_detections(
        &self,
        images: Tensor<Wgpu, 4>,
    ) -> Vec<Vec<Vec<crate::models::yolox::BoundingBox>>> {
        let output = self.forward(images);
        let [batch, anchors, _] = output.boxes.dims();
        let left_top = output.boxes.clone().slice([0..batch, 0..anchors, 0..2]);
        let right_bottom = output.boxes.slice([0..batch, 0..anchors, 2..4]);
        let center = (left_top.clone() + right_bottom.clone()) / 2.0;
        let size = right_bottom - left_top;
        crate::models::yolox::boxes::nms(
            Tensor::cat(vec![center, size], 2),
            output.scores,
            VALIDATION_IOU,
            VALIDATION_CONFIDENCE,
        )
    }
}

macro_rules! classic_detection_forward {
    ($($model:ty),+ $(,)?) => {$ (
        impl DetectionForward for $model {
            fn validation_detections(
                &self,
                images: Tensor<Wgpu, 4>,
            ) -> Vec<Vec<Vec<crate::models::yolox::BoundingBox>>> {
                let output = self.forward(images);
                crate::models::yolox::boxes::nms(
                    output.boxes,
                    output.scores,
                    VALIDATION_IOU,
                    VALIDATION_CONFIDENCE,
                )
            }
        }
    )+ };
}

classic_detection_forward!(
    crate::models::yolo11::Yolo11N<Wgpu>,
    crate::models::yolo11::Yolo11S<Wgpu>,
    crate::models::yolo11::Yolo11M<Wgpu>,
    crate::models::yolo11::Yolo11L<Wgpu>,
    crate::models::yolo11::Yolo11X<Wgpu>,
);

macro_rules! end_to_end_detection_forward {
    ($($model:ty),+ $(,)?) => {$ (
        impl DetectionForward for $model {
            fn validation_detections(
                &self,
                images: Tensor<Wgpu, 4>,
            ) -> Vec<Vec<Vec<crate::models::yolox::BoundingBox>>> {
                let output = self.forward(images);
                crate::end2end_topk_detections(
                    output.boxes,
                    output.scores,
                    VALIDATION_MAX_DETECTIONS,
                    VALIDATION_CONFIDENCE,
                )
            }
        }
    )+ };
}

end_to_end_detection_forward!(
    crate::models::yolov10::Yolov10N<Wgpu>,
    crate::models::yolov10::Yolov10S<Wgpu>,
    crate::models::yolov10::Yolov10M<Wgpu>,
    crate::models::yolov10::Yolov10B<Wgpu>,
    crate::models::yolov10::Yolov10L<Wgpu>,
    crate::models::yolov10::Yolov10X<Wgpu>,
    crate::models::yolo26::Yolo26N<Wgpu>,
    crate::models::yolo26::Yolo26S<Wgpu>,
    crate::models::yolo26::Yolo26M<Wgpu>,
    crate::models::yolo26::Yolo26L<Wgpu>,
    crate::models::yolo26::Yolo26X<Wgpu>,
);

fn validate_detection_model<M>(
    model: M,
    bytes: Vec<u8>,
    batches: Vec<DetectionBatch<Wgpu>>,
    model_id: ModelId,
) -> Result<ValidationSummary, Box<dyn Error + Send + Sync>>
where
    M: burn::module::Module<Wgpu> + DetectionForward,
{
    let device = batches[0].images.device();
    let model = model.load_record(decode_record::<Wgpu, _>(bytes, &device)?);
    let mut predictions = Vec::new();
    let mut targets = Vec::new();
    let mut image_count = 0;
    for batch in batches {
        let [batch_size, max_targets] = batch.classes.dims();
        let classes_data = batch.classes.into_data();
        let boxes_data = batch.boxes_xyxy.into_data();
        let valid_data = batch.valid.into_data();
        let classes = classes_data.as_slice::<i64>()?;
        let boxes = boxes_data.as_slice::<f32>()?;
        let valid = valid_data.as_slice::<bool>()?;
        let grouped = model.validation_detections(batch.images);
        if grouped.len() != batch.metadata.len() {
            return Err("detector output batch size differs from validation metadata".into());
        }
        for (image, (per_class, metadata)) in
            grouped.into_iter().zip(batch.metadata.iter()).enumerate()
        {
            let mut image_predictions = per_class
                .into_iter()
                .enumerate()
                .flat_map(|(class_id, boxes)| boxes.into_iter().map(move |bbox| (class_id, bbox)))
                .collect::<Vec<_>>();
            image_predictions.sort_unstable_by(|a, b| b.1.confidence.total_cmp(&a.1.confidence));
            image_predictions.truncate(VALIDATION_MAX_DETECTIONS);
            for (class_id, bbox) in image_predictions {
                if let Some(source_bbox) =
                    source_box(metadata, [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax])
                {
                    predictions.push(crate::training::metrics::detection::MetricPrediction {
                        image_id: metadata.image_id.clone(),
                        class_id,
                        confidence: bbox.confidence,
                        bbox: source_bbox,
                    });
                }
            }
            for target in 0..max_targets {
                let flat = image * max_targets + target;
                if !valid[flat] {
                    continue;
                }
                let class_id = usize::try_from(classes[flat])?;
                if let Some(bbox) =
                    source_box(metadata, boxes[flat * 4..flat * 4 + 4].try_into().unwrap())
                {
                    targets.push(crate::training::metrics::detection::MetricTarget {
                        image_id: metadata.image_id.clone(),
                        class_id,
                        bbox,
                        crowd: false,
                    });
                }
            }
        }
        image_count += batch_size;
    }
    Ok(ValidationSummary {
        model: model_id,
        split: "val".into(),
        images: image_count,
        mean_loss: None,
        top1_accuracy: None,
        top5_accuracy: None,
        box_metrics: Some(crate::training::metrics::detection::evaluate(
            &predictions,
            &targets,
        )),
    })
}

fn source_box(
    metadata: &ImageMeta,
    canvas: [f32; 4],
) -> Option<crate::training::geometry::BoxXyxy> {
    let [width, height] = metadata.source_size.map(|value| value as f32);
    let mapped = [
        ((canvas[0] - metadata.pad[0]) / metadata.scale[0]).clamp(0.0, width),
        ((canvas[1] - metadata.pad[1]) / metadata.scale[1]).clamp(0.0, height),
        ((canvas[2] - metadata.pad[0]) / metadata.scale[0]).clamp(0.0, width),
        ((canvas[3] - metadata.pad[1]) / metadata.scale[1]).clamp(0.0, height),
    ];
    crate::training::geometry::BoxXyxy::new(mapped).ok()
}

#[cfg(feature = "pretrained")]
pub fn export(
    checkpoint: PathBuf,
    output: PathBuf,
) -> Result<PathBuf, Box<dyn Error + Send + Sync>> {
    let manifest = crate::training::checkpoint::load(&checkpoint)?;
    if manifest.config.model.task != crate::training::TaskKind::Classify {
        return Err("native training export currently supports classification checkpoints".into());
    }
    if manifest.config.model.num_classes != crate::models::yolo26::classification::NUM_CLASSES {
        return Err("custom-class inference artifact metadata is not yet supported; refusing to emit an artifact that the public Predictor cannot reload safely".into());
    }
    let (device, _) = crate::default_wgpu_device();
    let bytes = std::fs::read(checkpoint.join("ema.bin"))
        .or_else(|_| std::fs::read(checkpoint.join("model.bin")))?;
    macro_rules! save {
        ($config:expr) => {{
            let model = $config.init::<Wgpu>(&device);
            let model = model.load_record(decode_record::<Wgpu, _>(bytes, &device)?);
            model.save_burnpack_weights(&output)?;
            Ok(output)
        }};
    }
    match manifest.config.model.architecture {
        ModelId::Yolo11NCls => save!(Yolo11ClsNConfig),
        ModelId::Yolo11SCls => save!(Yolo11ClsSConfig),
        ModelId::Yolo11MCls => save!(Yolo11ClsMConfig),
        ModelId::Yolo11LCls => save!(Yolo11ClsLConfig),
        ModelId::Yolo11XCls => save!(Yolo11ClsXConfig),
        ModelId::Yolo26NCls => save!(Yolo26ClsNConfig),
        ModelId::Yolo26SCls => save!(Yolo26ClsSConfig),
        ModelId::Yolo26MCls => save!(Yolo26ClsMConfig),
        ModelId::Yolo26LCls => save!(Yolo26ClsLConfig),
        ModelId::Yolo26XCls => save!(Yolo26ClsXConfig),
        ModelId::Yolov8NCls => save!(Yolov8ClsNConfig),
        ModelId::Yolov8SCls => save!(Yolov8ClsSConfig),
        ModelId::Yolov8MCls => save!(Yolov8ClsMConfig),
        ModelId::Yolov8LCls => save!(Yolov8ClsLConfig),
        ModelId::Yolov8XCls => save!(Yolov8ClsXConfig),
        _ => Err("checkpoint model is not a supported classifier".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_boxes_map_back_through_composed_geometry() {
        let metadata = ImageMeta {
            image_id: "image".into(),
            source_size: [40, 20],
            canvas_size: [32, 32],
            scale: [0.5, 0.5],
            pad: [6.0, 11.0],
        };
        let bbox = source_box(&metadata, [6.0, 11.0, 26.0, 21.0]).unwrap();
        assert_eq!(
            [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax],
            [0.0, 0.0, 40.0, 20.0]
        );
        assert!(source_box(&metadata, [0.0, 0.0, 5.0, 5.0]).is_none());
    }
}
