use std::{error::Error, path::PathBuf};

use burn::{
    backend::{Autodiff, Wgpu},
    optim::{AdamWConfig, Optimizer},
};

use crate::{
    ModelId,
    data::augmentation::{
        ByteImage, ClassificationPipeline, ColorOrder, FormattedClassificationSample, SeedKey,
    },
    models::{
        yolo11::{
            Yolo11ClsLConfig, Yolo11ClsMConfig, Yolo11ClsNConfig, Yolo11ClsSConfig,
            Yolo11ClsXConfig,
        },
        yolo26::{
            Yolo26ClsLConfig, Yolo26ClsMConfig, Yolo26ClsNConfig, Yolo26ClsSConfig,
            Yolo26ClsXConfig,
        },
        yolov8::{
            Yolov8ClsLConfig, Yolov8ClsMConfig, Yolov8ClsNConfig, Yolov8ClsSConfig,
            Yolov8ClsXConfig,
        },
    },
    training::{
        ModelSpec, Trainer,
        checkpoint::{CheckpointManifest, encode_record, replace_atomic, save_atomic},
        config::{OptimizerKind, ScheduleKind, TrainingConfig},
        data::{
            DatasetFormat, DatasetManifest,
            batch::{ClassificationBatch, FormattedClassificationBatch},
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
}

pub fn train(request: TrainingRequest) -> Result<PathBuf, Box<dyn Error>> {
    let dataset = DatasetManifest::load(&request.data)?;
    if dataset.format != DatasetFormat::ClassificationFolders {
        return Err("the production CLI currently dispatches classification-folder training; detection/segmentation model adapters are available through training::Trainer but their dataset orchestration is not yet exposed".into());
    }
    let spec = ModelSpec::new(
        request.model,
        dataset.class_names.clone(),
        request.image_size.map(|side| [side, side]),
    )?;
    if spec.task != crate::training::TaskKind::Classify {
        return Err("classification-folder datasets require a -cls model".into());
    }
    let mut config = TrainingConfig::yolox(spec, request.data.clone(), request.run_root);
    config.epochs = request.epochs;
    config.batch_size = request.batch_size;
    config.accumulation = request.accumulation;
    config.optimizer = OptimizerKind::AdamW;
    config.schedule = ScheduleKind::Cosine;
    config.initial_lr = 1e-3;
    config.augmentation.imgsz = config.model.input_size[0];
    config.validate()?;
    let (device, adapter) = crate::default_wgpu_device();
    eprintln!("Training adapter: {adapter}");
    let batches = build_classification_batches(&config, &dataset, &device, 0)?;
    let mut trainer = Trainer::create(config.clone(), &request.name, batches.len())?;
    trainer.run.write_dataset(&dataset)?;
    let classes = config.model.num_classes;

    macro_rules! run {
        ($config:expr) => {{
            let model = $config.init_with_classes(classes, &device);
            run_classification(model, trainer, batches, request.dry_run)
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
        _ => return Err("train CLI dispatch for this task is not complete".into()),
    }?;
    Ok(run)
}

fn run_classification<M>(
    mut model: M,
    mut trainer: Trainer,
    batches: Vec<ClassificationBatch<TrainBackend>>,
    dry_run: bool,
) -> Result<PathBuf, Box<dyn Error>>
where
    M: TrainableTask<TrainBackend, Batch = ClassificationBatch<TrainBackend>>,
{
    if dry_run {
        let output = model.forward_loss(&batches[0])?;
        if !output.finite {
            return Err("dry-run loss is non-finite".into());
        }
        let gradients = burn::optim::GradientsParams::from_grads(output.total.backward(), &model);
        if gradients.is_empty() {
            return Err("dry-run produced no gradients".into());
        }
        return Ok(trainer.run.root);
    }
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(trainer.config.weight_decay as f32)
        .init();
    while trainer.state.epoch < trainer.config.epochs {
        let result = trainer.train_epoch::<TrainBackend, _, _, _>(
            model,
            optimizer,
            &batches,
            |_model, _step| Ok(()),
        )?;
        model = result.0;
        optimizer = result.1;
        let model_bytes = encode_record::<TrainBackend, _>(model.clone().into_record())?;
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
                ("optimizer.bin", &optimizer_bytes),
            ],
        )?;
        replace_atomic(
            trainer.run.checkpoints.join("last"),
            manifest,
            &[
                ("model.bin", &model_bytes),
                ("optimizer.bin", &optimizer_bytes),
            ],
        )?;
        eprintln!(
            "epoch {}: loss {:.6}",
            trainer.state.epoch, result.2.mean_loss
        );
    }
    Ok(trainer.run.root)
}

fn build_classification_batches(
    config: &TrainingConfig,
    dataset: &crate::training::data::ResolvedDataset,
    device: &burn::tensor::Device<TrainBackend>,
    epoch: u64,
) -> Result<Vec<ClassificationBatch<TrainBackend>>, Box<dyn Error>> {
    let pipeline = ClassificationPipeline::new(
        config
            .augmentation
            .resolve(crate::training::TaskKind::Classify, true)?,
    )?;
    let order = crate::training::data::loader::epoch_permutation(
        dataset.train_images.len(),
        config.seed,
        epoch,
    );
    let mut formatted = Vec::<FormattedClassificationSample>::with_capacity(order.len());
    let mut metadata = Vec::<ImageMeta>::with_capacity(order.len());
    for (logical_position, index) in order.into_iter().enumerate() {
        let path = &dataset.train_images[index];
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

pub fn inspect_checkpoint(path: PathBuf) -> Result<CheckpointManifest, Box<dyn Error>> {
    Ok(crate::training::checkpoint::load(path)?)
}
