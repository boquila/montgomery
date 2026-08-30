use std::{
    collections::{HashMap, VecDeque},
    error::Error,
    path::PathBuf,
    sync::Arc,
};

use burn::{
    backend::{Autodiff, Wgpu},
    module::Module,
    optim::{Optimizer, SgdConfig, momentum::MomentumConfig},
    tensor::Tensor,
    tensor::backend::Backend,
};
use burn_store::{ModuleSnapshot, PathFilter};
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
        yolo12::{Yolo12LConfig, Yolo12MConfig, Yolo12NConfig, Yolo12SConfig, Yolo12XConfig},
        yolo26::{
            Yolo26ClsLConfig, Yolo26ClsMConfig, Yolo26ClsNConfig, Yolo26ClsSConfig,
            Yolo26ClsXConfig, Yolo26LConfig, Yolo26MConfig, Yolo26NConfig, Yolo26SConfig,
            Yolo26XConfig,
        },
        yolov3_tiny::Yolov3TinyConfig,
        yolov8::{
            Yolov8ClsLConfig, Yolov8ClsMConfig, Yolov8ClsNConfig, Yolov8ClsSConfig,
            Yolov8ClsXConfig, Yolov8LConfig, Yolov8MConfig, Yolov8NConfig, Yolov8SConfig,
            Yolov8SegLConfig, Yolov8SegMConfig, Yolov8SegNConfig, Yolov8SegSConfig,
            Yolov8SegXConfig, Yolov8XConfig,
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
            CheckpointManifest, decode_record, encode_record, replace_atomic_from_saved,
            save_atomic,
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
        engine::{EpochBatchSource, TrainableTask},
    },
};

type TrainBackend = Autodiff<Wgpu>;

fn prefetch_sample_capacity(batch_size: usize, prefetch: usize) -> usize {
    batch_size.saturating_mul(prefetch).max(1)
}

struct ClassificationBatchSource<'a, B: Backend> {
    config: &'a TrainingConfig,
    dataset: &'a crate::training::data::ResolvedDataset,
    images: &'a [PathBuf],
    device: &'a B::Device,
    pipeline: ClassificationPipeline,
    order: Vec<usize>,
    epoch: u64,
    next_sample: usize,
    pending: VecDeque<(FormattedClassificationBatch, Vec<ImageMeta>)>,
}

impl<'a, B: Backend> ClassificationBatchSource<'a, B> {
    fn new(
        config: &'a TrainingConfig,
        dataset: &'a crate::training::data::ResolvedDataset,
        images: &'a [PathBuf],
        device: &'a B::Device,
        epoch: u64,
        training: bool,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        Ok(Self {
            config,
            dataset,
            images,
            device,
            pipeline: ClassificationPipeline::new(
                config
                    .augmentation
                    .resolve(crate::training::TaskKind::Classify, training)?,
            )?,
            order: crate::training::data::loader::epoch_permutation(
                images.len(),
                config.seed,
                epoch,
            ),
            epoch,
            next_sample: 0,
            pending: VecDeque::with_capacity(config.prefetch),
        })
    }

    fn refill(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        if self.next_sample >= self.order.len() {
            return Ok(());
        }
        let sample_count = prefetch_sample_capacity(self.config.batch_size, self.config.prefetch);
        let end = (self.next_sample + sample_count).min(self.order.len());
        let indexed = (self.next_sample..end)
            .map(|logical_position| (logical_position, self.order[logical_position]))
            .collect::<Vec<_>>();
        let workers = self.config.workers.max(1).min(indexed.len());
        let worker_chunk = indexed.len().div_ceil(workers);
        let config = self.config;
        let dataset = self.dataset;
        let images = self.images;
        let epoch = self.epoch;
        let mut prepared =
            std::thread::scope(|scope| -> Result<Vec<_>, Box<dyn Error + Send + Sync>> {
                let handles = indexed
                    .chunks(worker_chunk.max(1))
                    .map(|chunk| {
                        let pipeline = self.pipeline.clone();
                        scope.spawn(move || {
                            chunk
                                .iter()
                                .map(|&(logical_position, index)| {
                                    prepare_classification_sample(
                                        config,
                                        dataset,
                                        images,
                                        &pipeline,
                                        epoch,
                                        logical_position,
                                        index,
                                    )
                                })
                                .collect::<Result<Vec<_>, _>>()
                        })
                    })
                    .collect::<Vec<_>>();
                let mut output = Vec::with_capacity(indexed.len());
                for handle in handles {
                    output.extend(handle.join().map_err(|_| {
                        Box::<dyn Error + Send + Sync>::from("classification data worker panicked")
                    })??);
                }
                Ok(output)
            })?;
        prepared.sort_unstable_by_key(|sample| sample.0);
        let mut prepared = prepared.into_iter();
        loop {
            let chunk = prepared
                .by_ref()
                .take(self.config.batch_size)
                .collect::<Vec<_>>();
            if chunk.is_empty() {
                break;
            }
            let (formatted, metadata): (Vec<_>, Vec<_>) = chunk
                .into_iter()
                .map(|(_, sample, metadata)| (sample, metadata))
                .unzip();
            self.pending
                .push_back((FormattedClassificationBatch::collate(&formatted)?, metadata));
        }
        self.next_sample = end;
        Ok(())
    }
}

impl<B: Backend> EpochBatchSource<ClassificationBatch<B>> for ClassificationBatchSource<'_, B> {
    fn batch_count(&self) -> usize {
        self.images.len().div_ceil(self.config.batch_size)
    }

    fn next_batch(&mut self) -> Result<Option<ClassificationBatch<B>>, String> {
        if self.pending.is_empty() {
            self.refill().map_err(|error| error.to_string())?;
        }
        self.pending
            .pop_front()
            .map(|(batch, metadata)| batch.into_device(metadata, self.device))
            .transpose()
    }
}

#[derive(Clone)]
enum VisionSampleLoader<'a> {
    Yolo {
        dataset: &'a crate::training::data::ResolvedDataset,
        images: &'a [PathBuf],
    },
    Coco {
        index: Arc<crate::training::data::coco::CocoIndex>,
    },
}

impl<'a> VisionSampleLoader<'a> {
    fn new(
        dataset: &'a crate::training::data::ResolvedDataset,
        images: &'a [PathBuf],
        training: bool,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        match dataset.format {
            DatasetFormat::Yolo => Ok(Self::Yolo { dataset, images }),
            DatasetFormat::Coco => {
                let annotation = if images == dataset.train_images {
                    dataset.train_annotations.as_ref()
                } else if images == dataset.val_images {
                    dataset.val_annotations.as_ref()
                } else if images == dataset.test_images {
                    dataset.test_annotations.as_ref()
                } else {
                    None
                }
                .ok_or("COCO split has no resolved annotation file")?;
                let images_root = images
                    .first()
                    .and_then(|path| path.parent())
                    .ok_or("COCO split contains no image root")?;
                let mut index = crate::training::data::coco::load_index(annotation, images_root)?;
                if index.class_names != dataset.class_names {
                    return Err("COCO category table differs from dataset names".into());
                }
                let mut by_path = index
                    .records
                    .drain(..)
                    .map(|record| (record.path.clone(), record))
                    .collect::<std::collections::BTreeMap<_, _>>();
                index.records = images
                    .iter()
                    .map(|path| {
                        by_path.remove(path).ok_or_else(|| {
                            format!(
                                "COCO annotations have no image record for {}",
                                path.display()
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if training {
                    for record in &mut index.records {
                        record.targets.retain(|target| !target.crowd);
                    }
                }
                Ok(Self::Coco {
                    index: Arc::new(index),
                })
            }
            DatasetFormat::ClassificationFolders => {
                Err("classification folders cannot feed a detector loader".into())
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Yolo { images, .. } => images.len(),
            Self::Coco { index } => index.records.len(),
        }
    }

    fn load(
        &self,
        index: usize,
    ) -> Result<crate::training::data::VisionSample, Box<dyn Error + Send + Sync>> {
        match self {
            Self::Yolo { dataset, images } => Ok(crate::training::data::loader::load_yolo_sample(
                dataset,
                &images[index],
            )?),
            Self::Coco { index: dataset } => Ok(dataset.load_sample(index, false)?),
        }
    }
}

#[derive(Clone)]
struct CachedAugSample {
    sample: AugSample,
    crowd: Vec<bool>,
}

struct BoundedLru<T> {
    entries: HashMap<usize, T>,
    order: VecDeque<usize>,
    capacity: usize,
    peak: usize,
}

impl<T: Clone> BoundedLru<T> {
    fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            capacity: capacity.max(1),
            peak: 0,
        }
    }

    fn get(&mut self, index: usize) -> Option<T> {
        let entry = self.entries.get(&index).cloned()?;
        self.touch(index);
        Some(entry)
    }

    fn insert(&mut self, index: usize, entry: T) {
        if self.entries.len() == self.capacity
            && !self.entries.contains_key(&index)
            && let Some(evicted) = self.order.pop_front()
        {
            self.entries.remove(&evicted);
        }
        self.entries.insert(index, entry);
        self.touch(index);
        self.peak = self.peak.max(self.entries.len());
    }

    fn touch(&mut self, index: usize) {
        if let Some(position) = self.order.iter().position(|cached| *cached == index) {
            self.order.remove(position);
        }
        self.order.push_back(index);
    }
}

struct LazyPartnerPool<'a> {
    loader: VisionSampleLoader<'a>,
    cache: BoundedLru<CachedAugSample>,
    imgsz: usize,
}

impl<'a> LazyPartnerPool<'a> {
    fn new(loader: VisionSampleLoader<'a>, capacity: usize, imgsz: usize) -> Self {
        Self {
            loader,
            cache: BoundedLru::new(capacity),
            imgsz,
        }
    }

    fn load(
        &mut self,
        index: usize,
    ) -> Result<CachedAugSample, crate::data::augmentation::AugmentationError> {
        if let Some(entry) = self.cache.get(index) {
            return Ok(entry);
        }
        let vision = self.loader.load(index).map_err(|error| {
            crate::data::augmentation::AugmentationError::new(error.to_string())
        })?;
        let crowd = vision.targets.iter().map(|target| target.crowd).collect();
        let sample = AugSample::from_vision(vision, index, self.imgsz, false)?;
        let entry = CachedAugSample { sample, crowd };
        self.cache.insert(index, entry.clone());
        Ok(entry)
    }
}

impl crate::data::augmentation::PartnerProvider for LazyPartnerPool<'_> {
    fn len(&self) -> usize {
        self.loader.len()
    }

    fn get(
        &mut self,
        index: usize,
    ) -> Result<AugSample, crate::data::augmentation::AugmentationError> {
        Ok(self.load(index)?.sample)
    }

    fn candidate_index(&self, logical_position: usize, draw: usize) -> usize {
        let len = self.len();
        if len == 0 {
            return 0;
        }
        let start = logical_position.saturating_sub(len - 1) % len;
        (start + draw.wrapping_mul(0x9e3779b9usize) % len) % len
    }
}

struct VisionEpochFormatter<'a> {
    config: &'a TrainingConfig,
    images: &'a [PathBuf],
    pipeline: AugmentationPipeline,
    pools: Vec<LazyPartnerPool<'a>>,
    order: Vec<usize>,
    epoch: u64,
    next_sample: usize,
    pending: VecDeque<FormattedVisionBatch>,
}

type FormattedVisionBatch = (Vec<FormattedDetectionSample>, Vec<ImageMeta>);

impl<'a> VisionEpochFormatter<'a> {
    fn new(
        config: &'a TrainingConfig,
        images: &'a [PathBuf],
        epoch: u64,
        task: crate::training::TaskKind,
        loader: VisionSampleLoader<'a>,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let capacity = prefetch_sample_capacity(config.batch_size, config.prefetch);
        let worker_count = config.workers.max(1).min(capacity);
        let capacity_per_worker = capacity / worker_count;
        let capacity_remainder = capacity % worker_count;
        let pools = (0..worker_count)
            .map(|worker| {
                LazyPartnerPool::new(
                    loader.clone(),
                    capacity_per_worker + usize::from(worker < capacity_remainder),
                    config.augmentation.imgsz,
                )
            })
            .collect();
        Ok(Self {
            config,
            images,
            pipeline: AugmentationPipeline::for_epoch(
                config.augmentation.resolve(task, true)?,
                epoch as usize,
                config.epochs,
            )?,
            pools,
            order: crate::training::data::loader::epoch_permutation(
                images.len(),
                config.seed,
                epoch,
            ),
            epoch,
            next_sample: 0,
            pending: VecDeque::with_capacity(config.prefetch),
        })
    }

    fn batch_count(&self) -> usize {
        self.images.len().div_ceil(self.config.batch_size)
    }

    fn next_formatted(&mut self) -> Result<Option<FormattedVisionBatch>, String> {
        if self.pending.is_empty() {
            self.refill().map_err(|error| error.to_string())?;
        }
        Ok(self.pending.pop_front())
    }

    fn refill(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let end = (self.next_sample
            + prefetch_sample_capacity(self.config.batch_size, self.config.prefetch))
        .min(self.order.len());
        if self.next_sample == end {
            return Ok(());
        }
        let indexed = (self.next_sample..end)
            .map(|logical_position| (logical_position, self.order[logical_position]))
            .collect::<Vec<_>>();
        let worker_count = self.pools.len().min(indexed.len());
        let worker_chunk = indexed.len().div_ceil(worker_count);
        let config = self.config;
        let images = self.images;
        let epoch = self.epoch;
        let pipeline = &self.pipeline;
        let mut prepared =
            std::thread::scope(|scope| -> Result<Vec<_>, Box<dyn Error + Send + Sync>> {
                let handles = self
                    .pools
                    .iter_mut()
                    .take(worker_count)
                    .zip(indexed.chunks(worker_chunk))
                    .map(|(pool, chunk)| {
                        let pipeline = pipeline.clone();
                        scope.spawn(move || {
                            chunk
                                .iter()
                                .map(|&(logical_position, index)| {
                                    prepare_vision_sample(
                                        config,
                                        images,
                                        &pipeline,
                                        pool,
                                        epoch,
                                        logical_position,
                                        index,
                                    )
                                })
                                .collect::<Result<Vec<_>, _>>()
                        })
                    })
                    .collect::<Vec<_>>();
                let mut output = Vec::with_capacity(indexed.len());
                for handle in handles {
                    output.extend(handle.join().map_err(|_| {
                        Box::<dyn Error + Send + Sync>::from("vision data worker panicked")
                    })??);
                }
                Ok(output)
            })?;
        prepared.sort_unstable_by_key(|(logical_position, _, _)| *logical_position);
        let mut prepared = prepared.into_iter();
        loop {
            let chunk = prepared
                .by_ref()
                .take(self.config.batch_size)
                .collect::<Vec<_>>();
            if chunk.is_empty() {
                break;
            }
            let (samples, metadata): (Vec<_>, Vec<_>) = chunk
                .into_iter()
                .map(|(_, sample, metadata)| (sample, metadata))
                .unzip();
            self.pending.push_back((samples, metadata));
        }
        self.next_sample = end;
        Ok(())
    }
}

fn prepare_vision_sample(
    config: &TrainingConfig,
    images: &[PathBuf],
    pipeline: &AugmentationPipeline,
    pool: &mut LazyPartnerPool<'_>,
    epoch: u64,
    logical_position: usize,
    index: usize,
) -> Result<(usize, FormattedDetectionSample, ImageMeta), Box<dyn Error + Send + Sync>> {
    let primary = pool.load(index)?;
    let path_text = images[index].to_string_lossy().into_owned();
    let (sample, _) = pipeline.apply(
        primary.sample,
        pool,
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
    let metadata = ImageMeta {
        image_id: path_text,
        source_size: [
            geometry.original_shape[1] as u32,
            geometry.original_shape[0] as u32,
        ],
        canvas_size: [canvas_width as u32, canvas_height as u32],
        scale: geometry.ratio,
        pad: geometry.pad,
        crowd: primary.crowd,
    };
    Ok((logical_position, sample, metadata))
}

struct DetectionBatchSource<'a, B: Backend> {
    formatter: VisionEpochFormatter<'a>,
    device: &'a B::Device,
}

impl<'a, B: Backend> DetectionBatchSource<'a, B> {
    fn new(
        config: &'a TrainingConfig,
        images: &'a [PathBuf],
        device: &'a B::Device,
        epoch: u64,
        loader: VisionSampleLoader<'a>,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        Ok(Self {
            formatter: VisionEpochFormatter::new(
                config,
                images,
                epoch,
                crate::training::TaskKind::Detect,
                loader,
            )?,
            device,
        })
    }
}

impl<B: Backend> EpochBatchSource<DetectionBatch<B>> for DetectionBatchSource<'_, B> {
    fn batch_count(&self) -> usize {
        self.formatter.batch_count()
    }

    fn next_batch(&mut self) -> Result<Option<DetectionBatch<B>>, String> {
        self.formatter
            .next_formatted()?
            .map(|(samples, metadata)| {
                FormattedDetectionBatch::collate(&samples)?.into_device(metadata, self.device)
            })
            .transpose()
    }
}

struct SegmentationBatchSource<'a, B: Backend> {
    formatter: VisionEpochFormatter<'a>,
    device: &'a B::Device,
}

impl<'a, B: Backend> SegmentationBatchSource<'a, B> {
    fn new(
        config: &'a TrainingConfig,
        images: &'a [PathBuf],
        device: &'a B::Device,
        epoch: u64,
        loader: VisionSampleLoader<'a>,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        Ok(Self {
            formatter: VisionEpochFormatter::new(
                config,
                images,
                epoch,
                crate::training::TaskKind::Segment,
                loader,
            )?,
            device,
        })
    }
}

impl<B: Backend> EpochBatchSource<SegmentationBatch<B>> for SegmentationBatchSource<'_, B> {
    fn batch_count(&self) -> usize {
        self.formatter.batch_count()
    }

    fn next_batch(&mut self) -> Result<Option<SegmentationBatch<B>>, String> {
        self.formatter
            .next_formatted()?
            .map(|(samples, metadata)| segmentation_into_device(&samples, metadata, self.device))
            .transpose()
    }
}

#[derive(Debug, Clone, Copy)]
enum ReplacedProjection {
    Classifier,
    Detector,
    Yolox,
    Yolov3,
    Yolo26Segment,
}

impl ReplacedProjection {
    fn official_classes(self) -> usize {
        match self {
            Self::Classifier => crate::models::yolo26::classification::NUM_CLASSES,
            _ => 80,
        }
    }

    fn filter(self) -> fn(&str, &str) -> bool {
        match self {
            Self::Classifier => keep_without_classifier,
            Self::Detector => keep_without_detector_classes,
            Self::Yolox => keep_without_yolox_classes,
            Self::Yolov3 => keep_without_yolov3_classes,
            Self::Yolo26Segment => keep_without_yolo26_segment_classes,
        }
    }

    fn is_replaced(self, path: &str) -> bool {
        !(self.filter())(path, "")
    }
}

fn keep_without_classifier(path: &str, _container: &str) -> bool {
    !path.starts_with("head.linear.")
}

fn keep_without_detector_classes(path: &str, _container: &str) -> bool {
    !path.contains(".cls_out.")
}

fn keep_without_yolox_classes(path: &str, _container: &str) -> bool {
    !path.starts_with("head.cls_preds.")
}

fn keep_without_yolov3_classes(path: &str, _container: &str) -> bool {
    !(path.starts_with("head.p4.cls_2.") || path.starts_with("head.p5.cls_2."))
}

fn keep_without_yolo26_segment_classes(path: &str, container: &str) -> bool {
    keep_without_detector_classes(path, container) && !path.starts_with("head.proto.sem_out.")
}

fn transfer_pretrained<B, M>(
    mut target: M,
    official: &M,
    projection: ReplacedProjection,
) -> Result<M, Box<dyn Error + Send + Sync>>
where
    B: Backend,
    M: Module<B>,
{
    let snapshots = official.collect(None, None, false);
    let result = target.apply(
        snapshots,
        Some(PathFilter::new().with_predicate(projection.filter())),
        None,
        false,
    );
    if !result.errors.is_empty() || !result.missing.is_empty() || !result.unused.is_empty() {
        return Err(format!(
            "pretrained transfer differed outside the documented class projections:\n{result}"
        )
        .into());
    }
    if result.skipped.is_empty()
        || result
            .skipped
            .iter()
            .any(|path| !projection.is_replaced(path))
    {
        return Err(format!(
            "pretrained transfer did not isolate the documented class projections:\n{result}"
        )
        .into());
    }
    eprintln!(
        "Initialized {} tensors from pretrained weights; freshly initialized {} class-projection tensors",
        result.applied.len(),
        result.skipped.len()
    );
    Ok(target)
}

#[derive(Debug, Clone)]
pub struct TrainingRequest {
    pub model: ModelId,
    pub data: PathBuf,
    pub epochs: usize,
    pub batch_size: usize,
    pub accumulation: usize,
    pub workers: usize,
    pub prefetch: usize,
    pub image_size: Option<usize>,
    pub seed: u64,
    pub run_root: PathBuf,
    pub name: String,
    pub dry_run: bool,
    pub resume: Option<PathBuf>,
    pub weights: Option<PathBuf>,
    pub val_confidence: Option<f32>,
    pub val_iou: Option<f32>,
    pub max_detections: Option<usize>,
}

pub fn train(request: TrainingRequest) -> Result<PathBuf, Box<dyn Error + Send + Sync>> {
    let worker = std::thread::Builder::new()
        .name("boquilens-training".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || train_inner(request))?;
    worker.join().map_err(|_| {
        Box::<dyn Error + Send + Sync>::from("native training worker thread panicked")
    })?
}

fn train_inner(request: TrainingRequest) -> Result<PathBuf, Box<dyn Error + Send + Sync>> {
    if request.resume.is_some() && request.weights.is_some() {
        return Err("--resume and --weights are mutually exclusive".into());
    }
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
        config.workers = request.workers;
        config.prefetch = request.prefetch;
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
        if let Some(value) = request.val_confidence {
            config.validation.confidence = value;
        }
        if let Some(value) = request.val_iou {
            config.validation.iou = value;
        }
        if let Some(value) = request.max_detections {
            config.validation.max_detections = value;
        }
    }
    config.validate()?;
    let (device, adapter) = crate::default_wgpu_device();
    TrainBackend::seed(&device, config.seed);
    eprintln!("Training adapter: {adapter}");
    let epoch = resume_manifest
        .as_ref()
        .map_or(0, |value| value.state.epoch as u64);
    let batch_count = dataset.train_images.len().div_ceil(config.batch_size);
    if batch_count == 0 {
        return Err("training split produces no batches".into());
    }
    let trainer = if let Some(resume) = &request.resume {
        Trainer::from_checkpoint(resume)?
    } else {
        let trainer = Trainer::create(config.clone(), &request.name, batch_count)?;
        trainer.run.write_dataset(&dataset)?;
        trainer.run.write_environment(&adapter, &dataset)?;
        trainer
    };
    let classes = config.model.num_classes;

    macro_rules! pretrained {
        ($target:expr, $official:expr, $projection:expr) => {{
            let mut target = $target;
            if let Some(weights) = request.weights.as_ref() {
                if classes == $projection.official_classes() {
                    target.load_pytorch_weights(weights)?;
                } else {
                    let mut official = $official;
                    official.load_pytorch_weights(weights)?;
                    target = transfer_pretrained(target, &official, $projection)?;
                }
            }
            target
        }};
    }

    macro_rules! run {
        ($config:expr) => {{
            let model = pretrained!(
                $config.init_with_classes(classes, &device),
                $config.init(&device),
                ReplacedProjection::Classifier
            );
            run_task(
                model,
                trainer,
                ClassificationBatchSource::new(
                    &config,
                    &dataset,
                    &dataset.train_images,
                    &device,
                    epoch,
                    true,
                )?,
                |epoch| {
                    ClassificationBatchSource::new(
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
        ($model:expr, $official:expr, $projection:expr) => {{
            let model = pretrained!($model, $official, $projection);
            let loader = VisionSampleLoader::new(&dataset, &dataset.train_images, true)?;
            run_task(
                model,
                trainer,
                DetectionBatchSource::new(
                    &config,
                    &dataset.train_images,
                    &device,
                    epoch,
                    loader.clone(),
                )?,
                |epoch| {
                    DetectionBatchSource::new(
                        &config,
                        &dataset.train_images,
                        &device,
                        epoch,
                        loader.clone(),
                    )
                },
                request.dry_run,
                request.resume.as_ref(),
                &device,
            )
        }};
    }
    macro_rules! run_segment {
        ($model:expr, $official:expr, $projection:expr) => {{
            let model = pretrained!($model, $official, $projection);
            let loader = VisionSampleLoader::new(&dataset, &dataset.train_images, true)?;
            run_task(
                model,
                trainer,
                SegmentationBatchSource::new(
                    &config,
                    &dataset.train_images,
                    &device,
                    epoch,
                    loader.clone(),
                )?,
                |epoch| {
                    SegmentationBatchSource::new(
                        &config,
                        &dataset.train_images,
                        &device,
                        epoch,
                        loader.clone(),
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
        ModelId::YoloxNano => run_detect!(
            Yolox::yolox_nano(classes, &device),
            Yolox::yolox_nano(80, &device),
            ReplacedProjection::Yolox
        ),
        ModelId::YoloxTiny => run_detect!(
            Yolox::yolox_tiny(classes, &device),
            Yolox::yolox_tiny(80, &device),
            ReplacedProjection::Yolox
        ),
        ModelId::YoloxS => run_detect!(
            Yolox::yolox_s(classes, &device),
            Yolox::yolox_s(80, &device),
            ReplacedProjection::Yolox
        ),
        ModelId::YoloxM => run_detect!(
            Yolox::yolox_m(classes, &device),
            Yolox::yolox_m(80, &device),
            ReplacedProjection::Yolox
        ),
        ModelId::YoloxL => run_detect!(
            Yolox::yolox_l(classes, &device),
            Yolox::yolox_l(80, &device),
            ReplacedProjection::Yolox
        ),
        ModelId::YoloxX => run_detect!(
            Yolox::yolox_x(classes, &device),
            Yolox::yolox_x(80, &device),
            ReplacedProjection::Yolox
        ),
        ModelId::Yolov3TinyU => run_detect!(
            Yolov3TinyConfig.init_with_classes(classes, &device),
            Yolov3TinyConfig.init(&device),
            ReplacedProjection::Yolov3
        ),
        ModelId::Yolov10N => run_detect!(
            Yolov10NConfig.init_with_classes(classes, &device),
            Yolov10NConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolov10S => run_detect!(
            Yolov10SConfig.init_with_classes(classes, &device),
            Yolov10SConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolov10M => run_detect!(
            Yolov10MConfig.init_with_classes(classes, &device),
            Yolov10MConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolov10B => run_detect!(
            Yolov10BConfig.init_with_classes(classes, &device),
            Yolov10BConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolov10L => run_detect!(
            Yolov10LConfig.init_with_classes(classes, &device),
            Yolov10LConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolov10X => run_detect!(
            Yolov10XConfig.init_with_classes(classes, &device),
            Yolov10XConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolo11N => {
            run_detect!(
                crate::models::yolo11::Yolo11NConfig.init_with_classes(classes, &device),
                crate::models::yolo11::Yolo11NConfig.init(&device),
                ReplacedProjection::Detector
            )
        }
        ModelId::Yolo11S => {
            run_detect!(
                crate::models::yolo11::Yolo11SConfig.init_with_classes(classes, &device),
                crate::models::yolo11::Yolo11SConfig.init(&device),
                ReplacedProjection::Detector
            )
        }
        ModelId::Yolo11M => {
            run_detect!(
                crate::models::yolo11::Yolo11MConfig.init_with_classes(classes, &device),
                crate::models::yolo11::Yolo11MConfig.init(&device),
                ReplacedProjection::Detector
            )
        }
        ModelId::Yolo11L => {
            run_detect!(
                crate::models::yolo11::Yolo11LConfig.init_with_classes(classes, &device),
                crate::models::yolo11::Yolo11LConfig.init(&device),
                ReplacedProjection::Detector
            )
        }
        ModelId::Yolo11X => {
            run_detect!(
                crate::models::yolo11::Yolo11XConfig.init_with_classes(classes, &device),
                crate::models::yolo11::Yolo11XConfig.init(&device),
                ReplacedProjection::Detector
            )
        }
        ModelId::Yolo26N => run_detect!(
            Yolo26NConfig.init_with_classes(classes, &device),
            Yolo26NConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolo26S => run_detect!(
            Yolo26SConfig.init_with_classes(classes, &device),
            Yolo26SConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolo26M => run_detect!(
            Yolo26MConfig.init_with_classes(classes, &device),
            Yolo26MConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolo26L => run_detect!(
            Yolo26LConfig.init_with_classes(classes, &device),
            Yolo26LConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolo26X => run_detect!(
            Yolo26XConfig.init_with_classes(classes, &device),
            Yolo26XConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolov8N => run_detect!(
            Yolov8NConfig.init_with_classes(classes, &device),
            Yolov8NConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolov8S => run_detect!(
            Yolov8SConfig.init_with_classes(classes, &device),
            Yolov8SConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolov8M => run_detect!(
            Yolov8MConfig.init_with_classes(classes, &device),
            Yolov8MConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolov8L => run_detect!(
            Yolov8LConfig.init_with_classes(classes, &device),
            Yolov8LConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolov8X => run_detect!(
            Yolov8XConfig.init_with_classes(classes, &device),
            Yolov8XConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolo12N => run_detect!(
            Yolo12NConfig.init_with_classes(classes, &device),
            Yolo12NConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolo12S => run_detect!(
            Yolo12SConfig.init_with_classes(classes, &device),
            Yolo12SConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolo12M => run_detect!(
            Yolo12MConfig.init_with_classes(classes, &device),
            Yolo12MConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolo12L => run_detect!(
            Yolo12LConfig.init_with_classes(classes, &device),
            Yolo12LConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolo12X => run_detect!(
            Yolo12XConfig.init_with_classes(classes, &device),
            Yolo12XConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolo11NSeg => run_segment!(
            crate::models::yolo11::Yolo11SegNConfig.init_with_classes(classes, &device),
            crate::models::yolo11::Yolo11SegNConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolo11SSeg => run_segment!(
            crate::models::yolo11::Yolo11SegSConfig.init_with_classes(classes, &device),
            crate::models::yolo11::Yolo11SegSConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolo11MSeg => run_segment!(
            crate::models::yolo11::Yolo11SegMConfig.init_with_classes(classes, &device),
            crate::models::yolo11::Yolo11SegMConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolo11LSeg => run_segment!(
            crate::models::yolo11::Yolo11SegLConfig.init_with_classes(classes, &device),
            crate::models::yolo11::Yolo11SegLConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolo11XSeg => run_segment!(
            crate::models::yolo11::Yolo11SegXConfig.init_with_classes(classes, &device),
            crate::models::yolo11::Yolo11SegXConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolov8NSeg => run_segment!(
            Yolov8SegNConfig.init_with_classes(classes, &device),
            Yolov8SegNConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolov8SSeg => run_segment!(
            Yolov8SegSConfig.init_with_classes(classes, &device),
            Yolov8SegSConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolov8MSeg => run_segment!(
            Yolov8SegMConfig.init_with_classes(classes, &device),
            Yolov8SegMConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolov8LSeg => run_segment!(
            Yolov8SegLConfig.init_with_classes(classes, &device),
            Yolov8SegLConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolov8XSeg => run_segment!(
            Yolov8SegXConfig.init_with_classes(classes, &device),
            Yolov8SegXConfig.init(&device),
            ReplacedProjection::Detector
        ),
        ModelId::Yolo26NSeg => run_segment!(
            crate::models::yolo26::Yolo26SegNConfig.init_with_classes(classes, &device),
            crate::models::yolo26::Yolo26SegNConfig.init(&device),
            ReplacedProjection::Yolo26Segment
        ),
        ModelId::Yolo26SSeg => run_segment!(
            crate::models::yolo26::Yolo26SegSConfig.init_with_classes(classes, &device),
            crate::models::yolo26::Yolo26SegSConfig.init(&device),
            ReplacedProjection::Yolo26Segment
        ),
        ModelId::Yolo26MSeg => run_segment!(
            crate::models::yolo26::Yolo26SegMConfig.init_with_classes(classes, &device),
            crate::models::yolo26::Yolo26SegMConfig.init(&device),
            ReplacedProjection::Yolo26Segment
        ),
        ModelId::Yolo26LSeg => run_segment!(
            crate::models::yolo26::Yolo26SegLConfig.init_with_classes(classes, &device),
            crate::models::yolo26::Yolo26SegLConfig.init(&device),
            ReplacedProjection::Yolo26Segment
        ),
        ModelId::Yolo26XSeg => run_segment!(
            crate::models::yolo26::Yolo26SegXConfig.init_with_classes(classes, &device),
            crate::models::yolo26::Yolo26SegXConfig.init(&device),
            ReplacedProjection::Yolo26Segment
        ),
    }?;
    Ok(run)
}

fn run_task<M, F, S>(
    model: M,
    trainer: Trainer,
    mut batches: S,
    rebuild_batches: F,
    dry_run: bool,
    resume: Option<&PathBuf>,
    device: &burn::tensor::Device<TrainBackend>,
) -> Result<PathBuf, Box<dyn Error + Send + Sync>>
where
    M: TrainableTask<TrainBackend> + Clone,
    F: FnMut(u64) -> Result<S, Box<dyn Error + Send + Sync>>,
    S: EpochBatchSource<M::Batch>,
{
    if dry_run {
        let batch = batches
            .next_batch()?
            .ok_or("dry-run batch source produced no batches")?;
        let output = model.forward_loss(
            &batch,
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
        let deferred_finite = output.deferred_component.is_none()
            || crate::training::loss::common::scalar_value(output.total.clone()).is_finite();
        if !output.finite || !deferred_finite {
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
    let weight_decay = trainer.config.weight_decay as f32;
    match trainer.config.optimizer {
        OptimizerKind::AdamW => run_task_with_optimizer(
            model,
            trainer,
            batches,
            rebuild_batches,
            resume,
            device,
            (
                crate::training::optimizer::selective_adamw::<TrainBackend, M>(
                    weight_decay,
                    gradient_clip,
                ),
                false,
            ),
        ),
        OptimizerKind::Sgd => run_task_with_optimizer(
            model,
            trainer,
            batches,
            rebuild_batches,
            resume,
            device,
            (
                SgdConfig::new()
                    .with_momentum(Some(
                        MomentumConfig::new()
                            .with_momentum(momentum)
                            .with_dampening(0.0)
                            .with_nesterov(true),
                    ))
                    .with_gradient_clipping(Some(
                        burn::grad_clipping::GradientClippingConfig::Norm(gradient_clip),
                    ))
                    .init(),
                true,
            ),
        ),
    }
}

fn run_task_with_optimizer<M, O, F, S>(
    mut model: M,
    mut trainer: Trainer,
    mut batches: S,
    mut rebuild_batches: F,
    resume: Option<&PathBuf>,
    device: &burn::tensor::Device<TrainBackend>,
    optimizer: (O, bool),
) -> Result<PathBuf, Box<dyn Error + Send + Sync>>
where
    M: TrainableTask<TrainBackend> + Clone,
    O: Optimizer<M, TrainBackend>,
    F: FnMut(u64) -> Result<S, Box<dyn Error + Send + Sync>>,
    S: EpochBatchSource<M::Batch>,
{
    let (mut optimizer, external_weight_decay) = optimizer;
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
        let result = trainer.train_epoch::<TrainBackend, _, _, _, _>(
            model,
            optimizer,
            &mut batches,
            external_weight_decay,
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
        let saved_manifest = save_atomic(
            &epoch_path,
            manifest.clone(),
            &[
                ("model.bin", &model_bytes),
                ("ema.bin", &ema_bytes),
                ("optimizer.bin", &optimizer_bytes),
            ],
        )?;
        replace_atomic_from_saved(
            trainer.run.checkpoints.join("last"),
            &epoch_path,
            &saved_manifest,
        )?;
        if improved {
            replace_atomic_from_saved(
                trainer.run.checkpoints.join("best"),
                &epoch_path,
                &saved_manifest,
            )?;
        }
        eprintln!(
            "epoch {}: loss {:.6}",
            trainer.state.epoch, result.2.mean_loss
        );
        if trainer.state.epoch < trainer.config.epochs {
            batches = rebuild_batches(trainer.state.epoch as u64)?;
            if batches.batch_count() == 0 {
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
    let indexed = order.into_iter().enumerate().collect::<Vec<_>>();
    let workers = config.workers.max(1).min(indexed.len().max(1));
    let chunk_size = indexed.len().div_ceil(workers);
    let mut prepared =
        std::thread::scope(|scope| -> Result<Vec<_>, Box<dyn Error + Send + Sync>> {
            let handles = indexed
                .chunks(chunk_size.max(1))
                .map(|chunk| {
                    let pipeline = pipeline.clone();
                    scope.spawn(move || {
                        chunk
                            .iter()
                            .map(|&(logical_position, index)| {
                                prepare_classification_sample(
                                    config,
                                    dataset,
                                    images,
                                    &pipeline,
                                    epoch,
                                    logical_position,
                                    index,
                                )
                            })
                            .collect::<Result<Vec<_>, _>>()
                    })
                })
                .collect::<Vec<_>>();
            let mut output = Vec::with_capacity(indexed.len());
            for handle in handles {
                let chunk = handle.join().map_err(|_| {
                    Box::<dyn Error + Send + Sync>::from("classification data worker panicked")
                })??;
                output.extend(chunk);
            }
            Ok(output)
        })?;
    prepared.sort_unstable_by_key(|sample| sample.0);
    let (formatted, metadata): (Vec<_>, Vec<_>) = prepared
        .into_iter()
        .map(|(_, sample, metadata)| (sample, metadata))
        .unzip();
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

fn prepare_classification_sample(
    config: &TrainingConfig,
    dataset: &crate::training::data::ResolvedDataset,
    images: &[PathBuf],
    pipeline: &ClassificationPipeline,
    epoch: u64,
    logical_position: usize,
    index: usize,
) -> Result<(usize, FormattedClassificationSample, ImageMeta), Box<dyn Error + Send + Sync>> {
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
    let path_text = path.to_string_lossy().into_owned();
    let sample = pipeline.apply(
        image,
        class_id as u32,
        SeedKey {
            run_seed: config.seed,
            epoch,
            logical_position: logical_position as u64,
            sample_index: index as u64,
            rank: 0,
            path: &path_text,
        },
    )?;
    let metadata = ImageMeta {
        image_id: path_text,
        source_size,
        canvas_size: [
            config.model.input_size[1] as u32,
            config.model.input_size[0] as u32,
        ],
        scale: [1.0, 1.0],
        pad: [0.0, 0.0],
        crowd: Vec::new(),
    };
    Ok((logical_position, sample, metadata))
}

fn build_detection_batches<B: Backend>(
    config: &TrainingConfig,
    dataset: &crate::training::data::ResolvedDataset,
    images: &[PathBuf],
    device: &burn::tensor::Device<B>,
    epoch: u64,
    training: bool,
) -> Result<Vec<DetectionBatch<B>>, Box<dyn Error + Send + Sync>> {
    let pipeline = AugmentationPipeline::for_epoch(
        config
            .augmentation
            .resolve(crate::training::TaskKind::Detect, training)?,
        epoch as usize,
        config.epochs,
    )?;
    let source_samples = load_vision_samples(dataset, images, training)?;
    let crowd_flags = source_samples
        .iter()
        .map(|sample| {
            sample
                .targets
                .iter()
                .map(|target| target.crowd)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut samples = Vec::with_capacity(source_samples.len());
    for (index, sample) in source_samples.into_iter().enumerate() {
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
            crowd: crowd_flags[index].clone(),
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
    let [height, width] = config.model.input_size;
    if height != width {
        return Err("YOLOX validation currently requires a square input".into());
    }
    let mut formatted = Vec::with_capacity(images.len());
    let mut metadata = Vec::with_capacity(images.len());
    for sample in load_vision_samples(dataset, images, false)? {
        let crowd = sample.targets.iter().map(|target| target.crowd).collect();
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
            crowd,
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
    let pipeline = AugmentationPipeline::for_epoch(
        config
            .augmentation
            .resolve(crate::training::TaskKind::Segment, training)?,
        epoch as usize,
        config.epochs,
    )?;
    let source_samples = load_vision_samples(dataset, images, training)?;
    let crowd_flags = source_samples
        .iter()
        .map(|sample| {
            sample
                .targets
                .iter()
                .map(|target| target.crowd)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut samples = Vec::with_capacity(source_samples.len());
    for (index, sample) in source_samples.into_iter().enumerate() {
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
            crowd: crowd_flags[index].clone(),
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

fn load_vision_samples(
    dataset: &crate::training::data::ResolvedDataset,
    images: &[PathBuf],
    training: bool,
) -> Result<Vec<crate::training::data::VisionSample>, Box<dyn Error + Send + Sync>> {
    match dataset.format {
        DatasetFormat::Yolo => images
            .iter()
            .map(|path| {
                crate::training::data::loader::load_yolo_sample(dataset, path)
                    .map_err(|error| -> Box<dyn Error + Send + Sync> { Box::new(error) })
            })
            .collect(),
        DatasetFormat::Coco => {
            let annotation = if images == dataset.train_images {
                dataset.train_annotations.as_ref()
            } else if images == dataset.val_images {
                dataset.val_annotations.as_ref()
            } else if images == dataset.test_images {
                dataset.test_annotations.as_ref()
            } else {
                None
            }
            .ok_or("COCO split has no resolved annotation file")?;
            let images_root = images
                .first()
                .and_then(|path| path.parent())
                .ok_or("COCO split contains no image root")?;
            let loaded = crate::training::data::coco::load(annotation, images_root)?;
            if loaded.class_names != dataset.class_names {
                return Err("COCO category table differs from dataset names".into());
            }
            let mut by_path = loaded
                .sample_paths
                .into_iter()
                .zip(loaded.samples)
                .collect::<std::collections::BTreeMap<_, _>>();
            let mut selected = Vec::with_capacity(images.len());
            for path in images {
                let mut sample = by_path.remove(path).ok_or_else(|| {
                    format!(
                        "COCO annotations have no image record for {}",
                        path.display()
                    )
                })?;
                if training {
                    sample.targets.retain(|target| !target.crowd);
                }
                selected.push(sample);
            }
            Ok(selected)
        }
        DatasetFormat::ClassificationFolders => {
            Err("classification folders cannot feed a detector loader".into())
        }
    }
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask_metrics: Option<crate::training::metrics::segmentation::SegmentationMetrics>,
}

pub fn validate(checkpoint: PathBuf) -> Result<ValidationSummary, Box<dyn Error + Send + Sync>> {
    let worker = std::thread::Builder::new()
        .name("boquilens-training-validation".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || validate_inner(checkpoint))?;
    worker.join().map_err(|_| {
        Box::<dyn Error + Send + Sync>::from("native validation worker thread panicked")
    })?
}

fn validate_inner(checkpoint: PathBuf) -> Result<ValidationSummary, Box<dyn Error + Send + Sync>> {
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
            ($model:expr) => {{
                validate_detection_model(
                    $model,
                    bytes,
                    batches,
                    manifest.config.model.architecture,
                    &manifest.config.validation,
                )
            }};
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
            ModelId::Yolov8N => run_detect!(Yolov8NConfig.init_with_classes(classes, &device)),
            ModelId::Yolov8S => run_detect!(Yolov8SConfig.init_with_classes(classes, &device)),
            ModelId::Yolov8M => run_detect!(Yolov8MConfig.init_with_classes(classes, &device)),
            ModelId::Yolov8L => run_detect!(Yolov8LConfig.init_with_classes(classes, &device)),
            ModelId::Yolov8X => run_detect!(Yolov8XConfig.init_with_classes(classes, &device)),
            ModelId::Yolo12N => run_detect!(Yolo12NConfig.init_with_classes(classes, &device)),
            ModelId::Yolo12S => run_detect!(Yolo12SConfig.init_with_classes(classes, &device)),
            ModelId::Yolo12M => run_detect!(Yolo12MConfig.init_with_classes(classes, &device)),
            ModelId::Yolo12L => run_detect!(Yolo12LConfig.init_with_classes(classes, &device)),
            ModelId::Yolo12X => run_detect!(Yolo12XConfig.init_with_classes(classes, &device)),
            _ => Err("checkpoint model is not a supported detector".into()),
        };
    }
    if manifest.config.model.task == crate::training::TaskKind::Segment {
        let batches = build_segmentation_batches(
            &manifest.config,
            &dataset,
            &dataset.val_images,
            &device,
            manifest.state.epoch as u64,
            false,
        )?;
        macro_rules! run_segment {
            ($model:expr) => {{
                validate_segmentation_model(
                    $model,
                    bytes,
                    batches,
                    manifest.config.model.architecture,
                    &manifest.config.validation,
                )
            }};
        }
        return match manifest.config.model.architecture {
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
            ModelId::Yolov8NSeg => {
                run_segment!(Yolov8SegNConfig.init_with_classes(classes, &device))
            }
            ModelId::Yolov8SSeg => {
                run_segment!(Yolov8SegSConfig.init_with_classes(classes, &device))
            }
            ModelId::Yolov8MSeg => {
                run_segment!(Yolov8SegMConfig.init_with_classes(classes, &device))
            }
            ModelId::Yolov8LSeg => {
                run_segment!(Yolov8SegLConfig.init_with_classes(classes, &device))
            }
            ModelId::Yolov8XSeg => {
                run_segment!(Yolov8SegXConfig.init_with_classes(classes, &device))
            }
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
            _ => Err("checkpoint model is not a supported segmenter".into()),
        };
    }
    if manifest.config.model.task != crate::training::TaskKind::Classify {
        return Err("checkpoint task is not supported by native validation".into());
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
        let labels = classes_data
            .as_slice::<i32>()?
            .iter()
            .map(|value| i64::from(*value))
            .collect::<Vec<_>>();
        let rows = values
            .chunks_exact(class_count)
            .map(Vec::from)
            .collect::<Vec<_>>();
        let labels = labels
            .into_iter()
            .map(usize::try_from)
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
        mask_metrics: None,
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

trait DetectionForward {
    fn validation_detections(
        &self,
        images: Tensor<Wgpu, 4>,
        validation: &crate::training::config::ValidationConfig,
    ) -> Vec<Vec<Vec<crate::postprocess::BoundingBox>>>;
}

impl DetectionForward for Yolox<Wgpu> {
    fn validation_detections(
        &self,
        images: Tensor<Wgpu, 4>,
        validation: &crate::training::config::ValidationConfig,
    ) -> Vec<Vec<Vec<crate::postprocess::BoundingBox>>> {
        let output = self.forward(images * 255.0);
        let [batch, anchors, outputs] = output.dims();
        let boxes = output.clone().slice([0..batch, 0..anchors, 0..4]);
        let objectness = output.clone().slice([0..batch, 0..anchors, 4..5]);
        let scores = output.slice([0..batch, 0..anchors, 5..outputs]) * objectness;
        crate::postprocess::nms(boxes, scores, validation.iou, validation.confidence)
    }
}

impl DetectionForward for crate::models::yolov3_tiny::Yolov3Tiny<Wgpu> {
    fn validation_detections(
        &self,
        images: Tensor<Wgpu, 4>,
        validation: &crate::training::config::ValidationConfig,
    ) -> Vec<Vec<Vec<crate::postprocess::BoundingBox>>> {
        let output = self.forward(images);
        let [batch, anchors, _] = output.boxes.dims();
        let left_top = output.boxes.clone().slice([0..batch, 0..anchors, 0..2]);
        let right_bottom = output.boxes.slice([0..batch, 0..anchors, 2..4]);
        let center = (left_top.clone() + right_bottom.clone()) / 2.0;
        let size = right_bottom - left_top;
        crate::postprocess::nms(
            Tensor::cat(vec![center, size], 2),
            output.scores,
            validation.iou,
            validation.confidence,
        )
    }
}

macro_rules! classic_detection_forward {
    ($($model:ty),+ $(,)?) => {$ (
        impl DetectionForward for $model {
            fn validation_detections(
                &self,
                images: Tensor<Wgpu, 4>,
                validation: &crate::training::config::ValidationConfig,
            ) -> Vec<Vec<Vec<crate::postprocess::BoundingBox>>> {
                let output = self.forward(images);
                crate::postprocess::nms(
                    output.boxes,
                    output.scores,
                    validation.iou,
                    validation.confidence,
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
    crate::models::yolov8::Yolov8N<Wgpu>,
    crate::models::yolov8::Yolov8S<Wgpu>,
    crate::models::yolov8::Yolov8M<Wgpu>,
    crate::models::yolov8::Yolov8L<Wgpu>,
    crate::models::yolov8::Yolov8X<Wgpu>,
    crate::models::yolo12::Yolo12N<Wgpu>,
    crate::models::yolo12::Yolo12S<Wgpu>,
    crate::models::yolo12::Yolo12M<Wgpu>,
    crate::models::yolo12::Yolo12L<Wgpu>,
    crate::models::yolo12::Yolo12X<Wgpu>,
);

macro_rules! end_to_end_detection_forward {
    ($($model:ty),+ $(,)?) => {$ (
        impl DetectionForward for $model {
            fn validation_detections(
                &self,
                images: Tensor<Wgpu, 4>,
                validation: &crate::training::config::ValidationConfig,
            ) -> Vec<Vec<Vec<crate::postprocess::BoundingBox>>> {
                let output = self.forward(images);
                crate::end2end_topk_detections(
                    output.boxes,
                    output.scores,
                    validation.max_detections,
                    validation.confidence,
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
    validation: &crate::training::config::ValidationConfig,
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
        let classes = classes_data
            .as_slice::<i32>()?
            .iter()
            .map(|value| i64::from(*value))
            .collect::<Vec<_>>();
        let boxes = boxes_data.as_slice::<f32>()?;
        let valid = valid_data
            .as_slice::<u32>()?
            .iter()
            .map(|value| *value != 0)
            .collect::<Vec<_>>();
        let grouped = model.validation_detections(batch.images, validation);
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
            image_predictions.truncate(validation.max_detections);
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
                        crowd: metadata.crowd.get(target).copied().unwrap_or(false),
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
        mask_metrics: None,
    })
}

trait SegmentationForward {
    fn validation_segmentations(
        &self,
        image: Tensor<Wgpu, 4>,
        validation: &crate::training::config::ValidationConfig,
    ) -> crate::SegmentationOutputCpu;
}

macro_rules! classic_segmentation_forward {
    ($($model:ty),+ $(,)?) => {$ (
        impl SegmentationForward for $model {
            fn validation_segmentations(
                &self,
                image: Tensor<Wgpu, 4>,
                validation: &crate::training::config::ValidationConfig,
            ) -> crate::SegmentationOutputCpu {
                crate::run_classic_segmentations(
                    self,
                    image * 255.0,
                    validation.iou,
                    validation.confidence,
                )
            }
        }
    )+ };
}

classic_segmentation_forward!(
    crate::models::yolo11::Yolo11SegN<Wgpu>,
    crate::models::yolo11::Yolo11SegS<Wgpu>,
    crate::models::yolo11::Yolo11SegM<Wgpu>,
    crate::models::yolo11::Yolo11SegL<Wgpu>,
    crate::models::yolo11::Yolo11SegX<Wgpu>,
    crate::models::yolov8::Yolov8SegN<Wgpu>,
    crate::models::yolov8::Yolov8SegS<Wgpu>,
    crate::models::yolov8::Yolov8SegM<Wgpu>,
    crate::models::yolov8::Yolov8SegL<Wgpu>,
    crate::models::yolov8::Yolov8SegX<Wgpu>,
);

macro_rules! end_to_end_segmentation_forward {
    ($($model:ty),+ $(,)?) => {$ (
        impl SegmentationForward for $model {
            fn validation_segmentations(
                &self,
                image: Tensor<Wgpu, 4>,
                validation: &crate::training::config::ValidationConfig,
            ) -> crate::SegmentationOutputCpu {
                crate::run_end_to_end_segmentations(
                    self,
                    image * 255.0,
                    validation.max_detections,
                    validation.confidence,
                )
            }
        }
    )+ };
}

end_to_end_segmentation_forward!(
    crate::models::yolo26::Yolo26SegN<Wgpu>,
    crate::models::yolo26::Yolo26SegS<Wgpu>,
    crate::models::yolo26::Yolo26SegM<Wgpu>,
    crate::models::yolo26::Yolo26SegL<Wgpu>,
    crate::models::yolo26::Yolo26SegX<Wgpu>,
);

fn validate_segmentation_model<M>(
    model: M,
    bytes: Vec<u8>,
    batches: Vec<SegmentationBatch<Wgpu>>,
    model_id: ModelId,
    validation: &crate::training::config::ValidationConfig,
) -> Result<ValidationSummary, Box<dyn Error + Send + Sync>>
where
    M: Module<Wgpu> + SegmentationForward,
{
    let device = batches[0].detection.images.device();
    let model = model.load_record(decode_record::<Wgpu, _>(bytes, &device)?);
    let mut box_predictions = Vec::new();
    let mut box_targets = Vec::new();
    let mut mask_predictions = Vec::new();
    let mut mask_targets = Vec::new();
    let mut image_count = 0;
    for batch in batches {
        let [batch_size, max_targets] = batch.detection.classes.dims();
        let [_, _, image_height, image_width] = batch.detection.images.dims();
        let [mask_batch, mask_count, mask_height, mask_width] = batch.masks.dims();
        if mask_batch != batch_size || mask_count != max_targets {
            return Err("segmentation target tensors disagree on batch/object shape".into());
        }
        let classes_data = batch.detection.classes.into_data();
        let boxes_data = batch.detection.boxes_xyxy.into_data();
        let valid_data = batch.detection.valid.into_data();
        let masks_data = batch.masks.into_data();
        let classes = classes_data
            .as_slice::<i32>()?
            .iter()
            .map(|value| i64::from(*value))
            .collect::<Vec<_>>();
        let boxes = boxes_data.as_slice::<f32>()?;
        let valid = valid_data
            .as_slice::<u32>()?
            .iter()
            .map(|value| *value != 0)
            .collect::<Vec<_>>();
        let masks = masks_data.as_slice::<f32>()?;
        for (image, metadata) in batch.detection.metadata.iter().enumerate() {
            let input = batch.detection.images.clone().slice([
                image..image + 1,
                0..3,
                0..image_height,
                0..image_width,
            ]);
            let mut output = model.validation_segmentations(input, validation);
            output
                .candidates
                .sort_unstable_by(|a, b| b.bbox.confidence.total_cmp(&a.bbox.confidence));
            output.candidates.truncate(validation.max_detections);
            for candidate in &output.candidates {
                let canvas_box = [
                    candidate.bbox.xmin,
                    candidate.bbox.ymin,
                    candidate.bbox.xmax,
                    candidate.bbox.ymax,
                ];
                let Some(bbox) = source_box(metadata, canvas_box) else {
                    continue;
                };
                let canvas_mask = crate::canvas_instance_mask(
                    &output,
                    candidate.anchor,
                    image_width,
                    image_height,
                    canvas_box,
                );
                let source_mask = source_mask(metadata, &canvas_mask, image_width, image_height);
                if !source_mask.iter().any(|value| *value) {
                    continue;
                }
                box_predictions.push(crate::training::metrics::detection::MetricPrediction {
                    image_id: metadata.image_id.clone(),
                    class_id: candidate.class_id,
                    confidence: candidate.bbox.confidence,
                    bbox,
                });
                mask_predictions.push(
                    crate::training::metrics::segmentation::MetricMaskPrediction {
                        image_id: metadata.image_id.clone(),
                        class_id: candidate.class_id,
                        confidence: candidate.bbox.confidence,
                        mask: source_mask,
                    },
                );
            }
            for target in 0..max_targets {
                let flat = image * max_targets + target;
                if !valid[flat] {
                    continue;
                }
                let class_id = usize::try_from(classes[flat])?;
                let Some(bbox) =
                    source_box(metadata, boxes[flat * 4..flat * 4 + 4].try_into().unwrap())
                else {
                    continue;
                };
                let mask_start = flat * mask_height * mask_width;
                let canvas_mask = masks[mask_start..mask_start + mask_height * mask_width]
                    .iter()
                    .map(|value| *value > 0.5)
                    .collect::<Vec<_>>();
                let mask = source_mask(metadata, &canvas_mask, mask_width, mask_height);
                box_targets.push(crate::training::metrics::detection::MetricTarget {
                    image_id: metadata.image_id.clone(),
                    class_id,
                    bbox,
                    crowd: metadata.crowd.get(target).copied().unwrap_or(false),
                });
                mask_targets.push(crate::training::metrics::segmentation::MetricMaskTarget {
                    image_id: metadata.image_id.clone(),
                    class_id,
                    mask,
                    crowd: metadata.crowd.get(target).copied().unwrap_or(false),
                });
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
            &box_predictions,
            &box_targets,
        )),
        mask_metrics: Some(crate::training::metrics::segmentation::evaluate(
            &mask_predictions,
            &mask_targets,
        )),
    })
}

fn source_mask(
    metadata: &ImageMeta,
    canvas_mask: &[bool],
    mask_width: usize,
    mask_height: usize,
) -> Vec<bool> {
    if canvas_mask.len() != mask_width * mask_height || mask_width == 0 || mask_height == 0 {
        return Vec::new();
    }
    let [source_width, source_height] = metadata.source_size.map(|value| value as usize);
    let canvas_width = metadata.canvas_size[0].max(1) as f32;
    let canvas_height = metadata.canvas_size[1].max(1) as f32;
    let mut output = vec![false; source_width * source_height];
    for y in 0..source_height {
        let canvas_y = y as f32 * metadata.scale[1] + metadata.pad[1];
        let mask_y = (canvas_y * mask_height as f32 / canvas_height + 0.5)
            .floor()
            .clamp(0.0, mask_height.saturating_sub(1) as f32) as usize;
        for x in 0..source_width {
            let canvas_x = x as f32 * metadata.scale[0] + metadata.pad[0];
            let mask_x = (canvas_x * mask_width as f32 / canvas_width + 0.5)
                .floor()
                .clamp(0.0, mask_width.saturating_sub(1) as f32) as usize;
            output[y * source_width + x] = canvas_mask[mask_y * mask_width + mask_x];
        }
    }
    output
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
    let worker = std::thread::Builder::new()
        .name("boquilens-training-export".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || export_inner(checkpoint, output))?;
    worker.join().map_err(|_| {
        Box::<dyn Error + Send + Sync>::from("native training export worker thread panicked")
    })?
}

#[cfg(feature = "pretrained")]
fn export_inner(
    checkpoint: PathBuf,
    output: PathBuf,
) -> Result<PathBuf, Box<dyn Error + Send + Sync>> {
    let manifest = crate::training::checkpoint::load(&checkpoint)?;
    if output.extension().and_then(|value| value.to_str()) != Some("bpk") {
        return Err("native training artifacts must use the .bpk extension".into());
    }
    let (device, _) = crate::default_wgpu_device();
    let bytes = std::fs::read(checkpoint.join("ema.bin"))
        .or_else(|_| std::fs::read(checkpoint.join("model.bin")))?;
    let classes = manifest.config.model.num_classes;
    macro_rules! save {
        ($config:expr) => {{
            let model = $config.init_with_classes::<Wgpu>(classes, &device);
            let model = model.load_record(decode_record::<Wgpu, _>(bytes, &device)?);
            save_training_artifact(&model, &manifest.config.model, &output)?;
            Ok(output)
        }};
    }
    macro_rules! save_yolox {
        ($model:expr) => {{
            let model = $model.load_record(decode_record::<Wgpu, _>(bytes, &device)?);
            save_training_artifact(&model, &manifest.config.model, &output)?;
            Ok(output)
        }};
    }
    let exported: Result<PathBuf, Box<dyn Error + Send + Sync>> =
        match manifest.config.model.architecture {
            ModelId::YoloxNano => save_yolox!(Yolox::yolox_nano(classes, &device)),
            ModelId::YoloxTiny => save_yolox!(Yolox::yolox_tiny(classes, &device)),
            ModelId::YoloxS => save_yolox!(Yolox::yolox_s(classes, &device)),
            ModelId::YoloxM => save_yolox!(Yolox::yolox_m(classes, &device)),
            ModelId::YoloxL => save_yolox!(Yolox::yolox_l(classes, &device)),
            ModelId::YoloxX => save_yolox!(Yolox::yolox_x(classes, &device)),
            ModelId::Yolov3TinyU => save!(Yolov3TinyConfig),
            ModelId::Yolov10N => save!(Yolov10NConfig),
            ModelId::Yolov10S => save!(Yolov10SConfig),
            ModelId::Yolov10M => save!(Yolov10MConfig),
            ModelId::Yolov10B => save!(Yolov10BConfig),
            ModelId::Yolov10L => save!(Yolov10LConfig),
            ModelId::Yolov10X => save!(Yolov10XConfig),
            ModelId::Yolo11N => save!(crate::models::yolo11::Yolo11NConfig),
            ModelId::Yolo11S => save!(crate::models::yolo11::Yolo11SConfig),
            ModelId::Yolo11M => save!(crate::models::yolo11::Yolo11MConfig),
            ModelId::Yolo11L => save!(crate::models::yolo11::Yolo11LConfig),
            ModelId::Yolo11X => save!(crate::models::yolo11::Yolo11XConfig),
            ModelId::Yolo11NSeg => save!(crate::models::yolo11::Yolo11SegNConfig),
            ModelId::Yolo11SSeg => save!(crate::models::yolo11::Yolo11SegSConfig),
            ModelId::Yolo11MSeg => save!(crate::models::yolo11::Yolo11SegMConfig),
            ModelId::Yolo11LSeg => save!(crate::models::yolo11::Yolo11SegLConfig),
            ModelId::Yolo11XSeg => save!(crate::models::yolo11::Yolo11SegXConfig),
            ModelId::Yolo11NCls => save!(Yolo11ClsNConfig),
            ModelId::Yolo11SCls => save!(Yolo11ClsSConfig),
            ModelId::Yolo11MCls => save!(Yolo11ClsMConfig),
            ModelId::Yolo11LCls => save!(Yolo11ClsLConfig),
            ModelId::Yolo11XCls => save!(Yolo11ClsXConfig),
            ModelId::Yolov8N => save!(Yolov8NConfig),
            ModelId::Yolov8S => save!(Yolov8SConfig),
            ModelId::Yolov8M => save!(Yolov8MConfig),
            ModelId::Yolov8L => save!(Yolov8LConfig),
            ModelId::Yolov8X => save!(Yolov8XConfig),
            ModelId::Yolov8NSeg => save!(Yolov8SegNConfig),
            ModelId::Yolov8SSeg => save!(Yolov8SegSConfig),
            ModelId::Yolov8MSeg => save!(Yolov8SegMConfig),
            ModelId::Yolov8LSeg => save!(Yolov8SegLConfig),
            ModelId::Yolov8XSeg => save!(Yolov8SegXConfig),
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
            ModelId::Yolo12N => save!(Yolo12NConfig),
            ModelId::Yolo12S => save!(Yolo12SConfig),
            ModelId::Yolo12M => save!(Yolo12MConfig),
            ModelId::Yolo12L => save!(Yolo12LConfig),
            ModelId::Yolo12X => save!(Yolo12XConfig),
            ModelId::Yolo26N => save!(Yolo26NConfig),
            ModelId::Yolo26S => save!(Yolo26SConfig),
            ModelId::Yolo26M => save!(Yolo26MConfig),
            ModelId::Yolo26L => save!(Yolo26LConfig),
            ModelId::Yolo26X => save!(Yolo26XConfig),
            ModelId::Yolo26NSeg => save!(crate::models::yolo26::Yolo26SegNConfig),
            ModelId::Yolo26SSeg => save!(crate::models::yolo26::Yolo26SegSConfig),
            ModelId::Yolo26MSeg => save!(crate::models::yolo26::Yolo26SegMConfig),
            ModelId::Yolo26LSeg => save!(crate::models::yolo26::Yolo26SegLConfig),
            ModelId::Yolo26XSeg => save!(crate::models::yolo26::Yolo26SegXConfig),
        };
    let exported = exported?;
    let predictor = crate::Predictor::<Wgpu>::from_trained_artifact_on_device(
        manifest.config.model.architecture,
        &exported,
        device,
        crate::PredictOptions::default(),
    )?;
    if predictor.class_names() != manifest.config.model.class_names {
        return Err("public predictor reloaded a different artifact class table".into());
    }
    if predictor.input_size() != manifest.config.model.input_size[0] {
        return Err("public predictor reloaded a different artifact input size".into());
    }
    Ok(exported)
}

#[cfg(feature = "pretrained")]
fn keep_inference_tensor(path: &str, _container: &str) -> bool {
    !path.contains(".o2m_") && !path.starts_with("head.proto.sem_")
}

#[cfg(feature = "pretrained")]
fn save_training_artifact<M>(
    model: &M,
    spec: &ModelSpec,
    output: &std::path::Path,
) -> Result<(), Box<dyn Error + Send + Sync>>
where
    M: Module<Wgpu>,
{
    let task = match spec.task {
        crate::training::TaskKind::Detect => "detect",
        crate::training::TaskKind::Segment => "segment",
        crate::training::TaskKind::Classify => "classify",
    };
    let mut store = burn_store::BurnpackStore::from_file(output)
        .metadata("boquilens.artifact-format", "boquilens-trained-v1")
        .metadata("boquilens.model", spec.architecture.as_str())
        .metadata("boquilens.task", task)
        .metadata(
            "boquilens.class-names-json",
            serde_json::to_string(&spec.class_names)?,
        )
        .metadata(
            "boquilens.input-size-json",
            serde_json::to_string(&spec.input_size)?,
        )
        .metadata("boquilens.precision", "f16")
        .with_filter(PathFilter::new().with_predicate(keep_inference_tensor))
        .with_to_adapter(burn_store::HalfPrecisionAdapter::new());
    model.save_into(&mut store)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_sample_cache_and_prefetch_are_bounded() {
        let mut cache = BoundedLru::new(4);
        for index in 0..100 {
            cache.insert(index, index);
            assert!(cache.entries.len() <= 4);
        }
        assert_eq!(cache.peak, 4);
        assert_eq!(prefetch_sample_capacity(8, 2), 16);
        assert_eq!(prefetch_sample_capacity(usize::MAX, 2), usize::MAX);
    }

    #[derive(Module, Debug)]
    struct TransferHead<B: Backend> {
        linear: burn::nn::Linear<B>,
    }

    #[derive(Module, Debug)]
    struct TransferModel<B: Backend> {
        body: burn::nn::Linear<B>,
        head: TransferHead<B>,
    }

    fn transfer_model<B: Backend>(classes: usize, device: &B::Device) -> TransferModel<B> {
        TransferModel {
            body: burn::nn::LinearConfig::new(2, 3).init(device),
            head: TransferHead {
                linear: burn::nn::LinearConfig::new(3, classes).init(device),
            },
        }
    }

    #[test]
    fn changed_class_transfer_preserves_only_fresh_classifier_projection() {
        let device = Default::default();
        let official = transfer_model::<burn_flex::Flex>(5, &device);
        let target = transfer_model::<burn_flex::Flex>(2, &device);
        let classifier_before = target.head.linear.weight.val().into_data();
        let transferred =
            transfer_pretrained(target, &official, ReplacedProjection::Classifier).unwrap();
        assert_eq!(
            classifier_before.as_slice::<f32>().unwrap(),
            transferred
                .head
                .linear
                .weight
                .val()
                .into_data()
                .as_slice::<f32>()
                .unwrap()
        );
        assert_eq!(
            official
                .body
                .weight
                .val()
                .into_data()
                .as_slice::<f32>()
                .unwrap(),
            transferred
                .body
                .weight
                .val()
                .into_data()
                .as_slice::<f32>()
                .unwrap()
        );
    }

    #[test]
    fn validation_boxes_map_back_through_composed_geometry() {
        let metadata = ImageMeta {
            image_id: "image".into(),
            source_size: [40, 20],
            canvas_size: [32, 32],
            scale: [0.5, 0.5],
            pad: [6.0, 11.0],
            crowd: Vec::new(),
        };
        let bbox = source_box(&metadata, [6.0, 11.0, 26.0, 21.0]).unwrap();
        assert_eq!(
            [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax],
            [0.0, 0.0, 40.0, 20.0]
        );
        assert!(source_box(&metadata, [0.0, 0.0, 5.0, 5.0]).is_none());
    }
}
