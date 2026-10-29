//! Offline dataset tools. Images are decoded only for verification or transformation.

mod annotations;
mod inspect;
pub mod manifest;
mod rle;
mod transform;

pub use inspect::{ClassCount, DatasetReport, Issue, SplitReport, inspect};
pub use transform::{
    Conversion, ImageSize, ResizeMode, TransformOptions, TransformReport, convert, resize,
};

use manifest::{DatasetError, ResolvedDataset};
use serde_json::Value;
use std::{
    path::PathBuf,
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

#[derive(Debug, Clone)]
struct Object {
    class_id: usize,
    /// Normalized continuous XYXY edges.
    bbox: [f64; 4],
    /// YOLO polygon, or COCO segmentation (pixel coordinates / RLE).
    polygon: Option<Vec<[f64; 2]>>,
    segmentation: Option<Value>,
    crowd: bool,
}

#[derive(Debug)]
struct Record {
    image: PathBuf,
    split: &'static str,
    size: [u32; 2],
    bytes: u64,
    objects: Vec<Object>,
    class_id: Option<usize>,
    missing_label: bool,
}

struct Index {
    dataset: ResolvedDataset,
    records: Vec<Record>,
    report: DatasetReport,
}

/// Bounded concurrency; results retain input order and errors never panic a worker.
fn parallel_map<T: Sync, U: Send>(
    items: &[T],
    workers: usize,
    f: impl Fn(&T) -> U + Sync,
) -> Vec<U> {
    let cursor = AtomicUsize::new(0);
    let results = Mutex::new((0..items.len()).map(|_| None).collect::<Vec<Option<U>>>());
    std::thread::scope(|scope| {
        for _ in 0..workers.max(1).min(items.len()) {
            scope.spawn(|| {
                loop {
                    let i = cursor.fetch_add(1, Ordering::Relaxed);
                    let Some(item) = items.get(i) else { break };
                    let value = f(item);
                    results.lock().expect("result mutex")[i] = Some(value);
                }
            });
        }
    });
    results
        .into_inner()
        .expect("result mutex")
        .into_iter()
        .map(|v| v.expect("worker result"))
        .collect()
}

/// Conservative default: at most four simultaneous decoded images.
pub fn default_workers() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get().min(4))
}

fn error(context: impl std::fmt::Display, source: impl std::fmt::Display) -> DatasetError {
    DatasetError::new(format!("{context}: {source}"))
}
