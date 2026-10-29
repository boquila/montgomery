use super::{
    Object, error,
    manifest::{DatasetError, ResolvedDataset, yolo_label_path},
};
use serde::Deserialize;
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

pub(super) fn yolo(image: &Path, classes: usize) -> Result<(Vec<Object>, bool), DatasetError> {
    let path = yolo_label_path(image);
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((Vec::new(), true)),
        Err(e) => return Err(error(path.display(), e)),
    };
    let mut objects = Vec::new();
    for (line, row) in text
        .lines()
        .enumerate()
        .filter(|(_, row)| !row.trim().is_empty())
    {
        let context = format!("{}: line {}", path.display(), line + 1);
        let values = row
            .split_whitespace()
            .map(str::parse::<f64>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| error(&context, e))?;
        if values.len() != 5 && (values.len() < 7 || values.len().is_multiple_of(2)) {
            return Err(error(
                context,
                "expected class cx cy width height, or class followed by at least three polygon points",
            ));
        }
        let class = values[0];
        if !class.is_finite() || class < 0. || class.fract() != 0. || class >= classes as f64 {
            return Err(error(
                context,
                format!("class ID must be an integer in 0..{classes}"),
            ));
        }
        if values[1..]
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err(error(
                context,
                "coordinates must be finite and normalized to [0, 1]",
            ));
        }
        let (bbox, polygon) = if values.len() == 5 {
            let [cx, cy, w, h] = [values[1], values[2], values[3], values[4]];
            ([cx - w / 2., cy - h / 2., cx + w / 2., cy + h / 2.], None)
        } else {
            let points = values[1..]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| [p[0], p[1]])
                .collect::<Vec<_>>();
            let bbox = [
                points.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min),
                points.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min),
                points
                    .iter()
                    .map(|p| p[0])
                    .fold(f64::NEG_INFINITY, f64::max),
                points
                    .iter()
                    .map(|p| p[1])
                    .fold(f64::NEG_INFINITY, f64::max),
            ];
            (bbox, Some(points))
        };
        let bbox = bbox.map(|v| v.clamp(0., 1.));
        if bbox[2] <= bbox[0] || bbox[3] <= bbox[1] {
            return Err(error(context, "zero-area box or polygon"));
        }
        objects.push(Object {
            class_id: class as usize,
            bbox,
            polygon,
            segmentation: None,
            crowd: false,
        });
    }
    Ok((objects, false))
}

#[derive(Deserialize)]
struct Coco {
    images: Vec<CocoImage>,
    annotations: Vec<CocoAnnotation>,
    categories: Vec<Category>,
}
#[derive(Deserialize)]
struct CocoImage {
    id: u64,
    file_name: String,
    width: u32,
    height: u32,
}
#[derive(Deserialize)]
struct Category {
    id: u64,
    name: String,
}
#[derive(Deserialize)]
struct CocoAnnotation {
    id: u64,
    image_id: u64,
    category_id: u64,
    bbox: [f64; 4],
    #[serde(default)]
    segmentation: Option<Value>,
    #[serde(default)]
    iscrowd: u8,
}

pub(super) struct CocoRecord {
    pub size: [u32; 2],
    pub objects: Vec<Object>,
}

/// Resolve against the manifest's actual images, including nested paths and list splits.
pub(super) fn coco(
    path: &Path,
    images: &[PathBuf],
    dataset: &ResolvedDataset,
) -> Result<BTreeMap<PathBuf, CocoRecord>, DatasetError> {
    let parsed: Coco =
        serde_json::from_slice(&fs::read(path).map_err(|e| error(path.display(), e))?)
            .map_err(|e| error(path.display(), e))?;
    let mut categories = parsed.categories;
    categories.sort_by_key(|c| c.id);
    if categories
        .iter()
        .map(|c| &c.name)
        .ne(dataset.class_names.iter())
    {
        return Err(error(
            path.display(),
            "COCO category names (ordered by ID) must match YAML names",
        ));
    }
    let mut classes = BTreeMap::new();
    for (class, category) in categories.iter().enumerate() {
        if classes.insert(category.id, class).is_some() {
            return Err(error(path.display(), "duplicate category ID"));
        }
    }
    let mut annotations = BTreeMap::<u64, Vec<CocoAnnotation>>::new();
    let mut annotation_ids = BTreeSet::new();
    for ann in parsed.annotations {
        if !annotation_ids.insert(ann.id) {
            return Err(error(path.display(), "duplicate annotation ID"));
        }
        annotations.entry(ann.image_id).or_default().push(ann);
    }
    let mut result = BTreeMap::new();
    let mut image_ids = BTreeSet::new();
    // Index only the suffixes requested by COCO (plus basename fallbacks),
    // avoiding repeated scans and storage of every absolute ancestor path.
    let mut requested = HashSet::new();
    for image in &parsed.images {
        let name = PathBuf::from(image.file_name.replace('\\', "/"));
        if let Some(basename) = name.file_name() {
            requested.insert(PathBuf::from(basename));
        }
        requested.insert(name);
    }
    let depths = requested
        .iter()
        .map(|name| name.components().count())
        .collect::<BTreeSet<_>>();
    let mut suffixes = HashMap::<PathBuf, Option<usize>>::new();
    for (id, image) in images.iter().enumerate() {
        let components = image.components().collect::<Vec<_>>();
        for &depth in depths.range(..=components.len()) {
            let suffix = components[components.len() - depth..]
                .iter()
                .collect::<PathBuf>();
            if !requested.contains(&suffix) {
                continue;
            }
            suffixes
                .entry(suffix)
                .and_modify(|existing| *existing = None)
                .or_insert(Some(id));
        }
    }
    drop(requested);
    for image in parsed.images {
        if !image_ids.insert(image.id) {
            return Err(error(path.display(), "duplicate image ID"));
        }
        if image.width == 0 || image.height == 0 {
            return Err(error(path.display(), "image dimensions must be positive"));
        }
        let name = image.file_name.replace('\\', "/");
        let name_path = Path::new(&name);
        let matched = suffixes.get(name_path).or_else(|| {
            name_path
                .file_name()
                .and_then(|basename| suffixes.get(Path::new(basename)))
        });
        let selected = match matched {
            Some(Some(id)) => Some(&images[*id]),
            Some(None) => {
                return Err(error(
                    path.display(),
                    format!("ambiguous COCO file_name {name}"),
                ));
            }
            None => None,
        };
        let mut objects = Vec::new();
        for ann in annotations.remove(&image.id).unwrap_or_default() {
            let context = format!("{}: annotation {}", path.display(), ann.id);
            let class_id = *classes
                .get(&ann.category_id)
                .ok_or_else(|| error(&context, "unknown category ID"))?;
            let [x, y, w, h] = ann.bbox;
            if ann.bbox.iter().any(|v| !v.is_finite()) || w <= 0. || h <= 0. {
                return Err(error(context, "invalid box"));
            }
            let bbox = [
                x / image.width as f64,
                y / image.height as f64,
                (x + w) / image.width as f64,
                (y + h) / image.height as f64,
            ]
            .map(|v| v.clamp(0., 1.));
            if bbox[2] <= bbox[0] || bbox[3] <= bbox[1] {
                return Err(error(context, "box outside image"));
            }
            let segmentation = ann
                .segmentation
                .filter(|v| !v.is_null() && !v.as_array().is_some_and(Vec::is_empty));
            if let Some(seg) = &segmentation {
                validate_segmentation(seg, [image.width, image.height])
                    .map_err(|e| error(&context, e))?;
            }
            objects.push(Object {
                class_id,
                bbox,
                polygon: None,
                segmentation,
                crowd: ann.iscrowd != 0,
            });
        }
        if let Some(selected) = selected
            && result
                .insert(
                    selected.clone(),
                    CocoRecord {
                        size: [image.width, image.height],
                        objects,
                    },
                )
                .is_some()
        {
            return Err(error(
                path.display(),
                format!("multiple COCO records match {}", selected.display()),
            ));
        }
    }
    if !annotations.is_empty() {
        return Err(error(
            path.display(),
            "annotation references an unknown image ID",
        ));
    }
    Ok(result)
}

fn validate_segmentation(seg: &Value, size: [u32; 2]) -> Result<(), DatasetError> {
    if let Some(polygons) = seg.as_array() {
        for polygon in polygons {
            let coordinates = polygon
                .as_array()
                .ok_or_else(|| DatasetError::new("invalid COCO polygon"))?;
            if coordinates.len() < 6
                || !coordinates.len().is_multiple_of(2)
                || coordinates
                    .iter()
                    .any(|v| !v.as_f64().is_some_and(f64::is_finite))
            {
                return Err(DatasetError::new(
                    "COCO polygons require at least three finite coordinate pairs",
                ));
            }
        }
    } else if seg.is_object() {
        if seg["size"] != serde_json::json!([size[1], size[0]])
            || !(seg["counts"].is_array() || seg["counts"].is_string())
        {
            return Err(DatasetError::new("invalid COCO RLE size/counts"));
        }
        super::rle::counts(seg, size)?;
    } else {
        return Err(DatasetError::new("expected COCO polygons or RLE"));
    }
    Ok(())
}
