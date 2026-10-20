use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt, fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DatasetFormat {
    Yolo,
    Coco,
    ClassificationFolders,
}

impl fmt::Display for DatasetFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Yolo => "yolo",
            Self::Coco => "coco",
            Self::ClassificationFolders => "classification-folders",
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum Names {
    List(Vec<String>),
    Map(BTreeMap<usize, String>),
}

impl Names {
    fn ordered(self) -> Result<Vec<String>, DatasetError> {
        match self {
            Self::List(names) => Ok(names),
            Self::Map(names) => {
                let len = names.len();
                if names.keys().copied().eq(0..len) {
                    Ok(names.into_values().collect())
                } else {
                    Err(DatasetError::new(
                        "dataset names map keys must be contiguous starting at zero",
                    ))
                }
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum Split {
    One(PathBuf),
    Many(Vec<PathBuf>),
}

impl Split {
    fn paths(&self) -> Vec<&Path> {
        match self {
            Self::One(path) => vec![path],
            Self::Many(paths) => paths.iter().map(PathBuf::as_path).collect(),
        }
    }
}

/// Ultralytics-compatible user dataset YAML.
#[derive(Debug, Clone, Deserialize)]
pub struct DatasetManifest {
    path: Option<PathBuf>,
    train: Split,
    val: Option<Split>,
    test: Option<Split>,
    names: Option<Names>,
    format: Option<DatasetFormat>,
    train_annotations: Option<PathBuf>,
    val_annotations: Option<PathBuf>,
    test_annotations: Option<PathBuf>,
    nc: Option<usize>,
    task: Option<String>,
    kpt_shape: Option<serde_yaml::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedDataset {
    pub manifest: PathBuf,
    pub root: PathBuf,
    pub format: DatasetFormat,
    pub class_names: Vec<String>,
    pub train_images: Vec<PathBuf>,
    pub val_images: Vec<PathBuf>,
    pub test_images: Vec<PathBuf>,
    pub train_annotations: Option<PathBuf>,
    pub val_annotations: Option<PathBuf>,
    pub test_annotations: Option<PathBuf>,
    pub fingerprint: String,
    /// Actual class directories, independent of split names or nested subdirectories.
    #[serde(skip)]
    pub(crate) class_folders: BTreeMap<PathBuf, usize>,
}

impl DatasetManifest {
    pub fn load(path: impl AsRef<Path>) -> Result<ResolvedDataset, DatasetError> {
        let manifest = fs::canonicalize(path.as_ref()).map_err(|error| {
            DatasetError::new(format!(
                "cannot open dataset manifest {}: {error}",
                path.as_ref().display()
            ))
        })?;
        let contents = fs::read_to_string(&manifest).map_err(|error| {
            DatasetError::new(format!(
                "cannot read dataset manifest {}: {error}",
                manifest.display()
            ))
        })?;
        let parsed: Self = serde_yaml::from_str(&contents).map_err(|error| {
            DatasetError::new(format!(
                "invalid dataset YAML {}: {error}",
                manifest.display()
            ))
        })?;
        parsed.resolve(manifest)
    }

    fn resolve(self, manifest: PathBuf) -> Result<ResolvedDataset, DatasetError> {
        if self.kpt_shape.is_some()
            || self
                .task
                .as_deref()
                .is_some_and(|task| !matches!(task, "detect" | "segment" | "classify"))
        {
            return Err(DatasetError::new(
                "dataset tools support detection, instance segmentation, and classification; pose, OBB, semantic, and depth labels are unsupported",
            ));
        }
        let manifest_dir = manifest.parent().expect("canonical file has a parent");
        let root = match self.path {
            Some(path) if path.is_absolute() => path,
            Some(path) => manifest_dir.join(path),
            None => manifest_dir.to_path_buf(),
        };
        let root = fs::canonicalize(&root).map_err(|error| {
            DatasetError::new(format!(
                "cannot resolve dataset root {}: {error}",
                root.display()
            ))
        })?;
        let train_images = resolve_split(&root, &self.train)?;
        let val_images = self
            .val
            .as_ref()
            .map(|split| resolve_split(&root, split))
            .transpose()?
            .unwrap_or_default();
        let test_images = self
            .test
            .as_ref()
            .map(|split| resolve_split(&root, split))
            .transpose()?
            .unwrap_or_default();
        if train_images.is_empty() {
            return Err(DatasetError::new(
                "training split contains no supported images",
            ));
        }
        let format = match self.format {
            Some(format) => format,
            None if self.train_annotations.is_some() => DatasetFormat::Coco,
            None if self.task.as_deref() == Some("classify") => {
                DatasetFormat::ClassificationFolders
            }
            None => detect_format(&root, &self.train, &train_images)?,
        };
        let resolve_annotation = |path: Option<PathBuf>| -> Result<Option<PathBuf>, DatasetError> {
            path.map(|path| {
                let path = if path.is_absolute() {
                    path
                } else {
                    root.join(path)
                };
                fs::canonicalize(&path).map_err(|error| {
                    DatasetError::new(format!(
                        "cannot resolve annotation file {}: {error}",
                        path.display()
                    ))
                })
            })
            .transpose()
        };
        let train_annotations = resolve_annotation(self.train_annotations)?;
        let val_annotations = resolve_annotation(self.val_annotations)?;
        let test_annotations = resolve_annotation(self.test_annotations)?;
        if format == DatasetFormat::Coco
            && (train_annotations.is_none()
                || (!val_images.is_empty() && val_annotations.is_none())
                || (!test_images.is_empty() && test_annotations.is_none()))
        {
            return Err(DatasetError::new(
                "COCO manifests require train_annotations and annotations for each non-empty val/test split",
            ));
        }
        let class_names = match self.names {
            Some(names) => names.ordered()?,
            None if format == DatasetFormat::ClassificationFolders => {
                classification_names(&root, &self.train)?
            }
            None => {
                return Err(DatasetError::new(
                    "detect/segment datasets require a names table",
                ));
            }
        };
        validate_names(&class_names)?;
        let mut class_folders = BTreeMap::new();
        if format == DatasetFormat::ClassificationFolders {
            for split in [Some(&self.train), self.val.as_ref(), self.test.as_ref()]
                .into_iter()
                .flatten()
            {
                for directory in split.paths().into_iter().map(|path| root.join(path)) {
                    if directory.is_dir() {
                        for (class, name) in class_names.iter().enumerate() {
                            let folder = directory.join(name);
                            if folder.is_dir() {
                                let folder = fs::canonicalize(&folder)
                                    .map_err(|error| DatasetError::new(error.to_string()))?;
                                class_folders.insert(folder, class);
                            }
                        }
                    }
                }
            }
        }
        if self.nc.is_some_and(|nc| nc != class_names.len()) {
            return Err(DatasetError::new(
                "nc differs from the number of class names",
            ));
        }
        let fingerprint = fingerprint(
            &train_images,
            &val_images,
            &test_images,
            [&train_annotations, &val_annotations, &test_annotations],
        )?;
        Ok(ResolvedDataset {
            manifest,
            root,
            format,
            class_names,
            train_images,
            val_images,
            test_images,
            train_annotations,
            val_annotations,
            test_annotations,
            fingerprint,
            class_folders,
        })
    }
}

fn resolve_split(root: &Path, split: &Split) -> Result<Vec<PathBuf>, DatasetError> {
    let mut images = BTreeSet::new();
    for value in split.paths() {
        let value = if value.is_absolute() {
            value.to_path_buf()
        } else {
            root.join(value)
        };
        if value.is_file()
            && value
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("txt"))
        {
            let list = fs::read_to_string(&value).map_err(|error| {
                DatasetError::new(format!(
                    "cannot read split list {}: {error}",
                    value.display()
                ))
            })?;
            for line in list.lines().map(str::trim).filter(|line| !line.is_empty()) {
                let item = Path::new(line);
                let item = if item.is_absolute() {
                    item.to_path_buf()
                } else {
                    value.parent().unwrap_or(root).join(item)
                };
                collect_images(&item, &mut images)?;
            }
        } else {
            collect_images(&value, &mut images)?;
        }
    }
    Ok(images.into_iter().collect())
}

fn collect_images(path: &Path, images: &mut BTreeSet<PathBuf>) -> Result<(), DatasetError> {
    collect_images_inner(path, images, &mut BTreeSet::new())
}

fn collect_images_inner(
    path: &Path,
    images: &mut BTreeSet<PathBuf>,
    visited: &mut BTreeSet<PathBuf>,
) -> Result<(), DatasetError> {
    if path.is_dir() {
        let canonical =
            fs::canonicalize(path).map_err(|error| DatasetError::new(error.to_string()))?;
        if !visited.insert(canonical) {
            return Ok(());
        }
        for entry in fs::read_dir(path).map_err(|error| {
            DatasetError::new(format!(
                "cannot read split directory {}: {error}",
                path.display()
            ))
        })? {
            let entry = entry.map_err(|error| DatasetError::new(error.to_string()))?;
            collect_images_inner(&entry.path(), images, visited)?;
        }
    } else if path.is_file() && is_image(path) {
        images
            .insert(fs::canonicalize(path).map_err(|error| DatasetError::new(error.to_string()))?);
    } else if !path.exists() {
        return Err(DatasetError::new(format!(
            "dataset split path does not exist: {}",
            path.display()
        )));
    }
    Ok(())
}

fn is_image(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "webp"
            )
        })
}

fn detect_format(
    root: &Path,
    train: &Split,
    images: &[PathBuf],
) -> Result<DatasetFormat, DatasetError> {
    if root.join("labels").is_dir() || images.iter().any(|image| yolo_label_path(image).exists()) {
        return Ok(DatasetFormat::Yolo);
    }
    if root.join("annotations").is_dir() {
        return Ok(DatasetFormat::Coco);
    }
    let training_dirs = train
        .paths()
        .into_iter()
        .map(|path| root.join(path))
        .filter(|path| path.is_dir() && !path.starts_with(root.join("images")))
        .map(fs::canonicalize)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| DatasetError::new(error.to_string()))?;
    if !training_dirs.is_empty()
        && images.iter().all(|image| {
            training_dirs.iter().any(|directory| {
                image
                    .strip_prefix(directory)
                    .is_ok_and(|relative| relative.components().count() >= 2)
            })
        })
    {
        return Ok(DatasetFormat::ClassificationFolders);
    }
    Err(DatasetError::new(
        "dataset format is ambiguous; set format to yolo, coco, or classification-folders",
    ))
}

pub fn yolo_label_path(image: &Path) -> PathBuf {
    let mut components: Vec<_> = image.components().collect();
    if let Some(index) = components
        .iter()
        .rposition(|value| value.as_os_str() == "images")
    {
        components[index] = std::path::Component::Normal("labels".as_ref());
        let mut path = PathBuf::new();
        path.extend(components);
        return path.with_extension("txt");
    }
    image.with_extension("txt")
}

fn classification_names(root: &Path, train: &Split) -> Result<Vec<String>, DatasetError> {
    let mut names = BTreeSet::new();
    for directory in train.paths().into_iter().map(|path| root.join(path)) {
        if !directory.is_dir() {
            return Err(DatasetError::new(
                "classification class inference requires training directories; provide names for image-list or individual-image splits",
            ));
        }
        for entry in fs::read_dir(&directory).map_err(|error| {
            DatasetError::new(format!(
                "cannot infer classification classes from {}: {error}",
                directory.display()
            ))
        })? {
            let entry = entry.map_err(|error| DatasetError::new(error.to_string()))?;
            if entry.path().is_dir() {
                names.insert(entry.file_name().to_string_lossy().into_owned());
            }
        }
    }
    Ok(names.into_iter().collect())
}

fn validate_names(names: &[String]) -> Result<(), DatasetError> {
    if names.is_empty() {
        return Err(DatasetError::new("dataset has no classes"));
    }
    let mut unique = BTreeSet::new();
    for name in names {
        if name.trim().is_empty() || !unique.insert(name) {
            return Err(DatasetError::new(
                "class names must be non-empty and unique",
            ));
        }
    }
    Ok(())
}

fn fingerprint(
    train: &[PathBuf],
    val: &[PathBuf],
    test: &[PathBuf],
    annotations: [&Option<PathBuf>; 3],
) -> Result<String, DatasetError> {
    let mut hash = Sha256::new();
    for (split, paths) in [("train", train), ("val", val), ("test", test)] {
        hash.update(split.as_bytes());
        for path in paths {
            let metadata =
                fs::metadata(path).map_err(|error| DatasetError::new(error.to_string()))?;
            hash.update(path.to_string_lossy().as_bytes());
            hash.update(metadata.len().to_le_bytes());
        }
    }
    for path in annotations.into_iter().flatten() {
        let metadata = fs::metadata(path).map_err(|error| DatasetError::new(error.to_string()))?;
        hash.update(path.to_string_lossy().as_bytes());
        hash.update(metadata.len().to_le_bytes());
    }
    Ok(format!("{:x}", hash.finalize()))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetError(String);

impl DatasetError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for DatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for DatasetError {}
