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
    pub fingerprint: String,
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
            None => detect_format(&root, &train_images)?,
        };
        let class_names = match self.names {
            Some(names) => names.ordered()?,
            None if format == DatasetFormat::ClassificationFolders => {
                classification_names(&root.join("train"))?
            }
            None => {
                return Err(DatasetError::new(
                    "detect/segment datasets require a names table",
                ));
            }
        };
        validate_names(&class_names)?;
        let fingerprint = fingerprint(&train_images, &val_images, &test_images)?;
        Ok(ResolvedDataset {
            manifest,
            root,
            format,
            class_names,
            train_images,
            val_images,
            test_images,
            fingerprint,
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
    if path.is_dir() {
        for entry in fs::read_dir(path).map_err(|error| {
            DatasetError::new(format!(
                "cannot read split directory {}: {error}",
                path.display()
            ))
        })? {
            let entry = entry.map_err(|error| DatasetError::new(error.to_string()))?;
            collect_images(&entry.path(), images)?;
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

fn detect_format(root: &Path, images: &[PathBuf]) -> Result<DatasetFormat, DatasetError> {
    if root.join("annotations").is_dir() {
        return Ok(DatasetFormat::Coco);
    }
    if root.join("train").is_dir()
        && images
            .iter()
            .all(|image| image.parent().and_then(Path::file_name).is_some())
        && !root.join("labels").exists()
    {
        return Ok(DatasetFormat::ClassificationFolders);
    }
    if root.join("labels").is_dir() || images.iter().any(|image| yolo_label_path(image).exists()) {
        return Ok(DatasetFormat::Yolo);
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

fn classification_names(train: &Path) -> Result<Vec<String>, DatasetError> {
    let mut names = Vec::new();
    for entry in fs::read_dir(train).map_err(|error| {
        DatasetError::new(format!(
            "cannot infer classification classes from {}: {error}",
            train.display()
        ))
    })? {
        let entry = entry.map_err(|error| DatasetError::new(error.to_string()))?;
        if entry.path().is_dir() {
            names.push(entry.file_name().to_string_lossy().into_owned());
        }
    }
    names.sort();
    Ok(names)
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
