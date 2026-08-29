use std::{fs, path::Path};

use crate::training::data::manifest::DatasetError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassificationItem {
    pub image: std::path::PathBuf,
    pub class_id: usize,
    pub image_id: String,
}

/// Enumerate a conventional class-folder split using the persisted class-name order.
pub fn enumerate(
    split: impl AsRef<Path>,
    class_names: &[String],
) -> Result<Vec<ClassificationItem>, DatasetError> {
    let split = split.as_ref();
    let mut items = Vec::new();
    for (class_id, name) in class_names.iter().enumerate() {
        let directory = split.join(name);
        if !directory.is_dir() {
            return Err(DatasetError::new(format!(
                "classification class directory is missing: {}",
                directory.display()
            )));
        }
        collect(&directory, class_id, split, &mut items)?;
    }
    items.sort_by(|a, b| a.image.cmp(&b.image));
    if items.is_empty() {
        return Err(DatasetError::new(format!(
            "classification split contains no supported images: {}",
            split.display()
        )));
    }
    Ok(items)
}

fn collect(
    path: &Path,
    class_id: usize,
    root: &Path,
    output: &mut Vec<ClassificationItem>,
) -> Result<(), DatasetError> {
    for entry in fs::read_dir(path)
        .map_err(|error| DatasetError::new(format!("cannot read {}: {error}", path.display())))?
    {
        let entry = entry.map_err(|error| DatasetError::new(error.to_string()))?;
        let path = entry.path();
        if path.is_dir() {
            collect(&path, class_id, root, output)?;
        } else if path.extension().and_then(|v| v.to_str()).is_some_and(|v| {
            matches!(
                v.to_ascii_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "webp"
            )
        }) {
            let image =
                fs::canonicalize(&path).map_err(|error| DatasetError::new(error.to_string()))?;
            let image_id = image
                .strip_prefix(root)
                .unwrap_or(&image)
                .to_string_lossy()
                .replace('\\', "/");
            output.push(ClassificationItem {
                image,
                class_id,
                image_id,
            });
        }
    }
    Ok(())
}
