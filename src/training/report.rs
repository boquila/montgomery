use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

use crate::training::{TrainingConfig, data::ResolvedDataset, optimizer::ParameterGroupManifest};

#[derive(Debug, Clone)]
pub struct RunDirectory {
    pub root: PathBuf,
    pub checkpoints: PathBuf,
    pub exports: PathBuf,
    events: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepEvent {
    pub epoch: usize,
    pub micro_step: u64,
    pub optimizer_step: u64,
    pub learning_rate: f64,
    pub total_loss: f32,
    pub components: std::collections::BTreeMap<String, f32>,
    pub targets: usize,
    pub foreground: usize,
}

impl RunDirectory {
    pub fn create(config: &TrainingConfig, name: &str) -> Result<Self, std::io::Error> {
        let clean_name: String = name
            .chars()
            .map(|value| {
                if value.is_ascii_alphanumeric() || matches!(value, '-' | '_') {
                    value
                } else {
                    '-'
                }
            })
            .collect();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let task = format!("{:?}", config.model.task).to_ascii_lowercase();
        let root = config.run_root.join(task).join(format!(
            "{}-{timestamp}-{:08x}",
            clean_name.trim_matches('-'),
            std::process::id()
        ));
        fs::create_dir(&root)?;
        let checkpoints = root.join("checkpoints");
        let exports = root.join("exports");
        fs::create_dir(&checkpoints)?;
        fs::create_dir(&exports)?;
        fs::write(
            root.join("config.resolved.json"),
            serde_json::to_vec_pretty(config).map_err(std::io::Error::other)?,
        )?;
        fs::write(
            root.join("config.requested.yaml"),
            serde_yaml::to_string(config).map_err(std::io::Error::other)?,
        )?;
        let events = root.join("events.jsonl");
        fs::File::create(&events)?;
        fs::write(
            root.join("metrics.csv"),
            b"epoch,loss,fitness,learning_rate\n",
        )?;
        Ok(Self {
            root,
            checkpoints,
            exports,
            events,
        })
    }

    pub fn open(root: impl AsRef<Path>) -> Result<Self, std::io::Error> {
        let root = root.as_ref().to_path_buf();
        let checkpoints = root.join("checkpoints");
        let exports = root.join("exports");
        let events = root.join("events.jsonl");
        for required in [&checkpoints, &exports, &events] {
            if !required.exists() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("run directory is missing {}", required.display()),
                ));
            }
        }
        Ok(Self {
            root,
            checkpoints,
            exports,
            events,
        })
    }

    pub fn write_dataset(&self, dataset: &ResolvedDataset) -> Result<(), std::io::Error> {
        fs::write(
            self.root.join("dataset.json"),
            serde_json::to_vec_pretty(dataset).map_err(std::io::Error::other)?,
        )
    }

    pub fn write_environment(
        &self,
        adapter: &str,
        dataset: &ResolvedDataset,
    ) -> Result<(), std::io::Error> {
        let metadata = serde_json::json!({
            "format": "boquilens-training-run-v1",
            "crate_version": env!("CARGO_PKG_VERSION"),
            "backend": "burn-wgpu",
            "adapter": adapter,
            "dataset_fingerprint": dataset.fingerprint,
            "references": {
                "ultralytics": { "version": "8.4.117", "commit": "461196cf0", "license": "AGPL-3.0" },
                "yolox": { "version": "0.1.1rc0", "license": "Apache-2.0" }
            }
        });
        fs::write(
            self.root.join("environment.json"),
            serde_json::to_vec_pretty(&metadata).map_err(std::io::Error::other)?,
        )
    }

    pub fn write_parameter_groups(
        &self,
        groups: &ParameterGroupManifest,
    ) -> Result<(), std::io::Error> {
        fs::write(
            self.root.join("parameter-groups.json"),
            serde_json::to_vec_pretty(groups).map_err(std::io::Error::other)?,
        )
    }

    pub fn append_event(&self, event: &StepEvent) -> Result<(), std::io::Error> {
        let mut file = OpenOptions::new().append(true).open(&self.events)?;
        serde_json::to_writer(&mut file, event).map_err(std::io::Error::other)?;
        file.write_all(b"\n")?;
        file.flush()
    }

    pub fn append_metrics(
        &self,
        epoch: usize,
        loss: f32,
        fitness: Option<f64>,
        learning_rate: f64,
    ) -> Result<(), std::io::Error> {
        let mut file = OpenOptions::new()
            .append(true)
            .open(self.root.join("metrics.csv"))?;
        writeln!(
            file,
            "{epoch},{loss},{},{learning_rate}",
            fitness.map(|v| v.to_string()).unwrap_or_default()
        )
    }
}
