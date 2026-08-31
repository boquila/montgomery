use std::{
    fmt::Write as _,
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
    validation_events: PathBuf,
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

/// One stable, task-neutral row in the Ultralytics-style epoch results table.
#[derive(Debug, Clone, Default)]
pub struct ResultsRow {
    pub epoch: usize,
    pub train_loss: f32,
    pub train_components: std::collections::BTreeMap<String, f32>,
    pub box_precision: Option<f32>,
    pub box_recall: Option<f32>,
    pub box_map50: Option<f32>,
    pub box_map50_95: Option<f32>,
    pub mask_precision: Option<f32>,
    pub mask_recall: Option<f32>,
    pub mask_map50: Option<f32>,
    pub mask_map50_95: Option<f32>,
    pub top1_accuracy: Option<f32>,
    pub top5_accuracy: Option<f32>,
    pub val_loss: Option<f32>,
    pub fitness: Option<f64>,
    pub learning_rate: f64,
}

const RESULTS_HEADER: &str = "epoch,train/loss,train/box_loss,train/iou_loss,train/cls_loss,train/obj_loss,train/dfl_loss,train/l1_loss,train/mask_loss,train/semantic_loss,train/one_to_many_box_loss,train/one_to_many_cls_loss,train/one_to_many_dfl_loss,train/one_to_many_l1_loss,train/one_to_many_mask_loss,train/one_to_many_semantic_loss,train/one_to_one_box_loss,train/one_to_one_cls_loss,train/one_to_one_dfl_loss,train/one_to_one_l1_loss,train/one_to_one_mask_loss,train/one_to_one_semantic_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),metrics/precision(M),metrics/recall(M),metrics/mAP50(M),metrics/mAP50-95(M),metrics/accuracy_top1,metrics/accuracy_top5,val/loss,fitness,lr/pg0\n";

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
        let task_root = config.run_root.join(task);
        fs::create_dir_all(&task_root)?;
        let root = task_root.join(format!(
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
        let validation_events = root.join("validation.jsonl");
        fs::File::create(&validation_events)?;
        fs::write(root.join("results.csv"), RESULTS_HEADER)?;
        Ok(Self {
            root,
            checkpoints,
            exports,
            events,
            validation_events,
        })
    }

    pub fn open(root: impl AsRef<Path>) -> Result<Self, std::io::Error> {
        let root = root.as_ref().to_path_buf();
        let checkpoints = root.join("checkpoints");
        let exports = root.join("exports");
        let events = root.join("events.jsonl");
        let validation_events = root.join("validation.jsonl");
        for required in [&checkpoints, &exports, &events] {
            if !required.exists() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("run directory is missing {}", required.display()),
                ));
            }
        }
        let results = root.join("results.csv");
        let migrate_legacy_metrics = !results.exists() && root.join("metrics.csv").exists();
        if !results.exists() {
            fs::write(&results, RESULTS_HEADER)?;
        }
        if !validation_events.exists() {
            fs::File::create(&validation_events)?;
        }
        let run = Self {
            root,
            checkpoints,
            exports,
            events,
            validation_events,
        };
        if migrate_legacy_metrics {
            let legacy = fs::read_to_string(run.root.join("metrics.csv"))?;
            for line in legacy.lines().skip(1) {
                let fields = line.split(',').collect::<Vec<_>>();
                if fields.len() != 4 {
                    continue;
                }
                let (Ok(epoch), Ok(train_loss), Ok(learning_rate)) = (
                    fields[0].parse::<usize>(),
                    fields[1].parse::<f32>(),
                    fields[3].parse::<f64>(),
                ) else {
                    continue;
                };
                run.append_results(&ResultsRow {
                    epoch: epoch + 1,
                    train_loss,
                    fitness: fields[2].parse::<f64>().ok(),
                    learning_rate,
                    ..ResultsRow::default()
                })?;
            }
        }
        Ok(run)
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
        self.append_events(std::slice::from_ref(event))
    }

    pub fn append_events(&self, events: &[StepEvent]) -> Result<(), std::io::Error> {
        let mut file = OpenOptions::new().append(true).open(&self.events)?;
        for event in events {
            serde_json::to_writer(&mut file, event).map_err(std::io::Error::other)?;
            file.write_all(b"\n")?;
        }
        file.flush()
    }

    pub fn append_validation<T: Serialize>(
        &self,
        epoch: usize,
        summary: &T,
    ) -> Result<(), std::io::Error> {
        let mut file = OpenOptions::new()
            .append(true)
            .open(&self.validation_events)?;
        serde_json::to_writer(
            &mut file,
            &serde_json::json!({ "epoch": epoch, "metrics": summary }),
        )
        .map_err(std::io::Error::other)?;
        file.write_all(b"\n")?;
        file.flush()
    }

    pub fn append_results(&self, row: &ResultsRow) -> Result<(), std::io::Error> {
        let mut file = OpenOptions::new()
            .append(true)
            .open(self.root.join("results.csv"))?;
        let component = |name: &str| {
            row.train_components
                .get(name)
                .map(|value| value.to_string())
                .unwrap_or_default()
        };
        let optional = |value: Option<f32>| value.map(|v| v.to_string()).unwrap_or_default();
        let values = [
            row.epoch.to_string(),
            row.train_loss.to_string(),
            component("box_loss"),
            component("iou_loss"),
            component("classification_loss"),
            component("objectness_loss"),
            component("dfl_loss"),
            component("l1_loss"),
            component("mask_loss"),
            component("semantic_loss"),
            component("one_to_many_box_loss"),
            component("one_to_many_classification_loss"),
            component("one_to_many_dfl_loss"),
            component("one_to_many_l1_loss"),
            component("one_to_many_mask_loss"),
            component("one_to_many_semantic_loss"),
            component("one_to_one_box_loss"),
            component("one_to_one_classification_loss"),
            component("one_to_one_dfl_loss"),
            component("one_to_one_l1_loss"),
            component("one_to_one_mask_loss"),
            component("one_to_one_semantic_loss"),
            optional(row.box_precision),
            optional(row.box_recall),
            optional(row.box_map50),
            optional(row.box_map50_95),
            optional(row.mask_precision),
            optional(row.mask_recall),
            optional(row.mask_map50),
            optional(row.mask_map50_95),
            optional(row.top1_accuracy),
            optional(row.top5_accuracy),
            optional(row.val_loss),
            row.fitness
                .map(|value| value.to_string())
                .unwrap_or_default(),
            row.learning_rate.to_string(),
        ];
        writeln!(file, "{}", values.join(","))?;
        file.flush()?;
        write_results_svg(
            &self.root.join("results.csv"),
            &self.root.join("results.svg"),
        )
    }
}

fn write_results_svg(csv: &Path, output: &Path) -> Result<(), std::io::Error> {
    let contents = fs::read_to_string(csv)?;
    let rows = contents
        .lines()
        .skip(1)
        .map(|line| line.split(',').collect::<Vec<_>>())
        .filter(|row| row.len() == 35)
        .collect::<Vec<_>>();
    let parse_series = |index: usize| {
        rows.iter()
            .map(|row| row[index].parse::<f32>().ok())
            .collect::<Vec<_>>()
    };
    let loss = parse_series(1);
    let fitness = parse_series(33);
    let mut svg = String::from(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="600" viewBox="0 0 1000 600"><rect width="1000" height="600" fill="#10141c"/><style>text{font-family:system-ui,sans-serif;fill:#dce4f2}.axis{stroke:#526174;stroke-width:1}.loss{fill:none;stroke:#ffb454;stroke-width:3}.fitness{fill:none;stroke:#62d6a8;stroke-width:3}</style><text x="60" y="42" font-size="26" font-weight="700">Training results</text><text x="60" y="78" font-size="15">train/loss</text><line class="axis" x1="60" y1="250" x2="960" y2="250"/><line class="axis" x1="60" y1="90" x2="60" y2="250"/><text x="60" y="328" font-size="15">fitness (validation quality)</text><line class="axis" x1="60" y1="530" x2="960" y2="530"/><line class="axis" x1="60" y1="340" x2="60" y2="530"/>"##,
    );
    let points = |values: &[Option<f32>], top: f32, bottom: f32, fixed_unit: bool| {
        let present = values.iter().flatten().copied().collect::<Vec<_>>();
        if present.is_empty() {
            return String::new();
        }
        let (minimum, maximum) = if fixed_unit {
            (0.0, 1.0)
        } else {
            let minimum = present.iter().copied().fold(f32::INFINITY, f32::min);
            let maximum = present.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            (minimum, maximum)
        };
        let span = (maximum - minimum).max(1e-9);
        values
            .iter()
            .enumerate()
            .filter_map(|(index, value)| {
                value.map(|value| {
                    let x = if values.len() <= 1 {
                        60.0
                    } else {
                        60.0 + 900.0 * index as f32 / (values.len() - 1) as f32
                    };
                    let y = bottom - (bottom - top) * (value - minimum) / span;
                    format!("{x:.1},{y:.1}")
                })
            })
            .collect::<Vec<_>>()
            .join(" ")
    };
    write!(
        svg,
        "<polyline class=\"loss\" points=\"{}\"/><polyline class=\"fitness\" points=\"{}\"/><text x=\"60\" y=\"574\" font-size=\"13\">{} epoch{}</text></svg>",
        points(&loss, 90.0, 250.0, false),
        points(&fitness, 340.0, 530.0, true),
        rows.len(),
        if rows.len() == 1 { "" } else { "s" },
    )
    .map_err(std::io::Error::other)?;
    fs::write(output, svg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ModelId,
        training::{ModelSpec, TrainingConfig},
    };

    #[test]
    fn epoch_results_are_tabular_and_render_a_plot() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("boquilens-report-{}-{nonce}", std::process::id()));
        let spec = ModelSpec::new(ModelId::Yolo26N, vec!["object".into()], Some([64, 64])).unwrap();
        let config = TrainingConfig::yolox(spec, "data.yaml".into(), root.clone());
        let run = RunDirectory::create(&config, "test").unwrap();
        let mut components = std::collections::BTreeMap::new();
        components.insert("box_loss".into(), 1.25);
        run.append_results(&ResultsRow {
            epoch: 1,
            train_loss: 2.5,
            train_components: components,
            box_map50_95: Some(0.42),
            fitness: Some(0.42),
            learning_rate: 1e-3,
            ..ResultsRow::default()
        })
        .unwrap();
        let csv = fs::read_to_string(run.root.join("results.csv")).unwrap();
        let rows = csv.lines().collect::<Vec<_>>();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].split(',').count(), rows[1].split(',').count());
        assert!(
            fs::read_to_string(run.root.join("results.svg"))
                .unwrap()
                .contains("Training results")
        );
        fs::remove_file(run.root.join("results.csv")).unwrap();
        fs::write(
            run.root.join("metrics.csv"),
            "epoch,loss,fitness,learning_rate\n0,3.0,0.2,0.001\n",
        )
        .unwrap();
        let reopened = RunDirectory::open(&run.root).unwrap();
        let migrated = fs::read_to_string(reopened.root.join("results.csv")).unwrap();
        assert!(migrated.lines().nth(1).unwrap().starts_with("1,3"));
        fs::remove_dir_all(root).unwrap();
    }
}
