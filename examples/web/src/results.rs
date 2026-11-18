//! Task-specific presentation of source-space predictions. No inference logic lives here.

use eframe::egui;
use montgomery::{Detection, Prediction};

use crate::app::{class_color, status_dot};

pub struct ResultView {
    pub prediction: Prediction,
    class_names: Vec<String>,
    overlay: Option<egui::TextureHandle>,
    overlay_threshold: Option<f32>,
}

impl ResultView {
    pub fn new(prediction: Prediction, class_names: Vec<String>) -> Self {
        Self {
            prediction,
            class_names,
            overlay: None,
            overlay_threshold: None,
        }
    }

    pub fn boxes(&self) -> Vec<Detection> {
        match &self.prediction {
            Prediction::Detections(items) => items.clone(),
            Prediction::Segmentations(items) => items
                .iter()
                .map(|item| Detection {
                    class_id: item.class_id,
                    class_name: item.class_name.clone(),
                    confidence: item.confidence,
                    xmin: item.xmin,
                    ymin: item.ymin,
                    xmax: item.xmax,
                    ymax: item.ymax,
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    pub fn summary(&self, threshold: f32) -> String {
        match &self.prediction {
            Prediction::Detections(_) | Prediction::Segmentations(_) => {
                let shown = self
                    .boxes()
                    .iter()
                    .filter(|d| d.confidence >= threshold)
                    .count();
                format!(
                    "{shown} objects shown at confidence ≥ {:.0}%",
                    threshold * 100.0
                )
            }
            Prediction::Classifications(_) => "Top 5 ImageNet classes · centered crop".into(),
            Prediction::Semantics(_) => "Semantic classes · Cityscapes".into(),
            Prediction::Depth(map) => {
                let stats = map.stats();
                format!(
                    "Depth {:.2}–{:.2} m · mean {:.2} m",
                    stats.min, stats.max, stats.mean
                )
            }
        }
    }

    /// Cache a single source-sized texture; only instance masks depend on confidence.
    pub fn overlay(&mut self, ctx: &egui::Context, threshold: f32) -> Option<&egui::TextureHandle> {
        let key = if matches!(self.prediction, Prediction::Segmentations(_)) {
            threshold
        } else {
            0.0
        };
        if self.overlay_threshold != Some(key) {
            self.overlay_threshold = Some(key);
            self.overlay = overlay_image(&self.prediction, threshold).map(|image| {
                ctx.load_texture("prediction-overlay", image, egui::TextureOptions::NEAREST)
            });
        }
        self.overlay.as_ref()
    }

    pub fn legend(&self, ui: &mut egui::Ui, threshold: f32) {
        ui.horizontal_wrapped(|ui| match &self.prediction {
            Prediction::Classifications(items) => {
                for (rank, item) in items.iter().enumerate() {
                    ui.label(format!(
                        "{}. {} {:.1}%",
                        rank + 1,
                        item.class_name,
                        item.confidence * 100.0
                    ));
                    ui.add_space(8.0);
                }
            }
            Prediction::Semantics(mask) => {
                for (id, count) in mask
                    .class_histogram(self.class_names.len())
                    .iter()
                    .enumerate()
                {
                    if *count == 0 {
                        continue;
                    }
                    status_dot(ui, class_color(id));
                    ui.label(format!(
                        "{} {:.1}%",
                        self.class_names[id],
                        *count as f64 / mask.data.len() as f64 * 100.0
                    ));
                    ui.add_space(8.0);
                }
            }
            Prediction::Depth(_) => {
                status_dot(ui, egui::Color32::from_rgb(253, 231, 37));
                ui.label("Near");
                status_dot(ui, egui::Color32::from_rgb(68, 1, 84));
                ui.label("Far · colors scaled to this image's depth range");
            }
            _ => {
                let boxes = self.boxes();
                let visible: Vec<_> = boxes.iter().filter(|d| d.confidence >= threshold).collect();
                if visible.is_empty() {
                    ui.label("No objects above the confidence threshold.");
                }
                for item in visible {
                    status_dot(ui, class_color(item.class_id));
                    ui.label(format!(
                        "{} {:.0}%",
                        item.class_name,
                        item.confidence * 100.0
                    ));
                    ui.add_space(8.0);
                }
            }
        });
    }
}

fn overlay_image(prediction: &Prediction, threshold: f32) -> Option<egui::ColorImage> {
    match prediction {
        Prediction::Segmentations(items) => {
            let first = items.first()?;
            let size = [first.mask.width as usize, first.mask.height as usize];
            let mut image = egui::ColorImage::filled(size, egui::Color32::TRANSPARENT);
            // Stronger instances win overlapping pixels; filtering also removes their masks.
            let mut visible: Vec<_> = items.iter().filter(|d| d.confidence >= threshold).collect();
            visible.sort_by(|a, b| a.confidence.total_cmp(&b.confidence));
            for item in visible {
                let color = class_color(item.class_id);
                let color =
                    egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 105);
                for (pixel, covered) in image.pixels.iter_mut().zip(&item.mask.data) {
                    if *covered {
                        *pixel = color;
                    }
                }
            }
            Some(image)
        }
        Prediction::Semantics(mask) => {
            let mut image = egui::ColorImage::filled(
                [mask.width as usize, mask.height as usize],
                egui::Color32::TRANSPARENT,
            );
            for (pixel, id) in image.pixels.iter_mut().zip(&mask.data) {
                let color = class_color(*id as usize);
                *pixel =
                    egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 128);
            }
            Some(image)
        }
        Prediction::Depth(map) => {
            let rgb = montgomery::annotate_depth(map).to_rgb8();
            Some(egui::ColorImage::from_rgb(
                [map.width as usize, map.height as usize],
                rgb.as_raw(),
            ))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use montgomery::{DepthMap, InstanceMask, SegmentationDetection, SemanticMask};

    #[test]
    fn instance_overlay_filters_masks_and_preserves_source_pixels() {
        let item = SegmentationDetection {
            class_id: 0,
            class_name: "person".into(),
            confidence: 0.6,
            xmin: 0.0,
            ymin: 0.0,
            xmax: 2.0,
            ymax: 1.0,
            mask: InstanceMask {
                width: 2,
                height: 1,
                data: vec![false, true],
            },
        };
        let prediction = Prediction::Segmentations(vec![item]);
        let shown = overlay_image(&prediction, 0.5).unwrap();
        assert_eq!(shown.size, [2, 1]);
        assert_eq!(shown.pixels[0], egui::Color32::TRANSPARENT);
        assert_eq!(shown.pixels[1].a(), 105);
        assert!(
            overlay_image(&prediction, 0.7)
                .unwrap()
                .pixels
                .iter()
                .all(|p| p.a() == 0)
        );
        assert!(overlay_image(&Prediction::Segmentations(vec![]), 0.5).is_none());
    }

    #[test]
    fn dense_overlays_preserve_dimensions_and_depth_direction() {
        let semantics = Prediction::Semantics(SemanticMask {
            width: 2,
            height: 1,
            data: vec![0, 1],
        });
        let image = overlay_image(&semantics, 0.5).unwrap();
        assert_eq!(image.size, [2, 1]);
        assert_ne!(image.pixels[0], image.pixels[1]);
        let depth = Prediction::Depth(DepthMap {
            width: 2,
            height: 1,
            data: vec![1.0, 10.0],
        });
        let image = overlay_image(&depth, 0.5).unwrap();
        assert_eq!(image.pixels[0], egui::Color32::from_rgb(253, 231, 37));
        assert_eq!(image.pixels[1], egui::Color32::from_rgb(68, 1, 84));
    }
}
