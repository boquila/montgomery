//! Montgomery web demo: pick an image on the left, run any supported YOLO26 nano task.
//!
//! The same Rust code builds natively (inference on a worker thread) and for the browser
//! (WebAssembly, inference on the main thread until a worker port lands).

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

use eframe::egui;
use montgomery::{Detection, ModelId, PredictOptions, Predictor};

use crate::results::ResultView;

const GITHUB_URL: &str = "https://github.com/boquila/montgomery";

/// Only nano variants already implemented by Montgomery are offered here.
#[derive(Clone, Copy)]
struct DemoModel {
    id: ModelId,
    label: &'static str,
    artifact: &'static str,
    boxes: bool,
}

const MODELS: &[DemoModel] = &[
    DemoModel {
        id: ModelId::Yolo26N,
        label: "Object detection",
        artifact: "yolo26n.bpk",
        boxes: true,
    },
    DemoModel {
        id: ModelId::Yolo26NSeg,
        label: "Instance segmentation",
        artifact: "yolo26n-seg.bpk",
        boxes: true,
    },
    DemoModel {
        id: ModelId::Yolo26NCls,
        label: "Classification",
        artifact: "yolo26n-cls.bpk",
        boxes: false,
    },
    DemoModel {
        id: ModelId::Yolo26NSem,
        label: "Semantic segmentation",
        artifact: "yolo26n-sem.bpk",
        boxes: false,
    },
    DemoModel {
        id: ModelId::Yolo26NDepth,
        label: "Depth estimation",
        artifact: "yolo26n-depth.bpk",
        boxes: false,
    },
];

/// One selectable demo image, embedded so the app also works fully offline.
struct DemoImage {
    name: &'static str,
    /// Short label shown under the thumbnail.
    label: &'static str,
    encoded: &'static [u8],
}

const IMAGES: &[DemoImage] = &[
    DemoImage {
        name: "dog_bike_man.jpg",
        label: "Dog & bike",
        encoded: include_bytes!("../assets/dog_bike_man.jpg"),
    },
    DemoImage {
        name: "bus.jpg",
        label: "Bus",
        encoded: include_bytes!("../assets/bus.jpg"),
    },
    DemoImage {
        name: "zidane.jpg",
        label: "Zidane",
        encoded: include_bytes!("../assets/zidane.jpg"),
    },
    DemoImage {
        name: "camera.jpg",
        label: "Cameraman",
        encoded: include_bytes!("../assets/camera.jpg"),
    },
    DemoImage {
        name: "astronaut.jpg",
        label: "Astronaut",
        encoded: include_bytes!("../assets/astronaut.jpg"),
    },
    DemoImage {
        name: "chelsea.jpg",
        label: "Chelsea",
        encoded: include_bytes!("../assets/chelsea.jpg"),
    },
    DemoImage {
        name: "coffee.jpg",
        label: "Coffee",
        encoded: include_bytes!("../assets/coffee.jpg"),
    },
];

/// Accent used for selection outlines and status text.
const ACCENT: egui::Color32 = egui::Color32::from_rgb(51, 218, 114);
/// Fill for the primary action button (dark enough for light label text).
const RUN_FILL: egui::Color32 = egui::Color32::from_rgb(27, 104, 62);
/// Success color for completed inference.
const SUCCESS: egui::Color32 = egui::Color32::from_rgb(102, 217, 138);
/// Warning color while a run is in flight.
const RUNNING: egui::Color32 = egui::Color32::from_rgb(240, 190, 80);

/// A finished inference run for one selected image.
struct Outcome {
    image_index: usize,
    model_index: usize,
    result: Result<ResultView, String>,
    elapsed_ms: f64,
}

/// Model availability shared between the UI thread and the loading/inference task.
enum ModelSlot {
    Unloaded,
    Loading,
    Running,
    Ready(Box<Predictor>),
    Failed(String),
}

/// One embedded image, decoded and uploaded as a texture at startup.
struct ImageView {
    image: image::DynamicImage,
    texture: egui::TextureHandle,
    width: u32,
    height: u32,
}

pub struct DemoApp {
    images: Vec<ImageView>,
    selected: usize,
    selected_model: usize,
    models: Vec<Arc<Mutex<ModelSlot>>>,
    busy: Arc<AtomicBool>,
    outcome: Arc<Mutex<Option<Outcome>>>,
    /// Finished runs indexed by model, then image.
    outcomes: Vec<Vec<Option<Outcome>>>,
    /// Model and image whose inference is executing, even if the selection changes.
    run_image: Option<(usize, usize)>,
    threshold: f32,
}

impl DemoApp {
    pub fn new(ctx: &egui::Context) -> Self {
        egui_extras::install_image_loaders(ctx);
        ctx.set_theme(egui::Theme::Dark);
        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = egui::Color32::from_rgb(15, 17, 21);
        visuals.window_fill = visuals.panel_fill;
        visuals.selection.bg_fill = egui::Color32::from_rgb(25, 65, 43);
        visuals.selection.stroke = egui::Stroke::new(1.0, ACCENT);
        ctx.set_visuals(visuals);
        ctx.style_mut_of(egui::Theme::Dark, |style| {
            style.spacing.item_spacing = egui::vec2(8.0, 7.0);
            style.spacing.button_padding = egui::vec2(12.0, 7.0);
            style
                .text_styles
                .insert(egui::TextStyle::Body, egui::FontId::proportional(14.0));
            style
                .text_styles
                .insert(egui::TextStyle::Button, egui::FontId::proportional(14.0));
        });

        let images = IMAGES
            .iter()
            .map(|entry| {
                let image = image::load_from_memory(entry.encoded)
                    .expect("embedded demo images must decode");
                let (width, height) = (image.width(), image.height());
                let texture = ctx.load_texture(
                    entry.name,
                    egui::ColorImage::from_rgb(
                        [width as usize, height as usize],
                        image.to_rgb8().as_raw(),
                    ),
                    egui::TextureOptions::default(),
                );
                ImageView {
                    image,
                    texture,
                    width,
                    height,
                }
            })
            .collect();

        let models: Vec<_> = MODELS
            .iter()
            .map(|_| Arc::new(Mutex::new(ModelSlot::Unloaded)))
            .collect();
        *models[0].lock().unwrap() = ModelSlot::Loading;
        spawn_model_load(ctx.clone(), models[0].clone(), MODELS[0]);

        Self {
            images,
            selected: 0,
            selected_model: 0,
            models,
            busy: Arc::new(AtomicBool::new(false)),
            outcome: Arc::new(Mutex::new(None)),
            outcomes: MODELS
                .iter()
                .map(|_| (0..IMAGES.len()).map(|_| None).collect())
                .collect(),
            run_image: None,
            threshold: 0.25,
        }
    }

    fn photo_card(&self, ui: &mut egui::Ui, index: usize, compact: bool) -> egui::Response {
        let view = &self.images[index];
        let selected = self.selected == index;
        let card = egui::Frame::NONE
            .inner_margin(egui::Margin::same(5))
            .corner_radius(6)
            .fill(if selected {
                egui::Color32::from_rgb(21, 45, 32)
            } else {
                egui::Color32::TRANSPARENT
            })
            .stroke(if selected {
                egui::Stroke::new(2.0, ACCENT)
            } else {
                egui::Stroke::NONE
            })
            .show(ui, |ui| {
                if compact {
                    ui.set_width(100.0);
                } else {
                    ui.set_min_width(ui.available_width());
                }
                ui.vertical_centered(|ui| {
                    ui.add(
                        egui::Image::from_texture(&view.texture)
                            .max_width(if compact { 96.0 } else { 150.0 })
                            .max_height(if compact { 56.0 } else { 96.0 })
                            .maintain_aspect_ratio(true),
                    );
                    let label =
                        ui.label(egui::RichText::new(IMAGES[index].label).size(14.0).color(
                            if selected {
                                egui::Color32::WHITE
                            } else {
                                egui::Color32::from_gray(190)
                            },
                        ));
                    if !compact
                        && self.outcomes[self.selected_model][index]
                            .as_ref()
                            .is_some_and(|o| o.result.is_ok())
                    {
                        let center = egui::pos2(label.rect.right() + 6.0, label.rect.center().y);
                        ui.painter().circle_filled(center, 2.5, SUCCESS);
                    }
                });
            });
        let response = ui
            .interact(
                card.response.rect,
                ui.id().with(("sample-image", index)),
                egui::Sense::click(),
            )
            .on_hover_text(IMAGES[index].name);
        if response.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
            if !selected {
                ui.painter().rect_stroke(
                    card.response.rect,
                    6.0,
                    egui::Stroke::new(1.5, egui::Color32::from_gray(120)),
                    egui::StrokeKind::Inside,
                );
            }
        }
        response
    }

    fn run_inference(&mut self, ctx: &egui::Context) {
        if self.busy.swap(true, Ordering::Relaxed) {
            return;
        }
        self.outcomes[self.selected_model][self.selected] = None;
        self.run_image = Some((self.selected_model, self.selected));
        ctx.request_repaint();
        let image = self.images[self.selected].image.clone();
        spawn_inference(
            ctx.clone(),
            self.models[self.selected_model].clone(),
            self.busy.clone(),
            self.outcome.clone(),
            image,
            self.selected,
            self.selected_model,
        );
    }
}

impl eframe::App for DemoApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        if let Some(outcome) = self.outcome.lock().unwrap().take() {
            let key = (outcome.model_index, outcome.image_index);
            self.outcomes[key.0][key.1] = Some(outcome);
            if self.run_image == Some(key) {
                self.run_image = None;
            }
        }
        let busy = self.busy.load(Ordering::Relaxed);
        let compact = ui.available_width() < 700.0;
        // A task can be selected while another model loads or a native run finishes.
        // Each slot owns its load result, so a late completion cannot replace another task.
        egui::Panel::top("task-selector").show(ui, |ui| {
            ui.add_space(5.0);
            ui.horizontal_wrapped(|ui| {
                ui.label("Task");
                egui::ComboBox::from_id_salt("task")
                    .selected_text(MODELS[self.selected_model].label)
                    .width(220.0)
                    .show_ui(ui, |ui| {
                        for (index, model) in MODELS.iter().enumerate() {
                            ui.selectable_value(&mut self.selected_model, index, model.label);
                        }
                    });
                ui.label(
                    egui::RichText::new(format!(
                        "{} · experimental",
                        MODELS[self.selected_model].id
                    ))
                    .weak(),
                );
                ui.add(
                    egui::Hyperlink::from_label_and_url("GitHub", GITHUB_URL).open_in_new_tab(true),
                )
                .on_hover_text("View the Montgomery source code");
            });
            ui.add_space(5.0);
        });
        let spec = MODELS[self.selected_model];
        let slot = &self.models[self.selected_model];
        let should_load = {
            let mut state = slot.lock().unwrap();
            if matches!(*state, ModelSlot::Unloaded) {
                *state = ModelSlot::Loading;
                true
            } else {
                false
            }
        };
        if should_load {
            spawn_model_load(ctx.clone(), slot.clone(), spec);
        }
        let (model_state, model_ready, model_failed, status_color) = match &*slot.lock().unwrap() {
            ModelSlot::Unloaded | ModelSlot::Loading => {
                (format!("Loading {}…", spec.id), false, false, RUNNING)
            }
            ModelSlot::Running => (format!("{} running…", spec.id), false, false, RUNNING),
            ModelSlot::Failed(error) => (
                format!("Model error: {error}"),
                false,
                true,
                egui::Color32::from_rgb(240, 100, 100),
            ),
            ModelSlot::Ready(_) => (format!("{} ready", spec.id), true, false, SUCCESS),
        };

        if compact {
            egui::Panel::top("compact-images").show(ui, |ui| {
                ui.add_space(5.0);
                ui.horizontal(|ui| {
                    wordmark(ui, 174.0);
                    ui.label("Choose a photo");
                });
                ui.add_space(4.0);
                egui::ScrollArea::horizontal().show(ui, |ui| {
                    ui.horizontal(|ui| {
                        for index in 0..self.images.len() {
                            if self.photo_card(ui, index, true).clicked() {
                                self.selected = index;
                            }
                        }
                    });
                });
                if model_failed && ui.button("Retry loading").clicked() {
                    *self.models[self.selected_model].lock().unwrap() = ModelSlot::Loading;
                    spawn_model_load(ctx.clone(), self.models[self.selected_model].clone(), spec);
                }
            });
        } else {
            egui::Panel::left("images")
                .resizable(false)
                .default_size(230.0)
                .show(ui, |ui| {
                    ui.add_space(18.0);
                    wordmark(ui, 206.0);
                    ui.add_space(12.0);
                    ui.label(
                        egui::RichText::new("Choose a photo").color(egui::Color32::from_gray(205)),
                    );
                    ui.add_space(4.0);
                    ui.separator();
                    ui.add_space(8.0);
                    let mut clicked_card = None;
                    // One column, scrollable so every sample stays reachable in a short window.
                    let status_height = if model_failed { 96.0 } else { 56.0 };
                    let list_height = (ui.available_height() - status_height).max(120.0);
                    egui::ScrollArea::vertical()
                        .max_height(list_height)
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            for index in 0..self.images.len() {
                                if self.photo_card(ui, index, false).clicked() {
                                    clicked_card = Some(index);
                                }
                                ui.add_space(6.0);
                            }
                        });
                    if let Some(index) = clicked_card {
                        self.selected = index;
                    }
                    ui.add_space(4.0);
                    ui.separator();
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        status_dot(ui, status_color);
                        ui.label(
                            egui::RichText::new(if model_failed {
                                "Model unavailable"
                            } else {
                                model_state.as_str()
                            })
                            .size(12.0)
                            .color(status_color),
                        );
                    });
                    if model_failed && ui.button("Retry loading").clicked() {
                        *self.models[self.selected_model].lock().unwrap() = ModelSlot::Loading;
                        spawn_model_load(
                            ctx.clone(),
                            self.models[self.selected_model].clone(),
                            spec,
                        );
                    }
                });
        }

        // Photo selection is processed above; derive status from the current selection.
        egui::CentralPanel::default().show(ui, |ui| {
            ui.add_space(8.0);
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new(spec.label).size(22.0).strong());
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new("runs entirely in your browser")
                        .size(12.5)
                        .color(egui::Color32::from_gray(190)),
                );
            });
            ui.add_space(6.0);
            let has_result = self.outcomes[self.selected_model][self.selected].is_some();
            ui.horizontal(|ui| {
                let enabled = model_ready && !busy;
                let run = ui.add_enabled(
                    enabled,
                    egui::Button::new(
                        egui::RichText::new(if has_result {
                            "Run again"
                        } else {
                            "Run inference"
                        })
                        .strong(),
                    )
                    .min_size(egui::vec2(160.0, 32.0))
                    .fill(if enabled {
                        RUN_FILL
                    } else {
                        egui::Color32::from_gray(45)
                    }),
                );
                if run.clicked() {
                    self.run_inference(&ctx);
                }
            });
            // Recompute after handling Run so dimming appears in the first painted frame.
            let busy = self.busy.load(Ordering::Relaxed);
            let running_here = busy && self.run_image == Some((self.selected_model, self.selected));
            ui.add_space(4.0);
            if spec.boxes {
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        egui::RichText::new("Show objects above")
                            .size(12.5)
                            .color(egui::Color32::from_gray(205)),
                    );
                    // A wide track is easy to grab; the percentage is easier to read than 0.25.
                    ui.spacing_mut().slider_width = if compact {
                        (ui.available_width() - 190.0).clamp(90.0, 220.0)
                    } else {
                        220.0
                    };
                    let slider = ui.add(
                        egui::Slider::new(&mut self.threshold, 0.05..=0.95)
                            .show_value(true)
                            .step_by(0.01)
                            .custom_formatter(|value, _| format!("{:.0}%", value * 100.0)),
                    );
                    if slider.hovered() {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
                        let wheel = ui.input(|input| input.smooth_scroll_delta.y);
                        if wheel != 0.0 {
                            let next = self.threshold + wheel * 0.0005;
                            self.threshold = (next.clamp(0.05, 0.95) * 100.0).round() / 100.0;
                        }
                    }
                });
            }
            ui.add_space(6.0);

            // Reserve the status row so starting inference does not move the image.
            let status_height = if compact { 72.0 } else { 42.0 };
            ui.allocate_ui(egui::vec2(ui.available_width(), status_height), |ui| {
                ui.set_min_height(status_height);
                let current = self.outcomes[self.selected_model][self.selected].as_ref();
                if running_here {
                    status_banner(ui, RUNNING, "Running inference…", "This can take a moment.");
                } else if let Some(outcome) = current {
                    match &outcome.result {
                        Ok(result) => status_banner(
                            ui,
                            SUCCESS,
                            "Inference complete",
                            &format!(
                                "{} · {:.0} ms",
                                result.summary(self.threshold),
                                outcome.elapsed_ms
                            ),
                        ),
                        Err(error) => {
                            status_banner(ui, egui::Color32::LIGHT_RED, "Inference failed", error)
                        }
                    }
                } else if model_failed {
                    status_banner(ui, status_color, "Model not available", &model_state);
                } else if !model_ready {
                    status_banner(
                        ui,
                        RUNNING,
                        "Loading model…",
                        "Run inference will be available when ready.",
                    );
                }
            });
            ui.add_space(8.0);

            let view = &self.images[self.selected];
            let available = ui.available_size();
            let footer = if spec.boxes { 54.0 } else { 90.0 };
            let scale = (available.x / view.width as f32)
                .min((available.y - footer).max(1.0) / view.height as f32)
                .max(0.01);
            let display = egui::vec2(view.width as f32 * scale, view.height as f32 * scale);
            ui.horizontal(|ui| {
                let offset = ((available.x - display.x) * 0.5).max(0.0);
                if offset > 0.0 {
                    ui.add_space(offset);
                }
                let response =
                    ui.add(egui::Image::from_texture(&view.texture).fit_to_exact_size(display));
                let image_rect = response.rect;
                if let Some(Ok(result)) = self.outcomes[self.selected_model][self.selected]
                    .as_mut()
                    .map(|o| &mut o.result)
                {
                    if let Some(texture) = result.overlay(&ctx, self.threshold) {
                        ui.painter_at(image_rect).image(
                            texture.id(),
                            image_rect,
                            egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                            egui::Color32::WHITE,
                        );
                    }
                    draw_detections(ui, &image_rect, view, &result.boxes(), self.threshold);
                } else if running_here || !model_ready {
                    ui.painter_at(image_rect).rect_filled(
                        image_rect,
                        0.0,
                        egui::Color32::from_black_alpha(120),
                    );
                }
            });
            ui.add_space(6.0);
            if let Some(Ok(result)) = self.outcomes[self.selected_model][self.selected]
                .as_ref()
                .map(|o| &o.result)
            {
                egui::ScrollArea::vertical()
                    .max_height(80.0)
                    .show(ui, |ui| result.legend(ui, self.threshold));
            }
        });
    }
}

/// A small painted status dot. The default egui font lacks U+25CF, which rendered as tofu.
pub(crate) fn status_dot(ui: &mut egui::Ui, color: egui::Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 12.0), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), 3.5, color);
}

fn wordmark(ui: &mut egui::Ui, width: f32) {
    let logo = ui
        .add(
            egui::Image::new(egui::include_image!("../assets/montgomery.svg"))
                .fit_to_exact_size(egui::vec2(width, width * 31.15 / 206.0))
                .alt_text("Montgomery on GitHub")
                .sense(egui::Sense::click()),
        )
        .on_hover_cursor(egui::CursorIcon::PointingHand)
        .on_hover_text("Montgomery on GitHub");
    if logo.clicked() {
        ui.ctx().open_url(egui::OpenUrl::new_tab(GITHUB_URL));
    }
}

/// One-line status strip under the toolbar, colored by state.
fn status_banner(ui: &mut egui::Ui, color: egui::Color32, title: &str, detail: &str) {
    egui::Frame::NONE
        .fill(egui::Color32::from_rgba_unmultiplied(
            color.r(),
            color.g(),
            color.b(),
            30,
        ))
        .stroke(egui::Stroke::new(1.0, color))
        .corner_radius(5)
        .inner_margin(egui::Margin::symmetric(10, 7))
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                status_dot(ui, color);
                ui.label(egui::RichText::new(title).strong().size(14.0).color(color));
                ui.label(
                    egui::RichText::new(detail)
                        .size(13.0)
                        .color(egui::Color32::from_gray(215)),
                );
            });
        });
}

/// Draw detection boxes and labels over the displayed image, in source-image coordinates.
fn draw_detections(
    ui: &mut egui::Ui,
    target: &egui::Rect,
    view: &ImageView,
    detections: &[Detection],
    threshold: f32,
) {
    let painter = ui.painter_at(*target);
    let scale = egui::vec2(
        target.width() / view.width as f32,
        target.height() / view.height as f32,
    );
    let font = egui::TextStyle::Body.resolve(ui.style());
    let visible: Vec<_> = detections
        .iter()
        .filter(|detection| detection.confidence >= threshold)
        .collect();

    // Boxes first, then labels, so no later box stroke crosses an earlier label.
    for detection in &visible {
        let color = class_color(detection.class_id);
        let rect = egui::Rect::from_min_max(
            target.left_top() + egui::vec2(detection.xmin * scale.x, detection.ymin * scale.y),
            target.left_top() + egui::vec2(detection.xmax * scale.x, detection.ymax * scale.y),
        );
        painter.rect_stroke(
            rect,
            0.0,
            egui::Stroke::new(3.0, color),
            egui::StrokeKind::Inside,
        );
    }

    for detection in visible {
        let color = class_color(detection.class_id);
        let rect = egui::Rect::from_min_max(
            target.left_top() + egui::vec2(detection.xmin * scale.x, detection.ymin * scale.y),
            target.left_top() + egui::vec2(detection.xmax * scale.x, detection.ymax * scale.y),
        );
        let text = format!(
            "{} {:.0}%",
            detection.class_name,
            detection.confidence * 100.0
        );
        let galley = painter.layout(text, font.clone(), egui::Color32::WHITE, f32::INFINITY);
        let pad = egui::vec2(8.0, 4.0);
        let size = galley.size() + pad * 2.0;
        let mut pos = egui::pos2(rect.left(), rect.top() - size.y);
        if pos.y < target.top() {
            pos.y = rect.top() + 2.0;
        }
        if pos.x + size.x > target.right() {
            pos.x = (target.right() - size.x).max(target.left());
        }
        let label_rect = egui::Rect::from_min_size(pos, size);
        painter.rect_filled(
            label_rect,
            3.0,
            egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 235),
        );
        painter.galley(label_rect.left_top() + pad, galley, egui::Color32::WHITE);
    }
}

/// Stable per-class box colors (tableau palette).
pub(crate) fn class_color(class_id: usize) -> egui::Color32 {
    const PALETTE: &[egui::Color32] = &[
        egui::Color32::from_rgb(31, 119, 180),
        egui::Color32::from_rgb(255, 127, 14),
        egui::Color32::from_rgb(44, 160, 44),
        egui::Color32::from_rgb(214, 39, 40),
        egui::Color32::from_rgb(148, 103, 189),
        egui::Color32::from_rgb(140, 86, 75),
        egui::Color32::from_rgb(227, 119, 194),
        egui::Color32::from_rgb(127, 127, 127),
        egui::Color32::from_rgb(188, 189, 34),
        egui::Color32::from_rgb(23, 190, 207),
        egui::Color32::from_rgb(174, 199, 232),
        egui::Color32::from_rgb(255, 187, 120),
        egui::Color32::from_rgb(152, 223, 138),
        egui::Color32::from_rgb(255, 152, 150),
        egui::Color32::from_rgb(197, 176, 213),
        egui::Color32::from_rgb(196, 156, 148),
        egui::Color32::from_rgb(247, 182, 210),
        egui::Color32::from_rgb(199, 199, 199),
        egui::Color32::from_rgb(219, 219, 141),
    ];
    PALETTE[class_id % PALETTE.len()]
}

/// Kick off model loading off the current call site.
fn spawn_model_load(ctx: egui::Context, slot: Arc<Mutex<ModelSlot>>, spec: DemoModel) {
    let fill = move |result: Result<Predictor, String>, slot: &Arc<Mutex<ModelSlot>>| {
        *slot.lock().unwrap() = match result {
            Ok(model) => {
                #[cfg(target_arch = "wasm32")]
                web_sys::console::log_1(&"montgomery: model ready".into());
                ModelSlot::Ready(Box::new(model))
            }
            Err(error) => {
                #[cfg(target_arch = "wasm32")]
                web_sys::console::error_1(
                    &format!("montgomery: model load failed: {error}").into(),
                );
                ModelSlot::Failed(error)
            }
        };
    };

    #[cfg(target_arch = "wasm32")]
    {
        wasm_bindgen_futures::spawn_local(async move {
            let result = load_model(spec).await;
            fill(result, &slot);
            ctx.request_repaint();
        });
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::spawn(move || {
            let result = load_model(spec);
            fill(result, &slot);
            ctx.request_repaint();
        });
    }
}

/// Run the selected task and publish its result without holding the UI model lock.
fn spawn_inference(
    ctx: egui::Context,
    slot: Arc<Mutex<ModelSlot>>,
    busy: Arc<AtomicBool>,
    outcome: Arc<Mutex<Option<Outcome>>>,
    image: image::DynamicImage,
    image_index: usize,
    model_index: usize,
) {
    let run = move || {
        let started = now_ms();
        let state = std::mem::replace(&mut *slot.lock().unwrap(), ModelSlot::Running);
        let result = match &state {
            ModelSlot::Ready(model) => model
                .inference(&image)
                .map(|prediction| ResultView::new(prediction, model.class_names().to_vec()))
                .map_err(|error| error.to_string()),
            _ => Err("The selected model is not ready. Retry once it has loaded.".into()),
        };
        *slot.lock().unwrap() = state;
        let elapsed_ms = now_ms() - started;
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(
            &format!(
                "montgomery: {} completed in {elapsed_ms:.0} ms",
                MODELS[model_index].id
            )
            .into(),
        );
        *outcome.lock().unwrap() = Some(Outcome {
            image_index,
            model_index,
            result,
            elapsed_ms,
        });
        busy.store(false, Ordering::Relaxed);
        ctx.request_repaint();
    };

    #[cfg(target_arch = "wasm32")]
    {
        wasm_bindgen_futures::spawn_local(async move {
            // Let egui paint the busy banner before the blocking CPU inference starts.
            yield_to_paint().await;
            run();
        });
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::spawn(run);
    }
}

/// Yield to the browser event loop so the UI can paint the running state.
#[cfg(target_arch = "wasm32")]
async fn yield_to_paint() {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| match web_sys::window() {
        Some(window) => {
            let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 120);
        }
        None => {
            let _ = resolve.call0(&wasm_bindgen::JsValue::NULL);
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

#[cfg(target_arch = "wasm32")]
async fn load_model(spec: DemoModel) -> Result<Predictor, String> {
    let bytes = fetch_model_bytes(spec.artifact).await?;
    predictor_from_bytes(&bytes, spec)
}

#[cfg(target_arch = "wasm32")]
async fn fetch_model_bytes(model_url: &str) -> Result<Vec<u8>, String> {
    use wasm_bindgen::{JsCast, JsValue};
    use wasm_bindgen_futures::JsFuture;

    let window = web_sys::window().ok_or("no browser window")?;
    let response = JsFuture::from(window.fetch_with_str(model_url))
        .await
        .map_err(|error| format!("fetching {model_url} failed: {:?}", JsValue::from(error)))?;
    let response: web_sys::Response = response
        .dyn_into()
        .map_err(|_| "fetch did not return a Response")?;
    if !response.ok() {
        return Err(format!(
            "Could not download the model (HTTP {}). Use Retry loading to try again.",
            response.status()
        ));
    }
    let buffer = JsFuture::from(
        response
            .array_buffer()
            .map_err(|error| JsValue::from(error).as_string().unwrap_or_default())?,
    )
    .await
    .map_err(|error| format!("reading {model_url} body failed: {error:?}"))?;
    Ok(js_sys::Uint8Array::new(&buffer).to_vec())
}

#[cfg(not(target_arch = "wasm32"))]
fn load_model(spec: DemoModel) -> Result<Predictor, String> {
    let mut candidates: Vec<std::path::PathBuf> = vec![
        spec.artifact.into(),
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target/models")
            .join(spec.artifact),
    ];
    if let Ok(directory) = std::env::var("MONTGOMERY_WEB_DEMO_MODELS") {
        candidates.insert(0, std::path::Path::new(&directory).join(spec.artifact));
    }
    // Preserve the original detection-only override.
    if spec.id == ModelId::Yolo26N
        && let Ok(path) = std::env::var("MONTGOMERY_WEB_DEMO_MODEL")
    {
        candidates.insert(0, path.into());
    }
    for path in candidates {
        if let Ok(bytes) = std::fs::read(&path) {
            return predictor_from_bytes(&bytes, spec)
                .map_err(|error| format!("{path:?}: {error}"));
        }
    }
    Err(format!(
        "{} not found. Run the demo's prepare-models.py script (see examples/web/README.md), or set MONTGOMERY_WEB_DEMO_MODELS to your artifact directory.",
        spec.artifact
    ))
}

fn predictor_from_bytes(bytes: &[u8], spec: DemoModel) -> Result<Predictor, String> {
    // Retain low-confidence objects so the display slider can reveal them without rerunning.
    let model = Predictor::with_options_from_bytes(
        bytes,
        PredictOptions {
            confidence: 0.05,
            ..PredictOptions::default()
        },
    )
    .map_err(|error| error.to_string())?;
    if model.model_id() != spec.id {
        return Err(format!(
            "Expected {}, found {} in {}",
            spec.id,
            model.model_id(),
            spec.artifact
        ));
    }
    Ok(model)
}

/// Milliseconds since the epoch; `std::time::Instant` is unavailable on wasm32-unknown-unknown.
#[cfg(target_arch = "wasm32")]
fn now_ms() -> f64 {
    js_sys::Date::now()
}

#[cfg(not(target_arch = "wasm32"))]
fn now_ms() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs_f64() * 1e3)
        .unwrap_or_default()
}

/// Exercise the same bytes loader and task dispatch used by the browser.
#[cfg(test)]
pub(crate) fn smoke_from_bytes(bytes: &[u8], model_name: &str) -> Result<String, String> {
    use montgomery::Prediction;

    let spec = *MODELS
        .iter()
        .find(|spec| spec.id.as_str() == model_name)
        .ok_or("Unsupported demo model")?;
    let id = spec.id;
    let model = predictor_from_bytes(bytes, spec)?;
    let image = image::load_from_memory(IMAGES[0].encoded).map_err(|error| error.to_string())?;
    let prediction = model.inference(&image).map_err(|error| error.to_string())?;
    let source_size = (image.width(), image.height());
    let valid = match &prediction {
        Prediction::Detections(items) => ["dog", "bicycle", "person"].iter().all(|name| {
            items
                .iter()
                .any(|item| item.class_name == *name && item.confidence >= 0.25)
        }),
        Prediction::Segmentations(items) => {
            !items.is_empty()
                && items.iter().all(|item| {
                    (item.mask.width, item.mask.height) == source_size
                        && item.mask.data.len() == (source_size.0 * source_size.1) as usize
                        && item.mask.data.iter().any(|covered| *covered)
                })
                && ["dog", "bicycle", "person"]
                    .iter()
                    .all(|name| items.iter().any(|item| item.class_name == *name))
        }
        Prediction::Classifications(items) => {
            items.len() == 5
                && items.iter().all(|item| {
                    !item.class_name.is_empty()
                        && item.confidence.is_finite()
                        && (0.0..=1.0).contains(&item.confidence)
                })
                && items
                    .windows(2)
                    .all(|pair| pair[0].confidence >= pair[1].confidence)
        }
        Prediction::Semantics(mask) => {
            (mask.width, mask.height) == source_size
                && mask.data.len() == (source_size.0 * source_size.1) as usize
                && mask
                    .data
                    .iter()
                    .all(|id| (*id as usize) < model.class_names().len())
                && mask.data.iter().any(|id| *id != mask.data[0])
        }
        Prediction::Depth(map) => {
            (map.width, map.height) == source_size
                && map.data.len() == (source_size.0 * source_size.1) as usize
                && map
                    .data
                    .iter()
                    .all(|value| value.is_finite() && *value > 0.0)
                && map.stats().max > map.stats().min
        }
    };
    if !valid {
        return Err(format!(
            "{id}: invalid or unexpected prediction on the reference image"
        ));
    }
    let view = ResultView::new(prediction, model.class_names().to_vec());
    Ok(format!("{id}: {}", view.summary(0.25)))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires packed YOLO26n weights"]
    fn yolo26n_from_bytes_detects_reference_image() {
        let model = load_model(MODELS[0]).expect("yolo26n.bpk must be packed before running tests");
        let image = image::load_from_memory(IMAGES[0].encoded).expect("embedded image decodes");
        let detections = model.predict(&image);
        let classes: Vec<&str> = detections
            .iter()
            .map(|detection| detection.class_name.as_str())
            .collect();
        for expected in ["dog", "bicycle", "person"] {
            assert!(
                classes.contains(&expected),
                "expected {expected} in detections: {classes:?}"
            );
        }
    }

    #[test]
    #[ignore = "requires all five packed YOLO26 nano artifacts"]
    fn all_nano_tasks_from_bytes() {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                for spec in MODELS {
                    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("target/models")
                        .join(spec.artifact);
                    let bytes = std::fs::read(path).expect("pack all demo artifacts first");
                    println!("{}", smoke_from_bytes(&bytes, spec.id.as_str()).unwrap());
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
