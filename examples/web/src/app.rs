//! Montgomery web demo: pick an image on the left, run YOLO26n detection, see the boxes.
//!
//! The same Rust code builds natively (inference on a worker thread) and for the browser
//! (WebAssembly, inference on the main thread until a worker port lands).

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

use eframe::egui;
use montgomery::{Detection, Predictor};

/// Artifact served next to the page (see `Trunk.toml` and the demo README).
const MODEL_URL: &str = "yolo26n.bpk";

/// One selectable demo image, embedded so the app also works fully offline.
struct DemoImage {
    name: &'static str,
    encoded: &'static [u8],
}

const IMAGES: &[DemoImage] = &[
    DemoImage {
        name: "dog_bike_man.jpg",
        encoded: include_bytes!("../assets/dog_bike_man.jpg"),
    },
    DemoImage {
        name: "bus.jpg",
        encoded: include_bytes!("../assets/bus.jpg"),
    },
    DemoImage {
        name: "zidane.jpg",
        encoded: include_bytes!("../assets/zidane.jpg"),
    },
];

/// Accent used for selection outlines and status text.
const ACCENT: egui::Color32 = egui::Color32::from_rgb(94, 203, 255);
/// Fill for the primary action button (dark enough for light label text).
const RUN_FILL: egui::Color32 = egui::Color32::from_rgb(27, 105, 148);
/// Success color for completed inference.
const SUCCESS: egui::Color32 = egui::Color32::from_rgb(102, 217, 138);
/// Warning color while a run is in flight.
const RUNNING: egui::Color32 = egui::Color32::from_rgb(240, 190, 80);
/// Which backend actually executes the graph.
#[cfg(target_arch = "wasm32")]
const BACKEND: &str = "Flex CPU · WebAssembly";
#[cfg(not(target_arch = "wasm32"))]
const BACKEND: &str = "Flex CPU · native";

/// A finished inference run for one selected image.
struct Outcome {
    image_index: usize,
    detections: Vec<Detection>,
    elapsed_ms: f64,
}

/// Model availability shared between the UI thread and the loading/inference task.
enum ModelSlot {
    Loading,
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
    model: Arc<Mutex<ModelSlot>>,
    busy: Arc<AtomicBool>,
    outcome: Arc<Mutex<Option<Outcome>>>,
    last_outcome: Option<Outcome>,
    threshold: f32,
}

impl DemoApp {
    pub fn new(ctx: &egui::Context) -> Self {
        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = egui::Color32::from_rgb(15, 17, 21);
        visuals.window_fill = visuals.panel_fill;
        visuals.selection.bg_fill = egui::Color32::from_rgb(30, 74, 96);
        visuals.selection.stroke = egui::Stroke::new(1.0, ACCENT);
        ctx.set_visuals(visuals);

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

        let model = Arc::new(Mutex::new(ModelSlot::Loading));
        spawn_model_load(ctx.clone(), model.clone());

        Self {
            images,
            selected: 0,
            model,
            busy: Arc::new(AtomicBool::new(false)),
            outcome: Arc::new(Mutex::new(None)),
            last_outcome: None,
            threshold: 0.25,
        }
    }

    /// Selecting a different image clears the previous result, so boxes never look cached.
    fn select_image(&mut self, index: usize) {
        if self.selected != index {
            self.selected = index;
            self.last_outcome = None;
        }
    }

    fn run_inference(&mut self, ctx: &egui::Context) {
        if self.busy.swap(true, Ordering::Relaxed) {
            return;
        }
        self.last_outcome = None;
        ctx.request_repaint();
        let image = self.images[self.selected].image.clone();
        spawn_inference(
            ctx.clone(),
            self.model.clone(),
            self.busy.clone(),
            self.outcome.clone(),
            image,
            self.selected,
        );
    }
}

impl eframe::App for DemoApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        if let Some(outcome) = self.outcome.lock().unwrap().take() {
            // Ignore a run that finished after the user switched images.
            if outcome.image_index == self.selected {
                self.last_outcome = Some(outcome);
            }
        }
        let busy = self.busy.load(Ordering::Relaxed);
        let (model_state, model_ready, status_color) = match &*self.model.lock().unwrap() {
            ModelSlot::Loading => ("loading model…".to_owned(), false, RUNNING),
            ModelSlot::Failed(error) => (
                format!("model error: {error}"),
                false,
                egui::Color32::from_rgb(240, 100, 100),
            ),
            ModelSlot::Ready(_) => ("YOLO26n ready".to_owned(), true, SUCCESS),
        };

        egui::Panel::left("images")
            .resizable(false)
            .default_size(200.0)
            .show(ui, |ui| {
                ui.add_space(10.0);
                ui.label(
                    egui::RichText::new("MONTGOMERY")
                        .size(11.0)
                        .strong()
                        .color(ACCENT),
                );
                ui.label(egui::RichText::new("YOLO26n demo").size(20.0).strong());
                ui.label(
                    egui::RichText::new("1. Pick a photo\n2. Press Run inference")
                        .size(12.5)
                        .color(egui::Color32::from_gray(205)),
                );
                ui.add_space(12.0);
                ui.separator();
                ui.add_space(8.0);
                let mut clicked_card = None;
                for (index, view) in self.images.iter().enumerate() {
                    let selected = self.selected == index;
                    let card = egui::Frame::NONE
                        .inner_margin(egui::Margin::same(5))
                        .corner_radius(6)
                        .fill(if selected {
                            egui::Color32::from_rgb(24, 45, 58)
                        } else {
                            egui::Color32::TRANSPARENT
                        })
                        .stroke(if selected {
                            egui::Stroke::new(2.0, ACCENT)
                        } else {
                            egui::Stroke::NONE
                        })
                        .show(ui, |ui| {
                            ui.set_min_width(ui.available_width());
                            ui.vertical_centered(|ui| {
                                ui.add(
                                    egui::Image::from_texture(&view.texture)
                                        .max_width(150.0)
                                        .max_height(96.0)
                                        .maintain_aspect_ratio(true),
                                );
                                ui.label(egui::RichText::new(IMAGES[index].name).small().color(
                                    if selected {
                                        egui::Color32::WHITE
                                    } else {
                                        egui::Color32::from_gray(190)
                                    },
                                ));
                            });
                        });
                    let response = ui.interact(
                        card.response.rect,
                        ui.id().with(("sample-image", index)),
                        egui::Sense::click(),
                    );
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
                    if response.clicked() {
                        clicked_card = Some(index);
                    }
                    ui.add_space(6.0);
                }
                if let Some(index) = clicked_card {
                    self.select_image(index);
                }
                ui.add_space(4.0);
                ui.separator();
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    status_dot(ui, status_color);
                    ui.label(
                        egui::RichText::new(model_state.as_str())
                            .size(12.0)
                            .color(status_color),
                    );
                });
                ui.label(
                    egui::RichText::new("Burnpack · ~5 MB · WebAssembly")
                        .small()
                        .weak(),
                );
            });

        egui::CentralPanel::default().show(ui, |ui| {
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("YOLO26n object detection")
                        .size(22.0)
                        .strong(),
                );
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new("runs entirely in your browser")
                        .size(12.5)
                        .color(egui::Color32::from_gray(190)),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new(model_state.as_str())
                            .size(12.5)
                            .color(status_color),
                    );
                    status_dot(ui, status_color);
                });
            });
            ui.add_space(6.0);
            let has_result = self
                .last_outcome
                .as_ref()
                .is_some_and(|outcome| outcome.image_index == self.selected);
            ui.horizontal(|ui| {
                let enabled = model_ready && !busy;
                let run = ui.add_enabled(
                    enabled,
                    egui::Button::new(
                        egui::RichText::new(if has_result {
                            "▶  Run again"
                        } else {
                            "▶  Run inference"
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
                if busy {
                    ui.spinner();
                    ui.label(egui::RichText::new("Running…").color(RUNNING).strong());
                }
                ui.add_space(8.0);
                ui.separator();
                ui.add(
                    egui::Slider::new(&mut self.threshold, 0.05..=0.95)
                        .text("confidence")
                        .fixed_decimals(2),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(egui::RichText::new(BACKEND).small().weak());
                });
            });
            ui.add_space(6.0);

            // Re-read the shared state: a click above just changed it.
            let busy = self.busy.load(Ordering::Relaxed);
            let current = self
                .last_outcome
                .as_ref()
                .filter(|outcome| outcome.image_index == self.selected);
            if busy {
                status_banner(
                    ui,
                    RUNNING,
                    "Running YOLO26n inference…",
                    "Flex CPU · WebAssembly — this can take a moment.",
                );
            } else if let Some(outcome) = current {
                let shown = outcome
                    .detections
                    .iter()
                    .filter(|detection| detection.confidence >= self.threshold)
                    .count();
                status_banner(
                    ui,
                    SUCCESS,
                    "Inference complete",
                    &format!(
                        "{} objects in {:.0} ms · {} · {} shown at ≥ {:.2}",
                        outcome.detections.len(),
                        outcome.elapsed_ms,
                        BACKEND,
                        shown,
                        self.threshold
                    ),
                );
            } else if model_ready {
                status_banner(
                    ui,
                    egui::Color32::from_gray(150),
                    "No inference yet",
                    "Press Run inference to detect objects in this photo.",
                );
            } else if let ModelSlot::Failed(_) = &*self.model.lock().unwrap() {
                status_banner(ui, status_color, "Model not available", &model_state);
            } else {
                status_banner(
                    ui,
                    RUNNING,
                    "Loading model…",
                    "Run inference unlocks when the Burnpack is ready.",
                );
            }
            ui.add_space(8.0);

            let view = &self.images[self.selected];
            let available = ui.available_size();
            let footer = 40.0;
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
                if let Some(outcome) = self
                    .last_outcome
                    .as_ref()
                    .filter(|outcome| outcome.image_index == self.selected)
                {
                    draw_detections(ui, &image_rect, view, outcome, self.threshold);
                    ui.painter().rect_stroke(
                        image_rect,
                        0.0,
                        egui::Stroke::new(2.0, SUCCESS),
                        egui::StrokeKind::Inside,
                    );
                } else if busy {
                    draw_status_overlay(
                        ui,
                        image_rect,
                        "Running YOLO26n inference…",
                        "Flex CPU · WebAssembly",
                        RUNNING,
                    );
                } else if model_ready {
                    draw_status_overlay(
                        ui,
                        image_rect,
                        "No inference yet",
                        "Press ▶ Run inference",
                        ACCENT,
                    );
                } else {
                    draw_status_overlay(
                        ui,
                        image_rect,
                        "Loading YOLO26n…",
                        "The Run button unlocks when ready.",
                        RUNNING,
                    );
                }
            });
            ui.add_space(6.0);
            if let Some(outcome) = self
                .last_outcome
                .as_ref()
                .filter(|outcome| outcome.image_index == self.selected)
            {
                let visible: Vec<&Detection> = outcome
                    .detections
                    .iter()
                    .filter(|detection| detection.confidence >= self.threshold)
                    .collect();
                if visible.is_empty() {
                    ui.label(
                        egui::RichText::new(if outcome.detections.is_empty() {
                            "No objects detected."
                        } else {
                            "No detections above the confidence threshold."
                        })
                        .small()
                        .weak(),
                    );
                } else {
                    ui.horizontal_wrapped(|ui| {
                        for detection in visible {
                            let color = class_color(detection.class_id);
                            status_dot(ui, color);
                            ui.label(
                                egui::RichText::new(format!(
                                    "{} {:.0}%",
                                    detection.class_name,
                                    detection.confidence * 100.0
                                ))
                                .size(12.5),
                            );
                            ui.add_space(8.0);
                        }
                    });
                }
            }
        });
    }
}

/// A small painted status dot. The default egui font lacks U+25CF, which rendered as tofu.
fn status_dot(ui: &mut egui::Ui, color: egui::Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 12.0), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), 3.5, color);
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

/// Dim the image and place a centered, high-contrast state message over it.
fn draw_status_overlay(
    ui: &egui::Ui,
    target: egui::Rect,
    title: &str,
    subtitle: &str,
    accent: egui::Color32,
) {
    let painter = ui.painter_at(target);
    painter.rect_filled(target, 0.0, egui::Color32::from_black_alpha(120));

    let title_font = egui::TextStyle::Heading.resolve(ui.style());
    let body_font = egui::TextStyle::Body.resolve(ui.style());
    let title_galley = painter.layout(
        title.to_owned(),
        title_font,
        egui::Color32::WHITE,
        target.width() - 40.0,
    );
    let sub_galley = painter.layout(
        subtitle.to_owned(),
        body_font,
        egui::Color32::from_gray(205),
        target.width() - 40.0,
    );
    let padding = egui::vec2(28.0, 20.0);
    let gap = 8.0;
    let size = egui::vec2(
        (title_galley.size().x.max(sub_galley.size().x) + padding.x * 2.0)
            .min(target.width() - 24.0),
        title_galley.size().y + gap + sub_galley.size().y + padding.y * 2.0,
    );
    let panel = egui::Rect::from_center_size(target.center(), size);
    painter.rect_filled(panel, 10.0, egui::Color32::from_black_alpha(225));
    painter.rect_stroke(
        panel,
        10.0,
        egui::Stroke::new(1.5, accent),
        egui::StrokeKind::Inside,
    );
    let title_pos = egui::pos2(
        panel.center().x - title_galley.size().x * 0.5,
        panel.top() + padding.y,
    );
    painter.galley(title_pos, title_galley.clone(), egui::Color32::WHITE);
    let sub_pos = egui::pos2(
        panel.center().x - sub_galley.size().x * 0.5,
        title_pos.y + title_galley.size().y + gap,
    );
    painter.galley(sub_pos, sub_galley, egui::Color32::from_gray(205));
}

/// Draw detection boxes and labels over the displayed image, in source-image coordinates.
fn draw_detections(
    ui: &mut egui::Ui,
    target: &egui::Rect,
    view: &ImageView,
    outcome: &Outcome,
    threshold: f32,
) {
    let painter = ui.painter_at(*target);
    let scale = egui::vec2(
        target.width() / view.width as f32,
        target.height() / view.height as f32,
    );
    let font = egui::TextStyle::Body.resolve(ui.style());
    let visible: Vec<_> = outcome
        .detections
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
fn class_color(class_id: usize) -> egui::Color32 {
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
    ];
    PALETTE[class_id % PALETTE.len()]
}

/// Kick off model loading off the current call site.
fn spawn_model_load(ctx: egui::Context, slot: Arc<Mutex<ModelSlot>>) {
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
            let result = load_model().await;
            fill(result, &slot);
            ctx.request_repaint();
        });
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::spawn(move || {
            let result = load_model();
            fill(result, &slot);
            ctx.request_repaint();
        });
    }
}

/// Run YOLO26n detection on the selected image and publish the outcome.
fn spawn_inference(
    ctx: egui::Context,
    slot: Arc<Mutex<ModelSlot>>,
    busy: Arc<AtomicBool>,
    outcome: Arc<Mutex<Option<Outcome>>>,
    image: image::DynamicImage,
    image_index: usize,
) {
    let run = move || {
        let started = now_ms();
        let detections = match &mut *slot.lock().unwrap() {
            ModelSlot::Ready(model) => model.predict(&image),
            _ => Vec::new(),
        };
        let elapsed_ms = now_ms() - started;
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(
            &format!(
                "montgomery: {} detections in {elapsed_ms:.0} ms",
                detections.len()
            )
            .into(),
        );
        *outcome.lock().unwrap() = Some(Outcome {
            image_index,
            detections,
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
async fn load_model() -> Result<Predictor, String> {
    let bytes = fetch_model_bytes().await?;
    Predictor::new_from_bytes(bytes.as_slice()).map_err(|error| error.to_string())
}

#[cfg(target_arch = "wasm32")]
async fn fetch_model_bytes() -> Result<Vec<u8>, String> {
    use wasm_bindgen::{JsCast, JsValue};
    use wasm_bindgen_futures::JsFuture;

    let window = web_sys::window().ok_or("no browser window")?;
    let response = JsFuture::from(window.fetch_with_str(MODEL_URL))
        .await
        .map_err(|error| format!("fetching {MODEL_URL} failed: {:?}", JsValue::from(error)))?;
    let response: web_sys::Response = response
        .dyn_into()
        .map_err(|_| "fetch did not return a Response")?;
    if !response.ok() {
        return Err(format!(
            "fetching {MODEL_URL} returned HTTP {} — pack yolo26n.bpk first (see the README)",
            response.status()
        ));
    }
    let buffer = JsFuture::from(
        response
            .array_buffer()
            .map_err(|error| JsValue::from(error).as_string().unwrap_or_default())?,
    )
    .await
    .map_err(|error| format!("reading {MODEL_URL} body failed: {error:?}"))?;
    Ok(js_sys::Uint8Array::new(&buffer).to_vec())
}

#[cfg(not(target_arch = "wasm32"))]
fn load_model() -> Result<Predictor, String> {
    let mut candidates: Vec<std::path::PathBuf> = vec![
        MODEL_URL.into(),
        "../../yolo26n.bpk".into(), // cargo run from examples/web
    ];
    if let Ok(path) = std::env::var("MONTGOMERY_WEB_DEMO_MODEL") {
        candidates.insert(0, path.into());
    }
    for path in candidates {
        if let Ok(bytes) = std::fs::read(&path) {
            return Predictor::new_from_bytes(bytes.as_slice())
                .map_err(|error| format!("{path:?}: {error}"));
        }
    }
    Err("yolo26n.bpk not found. From the repository root run \
         `montgomery pack-weights --architecture yolo26n --state target/yolo26n-state.pt`, \
         or point MONTGOMERY_WEB_DEMO_MODEL at an existing artifact."
        .to_owned())
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

/// The demo's headline claim, verified headlessly: bytes-loaded YOLO26n detects the reference
/// image's dog, bicycle, and person. Requires `yolo26n.bpk` at the repository root (see README).
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn yolo26n_from_bytes_detects_reference_image() {
        let model = load_model().expect("yolo26n.bpk must be packed before running tests");
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
}
