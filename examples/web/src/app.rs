//! Montgomery web demo: pick an image on the left, run YOLO26n detection, see the boxes.
//!
//! The same Rust code builds natively (inference on a worker thread) and for the browser
//! (WebAssembly, inference on the main thread until a worker port lands).

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
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
}

impl DemoApp {
    pub fn new(ctx: &egui::Context) -> Self {
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
        }
    }

    fn run_inference(&mut self, ctx: &egui::Context) {
        if self.busy.swap(true, Ordering::Relaxed) {
            return;
        }
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
            self.last_outcome = Some(outcome);
        }
        let busy = self.busy.load(Ordering::Relaxed);
        let (model_state, model_ready) = match &*self.model.lock().unwrap() {
            ModelSlot::Loading => ("loading yolo26n.bpk…".to_owned(), false),
            ModelSlot::Failed(error) => (format!("model error: {error}"), false),
            ModelSlot::Ready(_) => ("yolo26n ready".to_owned(), true),
        };

        egui::Panel::left("images")
            .resizable(false)
            .default_size(160.0)
            .show(ui, |ui| {
                ui.add_space(6.0);
                ui.heading("Images");
                ui.add_space(6.0);
                for (index, view) in self.images.iter().enumerate() {
                    let selected = self.selected == index;
                    ui.horizontal(|ui| {
                        if ui
                            .add(
                                egui::Image::from_texture(&view.texture)
                                    .max_width(110.0)
                                    .max_height(72.0)
                                    .maintain_aspect_ratio(true),
                            )
                            .clicked()
                        {
                            self.selected = index;
                        }
                        if ui.selectable_label(selected, IMAGES[index].name).clicked() {
                            self.selected = index;
                        }
                    });
                    ui.add_space(6.0);
                }
            });

        egui::CentralPanel::default().show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Montgomery — YOLO26n detection").strong());
                ui.separator();
                ui.label(&model_state);
                if model_ready && !busy {
                    if ui.button("Run inference").clicked() {
                        self.run_inference(&ctx);
                    }
                } else if busy {
                    ui.label("running inference…");
                }
                if let Some(outcome) = self.last_outcome.as_ref() {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(format!(
                            "{} detections in {:.0} ms",
                            outcome.detections.len(),
                            outcome.elapsed_ms
                        ));
                    });
                }
            });
            ui.add_space(4.0);

            let view = &self.images[self.selected];
            let available = ui.available_size();
            let scale = (available.x / view.width as f32).min(available.y / view.height as f32);
            let display = egui::vec2(view.width as f32 * scale, view.height as f32 * scale);
            let response = ui.add(
                egui::Image::from_texture(&view.texture).fit_to_exact_size(display),
            );

            let show_outcome = self
                .last_outcome
                .as_ref()
                .is_some_and(|outcome| outcome.image_index == self.selected);
            if show_outcome {
                let outcome = self.last_outcome.as_ref().unwrap();
                draw_detections(ui, &response.rect, view, outcome);
            }
        });
    }
}

/// Draw detection boxes and labels over the displayed image, in source-image coordinates.
fn draw_detections(ui: &mut egui::Ui, target: &egui::Rect, view: &ImageView, outcome: &Outcome) {
    let painter = ui.painter_at(*target);
    let scale = egui::vec2(
        target.width() / view.width as f32,
        target.height() / view.height as f32,
    );
    let font = egui::TextStyle::Body.resolve(ui.style());
    for detection in &outcome.detections {
        let color = class_color(detection.class_id);
        let rect = egui::Rect::from_min_max(
            target.left_top() + egui::vec2(detection.xmin * scale.x, detection.ymin * scale.y),
            target.left_top() + egui::vec2(detection.xmax * scale.x, detection.ymax * scale.y),
        );
        painter.rect_stroke(
            rect,
            0.0,
            egui::Stroke::new(2.5, color),
            egui::StrokeKind::Inside,
        );

        let text = format!("{} {:.0}%", detection.class_name, detection.confidence * 100.0);
        let galley = painter.layout(text, font.clone(), color, f32::INFINITY);
        let label_rect = egui::Rect::from_min_size(
            rect.left_top() - egui::vec2(0.0, galley.size().y + 4.0),
            galley.size() + egui::vec2(6.0, 2.0),
        );
        let label_rect = if label_rect.top() < target.top() {
            egui::Rect::from_min_size(
                rect.left_top() + egui::vec2(0.0, 2.0),
                galley.size() + egui::vec2(6.0, 2.0),
            )
        } else {
            label_rect
        };
        painter.rect_filled(label_rect, 0.0, egui::Color32::from_black_alpha(160));
        painter.galley(
            label_rect.left_top() + egui::vec2(3.0, 1.0),
            galley,
            color,
        );
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
            Ok(model) => ModelSlot::Ready(Box::new(model)),
            Err(error) => ModelSlot::Failed(error),
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
            run();
        });
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::spawn(run);
    }
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
    Err(format!(
        "yolo26n.bpk not found. From the repository root run \
         `montgomery pack-weights --architecture yolo26n --state target/yolo26n-state.pt`, \
         or point MONTGOMERY_WEB_DEMO_MODEL at an existing artifact."
    ))
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
