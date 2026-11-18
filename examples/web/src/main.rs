#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    eframe::run_native(
        "Montgomery — YOLO26n web demo",
        eframe::NativeOptions {
            viewport: eframe::egui::ViewportBuilder::default()
                .with_inner_size([1280.0, 800.0])
                .with_min_inner_size([640.0, 480.0]),
            ..Default::default()
        },
        Box::new(|cc| Ok(Box::new(app::DemoApp::new(&cc.egui_ctx)))),
    )
}

#[cfg(target_arch = "wasm32")]
fn main() {
    use eframe::wasm_bindgen::JsCast;
    // A wasm bin is started on instantiation. Headless runtimes (the Node smoke check) have
    // no `window`, so skip the browser bootstrap instead of panicking.
    let Some(window) = web_sys::window() else {
        return;
    };
    let Some(document) = window.document() else {
        return;
    };
    wasm_bindgen_futures::spawn_local(async move {
        let canvas = document
            .get_element_by_id("montgomery_canvas")
            .unwrap()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();
        let result = eframe::WebRunner::new()
            .start(
                canvas,
                eframe::WebOptions::default(),
                Box::new(|cc| Ok(Box::new(app::DemoApp::new(&cc.egui_ctx)))),
            )
            .await;
        if let Some(loading) = document.get_element_by_id("loading") {
            match result {
                Ok(()) => loading.remove(),
                Err(error) => {
                    loading.set_text_content(Some(
                        "The Montgomery demo could not start. Please enable WebGL or try \
                         another browser.",
                    ));
                    web_sys::console::error_1(&error);
                }
            }
        }
    });
}

/// Headless runtime check for the wasm build, invoked by `scripts/smoke.cjs`.
///
/// The JS side reads the Burnpack bytes and passes them in; this runs the full
/// bytes-to-detections path inside the wasm module and returns a summary string. Browsers
/// never call it.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn run_smoke(artifact: &[u8]) -> String {
    let model = match montgomery::Predictor::new_from_bytes(artifact) {
        Ok(model) => model,
        Err(error) => return format!("load failed: {error}"),
    };
    let image = image::load_from_memory(include_bytes!("../assets/dog_bike_man.jpg"))
        .expect("embedded reference image decodes");
    let detections = model.predict(&image);
    let summary: Vec<String> = detections
        .iter()
        .map(|detection| {
            format!(
                "{}:{:.0}%",
                detection.class_name,
                detection.confidence * 100.0
            )
        })
        .collect();
    format!("{} detections: {}", detections.len(), summary.join(", "))
}
