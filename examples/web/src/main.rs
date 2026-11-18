#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod results;

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
    // A wasm bin is started on instantiation; skip bootstrap outside a browser window.
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
