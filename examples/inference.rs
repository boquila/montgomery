use montgomery::{Model, Prediction};

fn main() -> montgomery::Result<()> {
    let mut args = std::env::args_os().skip(1);
    let weights = args.next().ok_or("usage: inference <model.bpk> <image>")?;
    let image = args.next().ok_or("usage: inference <model.bpk> <image>")?;

    let model = Model::new(weights)?;
    let prediction = model.inference(std::path::PathBuf::from(image))?;

    match prediction {
        Prediction::Detections(items) => println!("{} detections", items.len()),
        Prediction::Segmentations(items) => println!("{} segmented instances", items.len()),
        Prediction::Semantics(mask) => println!(
            "semantic mask {}x{} ({} labeled pixels)",
            mask.width,
            mask.height,
            mask.data.len()
        ),
        Prediction::Depth(map) => {
            let stats = map.stats();
            println!(
                "depth map {}x{} (meters): min {:.2}, max {:.2}, mean {:.2}",
                map.width, map.height, stats.min, stats.max, stats.mean
            );
        }
        Prediction::Classifications(items) => {
            for item in items {
                println!("{}: {:.1}%", item.class_name, item.confidence * 100.0);
            }
        }
    }
    Ok(())
}
