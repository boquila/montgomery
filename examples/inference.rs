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
        Prediction::Classifications(items) => {
            for item in items {
                println!("{}: {:.1}%", item.class_name, item.confidence * 100.0);
            }
        }
    }
    Ok(())
}
