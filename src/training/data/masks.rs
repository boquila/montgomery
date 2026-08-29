use crate::training::data::{SegmentationSource, manifest::DatasetError};

/// Decode a segmentation source into a row-major binary mask.
pub fn rasterize(
    source: &SegmentationSource,
    width: usize,
    height: usize,
) -> Result<Vec<u8>, DatasetError> {
    match source {
        SegmentationSource::Polygons(rings) => rasterize_polygons(rings, width, height),
        SegmentationSource::UncompressedRle { size, counts } => {
            validate_size(*size, width, height)?;
            decode_counts(counts, width, height)
        }
        SegmentationSource::CompressedRle { size, counts } => {
            validate_size(*size, width, height)?;
            decode_counts(&decode_compressed_counts(counts)?, width, height)
        }
    }
}

fn validate_size(size: [u32; 2], width: usize, height: usize) -> Result<(), DatasetError> {
    // COCO stores [height, width].
    if size != [height as u32, width as u32] {
        return Err(DatasetError::new(format!(
            "RLE size {:?} does not match image [{height}, {width}]",
            size
        )));
    }
    Ok(())
}

fn rasterize_polygons(
    rings: &[Vec<[f32; 2]>],
    width: usize,
    height: usize,
) -> Result<Vec<u8>, DatasetError> {
    if rings
        .iter()
        .any(|ring| ring.len() < 3 || ring.iter().flatten().any(|v| !v.is_finite()))
    {
        return Err(DatasetError::new(
            "polygon rings need at least three finite points",
        ));
    }
    let mut mask = vec![0_u8; width * height];
    for y in 0..height {
        for x in 0..width {
            let point = [x as f32 + 0.5, y as f32 + 0.5];
            let inside = rings
                .iter()
                .fold(false, |inside, ring| inside ^ point_in_ring(point, ring));
            mask[y * width + x] = u8::from(inside);
        }
    }
    Ok(mask)
}

fn point_in_ring([x, y]: [f32; 2], ring: &[[f32; 2]]) -> bool {
    let mut inside = false;
    let mut previous = ring[ring.len() - 1];
    for &current in ring {
        if (current[1] > y) != (previous[1] > y) {
            let crossing = (previous[0] - current[0]) * (y - current[1])
                / (previous[1] - current[1])
                + current[0];
            inside ^= x < crossing;
        }
        previous = current;
    }
    inside
}

fn decode_counts(counts: &[u32], width: usize, height: usize) -> Result<Vec<u8>, DatasetError> {
    let pixels = width
        .checked_mul(height)
        .ok_or_else(|| DatasetError::new("mask size overflow"))?;
    if counts.iter().map(|v| *v as usize).sum::<usize>() != pixels {
        return Err(DatasetError::new(
            "RLE counts do not cover the declared mask size",
        ));
    }
    // COCO RLE is column major. Convert explicitly to the row-major training contract.
    let mut output = vec![0_u8; pixels];
    let mut cursor = 0_usize;
    let mut value = 0_u8;
    for &run in counts {
        for column_major in cursor..cursor + run as usize {
            let x = column_major / height;
            let y = column_major % height;
            output[y * width + x] = value;
        }
        cursor += run as usize;
        value ^= 1;
    }
    Ok(output)
}

/// Decode pycocotools' variable-length signed delta representation.
fn decode_compressed_counts(encoded: &str) -> Result<Vec<u32>, DatasetError> {
    let bytes = encoded.as_bytes();
    let mut counts = Vec::<i64>::new();
    let mut cursor = 0;
    while cursor < bytes.len() {
        let mut value = 0_i64;
        let mut shift = 0;
        let mut more = true;
        while more {
            let byte = *bytes
                .get(cursor)
                .ok_or_else(|| DatasetError::new("truncated compressed RLE"))?;
            if !(48..=111).contains(&byte) {
                return Err(DatasetError::new("invalid compressed RLE character"));
            }
            let code = i64::from(byte - 48);
            value |= (code & 0x1f) << (5 * shift);
            more = code & 0x20 != 0;
            cursor += 1;
            shift += 1;
            if !more && code & 0x10 != 0 {
                value |= -1_i64 << (5 * shift);
            }
        }
        if counts.len() > 2 {
            value += counts[counts.len() - 2];
        }
        if value < 0 || value > u32::MAX as i64 {
            return Err(DatasetError::new("compressed RLE decoded an invalid run"));
        }
        counts.push(value);
    }
    Ok(counts.into_iter().map(|v| v as u32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uncompressed_rle_is_converted_from_column_major() {
        let mask = rasterize(
            &SegmentationSource::UncompressedRle {
                size: [2, 2],
                counts: vec![1, 2, 1],
            },
            2,
            2,
        )
        .unwrap();
        assert_eq!(mask, [0, 1, 1, 0]);
    }

    #[test]
    fn polygon_uses_pixel_centers_and_supports_holes() {
        let outer = vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
        let hole = vec![[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]];
        let mask = rasterize(&SegmentationSource::Polygons(vec![outer, hole]), 4, 4).unwrap();
        assert_eq!(mask.iter().filter(|v| **v != 0).count(), 12);
    }
}
