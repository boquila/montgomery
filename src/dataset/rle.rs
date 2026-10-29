use super::manifest::DatasetError;
use serde_json::Value;

/// Validate COCO counts without allocating a decoded mask.
pub(super) fn counts(seg: &Value, size: [u32; 2]) -> Result<Vec<u64>, DatasetError> {
    let counts = if let Some(values) = seg["counts"].as_array() {
        values
            .iter()
            .map(|v| {
                v.as_u64()
                    .ok_or_else(|| DatasetError::new("RLE counts must be non-negative integers"))
            })
            .collect::<Result<Vec<_>, _>>()?
    } else if let Some(encoded) = seg["counts"].as_str() {
        let mut counts = Vec::<u64>::new();
        let mut bytes = encoded.bytes();
        while let Some(first) = bytes.next() {
            let mut byte = first;
            let mut value = 0i64;
            let mut shift = 0;
            loop {
                if !(48..=111).contains(&byte) || shift >= 60 {
                    return Err(DatasetError::new("invalid compressed RLE"));
                }
                let code = (byte - 48) as i64;
                value |= (code & 31) << shift;
                shift += 5;
                if code & 32 == 0 {
                    if code & 16 != 0 {
                        value |= -1i64 << shift;
                    }
                    break;
                }
                byte = bytes
                    .next()
                    .ok_or_else(|| DatasetError::new("truncated compressed RLE"))?;
            }
            if counts.len() > 2 {
                value = value
                    .checked_add(counts[counts.len() - 2] as i64)
                    .ok_or_else(|| DatasetError::new("RLE run overflow"))?;
            }
            if value < 0 || value > u32::MAX as i64 {
                return Err(DatasetError::new("invalid compressed RLE run"));
            }
            counts.push(value as u64);
        }
        counts
    } else {
        return Err(DatasetError::new("invalid RLE counts"));
    };
    if counts.iter().try_fold(0u64, |a, b| a.checked_add(*b))
        != Some(size[0] as u64 * size[1] as u64)
    {
        return Err(DatasetError::new(
            "RLE counts do not cover the declared mask size",
        ));
    }
    Ok(counts)
}
