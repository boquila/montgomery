use serde::{Deserialize, Serialize};

use super::sample::AugmentationError;

pub type Polygon = Vec<[f32; 2]>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BoxFormat {
    Xyxy,
    Xywh,
    Ltwh,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BBox(pub [f32; 4]);

impl BBox {
    pub fn xyxy(self, format: BoxFormat) -> [f32; 4] {
        let [a, b, c, d] = self.0;
        match format {
            BoxFormat::Xyxy => self.0,
            BoxFormat::Xywh => [a - c / 2.0, b - d / 2.0, a + c / 2.0, b + d / 2.0],
            BoxFormat::Ltwh => [a, b, a + c, b + d],
        }
    }
    pub fn from_xyxy(value: [f32; 4], format: BoxFormat) -> Self {
        let [x1, y1, x2, y2] = value;
        Self(match format {
            BoxFormat::Xyxy => value,
            BoxFormat::Xywh => [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1],
            BoxFormat::Ltwh => [x1, y1, x2 - x1, y2 - y1],
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Instances {
    pub(super) boxes: Vec<BBox>,
    pub(super) format: BoxFormat,
    pub(super) normalized: bool,
    pub(super) segments: Option<Vec<Polygon>>,
}

impl Instances {
    pub fn new(
        boxes: Vec<BBox>,
        format: BoxFormat,
        normalized: bool,
        segments: Option<Vec<Polygon>>,
    ) -> Result<Self, AugmentationError> {
        let result = Self {
            boxes,
            format,
            normalized,
            segments,
        };
        result.validate()?;
        Ok(result)
    }
    pub fn empty() -> Self {
        Self {
            boxes: vec![],
            format: BoxFormat::Xyxy,
            normalized: false,
            segments: None,
        }
    }
    pub fn len(&self) -> usize {
        self.boxes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.boxes.is_empty()
    }
    pub fn boxes(&self) -> &[BBox] {
        &self.boxes
    }
    pub fn segments(&self) -> Option<&[Polygon]> {
        self.segments.as_deref()
    }
    pub fn normalized(&self) -> bool {
        self.normalized
    }
    pub fn format(&self) -> BoxFormat {
        self.format
    }
    pub fn validate(&self) -> Result<(), AugmentationError> {
        if self
            .segments
            .as_ref()
            .is_some_and(|s| s.len() != self.boxes.len())
        {
            return Err(AugmentationError::new("segment and box counts disagree"));
        }
        if self.boxes.iter().flat_map(|b| b.0).any(|v| !v.is_finite())
            || self
                .segments
                .iter()
                .flatten()
                .flatten()
                .flatten()
                .any(|v| !v.is_finite())
        {
            return Err(AugmentationError::new(
                "instance coordinates must be finite",
            ));
        }
        Ok(())
    }
    pub fn convert(&mut self, format: BoxFormat) {
        if self.format == format {
            return;
        }
        self.boxes = self
            .boxes
            .iter()
            .map(|b| BBox::from_xyxy(b.xyxy(self.format), format))
            .collect();
        self.format = format;
    }
    pub fn denormalize(&mut self, width: f32, height: f32) {
        if !self.normalized {
            return;
        }
        self.convert(BoxFormat::Xyxy);
        for b in &mut self.boxes {
            b.0[0] *= width;
            b.0[2] *= width;
            b.0[1] *= height;
            b.0[3] *= height;
        }
        if let Some(segments) = &mut self.segments {
            for p in segments.iter_mut().flatten() {
                p[0] *= width;
                p[1] *= height;
            }
        }
        self.normalized = false;
    }
    pub fn normalize(&mut self, width: f32, height: f32) {
        if self.normalized {
            return;
        }
        self.convert(BoxFormat::Xyxy);
        for b in &mut self.boxes {
            b.0[0] /= width;
            b.0[2] /= width;
            b.0[1] /= height;
            b.0[3] /= height;
        }
        if let Some(segments) = &mut self.segments {
            for p in segments.iter_mut().flatten() {
                p[0] /= width;
                p[1] /= height;
            }
        }
        self.normalized = true;
    }
    pub fn scale(&mut self, sx: f32, sy: f32) {
        self.convert(BoxFormat::Xyxy);
        for b in &mut self.boxes {
            b.0[0] *= sx;
            b.0[2] *= sx;
            b.0[1] *= sy;
            b.0[3] *= sy;
        }
        if let Some(segments) = &mut self.segments {
            for p in segments.iter_mut().flatten() {
                p[0] *= sx;
                p[1] *= sy;
            }
        }
    }
    pub fn pad(&mut self, x: f32, y: f32) -> Result<(), AugmentationError> {
        if self.normalized {
            return Err(AugmentationError::new(
                "cannot add absolute padding to normalized instances",
            ));
        }
        self.convert(BoxFormat::Xyxy);
        for b in &mut self.boxes {
            b.0[0] += x;
            b.0[2] += x;
            b.0[1] += y;
            b.0[3] += y;
        }
        if let Some(segments) = &mut self.segments {
            for p in segments.iter_mut().flatten() {
                p[0] += x;
                p[1] += y;
            }
        }
        Ok(())
    }
    pub fn clip(&mut self, width: f32, height: f32) {
        self.convert(BoxFormat::Xyxy);
        for b in &mut self.boxes {
            b.0[0] = b.0[0].clamp(0.0, width);
            b.0[2] = b.0[2].clamp(0.0, width);
            b.0[1] = b.0[1].clamp(0.0, height);
            b.0[3] = b.0[3].clamp(0.0, height);
        }
        if let Some(segments) = &mut self.segments {
            for p in segments.iter_mut().flatten() {
                p[0] = p[0].clamp(0.0, width);
                p[1] = p[1].clamp(0.0, height);
            }
        }
    }
    pub fn flip_horizontal(&mut self, width: f32) {
        self.convert(BoxFormat::Xywh);
        for b in &mut self.boxes {
            b.0[0] = width - b.0[0];
        }
        if let Some(segments) = &mut self.segments {
            for p in segments.iter_mut().flatten() {
                p[0] = width - p[0];
            }
        }
    }
    pub fn flip_vertical(&mut self, height: f32) {
        self.convert(BoxFormat::Xywh);
        for b in &mut self.boxes {
            b.0[1] = height - b.0[1];
        }
        if let Some(segments) = &mut self.segments {
            for p in segments.iter_mut().flatten() {
                p[1] = height - p[1];
            }
        }
    }
    pub fn select(&mut self, indexes: &[usize]) {
        self.boxes = indexes.iter().map(|&i| self.boxes[i]).collect();
        if let Some(segments) = &mut self.segments {
            *segments = indexes.iter().map(|&i| segments[i].clone()).collect();
        }
    }
    pub fn remove_zero_area(&mut self) -> Vec<usize> {
        let keep: Vec<_> = self
            .boxes
            .iter()
            .enumerate()
            .filter_map(|(i, b)| {
                let [x1, y1, x2, y2] = b.xyxy(self.format);
                (x2 > x1 && y2 > y1).then_some(i)
            })
            .collect();
        self.select(&keep);
        keep
    }
    pub fn concatenate(items: &[Self]) -> Result<Self, AugmentationError> {
        if items.is_empty() {
            return Ok(Self::empty());
        }
        let with_segments = items[0].segments.is_some();
        if items.iter().any(|i| i.segments.is_some() != with_segments) {
            return Err(AugmentationError::new(
                "cannot concatenate mixed polygon presence",
            ));
        }
        let mut boxes = Vec::new();
        let mut segments = with_segments.then(Vec::new);
        for item in items {
            let mut item = item.clone();
            item.convert(BoxFormat::Xyxy);
            boxes.extend(item.boxes);
            if let (Some(out), Some(value)) = (&mut segments, item.segments) {
                out.extend(value);
            }
        }
        Self::new(boxes, BoxFormat::Xyxy, false, segments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn round_trip_and_edge_clip() {
        let mut i = Instances::new(
            vec![BBox([0.5, 0.5, 1.0, 1.0])],
            BoxFormat::Xywh,
            true,
            None,
        )
        .unwrap();
        i.denormalize(100.0, 50.0);
        i.clip(100.0, 50.0);
        i.normalize(100.0, 50.0);
        i.convert(BoxFormat::Xywh);
        assert_eq!(i.boxes()[0].0, [0.5, 0.5, 1.0, 1.0]);
    }
}
