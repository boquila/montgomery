use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoxXyxy {
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoxXywh {
    pub cx: f32,
    pub cy: f32,
    pub width: f32,
    pub height: f32,
}

impl BoxXyxy {
    pub fn new(edges: [f32; 4]) -> Result<Self, &'static str> {
        let result = Self {
            xmin: edges[0],
            ymin: edges[1],
            xmax: edges[2],
            ymax: edges[3],
        };
        if !edges.into_iter().all(f32::is_finite) {
            return Err("box edges must be finite");
        }
        if result.xmax <= result.xmin || result.ymax <= result.ymin {
            return Err("box must have positive width and height");
        }
        Ok(result)
    }

    pub fn width(self) -> f32 {
        (self.xmax - self.xmin).max(0.0)
    }

    pub fn height(self) -> f32 {
        (self.ymax - self.ymin).max(0.0)
    }

    pub fn area(self) -> f32 {
        self.width() * self.height()
    }

    pub fn center(self) -> [f32; 2] {
        [(self.xmin + self.xmax) * 0.5, (self.ymin + self.ymax) * 0.5]
    }

    pub fn clip(self, width: f32, height: f32) -> Option<Self> {
        Self::new([
            self.xmin.clamp(0.0, width),
            self.ymin.clamp(0.0, height),
            self.xmax.clamp(0.0, width),
            self.ymax.clamp(0.0, height),
        ])
        .ok()
    }

    pub fn to_xywh(self) -> BoxXywh {
        BoxXywh {
            cx: (self.xmin + self.xmax) * 0.5,
            cy: (self.ymin + self.ymax) * 0.5,
            width: self.width(),
            height: self.height(),
        }
    }
}

impl BoxXywh {
    pub fn to_xyxy(self) -> BoxXyxy {
        let half_width = self.width * 0.5;
        let half_height = self.height * 0.5;
        BoxXyxy {
            xmin: self.cx - half_width,
            ymin: self.cy - half_height,
            xmax: self.cx + half_width,
            ymax: self.cy + half_height,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_edges_equal_to_canvas_are_valid() {
        let value = BoxXyxy::new([0.0, 0.0, 640.0, 480.0]).unwrap();
        assert_eq!(value.clip(640.0, 480.0), Some(value));
        assert_eq!(value.to_xywh().to_xyxy(), value);
    }
}
