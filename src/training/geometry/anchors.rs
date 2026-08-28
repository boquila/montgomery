use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureLevelLayout {
    pub height: usize,
    pub width: usize,
    pub stride: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AnchorPoint {
    /// Anchor center in feature-grid coordinates.
    pub grid_xy: [f32; 2],
    pub stride: f32,
    pub level: usize,
    pub index: usize,
}

pub fn make_anchors(levels: &[FeatureLevelLayout]) -> Vec<AnchorPoint> {
    let mut anchors =
        Vec::with_capacity(levels.iter().map(|level| level.height * level.width).sum());
    for (level_index, level) in levels.iter().enumerate() {
        for y in 0..level.height {
            for x in 0..level.width {
                anchors.push(AnchorPoint {
                    grid_xy: [x as f32 + 0.5, y as f32 + 0.5],
                    stride: level.stride as f32,
                    level: level_index,
                    index: anchors.len(),
                });
            }
        }
    }
    anchors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supports_two_level_heads() {
        let anchors = make_anchors(&[
            FeatureLevelLayout {
                height: 2,
                width: 3,
                stride: 16,
            },
            FeatureLevelLayout {
                height: 1,
                width: 1,
                stride: 32,
            },
        ]);
        assert_eq!(anchors.len(), 7);
        assert_eq!(anchors[6].grid_xy, [0.5, 0.5]);
        assert_eq!(anchors[6].level, 1);
    }
}
