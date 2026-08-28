use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::{TRACE_SCHEMA_VERSION, ULTRALYTICS_SOURCE_COMMIT, sample::AugmentationError};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TraceValue {
    Bool(bool),
    Integer(i64),
    Float(f64),
    Text(String),
    Integers(Vec<i64>),
    Floats(Vec<f64>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceEvent {
    pub path: String,
    pub transform: String,
    pub applied: bool,
    pub before_instances: usize,
    pub after_instances: usize,
    pub params: BTreeMap<String, TraceValue>,
    pub partners: Vec<usize>,
    pub children: Vec<TraceEvent>,
}

impl TraceEvent {
    pub fn new(
        path: impl Into<String>,
        transform: impl Into<String>,
        applied: bool,
        before: usize,
    ) -> Self {
        Self {
            path: path.into(),
            transform: transform.into(),
            applied,
            before_instances: before,
            after_instances: before,
            params: BTreeMap::new(),
            partners: vec![],
            children: vec![],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AugmentationTrace {
    pub schema_version: u32,
    pub compatibility: String,
    pub source_commit: String,
    pub sample_id: String,
    pub events: Vec<TraceEvent>,
}

impl AugmentationTrace {
    pub fn new(sample_id: impl Into<String>) -> Self {
        Self {
            schema_version: TRACE_SCHEMA_VERSION,
            compatibility: "ultralytics-8.4.117".into(),
            source_commit: ULTRALYTICS_SOURCE_COMMIT.into(),
            sample_id: sample_id.into(),
            events: vec![],
        }
    }
    pub fn from_json(bytes: &[u8]) -> Result<Self, AugmentationError> {
        let value: Self = serde_json::from_slice(bytes)
            .map_err(|e| AugmentationError::new(format!("invalid augmentation trace: {e}")))?;
        if value.schema_version != TRACE_SCHEMA_VERSION {
            return Err(AugmentationError::new(format!(
                "unsupported augmentation trace schema {}",
                value.schema_version
            )));
        }
        Ok(value)
    }
    pub fn to_json(&self) -> Result<Vec<u8>, AugmentationError> {
        serde_json::to_vec_pretty(self).map_err(|e| AugmentationError::new(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn round_trip() {
        let mut t = AugmentationTrace::new("x");
        t.events.push(TraceEvent::new("0", "noop", false, 2));
        assert_eq!(
            AugmentationTrace::from_json(&t.to_json().unwrap()).unwrap(),
            t
        );
    }
}
