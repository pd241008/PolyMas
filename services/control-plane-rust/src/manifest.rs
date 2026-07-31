use std::collections::HashMap;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct RunManifest {
    pub run_id: String,
    pub pipeline_version: String,
    pub status: RunStatus,
    pub input_checksums: HashMap<String, String>,
    pub output_checksums: HashMap<String, String>,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum RunStatus {
    Queued,
    Running,
    Completed,
    Failed,
}
