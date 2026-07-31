use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::RwLock;
use tracing::info;
use uuid::Uuid;

use crate::polymas::v1::{RunManifest, RunStatus};

pub(crate) type RunStore = Arc<RwLock<HashMap<String, RunManifest>>>;

#[derive(Clone)]
pub struct PipelineOrchestrator {
    store: RunStore,
}

impl PipelineOrchestrator {
    pub fn new() -> Self {
        Self {
            store: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn start_run(
        &self,
        locus_ids: &[String],
        model_version: &str,
        parameters: &HashMap<String, String>,
    ) -> Result<RunManifest> {
        let run_id = Uuid::new_v4().to_string();
        info!("Starting pipeline run {} for {} loci", run_id, locus_ids.len());

        let mut input_checksums = HashMap::new();
        input_checksums.insert(
            "locus_ids".to_string(),
            crate::checksum(locus_ids.join(",").as_bytes()),
        );
        input_checksums.insert(
            "model_version".to_string(),
            crate::checksum(model_version.as_bytes()),
        );
        input_checksums.insert(
            "parameters".to_string(),
            crate::checksum(
                parameters
                    .iter()
                    .map(|(k, v)| format!("{}={}", k, v))
                    .collect::<Vec<_>>()
                    .join(",")
                    .as_bytes(),
            ),
        );

        let manifest = RunManifest {
            run_id: run_id.clone(),
            pipeline_version: "0.1.0".to_string(),
            started_at: crate::now_timestamp(),
            completed_at: None,
            input_checksums,
            output_checksums: HashMap::new(),
            status: RunStatus::Running as i32,
            error_message: String::new(),
        };

        self.store.write().await.insert(run_id.clone(), manifest.clone());
        info!("Pipeline run {} started", run_id);
        Ok(manifest)
    }

    #[allow(dead_code)]
    pub async fn complete_run(
        &self,
        run_id: &str,
        output_checksums: HashMap<String, String>,
    ) -> Result<RunManifest> {
        let mut store = self.store.write().await;
        match store.get(run_id) {
            Some(existing) => {
                let mut manifest = existing.clone();
                manifest.status = RunStatus::Completed as i32;
                manifest.completed_at = crate::now_timestamp();
                manifest.output_checksums = output_checksums;
                store.insert(run_id.to_string(), manifest.clone());
                Ok(manifest)
            }
            None => anyhow::bail!("run not found: {}", run_id),
        }
    }

    pub async fn get_run_status(&self, run_id: &str) -> Result<Option<RunManifest>> {
        let store = self.store.read().await;
        Ok(store.get(run_id).cloned())
    }

    pub async fn list_runs(&self, page_size: u32) -> Result<Vec<RunManifest>> {
        let store = self.store.read().await;
        let mut all: Vec<RunManifest> = store.values().cloned().collect();
        all.sort_by(|a, b| {
            let a_secs = a.started_at.as_ref().map(|t| t.seconds).unwrap_or(0);
            let b_secs = b.started_at.as_ref().map(|t| t.seconds).unwrap_or(0);
            b_secs.cmp(&a_secs)
        });
        let end = page_size.min(all.len() as u32) as usize;
        Ok(all.into_iter().take(end).collect())
    }

    pub async fn cancel_run(&self, run_id: &str) -> Result<bool> {
        let mut store = self.store.write().await;
        if let Some(existing) = store.get(run_id) {
            let mut manifest = existing.clone();
            if manifest.status == RunStatus::Running as i32 {
                manifest.status = RunStatus::Failed as i32;
                manifest.error_message = "cancelled by user".to_string();
                store.insert(run_id.to_string(), manifest);
                return Ok(true);
            }
        }
        Ok(false)
    }
}
