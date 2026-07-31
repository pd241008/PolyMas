use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    routing::get,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;

use crate::orchestrator::PipelineOrchestrator;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifestJson {
    pub run_id: String,
    pub pipeline_version: String,
    pub status: String,
    pub input_checksums: std::collections::HashMap<String, String>,
    pub output_checksums: std::collections::HashMap<String, String>,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
    pub error_message: String,
}

#[derive(Debug, Deserialize)]
pub struct StartRunRequestJson {
    pub locus_ids: Vec<String>,
    pub model_version: String,
}

#[derive(Debug, Deserialize)]
pub struct ListRunsQueryJson {
    pub page_size: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct ListRunsResponseJson {
    pub runs: Vec<RunManifestJson>,
}

impl From<crate::polymas::v1::RunManifest> for RunManifestJson {
    fn from(m: crate::polymas::v1::RunManifest) -> Self {
        let status = match crate::polymas::v1::RunStatus::try_from(m.status).unwrap_or(crate::polymas::v1::RunStatus::Unspecified) {
            crate::polymas::v1::RunStatus::Queued => "queued",
            crate::polymas::v1::RunStatus::Running => "running",
            crate::polymas::v1::RunStatus::Completed => "completed",
            crate::polymas::v1::RunStatus::Failed => "failed",
            _ => "unknown",
        };
        Self {
            run_id: m.run_id,
            pipeline_version: m.pipeline_version,
            status: status.to_string(),
            input_checksums: m.input_checksums,
            output_checksums: m.output_checksums,
            started_at: m.started_at.map(|ts| format!("{}.{:09}Z", ts.seconds, ts.nanos)),
            completed_at: m.completed_at.map(|ts| format!("{}.{:09}Z", ts.seconds, ts.nanos)),
            error_message: m.error_message,
        }
    }
}

pub fn build_rest_router(orchestrator: Arc<PipelineOrchestrator>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
        .allow_headers([axum::http::header::CONTENT_TYPE]);

    Router::new()
        .route("/api/runs", get(list_runs_handler).post(start_run_handler))
        .route("/api/runs/{run_id}", get(get_run_handler))
        .layer(cors)
        .with_state(orchestrator)
}

async fn list_runs_handler(
    State(orchestrator): State<Arc<PipelineOrchestrator>>,
    Query(params): Query<ListRunsQueryJson>,
) -> Result<Json<ListRunsResponseJson>, StatusCode> {
    let page_size = params.page_size.unwrap_or(20).max(1) as usize;
    let manifests = orchestrator
        .list_runs(page_size as u32)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let runs = manifests.into_iter().map(RunManifestJson::from).collect();
    Ok(Json(ListRunsResponseJson { runs }))
}

async fn start_run_handler(
    State(orchestrator): State<Arc<PipelineOrchestrator>>,
    Json(req): Json<StartRunRequestJson>,
) -> Result<Json<RunManifestJson>, StatusCode> {
    let params = std::collections::HashMap::new();
    let manifest = orchestrator
        .start_run(&req.locus_ids, &req.model_version, &params)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(RunManifestJson::from(manifest)))
}

async fn get_run_handler(
    State(orchestrator): State<Arc<PipelineOrchestrator>>,
    Path(run_id): Path<String>,
) -> Result<Json<RunManifestJson>, StatusCode> {
    let manifest = orchestrator
        .get_run_status(&run_id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    match manifest {
        Some(m) => Ok(Json(RunManifestJson::from(m))),
        None => Err(StatusCode::NOT_FOUND),
    }
}
