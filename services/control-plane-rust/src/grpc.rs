use crate::polymas::v1::control_plane_service_server::ControlPlaneService;
use crate::polymas::v1::{
    CancelRunRequest, CancelRunResponse, GetRunStatusRequest, HealthCheckRequest,
    HealthCheckResponse, ListRunsRequest, ListRunsResponse, RunManifest, StartRunRequest,
};
use crate::orchestrator::PipelineOrchestrator;
use tonic::{Request, Response, Status};

pub struct ControlPlaneServiceHandler {
    orchestrator: PipelineOrchestrator,
}

impl ControlPlaneServiceHandler {
    pub fn new(orchestrator: PipelineOrchestrator) -> Self {
        Self { orchestrator }
    }
}

#[tonic::async_trait]
impl ControlPlaneService for ControlPlaneServiceHandler {
    async fn start_run(
        &self,
        request: Request<StartRunRequest>,
    ) -> Result<Response<RunManifest>, Status> {
        let req = request.into_inner();
        match self
            .orchestrator
            .start_run(&req.locus_ids, &req.model_version, &req.parameters)
            .await
        {
            Ok(manifest) => Ok(Response::new(manifest)),
            Err(e) => Err(Status::internal(e.to_string())),
        }
    }

    async fn get_run_status(
        &self,
        request: Request<GetRunStatusRequest>,
    ) -> Result<Response<RunManifest>, Status> {
        let req = request.into_inner();
        match self.orchestrator.get_run_status(&req.run_id).await {
            Ok(Some(manifest)) => Ok(Response::new(manifest)),
            Ok(None) => Err(Status::not_found(format!("run not found: {}", req.run_id))),
            Err(e) => Err(Status::internal(e.to_string())),
        }
    }

    async fn list_runs(
        &self,
        request: Request<ListRunsRequest>,
    ) -> Result<Response<ListRunsResponse>, Status> {
        let req = request.into_inner();
        match self.orchestrator.list_runs(req.page_size).await {
            Ok(runs) => Ok(Response::new(ListRunsResponse {
                runs,
                next_page_token: String::new(),
            })),
            Err(e) => Err(Status::internal(e.to_string())),
        }
    }

    async fn cancel_run(
        &self,
        request: Request<CancelRunRequest>,
    ) -> Result<Response<CancelRunResponse>, Status> {
        let req = request.into_inner();
        match self.orchestrator.cancel_run(&req.run_id).await {
            Ok(cancelled) => Ok(Response::new(CancelRunResponse {
                cancelled,
                message: if cancelled {
                    "run cancelled".to_string()
                } else {
                    "run not found or not running".to_string()
                },
            })),
            Err(e) => Err(Status::internal(e.to_string())),
        }
    }

    async fn health_check(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        Ok(Response::new(HealthCheckResponse {
            healthy: true,
            service_name: "polymas-control-plane".to_string(),
            version: "0.1.0".to_string(),
        }))
    }
}
