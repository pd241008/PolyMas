use anyhow::Result;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use tracing::info;

mod orchestrator;
mod manifest;

use orchestrator::PipelineOrchestrator;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,polymas_control_plane=debug".into()),
        )
        .init();

    let grpc_port = std::env::var("GRPC_PORT").unwrap_or_else(|_| "50053".to_string());
    let addr: std::net::SocketAddr = format!("0.0.0.0:{}", grpc_port).parse()?;

    info!("Polymas Control Plane starting on {}", addr);

    let orchestrator = PipelineOrchestrator::new();

    // TODO: Build tonic gRPC server with ControlPlaneService
    // let svc = control_plane_server::ControlPlaneServer::new(orchestrator);
    // Server::builder()
    //     .add_service(svc)
    //     .serve(addr)
    //     .await?;

    // Placeholder: run orchestrator in demo mode
    let manifest = orchestrator.start_run(
        &["rs12345", "rs67890"],
        "0.1.0",
        &HashMap::from([("mode".to_string(), "full_pipeline".to_string())]),
    ).await?;

    info!("Pipeline run completed: {:?}", manifest.run_id);
    info!("Input checksums:  {:?}", manifest.input_checksums);
    info!("Output checksums: {:?}", manifest.output_checksums);

    Ok(())
}

/// Compute SHA-256 checksum of arbitrary bytes.
pub fn checksum(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}
