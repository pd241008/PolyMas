use anyhow::Result;
use sha2::{Digest, Sha256};
use tracing::info;

pub mod polymas {
    pub mod v1 {
        tonic::include_proto!("polymas.v1");
    }
}

mod grpc;
mod orchestrator;

#[cfg(test)]
mod tests;

use grpc::ControlPlaneServiceHandler;
use orchestrator::PipelineOrchestrator;

pub fn checksum(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

fn now_timestamp() -> Option<prost_types::Timestamp> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?;
    Some(prost_types::Timestamp {
        seconds: now.as_secs() as i64,
        nanos: now.subsec_nanos() as i32,
    })
}

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
    let handler = ControlPlaneServiceHandler::new(orchestrator);

    tonic::transport::Server::builder()
        .add_service(polymas::v1::control_plane_service_server::ControlPlaneServiceServer::new(handler))
        .serve(addr)
        .await?;

    Ok(())
}
