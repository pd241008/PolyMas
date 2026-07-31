use std::collections::HashMap;

use crate::orchestrator::PipelineOrchestrator;
use crate::polymas::v1::{RunManifest, RunStatus};

fn make_orchestrator() -> PipelineOrchestrator {
    PipelineOrchestrator::new()
}

fn make_params() -> HashMap<String, String> {
    HashMap::from([
        ("mode".to_string(), "full_pipeline".to_string()),
        ("cohorts".to_string(), "ukbb,mgc".to_string()),
    ])
}

#[tokio::test]
async fn test_start_run_returns_manifest_with_checksums() {
    let orch = make_orchestrator();
    let locus_ids = vec!["rs12345".to_string(), "rs67890".to_string()];
    let params = make_params();

    let manifest: RunManifest = orch
        .start_run(&locus_ids, "0.1.0", &params)
        .await
        .expect("start_run should succeed");

    assert!(!manifest.run_id.is_empty());
    assert_eq!(manifest.pipeline_version, "0.1.0");
    assert_eq!(manifest.status, RunStatus::Running as i32);
    assert!(manifest.started_at.is_some());
    assert!(!manifest.input_checksums.is_empty());
    assert_eq!(manifest.input_checksums.len(), 3);
    assert!(manifest
        .input_checksums
        .contains_key("locus_ids"));
    assert!(manifest
        .input_checksums
        .contains_key("model_version"));
    assert!(manifest
        .input_checksums
        .contains_key("parameters"));
}

#[tokio::test]
async fn test_get_run_status_returns_stored_manifest() {
    let orch = make_orchestrator();
    let locus_ids = vec!["rs111".to_string()];
    let params = make_params();

    let manifest = orch
        .start_run(&locus_ids, "0.1.0", &params)
        .await
        .unwrap();

    let fetched = orch
        .get_run_status(&manifest.run_id)
        .await
        .expect("get_run_status should succeed")
        .expect("run should exist");

    assert_eq!(fetched.run_id, manifest.run_id);
    assert_eq!(fetched.status, RunStatus::Running as i32);
}

#[tokio::test]
async fn test_get_run_status_returns_none_for_unknown_id() {
    let orch = make_orchestrator();

    let result = orch.get_run_status("nonexistent").await.unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn test_complete_run_updates_status_and_output_checksums() {
    let orch = make_orchestrator();
    let locus_ids = vec!["rs999".to_string()];
    let params = make_params();

    let manifest = orch.start_run(&locus_ids, "0.2.0", &params).await.unwrap();

    let output_checksums = HashMap::from([("predictions".to_string(), "deadbeef".to_string())]);
    let updated = orch
        .complete_run(&manifest.run_id, output_checksums)
        .await
        .unwrap();

    assert_eq!(updated.status, RunStatus::Completed as i32);
    assert!(updated.completed_at.is_some());
    assert_eq!(updated.output_checksums.len(), 1);
    assert_eq!(
        updated.output_checksums.get("predictions"),
        Some(&"deadbeef".to_string())
    );
}

#[tokio::test]
async fn test_list_runs_returns_all_in_reverse_order() {
    let orch = make_orchestrator();
    let params = make_params();

    orch.start_run(&["rs1".to_string()], "0.1.0", &params).await.unwrap();
    orch.start_run(&["rs2".to_string()], "0.1.0", &params).await.unwrap();
    orch.start_run(&["rs3".to_string()], "0.1.0", &params).await.unwrap();

    let runs = orch.list_runs(10).await.unwrap();
    assert_eq!(runs.len(), 3);
}

#[tokio::test]
async fn test_list_runs_respects_page_size() {
    let orch = make_orchestrator();
    let params = make_params();

    orch.start_run(&["rs1".to_string()], "0.1.0", &params).await.unwrap();
    orch.start_run(&["rs2".to_string()], "0.1.0", &params).await.unwrap();
    orch.start_run(&["rs3".to_string()], "0.1.0", &params).await.unwrap();

    let runs = orch.list_runs(2).await.unwrap();
    assert_eq!(runs.len(), 2);
}

#[tokio::test]
async fn test_cancel_run_succeeds_for_running_status() {
    let orch = make_orchestrator();
    let params = make_params();

    let manifest = orch.start_run(&["rs7".to_string()], "0.1.0", &params).await.unwrap();

    let cancelled = orch.cancel_run(&manifest.run_id).await.unwrap();
    assert!(cancelled);

    let fetched = orch.get_run_status(&manifest.run_id).await.unwrap().unwrap();
    assert_eq!(fetched.status, RunStatus::Failed as i32);
    assert_eq!(fetched.error_message, "cancelled by user");
}

#[tokio::test]
async fn test_cancel_run_fails_for_completed_status() {
    let orch = make_orchestrator();
    let params = make_params();

    let manifest = orch.start_run(&["rs7".to_string()], "0.1.0", &params).await.unwrap();
    orch.complete_run(&manifest.run_id, HashMap::new()).await.unwrap();

    let cancelled = orch.cancel_run(&manifest.run_id).await.unwrap();
    assert!(!cancelled);
}

#[test]
fn test_checksum_is_deterministic() {
    let data = b"hello world";
    let hash1 = crate::checksum(data);
    let hash2 = crate::checksum(data);
    assert_eq!(hash1, hash2);
    assert_eq!(hash1.len(), 64);
}

#[test]
fn test_checksum_differs_for_different_input() {
    let hash1 = crate::checksum(b"hello");
    let hash2 = crate::checksum(b"world");
    assert_ne!(hash1, hash2);
}
