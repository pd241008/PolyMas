const GRPC_GATEWAY = process.env.NEXT_PUBLIC_GRPC_GATEWAY || "http://localhost:50053";

export interface RunManifest {
  run_id: string;
  pipeline_version: string;
  status: "queued" | "running" | "completed" | "failed";
  input_checksums: Record<string, string>;
  output_checksums: Record<string, string>;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
}

export interface DiseasePrediction {
  label: string;
  probability: number;
  confidence_lower: number;
  confidence_upper: number;
}

export interface ScoredPatient {
  patient_id: string;
  predictions: DiseasePrediction[];
}

export interface ClusterAssignment {
  cluster_id: string;
  member_patient_ids: string[];
  distance_to_centroid: number;
}

export interface FeatureAttribution {
  feature_id: string;
  feature_name: string;
  attribution_value: number;
  method: string;
}

/**
 * API client for the Polymas control plane.
 *
 * In production, this would use a gRPC-web proxy or connect
 * directly via a Next.js API route proxying to gRPC.
 */
export const api = {
  async listRuns(): Promise<RunManifest[]> {
    // TODO: Replace with gRPC-web fetch
    const res = await fetch(`${GRPC_GATEWAY}/api/runs`);
    if (!res.ok) throw new Error("Failed to list runs");
    return res.json();
  },

  async getRunStatus(runId: string): Promise<RunManifest> {
    const res = await fetch(`${GRPC_GATEWAY}/api/runs/${runId}`);
    if (!res.ok) throw new Error(`Failed to get run status: ${runId}`);
    return res.json();
  },

  async startRun(locusIds: string[], modelVersion: string): Promise<RunManifest> {
    const res = await fetch(`${GRPC_GATEWAY}/api/runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ locus_ids: locusIds, model_version: modelVersion }),
    });
    if (!res.ok) throw new Error("Failed to start run");
    return res.json();
  },

  async explainPatient(patientId: string, method: string): Promise<FeatureAttribution[]> {
    const res = await fetch(`${GRPC_GATEWAY}/api/patients/${patientId}/explain?method=${method}`);
    if (!res.ok) throw new Error(`Failed to explain patient: ${patientId}`);
    return res.json();
  },
};
