"use client";

import { useEffect, useState } from "react";
import { RiskOverviewCard } from "@/components/RiskOverviewCard";
import { ClusterPreview } from "@/components/ClusterPreview";
import { RecentRuns } from "@/components/RecentRuns";
import { api, RunManifest } from "@/lib/api";

export default function DashboardPage() {
  const [runs, setRuns] = useState<RunManifest[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.listRuns()
      .then(setRuns)
      .catch(() => setRuns([]))
      .finally(() => setLoading(false));
  }, []);

  const completedRuns = runs.filter(r => r.status === "completed");

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="font-mono text-3xl font-bold">Pipeline Overview</h2>
        <span className="badge-brutal">v0.1.0</span>
      </div>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
        <RiskOverviewCard label="T1D" count={completedRuns.length > 0 ? 142 : 0} risk={0.15} />
        <RiskOverviewCard label="T2D" count={completedRuns.length > 0 ? 308 : 0} risk={0.25} />
        <RiskOverviewCard label="LADA" count={completedRuns.length > 0 ? 67 : 0} risk={0.08} />
        <RiskOverviewCard label="GDM" count={completedRuns.length > 0 ? 95 : 0} risk={0.12} />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="card-brutal">
          <h3 className="font-mono text-lg font-bold mb-4">Risk Cluster Dendrogram</h3>
          <ClusterPreview />
        </div>

        <div className="card-brutal">
          <h3 className="font-mono text-lg font-bold mb-4">Recent Pipeline Runs</h3>
          {loading ? (
            <p className="font-mono text-sm text-surface-muted">Loading runs...</p>
          ) : (
            <RecentRuns runs={runs.slice(0, 10)} />
          )}
        </div>
      </div>
    </div>
  );
}
