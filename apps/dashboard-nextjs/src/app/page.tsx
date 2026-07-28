import { RiskOverviewCard } from "@/components/RiskOverviewCard";
import { ClusterPreview } from "@/components/ClusterPreview";
import { RecentRuns } from "@/components/RecentRuns";

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="font-mono text-3xl font-bold">Pipeline Overview</h2>
        <span className="badge-brutal">v0.1.0</span>
      </div>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
        <RiskOverviewCard label="T1D" count={142} risk={0.15} />
        <RiskOverviewCard label="T2D" count={308} risk={0.25} />
        <RiskOverviewCard label="LADA" count={67} risk={0.08} />
        <RiskOverviewCard label="GDM" count={95} risk={0.12} />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="card-brutal">
          <h3 className="font-mono text-lg font-bold mb-4">Risk Cluster Dendrogram</h3>
          <ClusterPreview />
        </div>

        <div className="card-brutal">
          <h3 className="font-mono text-lg font-bold mb-4">Recent Pipeline Runs</h3>
          <RecentRuns />
        </div>
      </div>
    </div>
  );
}
