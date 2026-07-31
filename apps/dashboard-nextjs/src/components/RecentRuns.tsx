"use client";

interface RecentRun {
  run_id: string;
  status: string;
  started_at?: string;
}

const statusStyles: Record<string, string> = {
  completed: "bg-green-200 text-green-900 border-green-600",
  running: "bg-blue-200 text-blue-900 border-blue-600",
  failed: "bg-red-200 text-red-900 border-red-600",
  queued: "bg-gray-200 text-gray-900 border-gray-600",
};

interface RecentRunsProps {
  runs: RecentRun[];
}

export function RecentRuns({ runs }: RecentRunsProps) {
  if (!runs.length) {
    return (
      <p className="font-mono text-sm text-surface-muted">No runs yet. Start a pipeline to see results here.</p>
    );
  }

  return (
    <div className="space-y-3">
      {runs.map((run) => (
        <div
          key={run.run_id}
          className="flex items-center justify-between border-2 border-border p-3 shadow-brutal-sm"
        >
          <div className="flex items-center gap-3">
            <span className={`inline-block px-2 py-0.5 text-xs font-bold border ${statusStyles[run.status] || statusStyles.queued}`}>
              {run.status.toUpperCase()}
            </span>
            <span className="font-mono text-sm font-bold">{run.run_id}</span>
          </div>
          <div className="flex items-center gap-4 text-sm text-surface-muted">
            <span>{run.started_at ? new Date(run.started_at).toLocaleString() : "—"}</span>
          </div>
        </div>
      ))}
    </div>
  );
}
