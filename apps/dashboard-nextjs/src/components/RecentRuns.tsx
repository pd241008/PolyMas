const MOCK_RUNS = [
  { id: "run_001", status: "completed" as const, models: 3, patients: 1247, time: "4m 32s" },
  { id: "run_002", status: "completed" as const, models: 3, patients: 986, time: "3m 18s" },
  { id: "run_003", status: "running" as const, models: 3, patients: 1502, time: "..." },
  { id: "run_004", status: "failed" as const, models: 2, patients: 0, time: "0m 12s" },
];

const statusStyles = {
  completed: "bg-green-200 text-green-900 border-green-600",
  running: "bg-blue-200 text-blue-900 border-blue-600",
  failed: "bg-red-200 text-red-900 border-red-600",
  queued: "bg-gray-200 text-gray-900 border-gray-600",
};

export function RecentRuns() {
  return (
    <div className="space-y-3">
      {MOCK_RUNS.map((run) => (
        <div
          key={run.id}
          className="flex items-center justify-between border-2 border-border p-3 shadow-brutal-sm"
        >
          <div className="flex items-center gap-3">
            <span className={`inline-block px-2 py-0.5 text-xs font-bold border ${statusStyles[run.status]}`}>
              {run.status.toUpperCase()}
            </span>
            <span className="font-mono text-sm font-bold">{run.id}</span>
          </div>
          <div className="flex items-center gap-4 text-sm text-surface-muted">
            <span>{run.patients} patients</span>
            <span>{run.time}</span>
          </div>
        </div>
      ))}
    </div>
  );
}
