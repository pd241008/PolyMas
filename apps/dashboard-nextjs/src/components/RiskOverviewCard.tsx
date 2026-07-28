interface RiskOverviewCardProps {
  label: string;
  count: number;
  risk: number;
}

export function RiskOverviewCard({ label, count, risk }: RiskOverviewCardProps) {
  return (
    <div className="card-brutal">
      <div className="flex items-center justify-between mb-2">
        <span className="font-mono text-sm font-bold uppercase">{label}</span>
        <span className="badge-brutal">
          {(risk * 100).toFixed(0)}%
        </span>
      </div>
      <p className="font-mono text-3xl font-bold">{count}</p>
      <p className="text-surface-muted text-sm mt-1">patients at risk</p>
      <div className="mt-3 h-2 w-full bg-gray-200 border border-border">
        <div
          className="h-full bg-surface-accent"
          style={{ width: `${risk * 100}%` }}
        />
      </div>
    </div>
  );
}
