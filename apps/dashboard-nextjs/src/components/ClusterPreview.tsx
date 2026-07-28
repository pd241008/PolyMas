"use client";

export function ClusterPreview() {
  return (
    <div className="flex items-center justify-center h-64 border-2 border-dashed border-border rounded bg-gray-50">
      <p className="font-mono text-sm text-surface-muted">
        Dendrogram visualization will render here via d3-hierarchy
      </p>
    </div>
  );
}
