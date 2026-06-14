import type { LucideIcon } from "lucide-react";

interface MetricCardProps {
  icon: LucideIcon;
  label: string;
  value: string;
  sublabel?: string;
  accent?: string;
  trend?: { value: string; positive: boolean };
}

export function MetricCard({ icon: Icon, label, value, sublabel, accent, trend }: MetricCardProps) {
  const color = accent ?? "var(--color-fedavg)";

  return (
    <div className="stat-card">
      <div className="flex items-center justify-between">
        <span
          className="text-xs font-semibold uppercase tracking-wider"
          style={{ color: "var(--color-text-muted)" }}
        >
          {label}
        </span>
        <div
          className="flex h-8 w-8 items-center justify-center rounded-lg"
          style={{ background: `color-mix(in srgb, ${color} 16%, transparent)`, color }}
        >
          <Icon className="h-4 w-4" strokeWidth={2.25} />
        </div>
      </div>
      <div className="flex items-baseline gap-2">
        <span className="text-2xl font-bold tabular-nums" style={{ color: "var(--color-text-primary)" }}>
          {value}
        </span>
        {trend && (
          <span
            className="text-xs font-semibold"
            style={{ color: trend.positive ? "var(--color-positive)" : "var(--color-negative)" }}
          >
            {trend.value}
          </span>
        )}
      </div>
      {sublabel && (
        <span className="text-xs" style={{ color: "var(--color-text-secondary)" }}>
          {sublabel}
        </span>
      )}
    </div>
  );
}
