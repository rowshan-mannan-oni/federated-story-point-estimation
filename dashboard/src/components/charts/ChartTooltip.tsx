interface TooltipPayloadEntry {
  name?: string;
  value?: number | string;
  color?: string;
  dataKey?: string | number;
}

interface ChartTooltipProps {
  active?: boolean;
  label?: string | number;
  payload?: TooltipPayloadEntry[];
  formatter?: (value: number | string, name: string) => string;
  labelFormatter?: (label: string | number) => string;
}

/** Shared Recharts tooltip styled to match the dark theme panels. */
export function ChartTooltip({ active, label, payload, formatter, labelFormatter }: ChartTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;

  return (
    <div
      className="rounded-lg border px-3 py-2.5 text-xs shadow-xl"
      style={{
        background: "var(--color-surface-3)",
        borderColor: "var(--color-border)",
        color: "var(--color-text-primary)",
      }}
    >
      {label !== undefined && (
        <p className="mb-1.5 font-semibold" style={{ color: "var(--color-text-secondary)" }}>
          {labelFormatter ? labelFormatter(label) : label}
        </p>
      )}
      <div className="flex flex-col gap-1">
        {payload.map((entry, idx) => (
          <div key={`${entry.dataKey ?? entry.name ?? idx}`} className="flex items-center justify-between gap-4">
            <span className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full" style={{ background: entry.color }} />
              <span style={{ color: "var(--color-text-secondary)" }}>{entry.name}</span>
            </span>
            <span className="font-mono font-medium tabular-nums">
              {entry.value !== undefined
                ? formatter
                  ? formatter(entry.value, entry.name ?? "")
                  : entry.value
                : "—"}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
