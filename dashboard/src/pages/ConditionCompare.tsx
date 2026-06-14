import { useMemo } from "react";
import { GitCompareArrows, Scale, Ruler } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useRun } from "../context/RunContext";
import { PageHeader } from "../components/PageHeader";
import { Panel } from "../components/Panel";
import { ChartTooltip } from "../components/charts/ChartTooltip";
import { CONDITION_META } from "../lib/conditions";
import { availableConditions, getProjectIds } from "../lib/metrics";
import { formatNumber, formatPercent, formatProjectName } from "../lib/format";

export function ConditionCompare() {
  const { runData } = useRun();
  const conditions = useMemo(() => (runData ? availableConditions(runData) : []), [runData]);

  const projectIds = useMemo(() => {
    if (!runData) return [];
    for (const key of conditions) {
      const ids = getProjectIds(runData.perProject[key]);
      if (ids.length) return ids;
    }
    return [];
  }, [runData, conditions]);

  if (!runData) return null;

  const f1Data = projectIds.map((id) => {
    const row: Record<string, string | number> = { name: formatProjectName(id) };
    conditions.forEach((key) => {
      row[CONDITION_META[key].shortLabel] = runData.perProject[key]?.[id]?.macro_f1 ?? 0;
    });
    return row;
  });

  const maeData = projectIds.map((id) => {
    const row: Record<string, string | number> = { name: formatProjectName(id) };
    conditions.forEach((key) => {
      row[CONDITION_META[key].shortLabel] = runData.perProject[key]?.[id]?.mae ?? 0;
    });
    return row;
  });

  const chartWidth = Math.max(projectIds.length * 90, 640);

  return (
    <div className="flex flex-col gap-5 pb-8">
      <PageHeader
        icon={GitCompareArrows}
        title="Condition Comparison"
        description="Side-by-side metrics for every client across all available training conditions."
      />

      {conditions.length === 0 || projectIds.length === 0 ? (
        <div className="px-4 sm:px-6">
          <Panel title="No data" icon={GitCompareArrows}>
            <p className="text-sm" style={{ color: "var(--color-text-secondary)" }}>
              No per-project results were found for this run.
            </p>
          </Panel>
        </div>
      ) : (
        <>
          <div className="px-4 sm:px-6">
            <Panel title="Macro-F1 by project" icon={Scale}>
              <div className="h-80 overflow-x-auto">
                <div style={{ width: chartWidth, height: "100%" }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={f1Data} barCategoryGap="24%">
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-soft)" vertical={false} />
                      <XAxis
                        dataKey="name"
                        tick={{ fill: "var(--color-text-secondary)", fontSize: 11 }}
                        axisLine={{ stroke: "var(--color-border)" }}
                        tickLine={false}
                        interval={0}
                        angle={-35}
                        textAnchor="end"
                        height={70}
                      />
                      <YAxis
                        domain={[0, 1]}
                        tickFormatter={(v) => `${Math.round(v * 100)}%`}
                        tick={{ fill: "var(--color-text-secondary)", fontSize: 12 }}
                        axisLine={false}
                        tickLine={false}
                        width={48}
                      />
                      <Tooltip
                        content={<ChartTooltip formatter={(v) => formatPercent(v as number)} />}
                        cursor={{ fill: "var(--color-surface-2)" }}
                      />
                      <Legend wrapperStyle={{ fontSize: 12, color: "var(--color-text-secondary)" }} />
                      {conditions.map((key) => (
                        <Bar
                          key={key}
                          dataKey={CONDITION_META[key].shortLabel}
                          fill={CONDITION_META[key].color}
                          radius={[4, 4, 0, 0]}
                        />
                      ))}
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </Panel>
          </div>

          <div className="px-4 sm:px-6">
            <Panel title="MAE by project (lower is better)" icon={Ruler}>
              <div className="h-80 overflow-x-auto">
                <div style={{ width: chartWidth, height: "100%" }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={maeData} barCategoryGap="24%">
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-soft)" vertical={false} />
                      <XAxis
                        dataKey="name"
                        tick={{ fill: "var(--color-text-secondary)", fontSize: 11 }}
                        axisLine={{ stroke: "var(--color-border)" }}
                        tickLine={false}
                        interval={0}
                        angle={-35}
                        textAnchor="end"
                        height={70}
                      />
                      <YAxis
                        tick={{ fill: "var(--color-text-secondary)", fontSize: 12 }}
                        tickFormatter={(v) => formatNumber(v, 1)}
                        axisLine={false}
                        tickLine={false}
                        width={48}
                      />
                      <Tooltip
                        content={<ChartTooltip formatter={(v) => formatNumber(v as number, 3)} />}
                        cursor={{ fill: "var(--color-surface-2)" }}
                      />
                      <Legend wrapperStyle={{ fontSize: 12, color: "var(--color-text-secondary)" }} />
                      {conditions.map((key) => (
                        <Bar
                          key={key}
                          dataKey={CONDITION_META[key].shortLabel}
                          fill={CONDITION_META[key].color}
                          radius={[4, 4, 0, 0]}
                        />
                      ))}
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </Panel>
          </div>
        </>
      )}
    </div>
  );
}
