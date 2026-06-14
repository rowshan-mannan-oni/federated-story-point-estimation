import { Fragment, useMemo, useState } from "react";
import { Boxes, ChevronDown, BarChart3 } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useRun } from "../context/RunContext";
import { PageHeader } from "../components/PageHeader";
import { Panel } from "../components/Panel";
import { ConditionBadge } from "../components/ConditionBadge";
import { ChartTooltip } from "../components/charts/ChartTooltip";
import { CONDITION_META } from "../lib/conditions";
import { availableConditions, getProjectIds } from "../lib/metrics";
import { formatInt, formatNumber, formatPercent, formatProjectName } from "../lib/format";

export function ProjectBreakdown() {
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

  const [selected, setSelected] = useState<string | undefined>(undefined);
  const activeProject = selected ?? projectIds[0];

  if (!runData) return null;

  const chartData = conditions.map((key) => {
    const m = activeProject ? runData.perProject[key]?.[activeProject] : undefined;
    return {
      name: CONDITION_META[key].shortLabel,
      "Macro-F1": m?.macro_f1 ?? 0,
      Accuracy: m?.accuracy ?? 0,
      "Cohen's Kappa": m?.cohen_kappa ?? 0,
    };
  });

  return (
    <div className="flex flex-col gap-5 pb-8">
      <PageHeader
        icon={Boxes}
        title="Per-Project Breakdown"
        description="Compare per-client metrics across training conditions."
        actions={
          projectIds.length > 0 && (
            <div className="relative">
              <select
                value={activeProject}
                onChange={(e) => setSelected(e.target.value)}
                className="appearance-none rounded-lg border pl-3 pr-9 py-2 text-sm font-medium outline-none"
                style={{
                  background: "var(--color-surface-2)",
                  borderColor: "var(--color-border)",
                  color: "var(--color-text-primary)",
                }}
              >
                {projectIds.map((id) => (
                  <option key={id} value={id}>
                    {formatProjectName(id)}
                  </option>
                ))}
              </select>
              <ChevronDown
                className="pointer-events-none absolute right-2.5 top-1/2 h-4 w-4 -translate-y-1/2"
                style={{ color: "var(--color-text-muted)" }}
              />
            </div>
          )
        }
      />

      {projectIds.length === 0 ? (
        <div className="px-4 sm:px-6">
          <Panel title="No per-project data" icon={Boxes}>
            <p className="text-sm" style={{ color: "var(--color-text-secondary)" }}>
              This run does not include per-project result files.
            </p>
          </Panel>
        </div>
      ) : (
        <>
          <div className="px-4 sm:px-6">
            <Panel title={`${formatProjectName(activeProject)} — condition comparison`} icon={BarChart3}>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} barCategoryGap="28%">
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-soft)" vertical={false} />
                    <XAxis
                      dataKey="name"
                      tick={{ fill: "var(--color-text-secondary)", fontSize: 12 }}
                      axisLine={{ stroke: "var(--color-border)" }}
                      tickLine={false}
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
                    <Bar dataKey="Macro-F1" fill="var(--color-fedprox)" radius={[6, 6, 0, 0]} />
                    <Bar dataKey="Accuracy" fill="var(--color-fedavg)" radius={[6, 6, 0, 0]} />
                    <Bar dataKey="Cohen's Kappa" fill="var(--color-centralized)" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Panel>
          </div>

          <div className="px-4 sm:px-6">
            <Panel title="All projects" icon={Boxes} bodyClassName="p-0" className="overflow-hidden">
              <div className="max-h-[28rem] overflow-auto">
                <table className="table-base">
                  <thead>
                    <tr>
                      <th>Project</th>
                      <th className="text-right">N test</th>
                      {conditions.map((key) => (
                        <th key={key} colSpan={3} className="text-center">
                          <ConditionBadge condition={key} short />
                        </th>
                      ))}
                    </tr>
                    <tr>
                      <th></th>
                      <th></th>
                      {conditions.map((key) => (
                        <Fragment key={key}>
                          <th className="text-right">F1</th>
                          <th className="text-right">Acc</th>
                          <th className="text-right">MAE</th>
                        </Fragment>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {projectIds.map((id) => {
                      const nTest = conditions.map((key) => runData.perProject[key]?.[id]?.n_test).find((n) => n !== undefined);
                      return (
                        <tr
                          key={id}
                          className="cursor-pointer"
                          onClick={() => setSelected(id)}
                          style={id === activeProject ? { background: "var(--color-surface-2)" } : undefined}
                        >
                          <td className="font-medium" style={{ color: "var(--color-text-primary)" }}>
                            {formatProjectName(id)}
                          </td>
                          <td className="text-right font-mono" style={{ color: "var(--color-text-secondary)" }}>
                            {formatInt(nTest)}
                          </td>
                          {conditions.map((key) => {
                            const m = runData.perProject[key]?.[id];
                            return (
                              <Fragment key={key}>
                                <td className="text-right font-mono tabular-nums">{formatPercent(m?.macro_f1)}</td>
                                <td className="text-right font-mono tabular-nums">{formatPercent(m?.accuracy)}</td>
                                <td className="text-right font-mono tabular-nums">{formatNumber(m?.mae, 2)}</td>
                              </Fragment>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Panel>
          </div>
        </>
      )}
    </div>
  );
}
