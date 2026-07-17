import { useMemo, useState } from "react";
import { Grid3x3, ChevronDown, BarChart3 } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useRun } from "../context/RunContext";
import { PageHeader } from "../components/PageHeader";
import { Panel } from "../components/Panel";
import { ChartTooltip } from "../components/charts/ChartTooltip";
import { CONDITION_META } from "../lib/conditions";
import { availableConditions, getAggregateMetrics, getProjectIds } from "../lib/metrics";
import { formatInt, formatPercent, formatProjectName } from "../lib/format";
import { STORY_POINT_CLASSES } from "../types/results";

/** Synthetic id for the pooled/aggregate matrix summed across all clients. */
const AGGREGATE_ID = "__aggregate__";

export function ConfusionMatrixPage() {
  const { runData } = useRun();
  const conditions = useMemo(() => (runData ? availableConditions(runData) : []), [runData]);

  const [condition, setCondition] = useState<string | undefined>(undefined);
  const activeCondition = condition ?? conditions[0];

  const results = activeCondition
    ? runData?.perProject[activeCondition as keyof typeof runData.perProject]
    : undefined;

  const projectIds = useMemo(() => {
    if (!results) return [];
    // Shared-head conditions expose a real pooled "global"; personalized ones get a
    // summed-across-clients aggregate instead. Either leads the project list.
    const lead = results.global !== undefined ? "global" : AGGREGATE_ID;
    return [lead, ...getProjectIds(results)];
  }, [results]);

  const [project, setProject] = useState<string | undefined>(undefined);
  const activeProject = project && projectIds.includes(project) ? project : projectIds[0];

  if (!runData) return null;

  const metrics =
    activeProject === AGGREGATE_ID
      ? getAggregateMetrics(results)
      : activeProject
        ? results?.[activeProject]
        : undefined;
  const matrix = metrics?.confusion_matrix;

  const maxValue = matrix ? Math.max(...matrix.flat()) : 0;

  const f1Data = metrics
    ? STORY_POINT_CLASSES.map((cls, idx) => ({
        name: `${cls} SP`,
        "F1 score": metrics.per_class_f1[idx] ?? 0,
      }))
    : [];

  return (
    <div className="flex flex-col gap-5 pb-8">
      <PageHeader
        icon={Grid3x3}
        title="Confusion Matrix"
        description="Per-class prediction errors for the selected condition and client."
        actions={
          conditions.length > 0 && (
            <div className="flex flex-wrap items-center gap-2">
              <SelectField
                value={activeCondition}
                onChange={setCondition}
                options={conditions.map((key) => ({ value: key, label: CONDITION_META[key].label }))}
              />
              <SelectField
                value={activeProject}
                onChange={setProject}
                options={projectIds.map((id) => ({ value: id, label: formatProjectName(id) }))}
              />
            </div>
          )
        }
      />

      {!matrix ? (
        <div className="px-4 sm:px-6">
          <Panel title="No data" icon={Grid3x3}>
            <p className="text-sm" style={{ color: "var(--color-text-secondary)" }}>
              No confusion matrix available for this selection.
            </p>
          </Panel>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 px-4 sm:px-6">
          <Panel title={`${formatProjectName(activeProject)} — confusion matrix`} icon={Grid3x3}>
            <div className="flex flex-col gap-2">
              <div className="flex items-center gap-2 pl-16 text-xs font-medium" style={{ color: "var(--color-text-muted)" }}>
                <span className="w-16 shrink-0" />
                <span className="flex-1 text-center uppercase tracking-wider">Predicted</span>
              </div>
              <div className="flex">
                <div className="flex w-16 shrink-0 items-center justify-center">
                  <span
                    className="text-xs font-medium uppercase tracking-wider"
                    style={{ color: "var(--color-text-muted)", writingMode: "vertical-rl", transform: "rotate(180deg)" }}
                  >
                    Actual
                  </span>
                </div>
                <div className="flex-1">
                  <div
                    className="grid gap-1"
                    style={{ gridTemplateColumns: `repeat(${STORY_POINT_CLASSES.length}, minmax(0, 1fr))` }}
                  >
                    {STORY_POINT_CLASSES.map((cls) => (
                      <div
                        key={`col-${cls}`}
                        className="pb-1 text-center text-xs font-semibold"
                        style={{ color: "var(--color-text-secondary)" }}
                      >
                        {cls}
                      </div>
                    ))}
                  </div>
                  <div className="flex flex-col gap-1">
                    {matrix.map((row, rowIdx) => (
                      <div
                        key={`row-${rowIdx}`}
                        className="grid gap-1"
                        style={{ gridTemplateColumns: `repeat(${STORY_POINT_CLASSES.length}, minmax(0, 1fr))` }}
                      >
                        {row.map((value, colIdx) => {
                          const intensity = maxValue > 0 ? value / maxValue : 0;
                          const isDiagonal = rowIdx === colIdx;
                          return (
                            <div
                              key={`cell-${rowIdx}-${colIdx}`}
                              className="flex aspect-square items-center justify-center rounded-md text-sm font-semibold tabular-nums transition-transform hover:scale-[1.04]"
                              style={{
                                background: isDiagonal
                                  ? `color-mix(in srgb, var(--color-fedprox) ${10 + intensity * 70}%, var(--color-surface-2))`
                                  : `color-mix(in srgb, var(--color-negative) ${intensity * 55}%, var(--color-surface-2))`,
                                color: intensity > 0.4 ? "#0a0e17" : "var(--color-text-secondary)",
                              }}
                              title={`Actual ${STORY_POINT_CLASSES[rowIdx]} -> Predicted ${STORY_POINT_CLASSES[colIdx]}: ${value}`}
                            >
                              {value}
                            </div>
                          );
                        })}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              <div className="mt-2 flex items-center justify-between text-xs" style={{ color: "var(--color-text-muted)" }}>
                <span>Row label = true story points, column label = predicted story points</span>
                <span>n = {formatInt(metrics.n_test)}</span>
              </div>
            </div>
          </Panel>

          <Panel title="Per-class F1 score" icon={BarChart3}>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={f1Data} barCategoryGap="32%">
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
                  <Bar dataKey="F1 score" fill="var(--color-fedprox)" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Panel>
        </div>
      )}
    </div>
  );
}

function SelectField({
  value,
  onChange,
  options,
}: {
  value: string | undefined;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <div className="relative">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="appearance-none rounded-lg border pl-3 pr-9 py-2 text-sm font-medium outline-none"
        style={{
          background: "var(--color-surface-2)",
          borderColor: "var(--color-border)",
          color: "var(--color-text-primary)",
        }}
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
      <ChevronDown
        className="pointer-events-none absolute right-2.5 top-1/2 h-4 w-4 -translate-y-1/2"
        style={{ color: "var(--color-text-muted)" }}
      />
    </div>
  );
}
