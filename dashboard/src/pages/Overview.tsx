import { LayoutDashboard, Target, CheckCircle2, Ruler, ShieldCheck, Sparkles, Settings2 } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useRun } from "../context/RunContext";
import { PageHeader } from "../components/PageHeader";
import { Panel } from "../components/Panel";
import { MetricCard } from "../components/MetricCard";
import { CONDITION_META } from "../lib/conditions";
import { availableConditions, getGlobalMetrics, percentChange } from "../lib/metrics";
import { formatNumber, formatPercent } from "../lib/format";
import { ChartTooltip } from "../components/charts/ChartTooltip";

export function Overview() {
  const { runData } = useRun();
  if (!runData) return null;

  const conditions = availableConditions(runData);
  const primary =
    getGlobalMetrics(runData.perProject.federated) ?? getGlobalMetrics(runData.perProject.federated_no_warmstart);

  const warmstartGain = percentChange(
    getGlobalMetrics(runData.perProject.federated)?.macro_f1 ?? NaN,
    getGlobalMetrics(runData.perProject.federated_no_warmstart)?.macro_f1 ?? NaN,
  );

  const chartData = conditions.map((key) => {
    const m = getGlobalMetrics(runData.perProject[key]);
    return {
      name: CONDITION_META[key].shortLabel,
      key,
      "Macro-F1": m?.macro_f1 ?? 0,
      Accuracy: m?.accuracy ?? 0,
      "Cohen's Kappa": m?.cohen_kappa ?? 0,
    };
  });

  return (
    <div className="flex flex-col gap-5 pb-8">
      <PageHeader
        icon={LayoutDashboard}
        title="Overview"
        description={`Headline results for "${runData.label}" — global metrics pooled across all FL clients.`}
      />

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 px-4 sm:px-6">
        <MetricCard
          icon={Target}
          label="Macro-F1 (Federated)"
          value={formatPercent(primary?.macro_f1)}
          sublabel="Primary metric — robust to class imbalance"
          accent="var(--color-fedprox)"
        />
        <MetricCard
          icon={CheckCircle2}
          label="Accuracy"
          value={formatPercent(primary?.accuracy)}
          sublabel={`n = ${primary?.n_test ?? "—"} test issues`}
          accent="var(--color-fedavg)"
        />
        <MetricCard
          icon={Ruler}
          label="MAE (story points)"
          value={formatNumber(primary?.mae, 3)}
          sublabel="Lower is better"
          accent="var(--color-local)"
        />
        <MetricCard
          icon={ShieldCheck}
          label="Cohen's Kappa"
          value={formatNumber(primary?.cohen_kappa, 3)}
          sublabel="Quadratic-weighted, ordinal-aware"
          accent="var(--color-centralized)"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 px-4 sm:px-6">
        <Panel title="Condition comparison (global)" icon={Sparkles} className="lg:col-span-2">
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
                  tickFormatter={(v) => `${Math.round(v * 100)}%`}
                  domain={[0, 1]}
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

        <Panel title="Run configuration" icon={Settings2}>
          <dl className="flex flex-col gap-3 text-sm">
            <ConfigRow label="Encoder" value={runData.config?.model_name} mono />
            <ConfigRow label="Federated condition" value={runData.config?.federated_condition} />
            <ConfigRow label="Rounds" value={runData.config?.rounds} />
            <ConfigRow label="Local epochs" value={runData.config?.local_epochs} />
            <ConfigRow label="Batch size" value={runData.config?.batch_size} />
            <ConfigRow label="Learning rate" value={runData.config?.learning_rate} />
            <ConfigRow label="Prox μ" value={runData.config?.prox_mu} />
            <ConfigRow
              label="LoRA"
              value={
                runData.config?.use_lora
                  ? `r=${runData.config.lora_r}, α=${runData.config.lora_alpha}${runData.config.ffa_lora ? ", FFA" : ""}`
                  : "disabled"
              }
            />
            <ConfigRow label="Warm-start project" value={runData.config?.warmstart_project} />
            <ConfigRow label="Split mode" value={runData.config?.split_mode} />
            <ConfigRow label="Seed" value={runData.config?.random_state} />
          </dl>
        </Panel>
      </div>

      {warmstartGain !== undefined && !Number.isNaN(warmstartGain) && (
        <div className="px-4 sm:px-6">
          <div
            className="panel flex items-center gap-4 p-5"
            style={{ borderColor: "color-mix(in srgb, var(--color-fedprox) 35%, var(--color-border-soft))" }}
          >
            <div
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl"
              style={{ background: "color-mix(in srgb, var(--color-fedprox) 18%, transparent)" }}
            >
              <Sparkles className="h-5 w-5" style={{ color: "var(--color-fedprox)" }} />
            </div>
            <div>
              <p className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>
                Warm-start improved global Macro-F1 by{" "}
                <span style={{ color: warmstartGain >= 0 ? "var(--color-positive)" : "var(--color-negative)" }}>
                  {warmstartGain >= 0 ? "+" : ""}
                  {warmstartGain.toFixed(1)}%
                </span>
              </p>
              <p className="text-xs mt-0.5" style={{ color: "var(--color-text-secondary)" }}>
                Compared against federated training from random initialization (no warm-start checkpoint).
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ConfigRow({ label, value, mono }: { label: string; value: string | number | undefined; mono?: boolean }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <dt style={{ color: "var(--color-text-secondary)" }}>{label}</dt>
      <dd
        className={`text-right font-medium truncate max-w-[60%] ${mono ? "font-mono text-xs" : ""}`}
        style={{ color: "var(--color-text-primary)" }}
        title={String(value ?? "")}
      >
        {value ?? "—"}
      </dd>
    </div>
  );
}
