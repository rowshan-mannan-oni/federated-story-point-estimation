import { LineChart as LineChartIcon, TrendingUp, Flame } from "lucide-react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useRun } from "../context/RunContext";
import { PageHeader } from "../components/PageHeader";
import { Panel } from "../components/Panel";
import { StatusScreen } from "../components/StatusScreen";
import { ChartTooltip } from "../components/charts/ChartTooltip";
import { formatNumber, formatPercent } from "../lib/format";
import type { RoundHistoryEntry } from "../types/results";

interface MergedRow {
  round: number;
  val_macro_f1?: number;
  val_accuracy?: number;
  mean_val_macro_f1?: number;
  mean_local_loss?: number;
  weighted_local_loss?: number;
  ws_val_macro_f1?: number;
  ws_val_accuracy?: number;
  ws_mean_local_loss?: number;
  ws_weighted_local_loss?: number;
}

/**
 * Shared-head runs record `val_macro_f1`/`val_accuracy` (one global model).
 * Personalized-head runs instead record per-client aggregates
 * (`weighted_val_macro_f1`, `mean_val_macro_f1`, `mean_val_accuracy`).
 * Best-on-val selection uses the weighted mean, so prefer it as the primary curve.
 */
function primaryValF1(r?: RoundHistoryEntry): number | undefined {
  return r?.val_macro_f1 ?? r?.weighted_val_macro_f1 ?? r?.mean_val_macro_f1;
}
function primaryValAcc(r?: RoundHistoryEntry): number | undefined {
  return r?.val_accuracy ?? r?.mean_val_accuracy;
}

function mergeHistories(primary?: RoundHistoryEntry[], secondary?: RoundHistoryEntry[]): MergedRow[] {
  const rounds = new Set<number>();
  primary?.forEach((r) => rounds.add(r.round));
  secondary?.forEach((r) => rounds.add(r.round));

  const primaryByRound = new Map(primary?.map((r) => [r.round, r]));
  const secondaryByRound = new Map(secondary?.map((r) => [r.round, r]));

  return Array.from(rounds)
    .sort((a, b) => a - b)
    .map((round) => {
      const p = primaryByRound.get(round);
      const s = secondaryByRound.get(round);
      return {
        round,
        val_macro_f1: primaryValF1(p),
        val_accuracy: primaryValAcc(p),
        // Only distinct from the weighted curve in personalized mode.
        mean_val_macro_f1: p?.val_macro_f1 === undefined ? p?.mean_val_macro_f1 : undefined,
        mean_local_loss: p?.mean_local_loss,
        weighted_local_loss: p?.weighted_local_loss,
        ws_val_macro_f1: primaryValF1(s),
        ws_val_accuracy: primaryValAcc(s),
        ws_mean_local_loss: s?.mean_local_loss,
        ws_weighted_local_loss: s?.weighted_local_loss,
      };
    });
}

export function RoundHistory() {
  const { runData } = useRun();
  if (!runData) return null;

  const hasFederated = !!runData.roundHistory?.length;
  const hasNoWarmstart = !!runData.noWarmstartRoundHistory?.length;

  if (!hasFederated && !hasNoWarmstart) {
    return (
      <div className="flex flex-col gap-5 pb-8">
        <PageHeader icon={LineChartIcon} title="Round History" description="Per-round validation curves for this run." />
        <div className="px-4 sm:px-6">
          <StatusScreen
            icon={LineChartIcon}
            title="No round history available"
            description="This run did not produce federated_round_history.json."
          />
        </div>
      </div>
    );
  }

  const data = mergeHistories(runData.roundHistory, runData.noWarmstartRoundHistory);
  const personalized = !!runData.roundHistory?.some((r) => r.weighted_val_macro_f1 !== undefined);
  const primaryLabel = personalized ? "Weighted (per client)" : "Federated";

  return (
    <div className="flex flex-col gap-5 pb-8">
      <PageHeader
        icon={LineChartIcon}
        title="Round History"
        description={
          personalized
            ? "Per-client validation aggregated across clients (weighted + mean) and training loss across federated rounds."
            : "Validation performance and training loss across federated rounds."
        }
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 px-4 sm:px-6">
        <Panel title="Validation Macro-F1" icon={TrendingUp}>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-soft)" vertical={false} />
                <XAxis
                  dataKey="round"
                  tick={{ fill: "var(--color-text-secondary)", fontSize: 12 }}
                  axisLine={{ stroke: "var(--color-border)" }}
                  tickLine={false}
                  label={{ value: "Round", position: "insideBottom", offset: -4, fill: "var(--color-text-muted)", fontSize: 11 }}
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
                  content={<ChartTooltip formatter={(v) => formatPercent(v as number)} labelFormatter={(l) => `Round ${l}`} />}
                />
                <Legend wrapperStyle={{ fontSize: 12, color: "var(--color-text-secondary)" }} />
                {hasFederated && (
                  <Line
                    type="monotone"
                    dataKey="val_macro_f1"
                    name={primaryLabel}
                    stroke="var(--color-fedprox)"
                    strokeWidth={2}
                    dot={false}
                    connectNulls
                  />
                )}
                {personalized && (
                  <Line
                    type="monotone"
                    dataKey="mean_val_macro_f1"
                    name="Mean (per client)"
                    stroke="var(--color-centralized)"
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    dot={false}
                    connectNulls
                  />
                )}
                {hasNoWarmstart && (
                  <Line
                    type="monotone"
                    dataKey="ws_val_macro_f1"
                    name="No warm-start"
                    stroke="var(--color-fedavg)"
                    strokeWidth={2}
                    strokeDasharray="5 4"
                    dot={false}
                    connectNulls
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="Validation Accuracy" icon={TrendingUp}>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-soft)" vertical={false} />
                <XAxis
                  dataKey="round"
                  tick={{ fill: "var(--color-text-secondary)", fontSize: 12 }}
                  axisLine={{ stroke: "var(--color-border)" }}
                  tickLine={false}
                  label={{ value: "Round", position: "insideBottom", offset: -4, fill: "var(--color-text-muted)", fontSize: 11 }}
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
                  content={<ChartTooltip formatter={(v) => formatPercent(v as number)} labelFormatter={(l) => `Round ${l}`} />}
                />
                <Legend wrapperStyle={{ fontSize: 12, color: "var(--color-text-secondary)" }} />
                {hasFederated && (
                  <Line
                    type="monotone"
                    dataKey="val_accuracy"
                    name={primaryLabel}
                    stroke="var(--color-fedprox)"
                    strokeWidth={2}
                    dot={false}
                    connectNulls
                  />
                )}
                {hasNoWarmstart && (
                  <Line
                    type="monotone"
                    dataKey="ws_val_accuracy"
                    name="No warm-start"
                    stroke="var(--color-fedavg)"
                    strokeWidth={2}
                    strokeDasharray="5 4"
                    dot={false}
                    connectNulls
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="Mean Local Loss" icon={Flame} className="lg:col-span-2">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-soft)" vertical={false} />
                <XAxis
                  dataKey="round"
                  tick={{ fill: "var(--color-text-secondary)", fontSize: 12 }}
                  axisLine={{ stroke: "var(--color-border)" }}
                  tickLine={false}
                  label={{ value: "Round", position: "insideBottom", offset: -4, fill: "var(--color-text-muted)", fontSize: 11 }}
                />
                <YAxis
                  tick={{ fill: "var(--color-text-secondary)", fontSize: 12 }}
                  tickFormatter={(v) => formatNumber(v, 2)}
                  axisLine={false}
                  tickLine={false}
                  width={48}
                />
                <Tooltip
                  content={<ChartTooltip formatter={(v) => formatNumber(v as number, 4)} labelFormatter={(l) => `Round ${l}`} />}
                />
                <Legend wrapperStyle={{ fontSize: 12, color: "var(--color-text-secondary)" }} />
                {hasFederated && (
                  <>
                    <Line
                      type="monotone"
                      dataKey="mean_local_loss"
                      name="Mean local loss (federated)"
                      stroke="var(--color-fedprox)"
                      strokeWidth={2}
                      dot={false}
                      connectNulls
                    />
                    <Line
                      type="monotone"
                      dataKey="weighted_local_loss"
                      name="Weighted local loss (federated)"
                      stroke="var(--color-centralized)"
                      strokeWidth={2}
                      dot={false}
                      connectNulls
                    />
                  </>
                )}
                {hasNoWarmstart && (
                  <>
                    <Line
                      type="monotone"
                      dataKey="ws_mean_local_loss"
                      name="Mean local loss (no warm-start)"
                      stroke="var(--color-fedavg)"
                      strokeWidth={2}
                      strokeDasharray="5 4"
                      dot={false}
                      connectNulls
                    />
                    <Line
                      type="monotone"
                      dataKey="ws_weighted_local_loss"
                      name="Weighted local loss (no warm-start)"
                      stroke="var(--color-local)"
                      strokeWidth={2}
                      strokeDasharray="5 4"
                      dot={false}
                      connectNulls
                    />
                  </>
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>
    </div>
  );
}
