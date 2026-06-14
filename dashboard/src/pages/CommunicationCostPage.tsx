import { Radio, Cpu, Layers, Gauge, Database } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useRun } from "../context/RunContext";
import { PageHeader } from "../components/PageHeader";
import { Panel } from "../components/Panel";
import { MetricCard } from "../components/MetricCard";
import { StatusScreen } from "../components/StatusScreen";
import { ChartTooltip } from "../components/charts/ChartTooltip";
import { formatBytes, formatCompactNumber, formatInt } from "../lib/format";

export function CommunicationCostPage() {
  const { runData } = useRun();
  if (!runData) return null;

  const cost = runData.communicationCost;

  if (!cost) {
    return (
      <div className="flex flex-col gap-5 pb-8">
        <PageHeader icon={Radio} title="Communication Cost" description="Bytes transmitted per federated round." />
        <div className="px-4 sm:px-6">
          <StatusScreen
            icon={Radio}
            title="No communication cost data"
            description="This run did not produce communication_cost.json."
          />
        </div>
      </div>
    );
  }

  const trainableFraction = cost.total_params > 0 ? cost.trainable_params / cost.total_params : 0;

  const perRoundData = [
    {
      name: "Per client / round",
      "LoRA (trainable only)": cost.bytes_per_client_per_round,
      "Full fine-tune": cost.bytes_per_client_per_round_full_finetune,
    },
  ];

  const totalData = [
    {
      name: `Total over ${cost.rounds} rounds x ${cost.num_clients} clients`,
      "LoRA (trainable only)": cost.total_upload_bytes,
      "Full fine-tune": cost.total_upload_bytes_full_finetune,
    },
  ];

  return (
    <div className="flex flex-col gap-5 pb-8">
      <PageHeader
        icon={Radio}
        title="Communication Cost"
        description="Upload volume per round, comparing LoRA parameter-efficient updates against full fine-tuning."
      />

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 px-4 sm:px-6">
        <MetricCard
          icon={Cpu}
          label="Trainable params"
          value={formatCompactNumber(cost.trainable_params)}
          sublabel={`of ${formatCompactNumber(cost.total_params)} total`}
          accent="var(--color-fedprox)"
        />
        <MetricCard
          icon={Layers}
          label="Trainable fraction"
          value={`${(trainableFraction * 100).toFixed(2)}%`}
          sublabel="Share of total model parameters"
          accent="var(--color-fedavg)"
        />
        <MetricCard
          icon={Database}
          label="Upload / client / round"
          value={formatBytes(cost.bytes_per_client_per_round)}
          sublabel={`vs ${formatBytes(cost.bytes_per_client_per_round_full_finetune)} full fine-tune`}
          accent="var(--color-local)"
        />
        <MetricCard
          icon={Gauge}
          label="Reduction factor"
          value={`${formatInt(cost.reduction_factor)}x`}
          sublabel="Smaller uploads vs. full fine-tuning"
          accent="var(--color-centralized)"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 px-4 sm:px-6">
        <Panel title="Bytes per client per round" icon={Database}>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={perRoundData} layout="vertical" barCategoryGap="40%">
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-soft)" horizontal={false} />
                <XAxis
                  type="number"
                  tickFormatter={(v) => formatBytes(v)}
                  tick={{ fill: "var(--color-text-secondary)", fontSize: 12 }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis type="category" dataKey="name" hide />
                <Tooltip content={<ChartTooltip formatter={(v) => formatBytes(v as number)} />} cursor={{ fill: "var(--color-surface-2)" }} />
                <Legend wrapperStyle={{ fontSize: 12, color: "var(--color-text-secondary)" }} />
                <Bar dataKey="LoRA (trainable only)" fill="var(--color-fedprox)" radius={[0, 6, 6, 0]} barSize={28} />
                <Bar dataKey="Full fine-tune" fill="var(--color-surface-3)" radius={[0, 6, 6, 0]} barSize={28} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="Total upload volume (entire run)" icon={Radio}>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={totalData} layout="vertical" barCategoryGap="40%">
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-soft)" horizontal={false} />
                <XAxis
                  type="number"
                  tickFormatter={(v) => formatBytes(v)}
                  tick={{ fill: "var(--color-text-secondary)", fontSize: 12 }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis type="category" dataKey="name" hide />
                <Tooltip content={<ChartTooltip formatter={(v) => formatBytes(v as number)} />} cursor={{ fill: "var(--color-surface-2)" }} />
                <Legend wrapperStyle={{ fontSize: 12, color: "var(--color-text-secondary)" }} />
                <Bar dataKey="LoRA (trainable only)" fill="var(--color-fedprox)" radius={[0, 6, 6, 0]} barSize={28} />
                <Bar dataKey="Full fine-tune" fill="var(--color-surface-3)" radius={[0, 6, 6, 0]} barSize={28} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>
    </div>
  );
}
