import type { LucideIcon } from "lucide-react";
import {
  Network,
  Server,
  Lock,
  Database,
  Cpu,
  ArrowDown,
  ArrowUp,
  ArrowRight,
  RotateCw,
  Send,
  GitMerge,
  Trophy,
  Combine,
  Layers,
  Sigma,
  Type,
  Flag,
  FileText,
} from "lucide-react";
import { useRun } from "../context/RunContext";
import { PageHeader } from "../components/PageHeader";
import { Panel } from "../components/Panel";
import { availableConditions, getProjectIds } from "../lib/metrics";

export function ArchitecturePage() {
  const { runData } = useRun();
  if (!runData) return null;

  const config = runData.config;
  const algo = config && config.prox_mu > 0 ? `FedProx (μ=${config.prox_mu})` : "FedAvg";
  const encoder = config?.model_name ?? "bert encoder";
  const adapter = config ? (config.use_lora ? `LoRA r=${config.lora_r}${config.ffa_lora ? " · FFA" : ""}` : "full FT") : "LoRA";
  const rounds = config?.rounds ?? 8;

  const conditions = availableConditions(runData);
  const clientCount =
    conditions.map((k) => getProjectIds(runData.perProject[k]).length).find((n) => n > 0) ?? 0;

  // Show up to 4 client nodes, then summarize the rest.
  const shownClients = Math.min(clientCount || 4, 4);
  const remaining = clientCount ? Math.max(clientCount - shownClients, 0) : 0;

  return (
    <div className="flex flex-col gap-5 pb-8">
      <PageHeader
        icon={Network}
        title="System Architecture"
        description="How the federated system is connected and how a training round flows between the server and clients."
      />

      {/* ---- Federated topology: server hub + client silos ---- */}
      <div className="px-4 sm:px-6">
        <Panel title="Federated topology" icon={Network}>
          <div className="flex flex-col items-center">
            {/* Global server node */}
            <NodeCard
              icon={Server}
              color="var(--color-fedprox)"
              title="Global Server"
              subtitle={algo}
              chips={["aggregated global model", `${adapter}`]}
              prominent
            />

            {/* Bidirectional communication channel */}
            <div className="flex items-stretch gap-6 py-2">
              <Channel icon={ArrowDown} label="broadcast global weights" color="var(--color-fedprox)" />
              <Channel icon={ArrowUp} label="upload local updates" color="var(--color-fedavg)" />
            </div>

            {/* Privacy note on the channel */}
            <span
              className="badge mb-2"
              style={{ background: "var(--color-surface-3)", color: "var(--color-text-secondary)" }}
            >
              <Lock className="h-3.5 w-3.5" strokeWidth={2} /> raw data never leaves a client
            </span>

            {/* Horizontal bus connecting the clients to the server */}
            <Bus count={shownClients} />

            {/* Client silos */}
            <div
              className="grid gap-3 w-full"
              style={{ gridTemplateColumns: `repeat(${shownClients}, minmax(0, 1fr))` }}
            >
              {Array.from({ length: shownClients }).map((_, i) => (
                <ClientNode key={i} index={i + 1} encoder={encoder} adapter={adapter} />
              ))}
            </div>

            {remaining > 0 && (
              <span
                className="badge mt-3"
                style={{ background: "var(--color-surface-3)", color: "var(--color-text-secondary)" }}
              >
                + {remaining} more project silos · {clientCount} clients total
              </span>
            )}
          </div>
        </Panel>
      </div>

      {/* ---- The round loop (how it works over time) ---- */}
      <div className="px-4 sm:px-6">
        <Panel title="One federated round" icon={RotateCw}>
          <div className="flex flex-wrap items-center justify-center gap-x-1 gap-y-3">
            <Step icon={Send} color="var(--color-fedprox)" label="Broadcast" />
            <Flow />
            <Step icon={Cpu} color="var(--color-fedavg)" label="Local train" />
            <Flow />
            <Step icon={Network} color="var(--color-fedavg)" label="+ Proximal term" />
            <Flow />
            <Step icon={ArrowUp} color="var(--color-fedavg)" label="Upload updates" />
            <Flow />
            <Step icon={GitMerge} color="var(--color-fedprox)" label="Weighted average" />
            <Flow />
            <Step icon={Trophy} color="var(--color-centralized)" label="Validate · keep best" />
          </div>
          <div className="mt-4 flex items-center justify-center gap-2">
            <RotateCw className="h-4 w-4" style={{ color: "var(--color-text-muted)" }} strokeWidth={2} />
            <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>
              repeats for {rounds} rounds
            </span>
          </div>
        </Panel>
      </div>

      {/* ---- How one issue flows through the model ---- */}
      <div className="px-4 sm:px-6">
        <Panel title="Inside a client: model forward pass" icon={Combine}>
          <div className="flex flex-col gap-4">
            {/* Three input branches */}
            <div className="flex flex-wrap items-stretch justify-center gap-3">
              <Mini icon={FileText} color="var(--color-fedprox)" title="title + description" sub={`${encoder} · ${adapter}`} />
              <Mini icon={Type} color="var(--color-local)" title="type" sub="embedding" />
              <Mini icon={Flag} color="var(--color-local)" title="priority" sub="embedding" />
            </div>

            <CenterArrow />

            <div className="flex justify-center">
              <Mini icon={Combine} color="var(--color-fedavg)" title="Fusion" sub="concat the three vectors" wide />
            </div>

            <CenterArrow />

            <div className="flex justify-center">
              <Mini icon={Layers} color="var(--color-fedavg)" title="Classification head" sub="LayerNorm → Linear → ReLU → Linear" wide />
            </div>

            <CenterArrow />

            <div className="flex justify-center">
              <Mini icon={Sigma} color="var(--color-centralized)" title="Story point" sub="∈ {1, 2, 3, 5, 8}" wide />
            </div>
          </div>
        </Panel>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Diagram primitives
// ---------------------------------------------------------------------------

function NodeCard({
  icon: Icon,
  color,
  title,
  subtitle,
  chips,
  prominent,
}: {
  icon: LucideIcon;
  color: string;
  title: string;
  subtitle: string;
  chips: string[];
  prominent?: boolean;
}) {
  return (
    <div
      className="flex flex-col items-center gap-2 rounded-2xl border px-6 py-4 text-center"
      style={{
        borderColor: prominent ? `color-mix(in srgb, ${color} 45%, var(--color-border))` : "var(--color-border-soft)",
        background: "var(--color-surface-2)",
        boxShadow: prominent ? `0 0 0 1px color-mix(in srgb, ${color} 20%, transparent)` : undefined,
        minWidth: 240,
      }}
    >
      <div
        className="flex h-11 w-11 items-center justify-center rounded-xl"
        style={{ background: `color-mix(in srgb, ${color} 16%, transparent)`, color }}
      >
        <Icon className="h-6 w-6" strokeWidth={2} />
      </div>
      <div className="flex flex-col">
        <span className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>
          {title}
        </span>
        <span className="text-xs font-mono" style={{ color }}>
          {subtitle}
        </span>
      </div>
      <div className="flex flex-wrap justify-center gap-1.5">
        {chips.map((c) => (
          <span
            key={c}
            className="badge"
            style={{ background: "var(--color-surface-3)", color: "var(--color-text-secondary)" }}
          >
            {c}
          </span>
        ))}
      </div>
    </div>
  );
}

function Channel({ icon: Icon, label, color }: { icon: LucideIcon; label: string; color: string }) {
  return (
    <div className="flex flex-col items-center gap-1">
      <Icon className="h-5 w-5" style={{ color }} strokeWidth={2.25} />
      <span className="text-[11px]" style={{ color: "var(--color-text-muted)" }}>
        {label}
      </span>
    </div>
  );
}

/** Horizontal connecting line with vertical stubs dropping to each client. */
function Bus({ count }: { count: number }) {
  return (
    <div className="relative mb-3 w-full" style={{ height: 22 }}>
      {/* horizontal line */}
      <div
        className="absolute left-0 right-0 top-0"
        style={{ height: 2, background: "var(--color-border)" }}
      />
      {/* vertical stubs */}
      <div className="grid w-full" style={{ gridTemplateColumns: `repeat(${count}, minmax(0, 1fr))` }}>
        {Array.from({ length: count }).map((_, i) => (
          <div key={i} className="flex justify-center">
            <div style={{ width: 2, height: 22, background: "var(--color-border)" }} />
          </div>
        ))}
      </div>
    </div>
  );
}

function ClientNode({ index, encoder, adapter }: { index: number; encoder: string; adapter: string }) {
  return (
    <div
      className="flex flex-col items-center gap-2 rounded-xl border p-3 text-center"
      style={{ borderColor: "var(--color-border-soft)", background: "var(--color-surface-2)" }}
    >
      <div
        className="flex h-9 w-9 items-center justify-center rounded-lg"
        style={{ background: "color-mix(in srgb, var(--color-fedavg) 16%, transparent)", color: "var(--color-fedavg)" }}
      >
        <Cpu className="h-5 w-5" strokeWidth={2} />
      </div>
      <span className="text-xs font-semibold" style={{ color: "var(--color-text-primary)" }}>
        Client {index}
      </span>
      <div className="flex flex-col items-center gap-1">
        <span className="badge" style={{ background: "var(--color-surface-3)", color: "var(--color-text-secondary)" }}>
          <Lock className="h-3 w-3" strokeWidth={2} /> private data
        </span>
        <span className="badge" style={{ background: "var(--color-surface-3)", color: "var(--color-text-secondary)" }}>
          <Database className="h-3 w-3" strokeWidth={2} /> local model
        </span>
        <span className="text-[10px] font-mono truncate max-w-[120px]" style={{ color: "var(--color-text-muted)" }} title={`${encoder} · ${adapter}`}>
          {adapter}
        </span>
      </div>
    </div>
  );
}

function Step({ icon: Icon, color, label }: { icon: LucideIcon; color: string; label: string }) {
  return (
    <div
      className="flex flex-col items-center gap-1.5 rounded-xl border px-3 py-2.5"
      style={{ borderColor: "var(--color-border-soft)", background: "var(--color-surface-2)", minWidth: 96 }}
    >
      <div
        className="flex h-8 w-8 items-center justify-center rounded-lg"
        style={{ background: `color-mix(in srgb, ${color} 16%, transparent)`, color }}
      >
        <Icon className="h-4 w-4" strokeWidth={2} />
      </div>
      <span className="text-xs font-medium text-center" style={{ color: "var(--color-text-secondary)" }}>
        {label}
      </span>
    </div>
  );
}

function Flow() {
  return <ArrowRight className="h-4 w-4 shrink-0" style={{ color: "var(--color-text-muted)" }} strokeWidth={2} />;
}

function CenterArrow() {
  return (
    <div className="flex justify-center" aria-hidden>
      <ArrowDown className="h-4 w-4" style={{ color: "var(--color-text-muted)" }} strokeWidth={2} />
    </div>
  );
}

function Mini({
  icon: Icon,
  color,
  title,
  sub,
  wide,
}: {
  icon: LucideIcon;
  color: string;
  title: string;
  sub: string;
  wide?: boolean;
}) {
  return (
    <div
      className="flex items-center gap-3 rounded-xl border px-4 py-3"
      style={{ borderColor: "var(--color-border-soft)", background: "var(--color-surface-2)", minWidth: wide ? 320 : 200 }}
    >
      <div
        className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg"
        style={{ background: `color-mix(in srgb, ${color} 16%, transparent)`, color }}
      >
        <Icon className="h-5 w-5" strokeWidth={2} />
      </div>
      <div className="flex flex-col">
        <span className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>
          {title}
        </span>
        <span className="text-xs font-mono" style={{ color: "var(--color-text-secondary)" }}>
          {sub}
        </span>
      </div>
    </div>
  );
}
