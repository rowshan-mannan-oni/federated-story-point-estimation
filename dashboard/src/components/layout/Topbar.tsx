import { useEffect, useRef } from "react";
import { AlertCircle, ChevronDown, Cpu, FolderOpen, GitBranch, Loader2, Shuffle } from "lucide-react";
import { useRun } from "../../context/RunContext";

export function Topbar() {
  const { runs, selectedRunId, setSelectedRunId, runData, importFolder, importing, importError } = useRun();
  const config = runData?.config;
  const folderInputRef = useRef<HTMLInputElement>(null);

  // `webkitdirectory` isn't a typed React prop; set it (plus the Firefox alias) on the DOM node.
  useEffect(() => {
    const input = folderInputRef.current;
    if (!input) return;
    input.setAttribute("webkitdirectory", "");
    input.setAttribute("directory", "");
  }, []);

  async function onFolderPicked(e: React.ChangeEvent<HTMLInputElement>) {
    const files = e.target.files;
    if (files && files.length > 0) await importFolder(files);
    e.target.value = ""; // allow re-picking the same folder
  }

  return (
    <header
      className="flex flex-wrap items-center justify-between gap-3 border-b px-4 sm:px-6 py-3.5"
      style={{ borderColor: "var(--color-border-soft)", background: "var(--color-canvas)" }}
    >
      <div className="flex items-center gap-3 min-w-0">
        <h1 className="text-base sm:text-lg font-semibold truncate" style={{ color: "var(--color-text-primary)" }}>
          Training Results
        </h1>
        {config?.federated_condition && (
          <span
            className="badge shrink-0"
            style={{
              background: "color-mix(in srgb, var(--color-fedprox) 18%, transparent)",
              color: "var(--color-fedprox)",
            }}
          >
            {config.federated_condition}
          </span>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {config && (
          <>
            <InfoPill icon={Cpu} text={config.model_name} title="Encoder model" />
            <InfoPill
              icon={GitBranch}
              text={config.use_lora ? `LoRA r=${config.lora_r}${config.ffa_lora ? " (FFA)" : ""}` : "Full fine-tune"}
              title="Adapter configuration"
            />
            <InfoPill icon={Shuffle} text={`${config.split_mode} split`} title="Train/val/test split mode" />
          </>
        )}

        {importError && (
          <span
            title={importError}
            className="badge max-w-[16rem] truncate"
            style={{ background: "color-mix(in srgb, var(--color-negative) 16%, transparent)", color: "var(--color-negative)" }}
          >
            <AlertCircle className="h-3.5 w-3.5 shrink-0" strokeWidth={2} />
            {importError}
          </span>
        )}

        <input ref={folderInputRef} type="file" multiple onChange={onFolderPicked} className="hidden" />
        <button
          type="button"
          onClick={() => folderInputRef.current?.click()}
          disabled={importing}
          title="Load a results folder from your computer (config.json, *_per_project.json, …)"
          className="flex items-center gap-1.5 rounded-lg border px-3 py-2 text-sm font-medium outline-none transition-colors disabled:opacity-60"
          style={{ borderColor: "var(--color-border)", background: "var(--color-surface-2)", color: "var(--color-text-primary)" }}
        >
          {importing ? (
            <Loader2 className="h-4 w-4 animate-spin" strokeWidth={2} />
          ) : (
            <FolderOpen className="h-4 w-4" strokeWidth={2} />
          )}
          {importing ? "Reading…" : "Load folder"}
        </button>

        {runs.length > 0 && (
          <div className="relative">
            <select
              value={selectedRunId ?? ""}
              onChange={(e) => setSelectedRunId(e.target.value)}
              className="appearance-none rounded-lg border px-3 py-2 pr-8 text-sm font-medium outline-none cursor-pointer"
              style={{
                borderColor: "var(--color-border)",
                background: "var(--color-surface-2)",
                color: "var(--color-text-primary)",
              }}
            >
              {runs.map((run) => (
                <option key={run.id} value={run.id}>
                  {run.source === "local" ? `📁 ${run.label}` : run.label}
                </option>
              ))}
            </select>
            <ChevronDown
              className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 h-4 w-4"
              style={{ color: "var(--color-text-muted)" }}
            />
          </div>
        )}
      </div>
    </header>
  );
}

function InfoPill({ icon: Icon, text, title }: { icon: typeof Cpu; text: string; title: string }) {
  return (
    <span
      title={title}
      className="badge"
      style={{ background: "var(--color-surface-2)", color: "var(--color-text-secondary)" }}
    >
      <Icon className="h-3.5 w-3.5" strokeWidth={2} />
      {text}
    </span>
  );
}
