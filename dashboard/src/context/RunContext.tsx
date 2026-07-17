import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { collectRunsFromFileList, loadManifest, loadRun, type RunMeta } from "../lib/loadResults";
import type { RunData } from "../types/results";

interface RunContextValue {
  runs: RunMeta[];
  selectedRunId: string | null;
  setSelectedRunId: (id: string) => void;
  runData: RunData | null;
  loading: boolean;
  error: string | null;
  /** Import result files from a folder the user picks (client-side, no server). */
  importFolder: (files: FileList) => Promise<void>;
  importing: boolean;
  importError: string | null;
}

const RunContext = createContext<RunContextValue | undefined>(undefined);

export function RunProvider({ children }: { children: ReactNode }) {
  const [manifestRuns, setManifestRuns] = useState<RunMeta[]>([]);
  const [localRuns, setLocalRuns] = useState<RunMeta[]>([]);
  const [localData, setLocalData] = useState<Record<string, RunData>>({});
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [runData, setRunData] = useState<RunData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);

  const runs = useMemo(() => [...localRuns, ...manifestRuns], [localRuns, manifestRuns]);

  // Load the manifest once on mount. A missing manifest is not fatal — the user
  // can still import a folder — so only surface the empty state, never an error.
  useEffect(() => {
    let cancelled = false;
    loadManifest()
      .then((runsFromManifest) => {
        if (cancelled) return;
        setManifestRuns(runsFromManifest);
        if (runsFromManifest.length > 0) {
          setSelectedRunId((current) => current ?? runsFromManifest[0].id);
        }
        setLoading(false);
      })
      .catch(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Resolve the selected run: in-memory for imported folders, fetched otherwise.
  useEffect(() => {
    if (!selectedRunId) return;

    if (localData[selectedRunId]) {
      setRunData(localData[selectedRunId]);
      setLoading(false);
      setError(null);
      return;
    }

    const meta = manifestRuns.find((r) => r.id === selectedRunId);
    if (!meta) return;

    let cancelled = false;
    setLoading(true);
    setError(null);

    loadRun(meta)
      .then((data) => {
        if (cancelled) return;
        setRunData(data);
        setLoading(false);
      })
      .catch(() => {
        if (!cancelled) {
          setError(`Failed to load run "${meta.label}"`);
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [selectedRunId, manifestRuns, localData]);

  async function importFolder(files: FileList) {
    setImporting(true);
    setImportError(null);
    try {
      const parsed = await collectRunsFromFileList(files);
      if (parsed.length === 0) {
        setImportError("No result files found in that folder. Pick a folder containing config.json / *_per_project.json.");
        return;
      }
      // Merge, replacing any run re-imported from the same path.
      const newIds = new Set(parsed.map((p) => p.meta.id));
      setLocalRuns((prev) => [...parsed.map((p) => p.meta), ...prev.filter((r) => !newIds.has(r.id))]);
      setLocalData((prev) => {
        const next = { ...prev };
        for (const p of parsed) next[p.meta.id] = p.data;
        return next;
      });
      setSelectedRunId(parsed[0].meta.id);
    } catch {
      setImportError("Could not read that folder. Make sure it contains a training results directory.");
    } finally {
      setImporting(false);
    }
  }

  const value = useMemo<RunContextValue>(
    () => ({
      runs,
      selectedRunId,
      setSelectedRunId,
      runData,
      loading,
      error,
      importFolder,
      importing,
      importError,
    }),
    [runs, selectedRunId, runData, loading, error, importing, importError],
  );

  return <RunContext.Provider value={value}>{children}</RunContext.Provider>;
}

export function useRun(): RunContextValue {
  const ctx = useContext(RunContext);
  if (!ctx) throw new Error("useRun must be used within a RunProvider");
  return ctx;
}
