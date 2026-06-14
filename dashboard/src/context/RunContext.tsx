import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { loadManifest, loadRun, type RunMeta } from "../lib/loadResults";
import type { RunData } from "../types/results";

interface RunContextValue {
  runs: RunMeta[];
  selectedRunId: string | null;
  setSelectedRunId: (id: string) => void;
  runData: RunData | null;
  loading: boolean;
  error: string | null;
}

const RunContext = createContext<RunContextValue | undefined>(undefined);

export function RunProvider({ children }: { children: ReactNode }) {
  const [runs, setRuns] = useState<RunMeta[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [runData, setRunData] = useState<RunData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load the manifest once on mount.
  useEffect(() => {
    let cancelled = false;
    loadManifest()
      .then((manifestRuns) => {
        if (cancelled) return;
        setRuns(manifestRuns);
        if (manifestRuns.length > 0) {
          setSelectedRunId(manifestRuns[0].id);
        } else {
          setLoading(false);
          setError("No result runs found. Run `npm run mock-data` or `npm run sync-data` to add one.");
        }
      })
      .catch(() => {
        if (!cancelled) {
          setLoading(false);
          setError("Failed to load public/data/manifest.json");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Load the selected run's data whenever it changes.
  useEffect(() => {
    if (!selectedRunId) return;
    const meta = runs.find((r) => r.id === selectedRunId);
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
  }, [selectedRunId, runs]);

  const value = useMemo<RunContextValue>(
    () => ({ runs, selectedRunId, setSelectedRunId, runData, loading, error }),
    [runs, selectedRunId, runData, loading, error],
  );

  return <RunContext.Provider value={value}>{children}</RunContext.Provider>;
}

export function useRun(): RunContextValue {
  const ctx = useContext(RunContext);
  if (!ctx) throw new Error("useRun must be used within a RunProvider");
  return ctx;
}
