import Papa from "papaparse";
import type {
  CommunicationCost,
  FLConfig,
  PerProjectResults,
  RoundHistoryEntry,
  RunData,
  SummaryRow,
} from "../types/results";

export interface RunMeta {
  id: string;
  label: string;
  path: string;
}

export interface Manifest {
  runs: RunMeta[];
}

const DATA_ROOT = "/data";

async function fetchJson<T>(path: string): Promise<T | undefined> {
  try {
    const res = await fetch(path);
    if (!res.ok) return undefined;
    return (await res.json()) as T;
  } catch {
    return undefined;
  }
}

async function fetchCsv<T>(path: string): Promise<T[] | undefined> {
  try {
    const res = await fetch(path);
    if (!res.ok) return undefined;
    const text = await res.text();
    const parsed = Papa.parse<T>(text, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
    });
    return parsed.data;
  } catch {
    return undefined;
  }
}

export async function loadManifest(): Promise<RunMeta[]> {
  const manifest = await fetchJson<Manifest>(`${DATA_ROOT}/manifest.json`);
  return manifest?.runs ?? [];
}

export async function loadRun(meta: RunMeta): Promise<RunData> {
  const base = `${DATA_ROOT}/${meta.path}`;

  const [
    config,
    summary,
    federated,
    federatedNoWarmstart,
    centralized,
    localOnly,
    roundHistory,
    noWarmstartRoundHistory,
    communicationCost,
  ] = await Promise.all([
    fetchJson<FLConfig>(`${base}/config.json`),
    fetchCsv<SummaryRow>(`${base}/summary.csv`),
    fetchJson<PerProjectResults>(`${base}/federated_per_project.json`),
    fetchJson<PerProjectResults>(`${base}/federated_no_warmstart_per_project.json`),
    fetchJson<PerProjectResults>(`${base}/centralized_per_project.json`),
    fetchJson<PerProjectResults>(`${base}/local_only_per_project.json`),
    fetchJson<RoundHistoryEntry[]>(`${base}/federated_round_history.json`),
    fetchJson<RoundHistoryEntry[]>(`${base}/federated_no_warmstart_round_history.json`),
    fetchJson<CommunicationCost>(`${base}/communication_cost.json`),
  ]);

  return {
    id: meta.id,
    label: meta.label,
    config,
    summary,
    perProject: {
      federated,
      federated_no_warmstart: federatedNoWarmstart,
      centralized,
      local_only: localOnly,
    },
    roundHistory,
    noWarmstartRoundHistory,
    communicationCost,
  };
}
