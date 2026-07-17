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
  /** Where the run came from: the bundled manifest, or a folder the user picked. */
  source?: "manifest" | "local";
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

/* ------------------------------------------------------------------ *
 * Loading a run from a folder the user picks in the browser.
 * Reads File objects directly (no server / no sync step). The same
 * result filenames written by train_federated_dl.py are recognised.
 * ------------------------------------------------------------------ */

/** Result filenames that mark a directory as a "run". */
const RESULT_FILES = [
  "config.json",
  "summary.csv",
  "federated_per_project.json",
  "federated_no_warmstart_per_project.json",
  "centralized_per_project.json",
  "local_only_per_project.json",
  "federated_round_history.json",
  "federated_no_warmstart_round_history.json",
  "communication_cost.json",
] as const;

/** A directory qualifies as a run if it has a config or any per-project results. */
const RUN_MARKERS = [
  "config.json",
  "federated_per_project.json",
  "centralized_per_project.json",
  "local_only_per_project.json",
];

async function readFileJson<T>(file?: File): Promise<T | undefined> {
  if (!file) return undefined;
  try {
    return JSON.parse(await file.text()) as T;
  } catch {
    return undefined;
  }
}

async function readFileCsv<T>(file?: File): Promise<T[] | undefined> {
  if (!file) return undefined;
  try {
    const parsed = Papa.parse<T>(await file.text(), {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
    });
    return parsed.data;
  } catch {
    return undefined;
  }
}

async function parseRunFromFiles(id: string, label: string, files: Map<string, File>): Promise<RunData> {
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
    readFileJson<FLConfig>(files.get("config.json")),
    readFileCsv<SummaryRow>(files.get("summary.csv")),
    readFileJson<PerProjectResults>(files.get("federated_per_project.json")),
    readFileJson<PerProjectResults>(files.get("federated_no_warmstart_per_project.json")),
    readFileJson<PerProjectResults>(files.get("centralized_per_project.json")),
    readFileJson<PerProjectResults>(files.get("local_only_per_project.json")),
    readFileJson<RoundHistoryEntry[]>(files.get("federated_round_history.json")),
    readFileJson<RoundHistoryEntry[]>(files.get("federated_no_warmstart_round_history.json")),
    readFileJson<CommunicationCost>(files.get("communication_cost.json")),
  ]);

  return {
    id,
    label,
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

/**
 * Group a picked directory (FileList from an <input webkitdirectory>) into runs.
 * Every directory that contains recognised result files becomes one run, so a
 * single `results/` folder yields one run and a whole `experiments/` tree yields
 * one run per seed/condition. Returns runs in a stable, human-friendly order.
 */
export async function collectRunsFromFileList(fileList: FileList): Promise<{ meta: RunMeta; data: RunData }[]> {
  // dir path -> (basename -> File), keeping only recognised result files.
  const byDir = new Map<string, Map<string, File>>();
  let root = "";

  for (const file of Array.from(fileList)) {
    const rel = (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name;
    const parts = rel.split("/");
    const base = parts[parts.length - 1];
    if (!root && parts.length > 1) root = parts[0];
    if (!RESULT_FILES.includes(base as (typeof RESULT_FILES)[number])) continue;
    const dir = parts.slice(0, -1).join("/") || root || ".";
    if (!byDir.has(dir)) byDir.set(dir, new Map());
    byDir.get(dir)!.set(base, file);
  }

  const runDirs = Array.from(byDir.entries())
    .filter(([, files]) => RUN_MARKERS.some((m) => files.has(m)))
    .sort(([a], [b]) => a.localeCompare(b));

  const runs = await Promise.all(
    runDirs.map(async ([dir, files]) => {
      // Label: folder name for a single run, else the path below the picked root.
      const label = dir === root || !root ? dir.split("/").pop() || dir : dir.startsWith(root + "/") ? dir.slice(root.length + 1) : dir;
      const meta: RunMeta = { id: `local:${dir}`, label, path: dir, source: "local" };
      const data = await parseRunFromFiles(meta.id, label, files);
      return { meta, data };
    }),
  );

  return runs;
}
