/**
 * Copies a real `results/` directory produced by train_federated_dl.py
 * (e.g. <save-dir>/results) into public/data/runs/<id>/ and registers it
 * in public/data/manifest.json so the dashboard can load it.
 *
 * Usage:
 *   node scripts/sync-data.mjs <path-to-results-dir> <run-id> ["Display label"]
 *
 * Example:
 *   node scripts/sync-data.mjs ../artifacts/results baseline "Baseline (CodeBERT, FedProx)"
 */
import { existsSync, mkdirSync, copyFileSync, readFileSync, writeFileSync } from "node:fs";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

const [, , sourceArg, idArg, labelArg] = process.argv;

if (!sourceArg || !idArg) {
  console.error("Usage: node scripts/sync-data.mjs <path-to-results-dir> <run-id> [\"Display label\"]");
  process.exit(1);
}

const sourceDir = resolve(sourceArg);
const runId = idArg;
const label = labelArg ?? runId;

if (!existsSync(sourceDir)) {
  console.error(`[sync] Source directory not found: ${sourceDir}`);
  process.exit(1);
}

const FILES = [
  "config.json",
  "summary.csv",
  "summary.md",
  "federated_per_project.json",
  "federated_no_warmstart_per_project.json",
  "centralized_per_project.json",
  "local_only_per_project.json",
  "federated_round_history.json",
  "federated_no_warmstart_round_history.json",
  "communication_cost.json",
];

const destDir = join(__dirname, "..", "public", "data", "runs", runId);
mkdirSync(destDir, { recursive: true });

let copied = 0;
for (const file of FILES) {
  const src = join(sourceDir, file);
  if (existsSync(src)) {
    copyFileSync(src, join(destDir, file));
    copied += 1;
  }
}

if (copied === 0) {
  console.error(`[sync] No known result files found in ${sourceDir}. Nothing copied.`);
  process.exit(1);
}

const manifestPath = join(__dirname, "..", "public", "data", "manifest.json");
let manifest = { runs: [] };
if (existsSync(manifestPath)) {
  manifest = JSON.parse(readFileSync(manifestPath, "utf-8"));
}

const entry = { id: runId, label, path: `runs/${runId}` };
const existingIndex = manifest.runs.findIndex((r) => r.id === runId);
if (existingIndex >= 0) {
  manifest.runs[existingIndex] = entry;
} else {
  manifest.runs.push(entry);
}

writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));

console.log(`[sync] Copied ${copied} file(s) from ${sourceDir} -> ${destDir}`);
console.log(`[sync] Registered run "${runId}" in manifest.json`);
