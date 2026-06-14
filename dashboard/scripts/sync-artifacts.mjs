/**
 * Builds a dashboard run directly from this project's real training output in
 * `../artifacts/results/` (produced by train_federated_dl.py).
 *
 * That run only writes accuracy / macro-F1 / per-class-F1 / confusion matrix for
 * the federated and centralized conditions. The dashboard additionally shows
 * `mae` and `cohen_kappa`, which are *exact, deterministic* functions of the
 * confusion matrix over the ordinal story-point values [1, 2, 3, 5, 8] — so we
 * derive them here rather than leaving the UI blank. Nothing is fabricated:
 * round history, communication cost and the local-only / no-warm-start
 * conditions were not produced by this run, so they are intentionally omitted
 * (the dashboard renders honest "not available" states for them).
 *
 * Usage:
 *   node scripts/sync-artifacts.mjs
 */
import { existsSync, mkdirSync, copyFileSync, readFileSync, writeFileSync } from "node:fs";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

const ARTIFACTS_DIR = resolve(__dirname, "..", "..", "artifacts", "results");
const RUN_ID = "fedprox-distilbert";
const RUN_LABEL = "FedProx · DistilBERT + LoRA (μ=0.01)";

const SP_LABELS = [1, 2, 3, 5, 8]; // ordinal story-point values, in class-index order
const SP_RANGE2 = (SP_LABELS[SP_LABELS.length - 1] - SP_LABELS[0]) ** 2; // (8-1)^2 = 49

const round = (v, d = 4) => Math.round(v * 10 ** d) / 10 ** d;

/** Mean absolute error in story-point units, read straight off the confusion matrix. */
function maeFromConfusion(cm) {
  let weighted = 0;
  let total = 0;
  for (let i = 0; i < cm.length; i++) {
    for (let j = 0; j < cm[i].length; j++) {
      weighted += cm[i][j] * Math.abs(SP_LABELS[i] - SP_LABELS[j]);
      total += cm[i][j];
    }
  }
  return total ? weighted / total : 0;
}

/** Quadratic-weighted (ordinal-aware) Cohen's kappa over the story-point values. */
function quadraticWeightedKappa(cm) {
  const k = cm.length;
  const rowTotals = cm.map((row) => row.reduce((a, b) => a + b, 0));
  const colTotals = Array.from({ length: k }, (_, j) => cm.reduce((a, row) => a + row[j], 0));
  const total = rowTotals.reduce((a, b) => a + b, 0);
  if (!total) return 0;

  let num = 0;
  let den = 0;
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      const w = (SP_LABELS[i] - SP_LABELS[j]) ** 2 / SP_RANGE2;
      const expected = (rowTotals[i] * colTotals[j]) / total;
      num += w * cm[i][j];
      den += w * expected;
    }
  }
  return den ? 1 - num / den : 0;
}

/** Adds derived `mae` / `cohen_kappa` to every project entry in a per-project file. */
function augment(perProject) {
  const out = {};
  for (const [project, m] of Object.entries(perProject)) {
    out[project] = {
      accuracy: m.accuracy,
      macro_f1: m.macro_f1,
      per_class_f1: m.per_class_f1,
      confusion_matrix: m.confusion_matrix,
      mae: round(maeFromConfusion(m.confusion_matrix)),
      cohen_kappa: round(quadraticWeightedKappa(m.confusion_matrix)),
      n_test: m.n_test,
    };
  }
  return out;
}

// ---------------------------------------------------------------------------

if (!existsSync(ARTIFACTS_DIR)) {
  console.error(`[sync-artifacts] Not found: ${ARTIFACTS_DIR}`);
  console.error("Run training first (train_federated_dl.py) so artifacts/results/ exists.");
  process.exit(1);
}

const readJson = (name) => JSON.parse(readFileSync(join(ARTIFACTS_DIR, name), "utf-8"));

const federated = augment(readJson("federated_per_project.json"));
const centralized = augment(readJson("centralized_per_project.json"));

// Pass the real config through, adding fields the dashboard displays that the
// training run records implicitly rather than in config.json.
const rawConfig = readJson("config.json");
const config = {
  split_mode: "random", // fl/data.py uses sklearn train_test_split per client
  ...rawConfig,
  federated_condition:
    rawConfig.prox_mu > 0 ? `FedProx (μ=${rawConfig.prox_mu})` : "FedAvg",
};

const destDir = join(__dirname, "..", "public", "data", "runs", RUN_ID);
mkdirSync(destDir, { recursive: true });

writeFileSync(join(destDir, "federated_per_project.json"), JSON.stringify(federated, null, 2));
writeFileSync(join(destDir, "centralized_per_project.json"), JSON.stringify(centralized, null, 2));
writeFileSync(join(destDir, "config.json"), JSON.stringify(config, null, 2));

for (const file of ["summary.csv", "summary.md"]) {
  const src = join(ARTIFACTS_DIR, file);
  if (existsSync(src)) copyFileSync(src, join(destDir, file));
}

const manifestPath = join(__dirname, "..", "public", "data", "manifest.json");
const manifest = { runs: [{ id: RUN_ID, label: RUN_LABEL, path: `runs/${RUN_ID}` }] };
writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));

console.log(`[sync-artifacts] Wrote run "${RUN_ID}" to ${destDir}`);
console.log(`[sync-artifacts] federated: ${Object.keys(federated).length - 1} projects + global`);
console.log(`[sync-artifacts] centralized: ${Object.keys(centralized).length - 1} projects + global`);
console.log(`[sync-artifacts] Registered as the sole run in manifest.json`);
