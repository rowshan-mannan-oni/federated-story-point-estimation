/**
 * Generates a realistic "demo" results run under public/data/runs/demo/ so the
 * dashboard has something to render before any real training run completes.
 * Mirrors the exact JSON/CSV schema written by save_evaluation_results() in
 * train_federated_dl.py.
 *
 * Usage: node scripts/generate-mock-data.mjs
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = join(__dirname, "..", "public", "data", "runs", "demo");

const PROJECTS = [
  "Apache_Mesos",
  "Appcelerator_Studio",
  "Aptana_Studio",
  "Atlassian_Confluence_Server",
  "DotNetNuke_Platform",
  "Hyperledger_Fabric",
  "Hyperledger_Indy_Node",
  "Hyperledger_Indy_SDK",
  "Hyperledger_Sawtooth",
  "Lyrasis_Dura_Cloud",
  "MongoDB_Core_Server",
  "Moodle",
  "Mule",
  "Sonatype_Nexus",
  "Spring_XD",
  "The_MongoDB_Engineering",
  "The_Titanium_SDK",
  "Titanium_Mobile_Platform",
];

const SP_LABELS = [1, 2, 3, 5, 8];

// Deterministic PRNG so the demo data is stable across regenerations.
function mulberry32(seed) {
  let a = seed;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(42);
const randInt = (min, max) => Math.floor(rand() * (max - min + 1)) + min;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const round = (v, d = 4) => Math.round(v * 10 ** d) / 10 ** d;

/** Build a plausible confusion matrix biased towards the diagonal. */
function makeConfusionMatrix(nTest, accuracy) {
  const n = SP_LABELS.length;
  const matrix = Array.from({ length: n }, () => Array(n).fill(0));
  // Rough per-class share, skewed towards small story points (1, 2, 3).
  const classShare = [0.32, 0.28, 0.22, 0.12, 0.06];
  for (let i = 0; i < n; i++) {
    const rowTotal = Math.round(nTest * classShare[i]);
    let remaining = rowTotal;
    for (let j = 0; j < n; j++) {
      if (j === n - 1) {
        matrix[i][j] += remaining;
        break;
      }
      let val;
      if (j === i) {
        val = Math.round(rowTotal * accuracy * (0.85 + rand() * 0.2));
      } else {
        const dist = Math.abs(j - i);
        val = Math.round(rowTotal * (1 - accuracy) * (1 / (dist + 1)) * (0.5 + rand()));
      }
      val = Math.max(0, Math.min(val, remaining));
      matrix[i][j] += val;
      remaining -= val;
    }
  }
  return matrix;
}

function makeMetrics({ nTest, accuracy, macroF1, mae, kappa }) {
  const confusion_matrix = makeConfusionMatrix(nTest, accuracy);
  const per_class_f1 = SP_LABELS.map((_, i) => {
    const diag = confusion_matrix[i][i];
    const rowSum = confusion_matrix[i].reduce((a, b) => a + b, 0);
    const colSum = confusion_matrix.reduce((a, r) => a + r[i], 0);
    const precision = colSum ? diag / colSum : 0;
    const recall = rowSum ? diag / rowSum : 0;
    const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
    return round(clamp(f1, 0, 1));
  });
  return {
    accuracy: round(accuracy),
    macro_f1: round(macroF1),
    per_class_f1,
    confusion_matrix,
    mae: round(mae),
    cohen_kappa: round(kappa),
    n_test: nTest,
  };
}

/** Build a per-project results object (incl. "global") for one condition. */
function makeConditionResults({ baseAcc, baseF1, baseMae, baseKappa, jitter }) {
  const results = {};
  let totalN = 0;
  const all = [];

  for (const project of PROJECTS) {
    const nTest = randInt(40, 320);
    totalN += nTest;
    const accuracy = clamp(baseAcc + (rand() - 0.5) * jitter, 0.15, 0.95);
    const macroF1 = clamp(baseF1 + (rand() - 0.5) * jitter, 0.1, 0.9);
    const mae = clamp(baseMae + (rand() - 0.5) * jitter * 2, 0.15, 2.5);
    const kappa = clamp(baseKappa + (rand() - 0.5) * jitter, 0.05, 0.85);
    const metrics = makeMetrics({ nTest, accuracy, macroF1, mae, kappa });
    results[project] = metrics;
    all.push(metrics);
  }

  // "global" pools the same shape of metrics, weighted towards the mean.
  const mean = (key) => all.reduce((s, m) => s + m[key], 0) / all.length;
  results.global = makeMetrics({
    nTest: totalN,
    accuracy: mean("accuracy"),
    macroF1: mean("macro_f1"),
    mae: mean("mae"),
    kappa: mean("cohen_kappa"),
  });

  return results;
}

function makeRoundHistory(rounds, { startF1, endF1, startAcc, endAcc, startLoss, endLoss }) {
  const history = [];
  for (let r = 1; r <= rounds; r++) {
    const t = (r - 1) / Math.max(rounds - 1, 1);
    const noise = () => (rand() - 0.5) * 0.015;
    history.push({
      round: r,
      mean_local_loss: round(startLoss + (endLoss - startLoss) * t + noise() * 2, 6),
      weighted_local_loss: round(startLoss + (endLoss - startLoss) * t + noise() * 2, 6),
      val_macro_f1: round(clamp(startF1 + (endF1 - startF1) * (1 - Math.exp(-2.5 * t)) + noise(), 0, 1)),
      val_accuracy: round(clamp(startAcc + (endAcc - startAcc) * (1 - Math.exp(-2.5 * t)) + noise(), 0, 1)),
    });
  }
  return history;
}

function buildSummary(fed, cen, local) {
  const ids = [...PROJECTS.slice().sort(), "global"];
  return ids.map((pid) => {
    const row = {
      project: pid,
      n_test: fed[pid].n_test,
      fed_acc: fed[pid].accuracy,
      fed_macro_f1: fed[pid].macro_f1,
    };
    SP_LABELS.forEach((sp, i) => (row[`fed_f1_sp${sp}`] = fed[pid].per_class_f1[i]));
    row.cen_acc = cen[pid].accuracy;
    row.cen_macro_f1 = cen[pid].macro_f1;
    SP_LABELS.forEach((sp, i) => (row[`cen_f1_sp${sp}`] = cen[pid].per_class_f1[i]));
    row.local_acc = local[pid].accuracy;
    row.local_macro_f1 = local[pid].macro_f1;
    SP_LABELS.forEach((sp, i) => (row[`local_f1_sp${sp}`] = local[pid].per_class_f1[i]));
    return row;
  });
}

function toCsv(rows) {
  const cols = Object.keys(rows[0]);
  const lines = [cols.join(",")];
  for (const row of rows) {
    lines.push(cols.map((c) => row[c]).join(","));
  }
  return lines.join("\n");
}

// ---------------------------------------------------------------------------

mkdirSync(OUT_DIR, { recursive: true });

const ROUNDS = 20;
const NUM_CLIENTS = PROJECTS.length;

const federated = makeConditionResults({ baseAcc: 0.58, baseF1: 0.5, baseMae: 0.62, baseKappa: 0.55, jitter: 0.18 });
const centralized = makeConditionResults({ baseAcc: 0.66, baseF1: 0.58, baseMae: 0.5, baseKappa: 0.63, jitter: 0.14 });
const localOnly = makeConditionResults({ baseAcc: 0.46, baseF1: 0.37, baseMae: 0.85, baseKappa: 0.4, jitter: 0.2 });
const federatedNoWarmstart = makeConditionResults({
  baseAcc: 0.5,
  baseF1: 0.41,
  baseMae: 0.78,
  baseKappa: 0.46,
  jitter: 0.2,
});

const roundHistory = makeRoundHistory(ROUNDS, {
  startF1: 0.25,
  endF1: federated.global.macro_f1,
  startAcc: 0.32,
  endAcc: federated.global.accuracy,
  startLoss: 1.55,
  endLoss: 0.71,
});

const noWarmstartHistory = makeRoundHistory(ROUNDS, {
  startF1: 0.16,
  endF1: federatedNoWarmstart.global.macro_f1,
  startAcc: 0.22,
  endAcc: federatedNoWarmstart.global.accuracy,
  startLoss: 1.85,
  endLoss: 0.94,
});

const trainable_params = 252613;
const total_params = 125045701;
const communicationCost = {
  trainable_params,
  total_params,
  bytes_per_client_per_round: trainable_params * 4,
  bytes_per_client_per_round_full_finetune: total_params * 4,
  rounds: ROUNDS,
  num_clients: NUM_CLIENTS,
  total_upload_bytes: trainable_params * 4 * ROUNDS * NUM_CLIENTS,
  total_upload_bytes_full_finetune: total_params * 4 * ROUNDS * NUM_CLIENTS,
  reduction_factor: round(total_params / trainable_params, 3),
};

const config = {
  data_dir: "data_to_train_on_clean",
  random_state: 42,
  test_size: 0.2,
  val_size: 0.1,
  split_mode: "random",
  model_name: "microsoft/codebert-base",
  max_length: 384,
  categorical_emb_dim: 16,
  hidden_dim: 128,
  dropout: 0.2,
  freeze_encoder: false,
  use_lora: true,
  lora_r: 8,
  lora_alpha: 16,
  lora_dropout: 0.05,
  lora_target_modules: ["query", "value"],
  ffa_lora: true,
  warmstart_project: "lsstcorp",
  warmstart_val_size: 0.15,
  warmstart_epochs: 10,
  warmstart_patience: 3,
  warmstart_lr: 3e-5,
  skip_centralized: false,
  skip_local_only: false,
  run_no_warmstart_fl: true,
  batch_size: 8,
  local_epochs: 1,
  rounds: ROUNDS,
  clients_per_round_fraction: 1.0,
  local_sample_ratio_per_epoch: 1.0,
  sample_with_replacement: false,
  learning_rate: 3e-5,
  weight_decay: 1e-4,
  prox_mu: 0.01,
  num_classes: 5,
  device: "cuda",
  federated_condition: "FedProx (mu=0.01)",
};

const summary = buildSummary(federated, centralized, localOnly);

writeFileSync(join(OUT_DIR, "config.json"), JSON.stringify(config, null, 2));
writeFileSync(join(OUT_DIR, "federated_per_project.json"), JSON.stringify(federated, null, 2));
writeFileSync(join(OUT_DIR, "centralized_per_project.json"), JSON.stringify(centralized, null, 2));
writeFileSync(join(OUT_DIR, "local_only_per_project.json"), JSON.stringify(localOnly, null, 2));
writeFileSync(
  join(OUT_DIR, "federated_no_warmstart_per_project.json"),
  JSON.stringify(federatedNoWarmstart, null, 2),
);
writeFileSync(join(OUT_DIR, "federated_round_history.json"), JSON.stringify(roundHistory, null, 2));
writeFileSync(
  join(OUT_DIR, "federated_no_warmstart_round_history.json"),
  JSON.stringify(noWarmstartHistory, null, 2),
);
writeFileSync(join(OUT_DIR, "communication_cost.json"), JSON.stringify(communicationCost, null, 2));
writeFileSync(join(OUT_DIR, "summary.csv"), toCsv(summary));

const manifestPath = join(__dirname, "..", "public", "data", "manifest.json");
const manifest = {
  runs: [
    {
      id: "demo",
      label: "Demo run (synthetic data)",
      path: "runs/demo",
    },
  ],
};
writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));

console.log(`[mock] Wrote demo run to ${OUT_DIR}`);
console.log(`[mock] Wrote manifest to ${manifestPath}`);
