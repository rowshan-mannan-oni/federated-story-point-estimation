import { CONDITION_ORDER } from "./conditions";
import { STORY_POINT_CLASSES } from "../types/results";
import type { ClassificationMetrics, ConditionKey, PerProjectResults, RunData } from "../types/results";

export function getGlobalMetrics(results?: PerProjectResults): ClassificationMetrics | undefined {
  return results?.global;
}

/** Project/client ids only, excluding the synthetic "global" entry, sorted alphabetically. */
export function getProjectIds(results?: PerProjectResults): string[] {
  if (!results) return [];
  return Object.keys(results)
    .filter((id) => id !== "global")
    .sort((a, b) => a.localeCompare(b));
}

/**
 * Aggregate metric across all clients, computed from the element-wise sum of the
 * per-project confusion matrices. For shared-head conditions this equals the stored
 * "global" entry exactly (the per-project test sets partition the pooled test set).
 * For personalized-head conditions there is no single global model, so this is the
 * test-size-weighted aggregate of the per-client predictions — reported explicitly as
 * an aggregate, never as a single global model metric.
 */
export function getAggregateMetrics(results?: PerProjectResults): ClassificationMetrics | undefined {
  if (!results) return undefined;
  const ids = getProjectIds(results);
  if (ids.length === 0) return results.global;

  const k = STORY_POINT_CLASSES.length;
  const cm: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
  let sawMatrix = false;

  for (const id of ids) {
    const m = results[id]?.confusion_matrix;
    if (!m) continue;
    sawMatrix = true;
    for (let r = 0; r < Math.min(k, m.length); r++) {
      for (let c = 0; c < Math.min(k, m[r].length); c++) cm[r][c] += m[r][c];
    }
  }

  // Fallback: no confusion matrices — test-size-weighted mean of scalar metrics.
  if (!sawMatrix) return weightedScalarAggregate(results, ids);

  return metricsFromConfusionMatrix(cm);
}

function weightedScalarAggregate(results: PerProjectResults, ids: string[]): ClassificationMetrics | undefined {
  let n = 0;
  let acc = 0;
  let mae = 0;
  let f1 = 0;
  let kappa = 0;
  for (const id of ids) {
    const m = results[id];
    if (!m) continue;
    n += m.n_test;
    acc += m.accuracy * m.n_test;
    mae += m.mae * m.n_test;
    f1 += m.macro_f1 * m.n_test;
    kappa += m.cohen_kappa * m.n_test;
  }
  if (n === 0) return undefined;
  return {
    accuracy: acc / n,
    macro_f1: f1 / n,
    per_class_f1: [],
    confusion_matrix: [],
    mae: mae / n,
    cohen_kappa: kappa / n,
    n_test: n,
  };
}

/** Derive the full metric set from a (pooled) confusion matrix, matching sklearn conventions. */
export function metricsFromConfusionMatrix(cm: number[][]): ClassificationMetrics {
  const k = cm.length;
  const values = STORY_POINT_CLASSES;
  const rowSum = cm.map((row) => row.reduce((a, b) => a + b, 0));
  const colSum = Array.from({ length: k }, (_, c) => cm.reduce((a, row) => a + row[c], 0));
  const total = rowSum.reduce((a, b) => a + b, 0);

  let correct = 0;
  let maeWeighted = 0;
  const perClassF1: number[] = [];
  for (let c = 0; c < k; c++) {
    const tp = cm[c][c];
    correct += tp;
    const precision = colSum[c] > 0 ? tp / colSum[c] : 0;
    const recall = rowSum[c] > 0 ? tp / rowSum[c] : 0;
    perClassF1.push(precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0);
    for (let p = 0; p < k; p++) maeWeighted += cm[c][p] * Math.abs(values[c] - values[p]);
  }

  const macroF1 = perClassF1.reduce((a, b) => a + b, 0) / k;

  // Quadratic-weighted Cohen's kappa over class indices (matches evaluate_classification).
  let observed = 0;
  let expected = 0;
  const denom = (k - 1) * (k - 1) || 1;
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      const w = ((i - j) * (i - j)) / denom;
      observed += w * cm[i][j];
      expected += w * ((rowSum[i] * colSum[j]) / (total || 1));
    }
  }
  const kappa = expected > 0 ? 1 - observed / expected : 0;

  return {
    accuracy: total > 0 ? correct / total : 0,
    macro_f1: macroF1,
    per_class_f1: perClassF1,
    confusion_matrix: cm,
    mae: total > 0 ? maeWeighted / total : 0,
    cohen_kappa: kappa,
    n_test: total,
  };
}

/** True when a condition produced any per-project or global result in this run. */
export function hasResults(results?: PerProjectResults): boolean {
  return !!results && Object.keys(results).length > 0;
}

/** Conditions that have data in this run, in a consistent display order. */
export function availableConditions(runData: RunData): ConditionKey[] {
  return CONDITION_ORDER.filter((key) => hasResults(runData.perProject[key]));
}

/** Whether this run uses per-client (personalized) heads — no single global model exists. */
export function isPersonalized(runData: RunData): boolean {
  if (runData.config?.personalized_head) return true;
  // Fall back to structure: federated present but without a pooled global entry.
  const fed = runData.perProject.federated;
  return !!fed && Object.keys(fed).length > 0 && fed.global === undefined;
}

/** Percent change of `value` relative to `base` (e.g. warm-start improvement). */
export function percentChange(value: number, base: number): number | undefined {
  if (!base) return undefined;
  return ((value - base) / base) * 100;
}
