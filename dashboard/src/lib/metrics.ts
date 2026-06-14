import { CONDITION_ORDER } from "./conditions";
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

/** Conditions that have data in this run, in a consistent display order. */
export function availableConditions(runData: RunData): ConditionKey[] {
  return CONDITION_ORDER.filter((key) => runData.perProject[key]?.global !== undefined);
}

/** Percent change of `value` relative to `base` (e.g. warm-start improvement). */
export function percentChange(value: number, base: number): number | undefined {
  if (!base) return undefined;
  return ((value - base) / base) * 100;
}
