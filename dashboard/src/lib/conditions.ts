import type { ConditionKey } from "../types/results";

export interface ConditionMeta {
  key: ConditionKey;
  label: string;
  shortLabel: string;
  color: string;
  description: string;
}

/**
 * Visual identity for each training condition, shared across charts, badges
 * and legends so the same color always means the same condition.
 */
export const CONDITION_META: Record<ConditionKey, ConditionMeta> = {
  federated: {
    key: "federated",
    label: "Federated",
    shortLabel: "Federated",
    color: "var(--color-fedprox)",
    description: "FedProx/FedAvg global model, warm-started and federated across clients.",
  },
  federated_no_warmstart: {
    key: "federated_no_warmstart",
    label: "Federated (no warm-start)",
    shortLabel: "No warm-start",
    color: "var(--color-fedavg)",
    description: "Same federated setup, initialized from random weights instead of the warm-start checkpoint.",
  },
  centralized: {
    key: "centralized",
    label: "Centralized",
    shortLabel: "Centralized",
    color: "var(--color-centralized)",
    description: "Upper-bound baseline: single model trained on pooled data from every client.",
  },
  local_only: {
    key: "local_only",
    label: "Local-only",
    shortLabel: "Local-only",
    color: "var(--color-local)",
    description: "Lower-bound baseline: each client trains independently with no federation.",
  },
};

export const CONDITION_ORDER: ConditionKey[] = ["local_only", "federated_no_warmstart", "federated", "centralized"];
