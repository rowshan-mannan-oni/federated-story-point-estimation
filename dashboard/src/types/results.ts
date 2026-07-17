/**
 * Mirrors the JSON/CSV schema written by save_evaluation_results() in
 * train_federated_dl.py and fl/metrics.py's evaluate_classification().
 */

export const STORY_POINT_CLASSES = [1, 2, 3, 5, 8] as const;
export type StoryPoint = (typeof STORY_POINT_CLASSES)[number];

export interface ClassificationMetrics {
  accuracy: number;
  macro_f1: number;
  per_class_f1: number[];
  confusion_matrix: number[][];
  mae: number;
  cohen_kappa: number;
  n_test: number;
}

/** Keyed by project/client id, plus a synthetic "global" entry. */
export type PerProjectResults = Record<string, ClassificationMetrics>;

export interface RoundHistoryEntry {
  round: number;
  mean_local_loss: number;
  weighted_local_loss: number;
  /** Shared-head mode: single global-model validation metrics. */
  val_macro_f1?: number;
  val_accuracy?: number;
  /** Personalized-head mode: per-client validation aggregated across clients. */
  mean_val_macro_f1?: number;
  weighted_val_macro_f1?: number;
  mean_val_accuracy?: number;
  /** Client ids that participated in this round (full-participation runs list all). */
  selected_clients?: string[];
}

export interface CommunicationCost {
  trainable_params: number;
  total_params: number;
  bytes_per_client_per_round: number;
  bytes_per_client_per_round_full_finetune: number;
  rounds: number;
  num_clients: number;
  total_upload_bytes: number;
  total_upload_bytes_full_finetune: number;
  reduction_factor: number;
}

export interface FLConfig {
  data_dir: string;
  random_state: number;
  test_size: number;
  val_size: number;
  split_mode: "random" | "temporal";

  model_name: string;
  max_length: number;
  categorical_emb_dim: number;
  hidden_dim: number;
  dropout: number;
  freeze_encoder: boolean;

  use_lora: boolean;
  lora_r: number;
  lora_alpha: number;
  lora_dropout: number;
  lora_target_modules: string[];
  ffa_lora: boolean;

  warmstart_project: string;
  warmstart_val_size: number;
  warmstart_epochs: number;
  warmstart_patience: number;
  warmstart_lr: number;

  skip_centralized: boolean;
  skip_local_only: boolean;
  run_no_warmstart_fl: boolean;

  batch_size: number;
  local_epochs: number;
  rounds: number;
  clients_per_round_fraction: number;
  local_sample_ratio_per_epoch: number;
  sample_with_replacement: boolean;
  learning_rate: number;
  weight_decay: number;
  prox_mu: number;

  num_classes: number;
  device: string;

  /** Ordinal head: "ce" = 5-logit softmax, "corn" = 4-logit CORN threshold head. */
  head_type?: "ce" | "corn";
  /** FedSP-PEFT-P: head stays local per client, never aggregated. */
  personalized_head?: boolean;
  /** Server keeps a one-way averaged head for onboarding new clients. */
  generic_head?: boolean;
  /** Leave-one-project-out: project excluded from the FL pool entirely. */
  holdout_project?: string | null;

  checkpoint_every?: number;
  checkpoint_keep?: number;
  resume?: boolean;

  federated_condition?: string;
}

/** One row of summary.csv — flattened metrics per project across conditions. */
export interface SummaryRow {
  project: string;
  n_test: number;
  fed_acc?: number;
  fed_macro_f1?: number;
  cen_acc?: number;
  cen_macro_f1?: number;
  local_acc?: number;
  local_macro_f1?: number;
  [key: string]: string | number | undefined;
}

export type ConditionKey = "federated" | "federated_no_warmstart" | "centralized" | "local_only";

export const CONDITION_LABELS: Record<ConditionKey, string> = {
  federated: "Federated",
  federated_no_warmstart: "Federated (no warm-start)",
  centralized: "Centralized",
  local_only: "Local-only",
};

/** Everything loaded for a single run (one --save-dir/results directory). */
export interface RunData {
  /** Display name / relative path under public/data used to load this run. */
  id: string;
  label: string;
  config?: FLConfig;
  summary?: SummaryRow[];
  perProject: Partial<Record<ConditionKey, PerProjectResults>>;
  roundHistory?: RoundHistoryEntry[];
  noWarmstartRoundHistory?: RoundHistoryEntry[];
  communicationCost?: CommunicationCost;
}
