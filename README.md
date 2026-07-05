# FedSP-PEFT — Privacy-Preserving Federated Story Point Estimation

A federated learning system that predicts agile **story points** from JIRA issue
text, where **each client is one software project** so raw issue data never
leaves the project. Built for a bachelor thesis on applying federated learning
to story point estimation (SPE) — a novel intersection — with
parameter-efficient fine-tuning.

**Task:** 5-class **ordinal** classification over the Fibonacci story-point deck
`{1, 2, 3, 5, 8}` (not regression). The ordinal structure is exploited by a CORN
loss head; MAE on class values and quadratic-weighted Cohen's κ are the primary
metrics (bridging to the regression-based SPE literature).

**Dataset:** TAWOS — real open-source JIRA projects, one cleaned CSV per project.

> Accuracy on SPE is inherently low for everyone (even sophisticated deep models
> barely beat naive baselines — Tawosi et al. 2023). The contribution is the
> **local-only → federated → centralized gap analysis** under a privacy
> constraint, the **personalization** finding, and **communication cost** — not
> absolute accuracy.

---

## What's in the framework (FedSP-PEFT)

```
Each client (= one project):
  Frozen encoder (DistilRoBERTa; bert-tiny for dev)
    + LoRA adapters — FFA-LoRA: A frozen, only B trained & aggregated (exact averaging)
    + categorical embeddings (issue type, priority)
    + head:  CE  (5-logit softmax + class-weighted CrossEntropy)   [--head-type ce]
             CORN (4-logit ordinal head + corn_loss)               [--head-type corn]
  Local loss + FedProx proximal term (over aggregatable params only)
Server:
  Weighted average of aggregatable params (LoRA-B + embeddings [+ head if shared])
```

Key capabilities:

- **FFA-LoRA** (Sun et al. 2024): freeze A, train/transmit only B → aggregation is
  mathematically exact and <1% of parameters are transmitted per round.
- **FedProx / FedAvg**: `--prox-mu` > 0 is FedProx; `--prox-mu 0` degrades to FedAvg
  (auto-labeled).
- **CORN ordinal head** (`--head-type corn`): penalizes distant story-point misses
  more than adjacent ones.
- **Personalized heads — FedSP-PEFT-P** (`--personalized-head`): the classification
  head stays **local per client** (never aggregated); only the shared representation
  (LoRA-B + embeddings) is federated. Targets the project-specific calibration of
  story points ("an 8 here is a 3 there"). Optional `--generic-head` saves an
  averaged head for onboarding new projects.
- **Warm-start**: centralized pre-training on the largest project, then excluded
  from the client pool.
- **Checkpoint / resume** (`--checkpoint-every`, `--resume`): bit-reproducible
  resume of federated rounds, centralized epochs, and local-only clients — warm-start
  is cached on resume. Built for time-limited / interruptible hardware.
- **Leave-one-project-out onboarding** (`--holdout-project` + `run_lopo.py`): how
  quickly a brand-new project can be onboarded via head-only adaptation.
- **Multi-seed experiments + statistics**: 10-seed runner and paired significance
  tests (Wilcoxon, Friedman, Nemenyi, effect sizes).

## Conditions compared

| Condition | Description |
|---|---|
| Majority baseline | Always predict the most frequent class |
| Local-only | Each client trains on its own data only (no federation) |
| Centralized | Pooled data, privacy ignored — the upper bound |
| Federated (FedAvg) | `--prox-mu 0` |
| Federated (FedProx) | `--prox-mu 0.01`, shared head |
| Federated (FedProx + P-head) | Personalized per-client head (`--personalized-head`) |

## Input features & metrics

- **Text:** `title + [SEP] + description`, tokenized (max 128 by default).
- **Categorical:** issue `type` and `priority` → embeddings, fused before the head.
- **Metrics** (per project + mean; `results/*.json`, `summary.csv`, `summary.md`):
  **MAE on class values** and **quadratic-weighted Cohen's κ** (primary), macro-F1,
  accuracy, per-class F1, confusion matrix, and **communication cost** vs. full
  fine-tuning.

---

## Install

```powershell
pip install -r requirements.txt
```

## Quick start

Smoke test (bert-tiny, plumbing only — never interpret its accuracy):

```powershell
python train_federated_dl.py --data-dir data_to_train_on `
  --model-name prajjwal1/bert-tiny --rounds 2 --local-epochs 1 --warmstart-epochs 1 `
  --skip-centralized --skip-local-only --save-dir artifacts_smoke
```

Full feature set (CORN + personalized head + generic head + checkpoints):

```powershell
python train_federated_dl.py --data-dir data_to_train_on `
  --model-name distilroberta-base --max-length 256 `
  --rounds 20 --local-epochs 1 --batch-size 16 --lr 3e-5 --warmstart-lr 3e-5 `
  --prox-mu 1e-2 --head-type corn --personalized-head --generic-head `
  --checkpoint-every 5 --save-dir artifacts_fedsp_peft_p --seed 42
```

Resume after an interruption — re-run the **same command** plus `--resume`
(finished phases are skipped; warm-start is loaded from cache).

**See [`commands.txt`](commands.txt) for the full runbook** — every flag explained,
plus ready-to-run commands for ablations, the 10-seed sweep, LOPO onboarding, and
inference.

## Key flags

| Flag | Purpose |
|---|---|
| `--head-type {ce,corn}` | CE softmax head (default) vs CORN ordinal head |
| `--personalized-head` | Keep the head local per client (FedSP-PEFT-P) |
| `--generic-head` | Also save an averaged head for onboarding (personalized mode) |
| `--prox-mu` | FedProx strength; `0` = FedAvg |
| `--holdout-project <name>` | Exclude a project from the pool for the LOPO experiment |
| `--checkpoint-every N` / `--resume` | Checkpointing (rounds/epochs) and resume |
| `--split-mode {random,temporal}` | Random stratified vs chronological split |
| `--no-ffa-lora` / `--run-no-warmstart-fl` | FFA-LoRA and warm-start ablations |

DistilBERT encoders need `--lora-target-modules q_lin v_lin`; BERT/RoBERTa
(bert-tiny, distilroberta-base, codebert-base) use the default `query value`.

## Artifacts & inference

Shared-head runs save `<save-dir>/federated/{model_state.pt, metadata.json, tokenizer/}`.
Personalized runs save `<save-dir>/federated/{shared_state.pt, heads/<project>.pt,
generic_head.pt?}`. Score new data:

```powershell
# shared-head artifact
python predict_saved_model.py --artifact-dir artifacts/federated `
  --data-dir data_to_test_on --out-csv predictions.csv

# personalized artifact — pick a project's head, or the generic head
python predict_saved_model.py --artifact-dir artifacts_fedsp_peft_p/federated `
  --data-dir data_to_test_on --head-project Moodle --out-csv predictions_moodle.csv
```

Adds `predicted_class` (0-4) and `predicted_story_point` (1/2/3/5/8) columns; prints
metrics if the input has `story_point`.

## Scripts

| Script | Purpose |
|---|---|
| `train_federated_dl.py` | Main pipeline: warm-start → local-only → centralized → federated → eval |
| `run_experiments.py` | Multi-seed runner (FedProx + FedAvg per seed); `--skip-baselines` for the RQ4 run |
| `compute_statistics.py` | Paired stats across seeds (Wilcoxon, Friedman, Nemenyi, effect sizes) |
| `run_lopo.py` / `compare_lopo.py` | Leave-one-project-out onboarding + tidy crossover CSV |
| `predict_saved_model.py` | Inference on saved artifacts |
| `export_issues.py` | DB → cleaned CSV export (text cleaning, normalization, stats) |
| `sanity_check_fl_randomness.py` | Reproducibility checks, incl. resume bit-reproducibility |

## Project structure

```
fl/config.py      FLConfig — single source of truth for all hyperparameters
fl/data.py        Loading, filtering, per-client train/val/test split, cleaned-data validation
fl/model.py       StoryPointClassifier — frozen encoder + LoRA + embeddings + CE/CORN head
fl/client.py      FederatedClient — local training (FedProx over aggregatable params only)
fl/server.py      FedProxServer — round orchestration, personalized per-client val, checkpoint hooks
fl/metrics.py     evaluate_classification / run_prediction (accuracy, macro-F1, MAE, κ, CM)
fl/checkpoint.py  Save/load/rotate + RNG capture/restore for resume
dashboard/        React results dashboard (see dashboard/README.md)
commands.txt      Full command runbook
```

Development uses `prajjwal1/bert-tiny`; final results use `distilroberta-base`
(6 layers, 768-dim) on a larger-VRAM machine. **bert-tiny is a plumbing model only.**