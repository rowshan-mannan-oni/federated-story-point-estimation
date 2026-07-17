# Experiment Run Plan — FedSP-PEFT / FedSP-PEFT-P

Maps every run to one of the five locked research questions. Commands are
PowerShell (backtick line-continuation); run from the project root with the
venv active, or replace `python` with
`& "d:/Federated Learning using DL/.venv/Scripts/python.exe"`.

## Locked RQs

- **RQ1** Feasibility & cost of privacy — local-only vs centralized vs federated
- **RQ2** Heterogeneity — FedProx vs FedAvg (RQ2.1) + μ sensitivity (RQ2.2)
- **RQ3** Personalization — shared head vs personalized head (FedSP-PEFT-P)
- **RQ4** Onboarding crossover — leave-one-project-out cold start
- **RQ5** SOTA positioning — published deep SOTA is citation-based (no runs); the
  classic within-project comparators (TF-IDF+SVM, median-SP) **are** run, produced
  automatically by every non-`--skip-baselines` root

## Fixed method defaults (constant across every run)

| Setting | Value | Why fixed |
|---|---|---|
| Encoder | `microsoft/codebert-base` | RoBERTa arch → default `query value` LoRA targets; the encoder your good results used |
| Head | `--head-type corn` | CORN ordinal head is the thesis default (not an RQ) |
| PEFT | FFA-LoRA **on** (default) | Not an RQ; comm cost reported descriptively |
| Warm-start | on (lsstcorp) | cached per run; reused on `--resume` |
| max-length | 256 (fallback 128 if VRAM-tight) | p90 token length ≈ 164 |
| rounds / local-epochs | 60 / 1 | 60 rounds @ 1 local epoch is the measured convergence plateau; best-on-val still selects within that |
| lr / warmstart-lr | 3e-5 / 3e-5 | " |
| batch-size | 64 | proven to fit 24 GB and used in prior runs |
| Seeds | 42–44 (3 seeds, all roots) | per-project paired tests carry significance (n=18 projects); 3 seeds give variance bands. Reduced-seed count noted as a thesis limitation |
| Checkpointing | `--checkpoint-every 5 --resume` | interrupt-safe daily sessions |

**Splits:** temporal = **primary** (deployment-realistic; reported results).
random = **robustness** (kept, 3 seeds). Every primary root is run once temporal,
then once random for the robustness appendix.

**Seed count:** 3 seeds (42–44) on every root. Significance comes from the
per-project paired tests (n=18 projects), not the seed count; 3 seeds add a
variance band and guard against one unlucky init. This is a deliberate
time/rigor tradeoff — stated as a limitation in the thesis.

## RQ → root mapping (overview)

| RQ | Root(s) | Seeds | Conditions produced |
|---|---|---|---|
| RQ1, RQ2.1, RQ5 (classic) | `experiments_temporal_shared` | 42–44 | majority, median-SP, TF-IDF+SVM, local-only, centralized, FedProx(0.01), FedAvg(0) |
| RQ3 | `experiments_temporal_personalized` | 42–44 | FedProx(0.01)+P-head+generic |
| RQ2.2 | `experiments_temporal_mu0.001`, `_mu0.1` | 42–44 | FedProx(0.001), FedProx(0.1) |
| RQ4 | `artifacts_lopo_*` + `lopo_results` | 42(+43,44) | LOPO onboarding, Sawtooth held out |
| robustness | `experiments_random_shared`, `_personalized` | 42–44 | mirrors of the temporal roots |
| RQ5 | (shared root) | 42–44 | classic baselines run there; deep SOTA = citation table |

---

## Stage 0 — Smoke (bert-tiny, plumbing only; never interpret accuracy)

```powershell
python train_federated_dl.py `
    --data-dir data_to_train_on `
    --model-name prajjwal1/bert-tiny --max-length 64 `
    --rounds 2 --local-epochs 1 --warmstart-epochs 1 `
    --head-type corn --personalized-head --generic-head --checkpoint-every 1 `
    --save-dir artifacts_smoke --seed 42
```

---

## RQ1 + RQ2.1 — primary shared root (temporal)

One `run_experiments.py` call gives, per seed: majority, local-only,
centralized, FedProx(μ=0.01, shared head), FedAvg(μ=0, shared head). This
single root answers RQ1 (the privacy-cost gap) **and** RQ2.1 (FedProx vs
FedAvg).

```powershell
python run_experiments.py `
    --seeds 42-44 --results-root experiments_temporal_shared `
    --data-dir data_to_train_on `
    --model-name microsoft/codebert-base --max-length 256 `
    --rounds 60 --local-epochs 1 --batch-size 64 `
    --lr 3e-5 --warmstart-lr 3e-5 --warmstart-epochs 10 `
    --prox-mu 1e-2 --split-mode temporal --head-type corn `
    --checkpoint-every 5 --resume
```

Stats (run per metric; MAE and κ are primary, macro-F1 secondary):

```powershell
python compute_statistics.py --experiments-root experiments_temporal_shared --metric mae
python compute_statistics.py --experiments-root experiments_temporal_shared --metric cohen_kappa
python compute_statistics.py --experiments-root experiments_temporal_shared --metric macro_f1
```

Pairs FedProx vs {Local-only, Centralized, FedAvg} (Wilcoxon), Friedman across
all conditions, Nemenyi post-hoc, Vargha-Delaney Â. Communication cost for the
method section is in each run's `results/communication_cost.json`.

---

## RQ3 — personalized root (temporal)

Same 10 seeds, personalized per-client heads (only LoRA-B + embeddings
federated). `--skip-fedavg` (no FedAvg+P-head needed) and `--skip-baselines`
(local-only + centralized are identical to the shared root — personalization
only changes federated aggregation).

```powershell
python run_experiments.py `
    --seeds 42-44 --results-root experiments_temporal_personalized `
    --data-dir data_to_train_on `
    --model-name microsoft/codebert-base --max-length 256 `
    --rounds 60 --local-epochs 1 --batch-size 64 `
    --lr 3e-5 --warmstart-lr 3e-5 --warmstart-epochs 10 `
    --prox-mu 1e-2 --split-mode temporal --head-type corn `
    --personalized-head --generic-head --skip-fedavg --skip-baselines `
    --checkpoint-every 5 --resume
```

RQ3 = personalized federated (this root) vs shared federated
(`experiments_temporal_shared`), per-project, seed-matched. Personalized mode
has **no pooled "global" row** by design — report per-project + mean/median.
`compute_statistics.py` pairs *within* one root, so this cross-root comparison
uses the small helper in the Analysis section below.

---

## RQ2.2 — μ sensitivity sweep (3 seeds)

μ=0 (FedAvg) and μ=0.01 already exist in the shared root. Add the
two ends at the same 3 seeds, each in its **own root** (the runner names the federated
dir `fedprox` regardless of μ, so separate roots avoid collisions).

```powershell
python run_experiments.py `
    --seeds 42-44 --results-root experiments_temporal_mu0.001 `
    --data-dir data_to_train_on `
    --model-name microsoft/codebert-base --max-length 256 `
    --rounds 60 --local-epochs 1 --batch-size 64 `
    --lr 3e-5 --warmstart-lr 3e-5 --warmstart-epochs 10 `
    --prox-mu 0.001 --split-mode temporal --head-type corn `
    --skip-fedavg --skip-baselines --checkpoint-every 5 --resume

python run_experiments.py `
    --seeds 42-44 --results-root experiments_temporal_mu0.1 `
    --data-dir data_to_train_on `
    --model-name microsoft/codebert-base --max-length 256 `
    --rounds 60 --local-epochs 1 --batch-size 64 `
    --lr 3e-5 --warmstart-lr 3e-5 --warmstart-epochs 10 `
    --prox-mu 0.1 --split-mode temporal --head-type corn `
    --skip-fedavg --skip-baselines --checkpoint-every 5 --resume
```

Report μ ∈ {0, 0.001, 0.01, 0.1} as mean ± sd per metric across the common 5
common 3 seeds, plus a Friedman-across-μ. That cross-root μ aggregation is a
one-off (compute_statistics doesn't compare μ across roots) — flagged as a small
helper in Analysis.

---

## RQ4 — leave-one-project-out onboarding (Hyperledger_Sawtooth)

Sawtooth (948 rows, mid-sized, high description-missingness → a *hard*
onboarding case). Train two artifacts with it held out entirely, then adapt
head-only on its earliest {0,10,25,50,100} issues and test on its latest split.
The crossover between the generic/shared cold start and personalized adaptation
is the deliverable. LOPO adaptation is temporal by construction (earliest →
adapt, latest → test), so train the artifacts temporal.

**Step 1 — train both conditions, Sawtooth held out** (seed 42; add 43, 44 for a band):

```powershell
python train_federated_dl.py `
    --data-dir data_to_train_on --model-name microsoft/codebert-base --max-length 256 `
    --rounds 60 --local-epochs 1 --batch-size 64 --lr 3e-5 --warmstart-lr 3e-5 `
    --prox-mu 1e-2 --split-mode temporal --head-type corn --personalized-head --generic-head `
    --holdout-project Hyperledger_Sawtooth --checkpoint-every 5 --resume `
    --save-dir artifacts_lopo_personalized_s42 --seed 42

python train_federated_dl.py `
    --data-dir data_to_train_on --model-name microsoft/codebert-base --max-length 256 `
    --rounds 60 --local-epochs 1 --batch-size 64 --lr 3e-5 --warmstart-lr 3e-5 `
    --prox-mu 1e-2 --split-mode temporal --head-type corn `
    --holdout-project Hyperledger_Sawtooth --checkpoint-every 5 --resume `
    --save-dir artifacts_lopo_shared_s42 --seed 42
```

**Step 2 — head-only onboarding over budgets:**

```powershell
python run_lopo.py `
    --artifact-dir artifacts_lopo_personalized_s42/federated --data-dir data_to_train_on `
    --holdout-project Hyperledger_Sawtooth --head-init generic --out-dir lopo_results --seed 42

python run_lopo.py `
    --artifact-dir artifacts_lopo_shared_s42/federated --data-dir data_to_train_on `
    --holdout-project Hyperledger_Sawtooth --out-dir lopo_results --seed 42
```

**Step 3 — tidy crossover CSV:**

```powershell
python compare_lopo.py --results-dir lopo_results --out lopo_results/lopo_comparison.csv
```

---

## Robustness — random split (3 seeds)

Mirrors of the two temporal primary roots, `--split-mode random`, seeds 42–44.
Supports RQ1/RQ2.1/RQ3 as a robustness appendix.

```powershell
python run_experiments.py `
    --seeds 42-44 --results-root experiments_random_shared `
    --data-dir data_to_train_on `
    --model-name microsoft/codebert-base --max-length 256 `
    --rounds 60 --local-epochs 1 --batch-size 64 `
    --lr 3e-5 --warmstart-lr 3e-5 --warmstart-epochs 10 `
    --prox-mu 1e-2 --split-mode random --head-type corn `
    --checkpoint-every 5 --resume

python run_experiments.py `
    --seeds 42-44 --results-root experiments_random_personalized `
    --data-dir data_to_train_on `
    --model-name microsoft/codebert-base --max-length 256 `
    --rounds 60 --local-epochs 1 --batch-size 64 `
    --lr 3e-5 --warmstart-lr 3e-5 --warmstart-epochs 10 `
    --prox-mu 1e-2 --split-mode random --head-type corn `
    --personalized-head --generic-head --skip-fedavg --skip-baselines `
    --checkpoint-every 5 --resume
```

---

## RQ5 — SOTA positioning (classic baselines run; deep SOTA cited)

Two halves, deliberately treated differently:

**Run here (no extra command).** The classic within-project comparators — median-SP
and TF-IDF+LinearSVM — are computed by every root that doesn't pass
`--skip-baselines`, on the *same* split as the deep conditions, and land as
`results/{median,tfidf_svm}_per_project.json`. `compute_statistics.py` picks them
up automatically as ordinary conditions, so they enter the Wilcoxon/Friedman/Nemenyi
tests alongside FedProx. These are what make "competitive" testable: majority-class
alone is too weak a comparator given Tawosi et al. (2023).

They cost ~2 min of CPU and run **before** any GPU training, so they are already on
disk even if a run is interrupted.

**Cited, not run.** Published deep models (Deep-SE, GPT2SP, EGPT-SPE, Llama3SP) go in
a citation table on the TAWOS benchmark, with explicit caveats about split and
label-set differences (we filter to `{1,2,3,5,8}`; deep SOTA regresses over all SP
values). Lives in related-work/discussion, **not** statistically tested.

**Reporting caveat (important).** These baselines are per-project models and report
**no pooled "global" row** — by design. A pooled metric rewards a model for encoding
project identity: the constant median predictor pools to quadratic κ ≈ 0.50 while
scoring exactly 0.00 in all 18 projects (pooled macro-F1 is likewise inflated, ~0.30
vs ~0.10 per-project). Report per-project + mean/median only. Do not quote a pooled
κ for these in the thesis.

---

## Analysis / helpers

- **Per-root stats** (RQ1, RQ2.1): `compute_statistics.py --experiments-root <root> --metric {mae,cohen_kappa,macro_f1}`.
- **RQ3 cross-root** (shared vs personalized federated): small one-off — load
  `experiments_temporal_shared/seed_*/fedprox/results/federated_per_project.json`
  vs `experiments_temporal_personalized/seed_*/fedprox/results/federated_per_project.json`,
  pair per (seed, project), Wilcoxon + Â. **Needs a ~30-line helper** (not yet in repo).
- **RQ2.2 μ sweep** (Friedman across μ): gather `fedprox` per-project from the
  shared root (μ=0.01), its `fedavg` (μ=0), and the two μ roots, on the common
  seeds 42–46. **Same ~30-line helper**, μ as the grouping factor.
- **RQ4**: `compare_lopo.py` → `lopo_comparison.csv` (LONG: holdout, condition,
  budget, metric, value) → crossover plot.

## Execution order (compute-aware)

1. Stage 0 smoke (minutes) — after any code change.
2. `experiments_temporal_shared` (RQ1 + RQ2.1) — **run first**: produces the
   baselines everything else reuses.
3. `experiments_temporal_personalized` (RQ3) — `--skip-baselines` reuses step 2.
4. RQ4 LOPO artifacts + onboarding (independent; can run in parallel sessions).
5. RQ2.2 μ sweep (5 seeds × 2 μ).
6. Random-split robustness roots (lowest priority).
7. Write the two ~30-line helpers (RQ3 cross-root, RQ2.2 μ) and the RQ5 table.

**Compute note:** 3 seeds + batch 64 puts the full plan (temporal primary +
random robustness + μ sweep + LOPO) at roughly ~2,000 round-equivalents ≈
**~25–35 GPU-hours**. At 25 GPU-hours/week (5 days × 5 hours) that is ~1–1.5
weeks of machine time — inside the 2-week target with buffer for reruns. The
60-round val history (`results/federated_round_history.json`) also gives a clean
convergence-curve figure (plateau ≈ round 60). If anything slips, shorten the
*random robustness* roots first; keep the temporal primary roots and LOPO.

**Daily sessions:** Ctrl-C anytime; re-run the exact same command next day.
Warm-start is cached (not retrained), finished seed/condition pairs are skipped,
an interrupted run continues from its last checkpointed round. Add `--dry-run`
to any `run_experiments.py` line to preview, `--force` to redo.
