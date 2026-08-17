# Experiment Run Plan — FedSP-PEFT / FedSP-PEFT-P

Maps every run to one of the three research questions in
[RQs.md](RQs.md) — **that file is the source of truth for the RQs; this one is
the source of truth for what to run.** Commands are PowerShell (backtick
line-continuation); run from the project root with the venv active, or replace
`python` with `& "d:/Federated Learning using DL/.venv/Scripts/python.exe"`.

## The three RQs

- **RQ1** Feasibility & cost of privacy — federated vs centralized pooling vs
  per-project local training
- **RQ2** Personalization — per-project prediction head vs a fully shared model
- **RQ3** Parameter efficiency — federating LoRA adapters only vs fully
  fine-tuning and federating the whole encoder, in **quality and communication cost**

Everything else that gets run (FedProx vs FedAvg, the μ sweep, the classic
baselines, the onboarding case study) is a **supporting result**, not an RQ.
They still go in the thesis; they just don't carry an RQ number.

## Fixed method defaults (constant across every run)

| Setting | Value | Why fixed |
|---|---|---|
| Encoder | `microsoft/codebert-base` | RoBERTa arch → default `query value` LoRA targets; the encoder the good results used |
| Head | `--head-type corn` | CORN ordinal head is the thesis default (not an RQ) |
| PEFT | LoRA on, FFA-LoRA on (A frozen) — **except the RQ3 full-fine-tuning arm** | freezing A makes `avg(B·A) = avg(B)·A` exact. LoRA vs full FT is RQ3, so it is a *variable* there and a default everywhere else |
| Warm-start | on (lsstcorp) | cached per run; reused on `--resume` |
| max-length | 256 (fallback 128 if VRAM-tight) | p90 token length ≈ 164 |
| rounds / local-epochs | 60 / 1 | measured convergence plateau; best-on-val still selects within that |
| lr / warmstart-lr | 3e-5 / 3e-5 | also a conventional full-fine-tuning rate, so RQ3 is not rigged against the full-FT arm |
| batch-size | 64 | proven to fit 24 GB with LoRA — see the RQ3 VRAM note |
| Seeds | 42–44 (3 seeds, all roots) | per-project paired tests carry significance (n=18 projects); 3 seeds give variance bands. Reduced seed count noted as a thesis limitation |
| Checkpointing | `--checkpoint-every 5 --resume` | interrupt-safe daily sessions |

**Splits:** temporal = **primary** (deployment-realistic; reported results).
random = **robustness** (kept, 3 seeds). Every primary root is run once temporal,
then once random for the robustness appendix.

## RQ → root mapping (overview)

| RQ | Root(s) | Seeds | Conditions produced |
|---|---|---|---|
| RQ1 | `experiments_temporal_shared` | 42–44 | majority, median-SP, TF-IDF+SVM, local-only, centralized, FedProx(0.01), FedAvg(0) |
| RQ2 | `experiments_temporal_personalized` **vs** the shared root | 42–44 | FedProx(0.01) + P-head + generic head |
| RQ3 | `experiments_temporal_fullft` **vs** the shared root | 42–44 | FedProx(0.01) with `--no-lora`, plus `communication_cost.json` from both |
| supporting | shared root (FedAvg is free); `experiments_temporal_mu0.001`, `_mu0.1` | 42–44 | FedProx vs FedAvg; μ sensitivity |
| case study | `artifacts_lopo_*` + `lopo_results` | 42 | LOPO onboarding, Sawtooth held out |
| robustness | `experiments_random_shared`, `_personalized` | 42–44 | mirrors of the temporal roots |

RQ2 and RQ3 are both **cross-root** comparisons against the same shared root, so
one analysis helper serves both (see Analysis).

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

## RQ1 — primary shared root (temporal)

One `run_experiments.py` call gives, per seed: majority, median-SP, TF-IDF+SVM,
local-only, centralized, FedProx(μ=0.01, shared head), FedAvg(μ=0). This single
root answers RQ1 **and** produces the FedProx-vs-FedAvg supporting result for
free, **and** is the comparison arm for both RQ2 and RQ3. Run it first.

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

Pairs FedProx vs {Median, TF-IDF+SVM, Local-only, Centralized, FedAvg} with
Wilcoxon, Friedman across all conditions, Nemenyi post-hoc, Vargha-Delaney Â.

---

## RQ2 — personalized root (temporal)

Same 3 seeds, personalized per-client heads (only LoRA-B + embeddings
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

RQ2 = personalized federated (this root) vs shared federated
(`experiments_temporal_shared`), per-project, seed-matched. Personalized mode has
**no pooled "global" row** by design — report per-project + mean/median.
`compute_statistics.py` pairs *within* one root, so this uses the cross-root
helper in Analysis.

---

## RQ3 — full fine-tuning arm (temporal)

The comparison arm for RQ3: identical in every respect except `--no-lora`, which
leaves every encoder parameter trainable (`freeze_encoder` defaults to `False`,
so the `elif freeze_encoder` branch in `fl/model.py` is skipped). Federated only —
RQ3 is about the federated method, and the baselines are already on disk from
the shared root.

```powershell
python run_experiments.py `
    --seeds 42-44 --results-root experiments_temporal_fullft `
    --data-dir data_to_train_on `
    --model-name microsoft/codebert-base --max-length 256 `
    --rounds 60 --local-epochs 1 --batch-size 64 `
    --lr 3e-5 --warmstart-lr 3e-5 --warmstart-epochs 10 `
    --prox-mu 1e-2 --split-mode temporal --head-type corn `
    --no-lora `
    --skip-fedavg --skip-baselines --checkpoint-every 5 --resume
```

**The communication half of RQ3 needs no run at all.**
`compute_communication_cost()` writes `results/communication_cost.json` in every
run from the probe model's trainable/total parameter counts. The LoRA root's
file gives per-client per-round upload bytes and the reduction factor; the
full-FT root's file is the denominator (trainable ≈ total). Quote both — the
expensive part of RQ3 is only the **quality** half.

### ⚠ VRAM — check before committing 3 seeds

**Target hardware: RTX 3090 Ti, 24 GB.** Everything runs in fp32 (there is no
AMP/autocast, no TF32 setting and no gradient checkpointing in the codebase), so
the arithmetic is predictable. At batch 64 / seq 256, RoBERTa-base stores roughly
1.3 GB of activations per layer — attention scores 201 MB, softmax output 201 MB,
FFN intermediate 201 MB, GELU output 201 MB, plus projections and LayerNorms —
so ~16 GB across 12 layers. That term is the same for both arms.

| | LoRA arm | Full-FT arm |
|---|---|---|
| Stored activations | ~16 GB | ~16 GB |
| Params | 0.5 GB | 0.5 GB |
| Gradients | ~0 | 0.5 GB |
| AdamW moments | ~0 | 1.0 GB |
| CUDA context + fragmentation | ~1 GB | ~1 GB |
| **Total** | **~17.5 GB** | **~19 GB** |

So full fine-tuning should fit with ~5 GB spare. Two Windows caveats eat into
that: if the 3090 Ti also drives the display, the desktop and a browser hold
1–2 GB; and under WDDM, Windows **spills to shared system memory instead of
raising OOM** — the run doesn't crash, it just goes 5–20× slower. Watch "Shared
GPU memory" in Task Manager during the pre-check; non-zero means you're spilling.

Verify before committing 3 seeds:

```powershell
python train_federated_dl.py `
    --data-dir data_to_train_on `
    --model-name microsoft/codebert-base --max-length 256 `
    --rounds 2 --local-epochs 1 --warmstart-epochs 1 --batch-size 64 `
    --no-lora --head-type corn `
    --skip-centralized --skip-local-only --skip-classic-baselines `
    --save-dir artifacts_fullft_vramcheck --seed 42
```

This is a fit check, **not** a metrics run — 2 rounds says nothing about quality.

**If it OOMs (or spills), work down this ladder — best comparability first:**

1. **Gradient checkpointing on the full-FT arm only** —
   `self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})`
   in `fl/model.py`. Activations drop to roughly one layer's worth (~16 GB → ~2 GB)
   at the cost of ~30% more compute. It recomputes the forward pass during
   backward and is **numerically identical** (`preserve_rng_state` restores the
   dropout stream), so unlike every option below it can be applied to **one arm
   only** without confounding the comparison. Given this codebase's documented
   RNG sensitivity, confirm bit-reproducibility with a short A/B and
   `sanity_check_fl_randomness.py` before trusting it.
2. **`--max-length 128` on both arms** — halves activations, but changes which
   text the model sees (more truncation), so both arms must move together and the
   shared root must be re-run.
3. **Lower `--batch-size` on both arms** — last resort. There is no gradient
   accumulation in the client loop (`fl/client.py` steps once per batch), so this
   changes the optimization itself, and the shared root must be re-run. Never
   compare a batch-32 full-FT run against a batch-64 LoRA run.

**Speed, if the ~22 GPU-h is tight.** Two options, neither currently in the code,
both of which must be applied to **every** arm to keep runs comparable — and
therefore decided *before* the first real run, not after:
`torch.backends.cuda.matmul.allow_tf32 = True` (Ampere TF32; ~1.5–2× on matmuls,
no memory change, negligible accuracy cost), or bf16 `autocast` (roughly halves
activation memory and 2–3× faster; params stay fp32 so the FedProx proximal term
is unaffected).

**Also expect:** checkpoints grow from ~1 MB to ~500 MB (trainable-only now means
everything), so `--checkpoint-keep 2` plus latest and best is ~2 GB per seed,
~6 GB for the root. Budget the disk.

**Optional extra (not required for RQ3):** a centralized full-FT arm would show
whether the LoRA-vs-full-FT gap is federation-specific (client drift with 125M
free parameters) or exists centrally too. Costs another ~3 GPU-h; drop
`--skip-baselines` on a single seed if you want it.

---

## Supporting results (run and reported, but not RQs)

**FedProx vs FedAvg** — free from the shared root (both conditions are produced
per seed) and already covered by `compute_statistics.py`.

**μ sensitivity** — μ=0 (FedAvg) and μ=0.01 already exist in the shared root. Add
the two ends, each in its **own** root (the runner names the federated dir
`fedprox` regardless of μ, so separate roots avoid collisions).
**This is the cheapest thing to cut when compute is tight — cut it first.**

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

**Classic within-project comparators** — median-SP and TF-IDF+LinearSVM are
computed by every root that doesn't pass `--skip-baselines`, on the *same* split
as the deep conditions, and land as `results/{median,tfidf_svm}_per_project.json`.
`compute_statistics.py` picks them up automatically, so they enter the
Wilcoxon/Friedman/Nemenyi tests. They cost ~2 min of CPU and run **before** any
GPU training. These are what make "competitive" testable given Tawosi et al. (2023).

**Published deep SOTA** (Deep-SE, GPT2SP, EGPT-SPE, Llama3SP) — a citation table
on the TAWOS benchmark with explicit caveats about split and label-set
differences (we filter to `{1,2,3,5,8}`; deep SOTA regresses over all SP values).
Related-work/discussion, **not** statistically tested.

**Reporting caveat (important).** The classic baselines are per-project models and
report **no pooled "global" row** — by design. A pooled metric rewards a model for
encoding project identity: the constant median predictor pools to quadratic κ ≈ 0.50
while scoring exactly 0.00 in all 18 projects (pooled macro-F1 inflates the same
way, ~0.30 vs ~0.10 per-project). Report per-project + mean/median only.

---

## Case study — new-project onboarding (LOPO, not an RQ)

Held out entirely from training, then adapted head-only on its earliest issues
and tested on its latest split. **One holdout cannot support a general claim**, so
this is written up as a case study — do not phrase its result as "a new project
needs N issues". Seed 42 only; ~2 GPU-h.

Sawtooth (948 rows → 758 adaptation pool, 190 test) is a reasonable mid-sized
choice. Note its old justification was wrong: `preprocessing_report.json` reports
`desc_nan_pct = 0.0` for **all** 19 projects, so "high description-missingness"
is not why it was picked.

**Step 1 — train both conditions, Sawtooth held out:**

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

**Step 2 — head-only onboarding over budgets.** Extend past the default 100: the
head is the only thing training, so extra budgets cost minutes. Budgets clip to
pool size (`min(budget, len(pool))`), so 758 is Sawtooth's ceiling.

```powershell
python run_lopo.py `
    --artifact-dir artifacts_lopo_personalized_s42/federated --data-dir data_to_train_on `
    --holdout-project Hyperledger_Sawtooth --head-init generic `
    --budgets 0,10,25,50,100,200,400,758 --out-dir lopo_results --seed 42

python run_lopo.py `
    --artifact-dir artifacts_lopo_shared_s42/federated --data-dir data_to_train_on `
    --holdout-project Hyperledger_Sawtooth `
    --budgets 0,10,25,50,100,200,400,758 --out-dir lopo_results --seed 42
```

**Free extra arm:** `--head-init random` on the personalized artifact isolates
whether the server-side generic head earns its one-time transmission — same
representation, only the starting head differs. Minutes to run.

**Step 3 — tidy crossover CSV:**

```powershell
python compare_lopo.py --results-dir lopo_results --out lopo_results/lopo_comparison.csv
```

**Free ceiling line:** the holdout is an ordinary client in the shared and
personalized roots, and both sides select the same test rows — `fl/data.py`'s
temporal carve and `run_lopo.py` both take `max(1, int(round(n * 0.2)))` from the
tail of the same row order. So Sawtooth's row in
`experiments_temporal_shared/seed_42/fedprox/results/federated_per_project.json`
is a directly comparable "if it had been a member all along" reference.

---

## Robustness — random split (3 seeds)

Mirrors of the two temporal primary roots, `--split-mode random`, seeds 42–44.
Supports RQ1/RQ2 as a robustness appendix. Lowest priority.

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

## Analysis / helpers

- **RQ1** (and FedProx vs FedAvg): `compute_statistics.py --experiments-root experiments_temporal_shared --metric {mae,cohen_kappa,macro_f1}`.
- **RQ2 and RQ3 both need the same cross-root helper — write it early.** ~30 lines:
  load `<root_a>/seed_*/fedprox/results/federated_per_project.json` and the same
  from `<root_b>`, pair per (seed, project), Wilcoxon + Vargha-Delaney Â.
  - RQ2: `experiments_temporal_shared` vs `experiments_temporal_personalized`
  - RQ3: `experiments_temporal_shared` vs `experiments_temporal_fullft`
  It is now load-bearing for two of the three RQs, not a nice-to-have.
- **RQ3 communication table:** `results/communication_cost.json` from the shared
  root (LoRA) and the fullft root. No code needed.
- **μ sweep** (supporting): same cross-root helper, μ as the grouping factor,
  gathering the shared root's `fedprox` (μ=0.01), its `fedavg` (μ=0), and the two
  μ roots.
- **Case study:** `compare_lopo.py` → `lopo_comparison.csv` (LONG: holdout,
  condition, budget, metric, value) → crossover plot, plus the ceiling line above.

**Unit of analysis is per-project, always.** Pooled "global" entries are ignored on
load — pooling rewards a model for encoding project identity.

## Execution order (compute-aware)

1. Stage 0 smoke (minutes) — after any code change.
2. **Full-FT VRAM check** (minutes) — before committing to the RQ3 root.
3. `experiments_temporal_shared` (RQ1) — **run first**: it produces the baselines
   everything reuses and is the comparison arm for RQ2 and RQ3.
4. `experiments_temporal_personalized` (RQ2).
5. `experiments_temporal_fullft` (RQ3).
6. Write the cross-root helper (can be done while runs are going — it needs no GPU).
7. LOPO case study (independent; can run in a parallel session).
8. Random-split robustness roots.
9. μ sweep — **cut this first if compute runs short.**

**Compute budget.** At roughly 1 GPU-hour per 60-round artifact:

| Block | ~GPU-h |
|---|---|
| `experiments_temporal_shared` (fedprox + fedavg + centralized + local-only, 3 seeds) | ~12 |
| `experiments_temporal_personalized` (3 seeds) | ~3 |
| `experiments_temporal_fullft` (3 seeds, ~1.5–2× per round) | ~5 |
| LOPO case study (2 artifacts, seed 42) | ~2 |
| **Subtotal — everything the three RQs need** | **~22** |
| Random-split robustness | ~15 |
| μ sweep (cut first) | ~6 |

At 25 GPU-hours/week that puts the three RQs inside one week, with the
robustness and μ blocks as stretch. If anything slips, cut μ, then the random
robustness roots; keep the three temporal primary roots and the case study.

**Daily sessions:** Ctrl-C anytime; re-run the exact same command next day.
Warm-start is cached (not retrained), finished seed/condition pairs are skipped,
an interrupted run continues from its last checkpointed round. Add `--dry-run` to
any `run_experiments.py` line to preview, `--force` to redo.
