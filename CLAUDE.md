# CLAUDE.md — Federated Story Point Estimation (Bachelor Thesis)

This file gives you full context for every Claude Code session. Read it before touching any file.

---

## What this project is

A **Bachelor thesis** on **Federated Learning for Story Point Estimation (SPE)**. The goal is to predict agile story point values from JIRA issue text using a federated learning setup where each client is one software project — so raw issue data never leaves the project. This is a novel intersection: federated learning has not been applied to SPE before (re-verified via literature search, July 2026), which is the core novelty claim.

**Dataset:** TAWOS — a large collection of JIRA issues from real open-source projects, pre-cleaned and split into one CSV/XLSX file per project.

**Task framing:** 5-class **ordinal** classification over the Fibonacci story point values `{1, 2, 3, 5, 8}`. Issues with other SP values are filtered out and logged. This is a deliberate design choice: story points are ordinal categories chosen from a fixed deck, not continuous quantities. The ordinal structure is exploited via a CORN loss head (see below); plain CrossEntropy remains as an ablation arm. A bridge metric (MAE on predicted class values) is reported as a **primary** metric to keep results comparable to the regression-based SPE literature.

**Important framing note (accuracy expectations):** Tawosi et al. (2023) showed that even sophisticated deep SPE models barely beat naive baselines — absolute accuracy on this task is low for *everyone*. The thesis contribution is NOT absolute accuracy; it is (a) the local-only → federated → centralized **gap analysis** under a privacy constraint, (b) the personalization finding under project-specific label semantics, and (c) communication cost. Do not panic-refactor because macro-F1 looks low; check the centralized ceiling and the confusion-matrix adjacency structure first.

**Second framing note (project-specific semantics):** Story points are unitless and team-calibrated — an "8" in one project can be a "3" in another. Cross-project SPE transfer is known to be weak (GPT2SP / FastText-SVM only predict accurately within-project). This is simultaneously (a) the likely explanation for weak shared-head federated accuracy and (b) the motivation for the personalized-head variant. Treat non-IID label distributions and project-specific SP semantics as the *central challenge the design responds to*, not as an inconvenience.

---

## ⚠️ CURRENT WORK — running the experiments (branch `version-2`)

**Implementation is done.** Gaps #1–#14 are all implemented. The build-out that used to
be tracked here as Tasks 1–4 (CORN head, personalized head, checkpoint/resume, LOPO) is
complete and tagged `task-1-done` … `task-4-done`; the per-step checklist and its
verification notes live in the git log of `feat/personalization-corn-checkpointing`. The
classic within-project baselines (TF-IDF+SVM, median-SP) landed on top of that.

**The work now is running and analysing experiments, not writing features.**

**Branch:** work happens on **`version-2`**. It branches off
`feat/personalization-corn-checkpointing` (NOT off `main`), so it carries gaps #1–#14
plus the classic baselines. `main` is stale — it is the known-good state of gaps #1–#10
only. Whether/when to merge `version-2` into `main` is an open decision: ask, don't
assume.

**[RUN_PLAN.md](RUN_PLAN.md) is the source of truth for what to run.** It maps every run
to one of the five locked RQs, fixes the method defaults that stay constant across runs
(codebert-base, CORN head, FFA-LoRA, temporal primary + random robustness, seeds 42–44),
and gives the compute-aware execution order. If RUN_PLAN.md and this file disagree about
a *run*, RUN_PLAN.md wins and this file must be corrected. This file remains the source
of truth for *architecture, data, and framing*.

### Session protocol — every Claude Code session MUST follow this

1. **Orient first:** `git status`, `git branch --show-current`, `git log --oneline -8`.
   Expect `version-2`. Never work with a dirty tree from a previous session without
   understanding what's in it.
2. **One concern = one commit.** Do not batch unrelated changes. Small commits are the
   rollback mechanism on a low-resource machine.
3. **Verify before committing.** The default check is the smoke run below. Never commit
   red; fix first.
4. **Scope discipline:** if you notice an unrelated bug mid-task, note it under "Parking
   lot" below; do not fix it inside the current commit unless it blocks the work.
5. **Never interpret bert-tiny metrics.** Only codebert-base runs support conclusions.
   Before concluding a number looks "too low", re-read the accuracy-expectations note
   above and check the centralized ceiling + confusion-matrix adjacency.

**Default smoke run (fast, bert-tiny, ~minutes — plumbing only):**
```
python train_federated_dl.py --data-dir data_to_train_on \
  --model-name prajjwal1/bert-tiny --max-length 64 \
  --rounds 2 --local-epochs 1 --warmstart-epochs 1 \
  --skip-centralized --skip-local-only
```
The encoder flag is `--model-name` (there is no `--encoder`). Adapt per change: add
`--head-type corn`, `--personalized-head --generic-head`, `--checkpoint-every 1`,
`--skip-classic-baselines`. A smoke run verifies plumbing only — never interpret its
metrics. Note that at 1–2 rounds bert-tiny predicts essentially one class, so FedProx and
FedAvg produce *identical* metrics; that is a degeneracy artifact, not a bug (the
proximal term is live — verified at the weight level).

### Parking lot (unrelated issues noticed mid-task — do not fix in-step)

- (empty)

---

## Architecture — FedSP-PEFT

The framework name is **FedSP-PEFT** (Parameter-Efficient Federated Story Point Estimation). The personalized variant is referred to as **FedSP-PEFT-P** in the thesis.

### Core design

```
Each client (= one TAWOS project):
  Local private data (never transmitted)
       ↓
  Frozen encoder (CodeBERT for final runs / bert-tiny for dev)
       ↓
  LoRA adapters — trainable (FFA-LoRA: A frozen, B trained & transmitted)
  Categorical embeddings (issue type, priority) — trainable, transmitted (default; local-embeddings is an ablation)
  Head — trainable:
      shared-head mode:        transmitted & aggregated (original design)
      personalized-head mode:  stays LOCAL, never aggregated (FedPer/FedRep-style)
  Head type:
      "ce":    5-logit softmax head + class-weighted CrossEntropyLoss (baseline/ablation)
      "corn":  4-logit CORN ordinal head + corn_loss (default for final experiments)
       ↓
  Local loss (CE or CORN)
  + FedProx proximal term: (μ/2) · ‖w − w_global‖²
      — computed over SHARED (aggregatable) trainable params ONLY.
        In personalized-head mode the local head has no global reference point;
        pulling it toward anything global is meaningless and undoes personalization.

Server:
  Weighted average of AGGREGATABLE params only (weight = client's actual sampled examples)
  Frozen backbone and frozen LoRA-A matrices are excluded from aggregation
  In personalized-head mode, head params are additionally excluded by name
  Optional: server may keep a "generic head" (one-way average of client heads, used ONLY
  as initialization for new/external clients; never pushed back to participants)
```

### Why FFA-LoRA specifically

Standard federated LoRA averages A and B matrices separately, but the actual weight update is the product B·A. Because avg(Bᵢ·Aᵢ) ≠ avg(Bᵢ)·avg(Aᵢ), naive averaging introduces an aggregation error term.

**FFA-LoRA** (Sun et al., ICLR 2024) fixes this by fixing A as a shared random projection at init and training only B. With A identical across all clients, avg(B·A) = avg(B)·A — the aggregation is mathematically exact. This is not just an engineering trick; it is a principled contribution and must be cited and explained in the thesis methodology section. Note: the exactness argument holds unchanged in personalized-head mode (only B matrices + embeddings are averaged).

The flag in code: `ffa_lora=True` in `FLConfig`. The server derives `aggregatable_keys` from `requires_grad`, which automatically excludes frozen A matrices and the frozen backbone. Personalized-head mode further removes head parameters (by name prefix) from `aggregatable_keys`.

### Why FedProx over plain FedAvg

TAWOS projects have wildly different label distributions — some cluster on SP 1–3, others spread out. This is textbook non-IID heterogeneity. FedProx's proximal term pulls each client's update toward the global model, reducing client drift. The μ hyperparameter controls the strength of this pull. Setting `prox_mu=0.0` degrades to FedAvg, which is an explicit experimental condition. **Scope:** the proximal term applies to shared/aggregatable params only (see diagram above).

### Why personalized heads (FedPer/FedRep)

Shared LoRA-B matrices learn a common *representation* ("what a complex issue looks like linguistically"); each local head learns the project-specific *calibration* ("what this team calls an 8"). This directly targets the project-specific SP semantics problem. Cite Arivazhagan et al. (2019, FedPer) and Collins et al. (2021, FedRep).

Consequences for evaluation (all three matter — do not skip):
- There is no single "global model" in personalized mode. Per-round validation = each client evaluates shared-representation + its own head on its own val split; the server tracks mean and weighted val macro-F1 across clients per round. "Best-on-val" selection = best shared state by (weighted) average client val performance, with the per-client heads snapshotted at that round.
- The pooled "global" test entry is meaningless with per-client heads. Report per-project metrics and their mean/median. (This matches how compute_statistics.py already works — per-project paired tests.)
- New/external clients: see the onboarding protocol below.

### Why CORN loss (ordinal head)

Plain CE treats predicting 8-when-truth-is-3 the same as 2-when-truth-is-3. CORN (Shi et al., 2021) decomposes K classes into K−1 conditional binary threshold questions (SP>1? SP>2? SP>3? SP>5?), trained with BCE per threshold; distant misses are penalized at more thresholds, so the loss scales with ordinal distance — without leaving classification space or needing continuous-prediction rounding hacks. Implementation via `coral-pytorch` (`corn_loss`, `corn_label_from_logits`). Head = `nn.Linear(hidden_dim, num_classes - 1)`.

Config: `head_type` in `FLConfig` ∈ {"ce", "corn"}; CLI `--head-type`. CE remains the ablation baseline. Note: inverse-frequency class weights apply to the CE arm; CORN handles the (mild, <3:1) imbalance largely through its threshold structure.

### New-client onboarding protocol (external / non-participating projects)

A shared head does NOT solve the cold-start problem — its zero-shot predictions average away exactly the per-project calibration that matters. The protocol instead:

1. New client downloads shared parts: frozen backbone + frozen A + aggregated B + embeddings.
2. Head initialized randomly, or from the optional server-side "generic head".
3. Head-only fine-tuning on the client's own small labeled history (head is ~hidden_dim×4 params; trains in seconds on CPU; no federation round; no data sharing).

This mirrors how a new human team calibrates planning poker over its first sprints — use this analogy in the discussion chapter. Evaluated via the leave-one-project-out experiment (gap #14).

### Encoder choice

- **Final experiments — LOCKED to `microsoft/codebert-base`** (RUN_PLAN.md fixed default; issues contain stack traces / class names, and it is the encoder the good results used). RoBERTa-architecture → LoRA target modules are `query`, `value`, which is already the CLI default. Held constant across every run: the encoder is **not** an RQ.
- **Development / smoke:** `prajjwal1/bert-tiny` (2 layers, 128-dim, fast iteration). **bert-tiny numbers are plumbing checks only — never interpret accuracy from them.** At 1–2 rounds it collapses to predicting ~one class, which makes distinct conditions (e.g. FedProx vs FedAvg) produce identical metrics.
- **Alternatives, only if the locked choice has to change:** `distilroberta-base` (target modules `query`, `value`) or `distilbert-base-uncased` — note DistilBERT's target modules are `q_lin`, `v_lin`, **not** `query`/`value`, so `--lora-target-modules` must change with it.
- The flag is `--model-name`. There is no `--encoder`.

### Input features

`text = title + " [SEP] " + description` → tokenized at max 128 tokens (verify distribution first)
`type_id` → categorical embedding (issue type, e.g. Bug / Story / Task)
`priority_id` → categorical embedding (e.g. High / Medium / Low / Unknown)

All three are fused before the head. The categorical embeddings are a small contribution: they capture project metadata that plain text-only models ignore. Default: embeddings are shared/aggregated (export normalizes Priority to a canonical vocabulary, so sharing is defensible); shared-vs-local embeddings is an optional micro-ablation — do not block on it.

---

## Data & text preprocessing

### The actual dataset (profiled June 2026)

19 project CSVs, 42,002 issues total. Schema: `Issue_Key, Title, Description, Story_Point, Type, Priority, Creation_Date`.

- **Already filtered to the 5 classes** — 100% of rows have SP ∈ {1,2,3,5,8}. Distribution: 1 → 29.9%, 2 → 25.5%, 3 → 17.6%, 5 → 15.6%, 8 → 11.3%. **Note: this is <3:1 max/min — mild imbalance, not severe. Do not attribute weak accuracy to skew without evidence; check the centralized ceiling and confusion-matrix adjacency first.**
- `Creation_Date` is 100% populated → temporal split is feasible (gap #9).
- Description is NaN in 0–42.7% of rows depending on project (worst: Hyperledger_Sawtooth). **Keep these rows as title-only; never drop for missing description.**
- Largest project: Lsstcorp_Data_management (10,052) — this is the warm-start project, excluded from the FL pool.
- No duplicate Issue_Keys; no rows with both title and description empty.

### Measured noise (share of descriptions affected)

| Artifact | Share | Handling |
|---|---|---|
| Literal quote-wrapping `"..."` (export artifact) | ~90% of descriptions, 100% of titles | Strip wrapping quotes FIRST, before any other rule |
| URLs | 18.2% | Replace with `[URL]` token (signal: references external context) |
| HTML tags | 12.9% | `html.unescape()` then strip tags |
| Issue refs `ABC-123` | 11.8% | Replace with `[ISSUE_REF]` token |
| `{code}...{code}` blocks | 11.5% | Replace whole block with `[CODE]` token (don't delete silently) |
| Wiki headings `h1.`–`h6.` | 5.4% | Strip marker, keep heading text |
| `{noformat}` blocks | 3.0% | Replace with `[CODE]` token |
| Other Jira macros `{color}` `{panel}` etc. | 1.7% | Strip macro braces, keep inner text where sensible |
| Wiki tables `\|\|` | 0.4% | Replace pipes with spaces |

### Cleaning rules — what we deliberately do NOT do (thesis must state this)

- **No lowercasing** — encoder is cased; `NullPointerException` carries signal.
- **No stopword removal, no stemming/lemmatization** — transformers need full natural language; that advice is for TF-IDF-era models.
- **No removal of numbers or normal punctuation** — "timeout from 30s to 300s" is effort signal.
- Cleaning happens **once at export time** in `export_issues.py` (DB → CSV). The training pipeline (`fl/data.py`) consumes the cleaned CSVs as-is and must NOT re-clean — it should only *validate* (e.g. assert no `{code` remnants, no wrapping quotes) and fail loudly if handed raw data.
- The model-dependent title/description **join stays in the pipeline** (`[SEP]` for BERT-family, `</s>` for RoBERTa-family — depends on the loaded tokenizer). The export writes cleaned `Title` and `Description` as separate columns for exactly this reason.

### Order of operations (implemented in `export_issues.py`)

0. **Decode legacy export quoting** (`decode_export_quoting`): strip one wrapping-quote layer and collapse doubled interior quotes (`""` → `"`), applied only when the field shows evidence of wrapping. Raw-input-only step — never re-applied to cleaned text. No-op on fresh DB exports.
1. HTML: unescape entities, then strip tags
2. Jira markup: `{code}`/`{noformat}` (incl. unclosed openers) → `[CODE]`; known macros (`{color}`, `{panel}`, …) stripped by name — NOT a generic `{...}` regex, so inline JSON survives; headings `h1.`–`h6.` stripped; line-leading bullet markers (`*`, `**`, `#`, `-`) removed BEFORE bold unwrapping (otherwise collapsed bullets pair into false `*bold*` matches); `*bold*` unwrapped; `_italic_` left alone (snake_case risk); `||` table markers removed
3. URLs → `[URL]`; issue keys `ABC-123` → `[ISSUE_REF]`
4. Whitespace normalization (newlines/tabs → space, collapse runs)
5. Length floor: post-cleaning combined text < 10 chars → drop row, log per project
6. Categorical normalization: Priority → {Highest, High, Medium, Low, Lowest, Unknown} via the documented `PRIORITY_BASE_MAP` ("Major - P3" → High, Critical → Highest, NaN → Unknown); Type stripped, NaN → Unknown; Story_Point cast to int
7. Per-project + global stats written to `data_to_train_on/preprocessing_report.json` — the thesis Data chapter cites this file verbatim

**Validated against the real corpus (June 2026):** 42,002 → 41,995 rows (7 dropped by length floor, 0.02%); substitutions: 8,862 code blocks, 11,480 URLs, 6,513 issue refs. If a future export deviates wildly from these numbers, suspect a regex regression.

### Known residual per-row artifacts (post-export, do not "fix" by re-cleaning)

A handful of rows survive export with patterns that *look* like uncleaned artifacts but are either correct decodes or rare typo'd source markup. `validate_cleaned_dataframe` tolerates these via a >1% threshold per check rather than zero-tolerance, so a genuinely un-exported file (where these patterns are systemic, per the table above) still fails loudly.

- `Appcelerator_Studio.csv`, title *"Installed Node.js is an unsupported version"* — the title's real content itself starts/ends with a quote character (quoting an error message verbatim). `decode_export_quoting` correctly strips the legacy CSV-escaping wrapper, leaving these genuine quote marks behind. This is correct output, not an artifact.
- `Spring_XD.csv`, one Description contains `{noformat)` (closing paren, not brace) — a typo in the original Jira text that `RE_NOFORMAT_ORPHAN`/`RE_NOFORMAT_BLOCK` (which require a closing `}`) don't match. 1 of 2788 rows.

### Token length note

Combined title+description: p50 = 48 words, p90 = 164, p95 = 234. `max_length=128` truncates roughly the longest 10–15% of issues. Run a 128 vs 256 ablation on the big-VRAM machine before final results (the truncated long issues are disproportionately the complex/high-SP ones).

---

## File map

```
train_federated_dl.py   Main training script. Runs warmstart → local-only → centralized →
                        federated, then per-project evaluation and summary tables.
fl/config.py            FLConfig dataclass. Single source of truth for all hyperparameters
                        (incl. val_size, skip_local_only, skip_centralized; NEW:
                        head_type, personalized_head, checkpoint_every, resume,
                        checkpoint_keep).
fl/data.py              Data loading (multi-format), filtering, per-client stratified
                        train/val/test split (split_per_client + split_train_val).
fl/model.py             StoryPointClassifier — frozen encoder + LoRA + embeddings + head.
                        Head is CE (5 logits) or CORN (4 logits) per config.head_type.
fl/client.py            FederatedClient — local training with FedProx proximal term
                        (over shared/aggregatable params only). train_local() returns
                        ONLY trainable params (not the full state dict), keeping
                        per-client transfer small for larger encoders.
fl/server.py            FedProxServer — round orchestration. Aggregates client updates
                        incrementally (one client's tiny trainable-only state dict at a
                        time, never a list of full state dicts — bounds CPU RAM).
                        Personalized mode: head keys excluded from aggregation; per-round
                        val = mean/weighted across clients (own head + own val split);
                        best-on-val = best shared state + per-client head snapshots.
                        Calls fl/checkpoint.py hooks after each round.
fl/metrics.py           evaluate_classification(), run_prediction() — accuracy, macro-F1,
                        per-class F1, CM, MAE, quadratic-weighted Cohen's Kappa.
fl/checkpoint.py        Save/load/resume logic shared by centralized, local-only, and
                        federated paths (gap #13). See gap #13 for the contract.
fl/classic_baselines.py Classic within-project comparators for RQ5: per-project
                        TF-IDF+LinearSVM and a median-SP constant predictor, on the
                        SAME split as the deep conditions. Per-project entries only —
                        NO pooled "global" (pooling inflates per-project models; see
                        the reporting rule under External baselines). Runs before any
                        deep training; RNG-neutral. Toggle: --skip-classic-baselines.
predict_saved_model.py  Inference on saved artifacts.
export_issues.py        DB → CSV export WITH full preprocessing (text cleaning,
                        categorical normalization, length floor, stats report).
                        Cleaning functions are pure and importable for testing
                        (mysql.connector is lazily imported).
sanity_check_fl_randomness.py  Verifies client selection and epoch sampling are reproducible.
                        MUST also cover the resume path (selection sequence identical
                        whether run straight through or interrupted+resumed).
run_experiments.py      Multi-seed runner (gap #4). For each seed, runs train_federated_dl.py
                        for the FedProx (+baselines) condition and the FedAvg condition,
                        writing to <results-root>/seed_<N>/{fedprox,fedavg}/.
compute_statistics.py   Statistics script (gap #5). Loads per-project JSON results across
                        seeds and runs Wilcoxon signed-rank, Friedman, Nemenyi post-hoc, and
                        Vargha-Delaney A12/Cliff's delta; writes a LaTeX-ready summary table.
                        CONDITIONS covers Median, TF-IDF+SVM, Local-only, Centralized,
                        FedAvg, FedProx; absent files are skipped. Pairs per (seed, project).
run_lopo.py             Leave-one-project-out onboarding (gap #14). Loads a trained artifact
                        (shared, or personalized + generic/random head init), adapts head-only
                        over budgets {0,10,25,50,100} of the holdout's earliest issues,
                        evaluates on its test split -> results/lopo_<project>_<tag>.json.
compare_lopo.py         Aggregates lopo_*.json into a tidy LONG CSV for the crossover plot.
RUN_PLAN.md             THE source of truth for which experiments to run: RQ -> results-root
                        mapping, fixed method defaults, execution order, compute budget.
```

---

## Experimental design (what the thesis needs)

The thesis argument rests on comparing training conditions on the same data and model:

| Condition | Description | Status |
|---|---|---|
| Majority baseline | Always predict most frequent class | ✅ Implemented |
| Median-SP baseline | Per-project constant = class nearest the project's median train SP | ✅ Implemented (`fl/classic_baselines.py`) |
| TF-IDF + LinearSVM | Per-project TF-IDF + LinearSVC — the within-project simple-model comparator | ✅ Implemented (`fl/classic_baselines.py`) |
| Local-only | Each client trains on its own data only, no federation | ✅ Implemented |
| Centralized | Pooled data, privacy ignored — upper bound | ✅ Implemented |
| Federated (FedAvg) | `prox_mu=0`, no proximal term | ✅ Implemented (run with `--prox-mu 0`, auto-labeled "FedAvg") |
| Federated (FedProx) | `prox_mu=0.01`, proximal regularisation, shared head | ✅ Implemented (auto-labeled "FedProx (mu=X)") |
| Federated (FedProx + personalized head) | Head local per client; B + embeddings aggregated | ✅ Implemented (`--personalized-head`, auto-labeled "… + P-head") |

Local-only (gap #1) is now implemented: each client trains from the same warm-start checkpoint as federated, with `rounds * local_epochs` total local epochs (matching federated's per-client exposure) and no aggregation. The run output prints "<condition> Macro-F1 improvement vs local-only" automatically. The federated-vs-centralized gap measures the cost of privacy preservation. The personalized-vs-shared-head comparison (everything else identical) is the RQ4 experiment.

### External baselines for empirical comparison (RQ5)

**Why they exist:** majority-class alone is too weak a floor. Tawosi et al. (2023) showed simple models match deep ones on this task, so "FL is competitive" is meaningless without a *simple* comparator ("competitive with what?"). This is the weakness an empirical-SE examiner probes first.

**✅ Implemented and run** — `fl/classic_baselines.py`, computed by every root that doesn't pass `--skip-baselines` / `--skip-classic-baselines`, and picked up automatically by `compute_statistics.py` as ordinary conditions (so they enter Wilcoxon/Friedman/Nemenyi):

1. **TF-IDF + LinearSVM, per-project** — the canonical "simple model" from the SPE replication literature; trains within each project, so it's a within-project ceiling for the text signal. Writes `results/tfidf_svm_per_project.json`.
2. **Median-SP predictor, per-project** — the trivial MAE floor: the class nearest each project's median train SP. A deep model that can't beat it is a red flag. Writes `results/median_per_project.json`.

Both train on the SAME split as the deep conditions (`bundle.train_df`/`test_df`), so per-project pairing is identical. They run BEFORE any deep training (~2 min, CPU) and are verified RNG-neutral — federated results are bit-identical with and without them. FastText+SVM (Tawosi's specific comparator) is an optional future addition.

**⬜ Cited, not run** — reimplementation is not worth the time; compare on the same projects and state split/label-set differences honestly:

3. **Deep-SE (Choetkiertikul et al. 2018)** — LSTM+RHN, the canonical deep SPE baseline. The Tawosi et al. (2022/2023) replication publishes Deep-SE numbers per TAWOS project — CITE those.
4. **GPT2SP (Fu & Tantithamthavorn 2022)** — transformer regression SPE, within-project. Cite published numbers. Strongest within-project deep comparator; also supports the "prior SPE is within-project only" motivation for personalization (RQ3/RQ4).

**⚠️ Reporting rule — these have NO pooled "global" row, by design.** They are per-project models, so a pooled metric rewards them for merely encoding project identity. Measured on the real corpus: the constant median predictor scores quadratic κ **0.0000 in all 18 projects** (correct — it is exactly chance) but pools to **0.5006**; pooled macro-F1 inflates the same way (~0.10 → 0.3023). Quoting a pooled κ for these would put a trivial baseline next to the deep models on an inflated scale. Report per-project + mean/median only.

**Framing (keep honest):** these are *empirical-comparison* baselines, not FL conditions — they train centrally/per-project with no federation and no privacy constraint, so they contextualize the FL numbers rather than compete on privacy. The goal is NOT to beat them outright but to show the federated (cross-project, privacy-preserving) model lands in the same ballpark without sharing raw data — and that personalization (RQ4) closes the within-project gap. Deep-SE / GPT2SP / TF-IDF-SVM are all already in the "Key papers to cite" list.

### Ablations

**Promoted to an RQ — not an ablation any more:**
- `prox_mu` sensitivity 0.0 / 0.001 / 0.01 / 0.1 → **RQ2.2**, with its own roots in RUN_PLAN.md.

**Fixed method defaults — deliberately NOT ablated** (holding them constant is what makes the RQ comparisons clean):
- Encoder: locked to `microsoft/codebert-base`. bert-tiny is the smoke model, never a comparison arm.
- FFA-LoRA: on. Communication cost is reported descriptively, not as an RQ.
- Head: CORN.

**Still open, in rough priority order (all optional — the five RQs come first):**
- FFA-LoRA on vs off (`--no-ffa-lora`) — the aggregation-exactness argument is theoretical; showing it empirically is a bonus.
- Warm-start on vs off (`--run-no-warmstart-fl`, roughly doubles federated time)
- **Head/loss: CE vs CORN** (`--head-type ce`) — expect CORN to move MAE/Kappa more than macro-F1; that pattern is itself a finding (the model learns ordinal structure)
- **max_length 128 vs 256** (big-VRAM machine; RUN_PLAN locks 256)
- Clients per round: full participation vs 50% fraction (`--clients-per-round-fraction`)
- Local epochs: 1, 3, 5
- **Shared vs local categorical embeddings** (micro-ablation, only if time)

### Statistical testing ✅ Implemented (gaps #4/#5)

Run **3 seeds (42–44)** per condition via `python run_experiments.py --data-dir <...> --seeds 42-44 <other train_federated_dl.py flags>` — see [RUN_PLAN.md](RUN_PLAN.md) for the exact per-RQ commands. Per-project per-run results are saved to JSON automatically. Then `python compute_statistics.py --experiments-root <root> --metric mae` (also `cohen_kappa`, `macro_f1`) computes:
- **Wilcoxon signed-rank test** (paired, per (seed, project)): FedProx vs. each of Median, TF-IDF+SVM, Local-only, Centralized, FedAvg
- **Friedman test** across all conditions
- **Nemenyi post-hoc** for pairwise significance
- **Vargha-Delaney Â** or Cliff's delta as effect size (not just p-values — examiners in empirical SE require this)

**Seed count — deliberate change from the original 10.** RUN_PLAN.md locks **3 seeds (42–44)**, not 42–51. Significance comes from the per-project paired tests (n=18 projects per seed), not the seed count; 3 seeds add a variance band and guard against one unlucky init. This is a time/rigor tradeoff on constrained hardware and **must be stated as a limitation in the thesis**. If compute frees up, raising it is the cheapest rigor win available.

**Unit of analysis is per-project, always.** Pooled "global" entries are ignored on load, and that is not a convention — pooling rewards a model for encoding project identity (see the reporting rule under External baselines). `compute_statistics.py` warns if a condition is missing for only some seeds, because `dropna()` would otherwise silently shrink the analysis to a subset.

---

## Metrics to report

| Metric | Notes |
|---|---|
| **MAE on class values** | **PRIMARY.** Bridge to regression literature (Deep-SE/GPT2SP comparability). Map argmax (or CORN decode) → INV_LABEL_MAP → compute \|ŷ−y\|. ✅ `evaluate_classification["mae"]`. |
| **Cohen's Kappa (quadratic-weighted)** | **PRIMARY.** Chance-corrected, penalizes larger ordinal misses more — the right lens for the Fibonacci deck. ✅ `evaluate_classification["cohen_kappa"]`. |
| Macro-F1 | Secondary (was primary). Robust to class imbalance, but blind to ordinal adjacency. |
| Accuracy | Secondary. |
| Per-class F1 (SP 1/2/3/5/8) | Shows which classes are hard. |
| Weighted F1 | For completeness. |
| Communication cost | Bytes uploaded per round × rounds × clients, vs. full fine-tuning. ✅ `compute_communication_cost()` → `results/communication_cost.json`. Personalized mode transmits even less (no head). |
| Confusion matrix | Per condition, shown as figure in thesis. Check adjacency structure — errors concentrated on neighboring classes mean the model learns the ordinal scale even when accuracy looks weak. |

**⚠️ Aggregation rule — report per-project, not pooled.** Every metric above is computed per-project; the pooled "global" entry is a convenience printout, NOT the unit of analysis. `compute_statistics.py` pairs per (seed, project) and ignores `global` on load. This matters most for κ: pooling computes chance agreement from the *pooled* marginals, so any predictor that varies by project gets credit for encoding project identity. The constant median predictor scores κ **0.0000 in all 18 projects** yet pools to **0.5006**. MAE is not affected this way (it is a linear average); κ and macro-F1 are. Conditions whose model is per-project (personalized-head, median, TF-IDF+SVM) therefore emit **no pooled entry at all**.

---

## Known gaps — ALL IMPLEMENTED (#1–#14)

**Note:** every gap below is done; the entries are kept as the *specification* of how each feature behaves, which is still the reference when touching that code. Gaps #11–#14 map to the old Task Checklist (Task 1 = #12 CORN, Task 2 = #11 personalized head, Task 3 = #13 checkpointing, Task 4 = #14 LOPO) and are tagged `task-1-done` … `task-4-done`. Beyond these, the classic RQ5 baselines (`fl/classic_baselines.py`) are implemented too. If a spec here and the code ever conflict, trust the code and fix the spec.

**1. Local-only training condition** ✅ Implemented
Each client (`train_local_only` in `train_federated_dl.py`) trains independently from the same warm-start checkpoint as federated, for `rounds * local_epochs` total epochs, with no server/aggregation. Each client's model is evaluated on its own test split via `evaluate_local_only`, which also produces a "global" pooled entry. Results: `results/local_only_per_project.json`, plus `local_acc`/`local_macro_f1`/`local_f1_sp{1,2,3,5,8}` columns in `summary_df`/`summary.csv`. Toggle: `--skip-local-only`.

**2. FedAvg as explicit condition** ✅ Implemented
`federated_condition_name(prox_mu)` in `train_federated_dl.py` returns `"FedAvg"` when `prox_mu == 0.0`, else `"FedProx (mu=X)"`. This label (`fl_condition`) replaces the generic "Federated DL"/"Federated" text everywhere it's printed — the metrics line, per-project summary header, loss/val-F1 history lines, saved-artifact line, and the improvement-vs-baseline/local-only lines — and is written to `results/config.json` as `federated_condition` and prepended to `results/summary.md`. Run with `--prox-mu 0` to get the FedAvg condition; no extra training needed since `prox_mu=0` already degrades FedProx to FedAvg mathematically. (ASCII "mu" used instead of μ/µ to avoid `UnicodeEncodeError` on Windows consoles with cp1252 stdout.)

**3. Validation split + per-round eval tracking** ✅ Implemented
`prepare_tabular_bundle(..., val_size=config.val_size)` carves a per-client validation split from the train portion (`fl/data.py::split_train_val` — stratified by story_point in "random" mode, chronological-tail in "temporal" mode, sharing edge-case handling with `split_per_client` via `_carve_fraction`). `FedProxServer.train()` now accepts `val_loader`/`val_labels`/`num_classes`, evaluates the global model on this val set after every round, records `{round, mean_local_loss, weighted_local_loss, val_macro_f1, val_accuracy}` per round (saved to `results/federated_round_history.json` for the convergence-curve figure), and returns the **best-on-val** global state rather than the final-round state. The test set is still evaluated exactly once. `run_prediction` moved to `fl/metrics.py` (shared by server and main script). Toggle: `--val-size` (default 0.1). **Note: gap #11 changes the per-round val protocol in personalized mode — see architecture section.**

**4. Multi-seed runner script** ✅ Implemented
`run_experiments.py` loops over `--seeds` (RUN_PLAN.md locks `42-44`). For each seed it runs `train_federated_dl.py` twice: once for the "fedprox" condition (configurable `--prox-mu`, includes the majority/median/TF-IDF+SVM/local-only/centralized baselines) and once for "fedavg" (`--prox-mu 0`, `--skip-centralized --skip-local-only --skip-classic-baselines`, reusing run 1's baselines for that seed). Unrecognized CLI args are forwarded verbatim to both runs. Output: `<results-root>/seed_<N>/{fedprox,fedavg}/results/*.json` plus `<results-root>/manifest.json`. Supports `--dry-run` (prints commands, touches nothing), `--force`, `--skip-fedavg`, `--skip-baselines`, and resumes at two layers: coarse (skips a seed/condition whose `federated_per_project.json` exists) and fine (`--checkpoint-every`/`--resume` within a run).

**5. Statistics script** ✅ Implemented
`compute_statistics.py` loads `local_only_per_project.json` / `centralized_per_project.json` / `federated_per_project.json` (fedprox and fedavg dirs) across `<results-root>/seed_*`, builds a long table of per-(seed, project) values for `--metric` (default `macro_f1`; also `accuracy`, `mae`, `cohen_kappa`), then computes: Wilcoxon signed-rank (paired) FedProx vs. {Local-only, Centralized, FedAvg}; Vargha-Delaney A12 / Cliff's delta (via `scipy.stats.mannwhitneyu`); Friedman test across all conditions; Nemenyi post-hoc pairwise p-values (via `scipy.stats.studentized_range`, no extra dependencies). Writes `results_long.csv`, `pairwise_vs_fedprox.csv`, `friedman.json`, `nemenyi.csv`, and `summary_table.tex` to `<results-root>/statistics/`.

**6. MAE + Cohen's Kappa + communication cost metrics** ✅ Implemented
`evaluate_classification` (in `fl/metrics.py`) now also returns `mae` (predictions mapped back through `INV_LABEL_MAP` to story-point values, mean absolute error vs. true SP) and `cohen_kappa` (quadratic-weighted — appropriate for the ordinal Fibonacci classes). `format_metrics` prints both. New `compute_communication_cost(trainable_params, total_params, rounds, num_clients)` returns per-client per-round upload bytes (float32) for the trainable-only vs. full-fine-tuning case, total upload bytes, and the reduction factor; computed once in `train_federated_dl.py` from the probe model and saved to `results/communication_cost.json`, with a summary printed at the end of the run.

**7. Warm-start ablation** ✅ Implemented
`--run-no-warmstart-fl` runs a second `FedProxServer.train()` with `initial_state=None` (random init), using a fresh server seeded with the same `random_state` as the warm-started run so both see identical per-round client selections — isolating the effect of the warm-start checkpoint. Results: `results/federated_no_warmstart_per_project.json` and `results/federated_no_warmstart_round_history.json`; artifact saved to `<save-dir>/federated_no_warmstart`. Printed alongside the warm-started run as `"<condition> (no warmstart)"` in the metrics line, val-F1 history, and saved-artifacts list, plus a new `"<condition> Macro-F1 improvement from warm-start: X%"` summary line. Off by default (roughly doubles federated training time when enabled).

**8. Scale to a properly-sized encoder** — resolved: `microsoft/codebert-base` (see Encoder choice)
Development uses bert-tiny; final reported results use `--model-name microsoft/codebert-base` on the bigger-VRAM machine. Its LoRA target modules are `query`, `value` — already the CLI default, so no flag change is needed. **This is settled, not open: the encoder is a fixed method default in RUN_PLAN.md, held constant across every run.** The standing rule remains: **draw NO accuracy conclusions from bert-tiny — it cannot do this task.**

**9. Temporal split robustness check** ✅ Implemented
`--split-mode temporal` (also `FLConfig.split_mode`) is wired through `split_per_client`, `split_train_val`, and `_carve_fraction` in `fl/data.py`: train on each client's earliest issues, val/test on the latest (export already `ORDER BY Creation_Date`). `--split-mode random` (default) keeps the stratified random split. See commands.txt section 4 for a runnable example.

**10. Loader-side validation** ✅ Implemented
`fl/data.py::validate_cleaned_dataframe` asserts it received *cleaned* CSVs: `{code`/`{noformat` markup, quote-wrapped titles, and non-canonical Priority/Story_Point values. Fails loudly with a pointer to export_issues.py if violated. This enforces the export-time-cleaning contract.
The `{code`/`{noformat`/quote-wrap checks use a **>1% threshold**, not zero-tolerance — see "Known residual per-row artifacts" below for why.

**11. Personalized-head federated condition (FedSP-PEFT-P)** ✅ Implemented (tag `task-2-done`)
Spec below kept as the reference for how it behaves; `--personalized-head` (+ optional `--generic-head`).
- `FLConfig.personalized_head: bool = False`; CLI `--personalized-head`.
- Server: remove head params (by name prefix, e.g. `classifier.` / `head.`) from `aggregatable_keys`. Everything else (B matrices, embeddings) aggregates as before.
- Client: FedProx proximal term restricted to aggregatable params (this restriction should be unconditional — it's correct in shared mode too, where head IS aggregatable, so behavior there is unchanged).
- Per-round val (personalized mode): each client evaluates shared+own-head on own val split; server records `{round, mean_local_loss, weighted_local_loss, mean_val_macro_f1, weighted_val_macro_f1, mean_val_accuracy}`; best-on-val = best shared state by weighted mean, snapshotting all client heads at that round.
- Test eval (personalized mode): per-project only (own head on own test split); NO pooled-global entry. Summary tables must handle its absence.
- Optional `--generic-head`: server additionally keeps a one-way weighted average of heads, saved with artifacts, used only for gap #14 initialization. Never pushed to participants.
- Condition label: `"FedProx (mu=X) + P-head"` / `"FedAvg + P-head"` via `federated_condition_name`.
- Saved artifact layout: `<save-dir>/federated/{shared_state.pt, heads/<project>.pt, generic_head.pt?}`.

**12. CORN ordinal head** ✅ Implemented (tag `task-1-done`)
`--head-type corn`; CE remains the ablation arm. CORN head = 4×hidden+4 params vs CE's 5×hidden+5.
- `pip install coral-pytorch` (add to requirements.txt).
- `FLConfig.head_type: str = "ce"` ∈ {"ce", "corn"}; CLI `--head-type`. (Default is `"ce"` to preserve backward-compatible behavior of existing runs; pass `--head-type corn` for the final ordinal experiments.)
- `fl/model.py`: head output dim = `num_classes - 1` when corn; loss = `corn_loss(logits, labels, num_classes)`; prediction decode = `corn_label_from_logits`.
- `fl/metrics.py::run_prediction`: decode path must branch on head_type (argmax for CE, CORN decode for corn). Metrics downstream unchanged (they consume predicted class indices).
- Class weights: CE arm only.
- Sanity check: CORN head has 4×hidden+4 params vs CE's 5×hidden+5 — communication-cost numbers change slightly; recompute, don't reuse.

**13. Checkpoint / resume (low-resource requirement)** ✅ Implemented (tag `task-3-done`)
`--checkpoint-every N --checkpoint-keep K --resume|--resume-from`. Resume is bit-reproducible: a 1-round + resume run reproduced an uninterrupted 3-round run exactly (max metric diff 0.00e+00), and `sanity_check_fl_randomness.py` proves the client-selection sequence matches.
Motivation: training runs on constrained hardware and may be interrupted; must be resumable without redoing completed work, and resume must be *bit-reproducible* with respect to client selection and data sampling (extend `sanity_check_fl_randomness.py` to prove this).

New module `fl/checkpoint.py`, used by all three training paths:
- CLI / config: `--checkpoint-every N` (default 0 = off; unit = **epochs** for centralized and local-only, **global rounds** for federated), `--resume` (auto-detect latest checkpoint in save-dir) or `--resume-from <path>`, `--checkpoint-keep K` (default 2; rotate, always keep `latest/` plus `best/`).
- Federated round flow: round starts → selected clients train → server aggregates → per-round val → **then**, if `round % N == 0`, checkpoint. Checkpoint contents:
  - aggregated shared trainable state (best AND latest)
  - per-client local heads if `personalized_head` (all clients' current heads)
  - round index, round history so far, best-on-val bookkeeping (best round, best metric)
  - RNG states: `torch`, `numpy`, `random`, and the server's client-selection `Generator` state
  - a `meta.json` with the full FLConfig + code-relevant versions, validated on resume (refuse to resume across incompatible configs, e.g. different head_type/encoder/personalized_head; allow overriding rounds upward)
- Centralized / local-only flow: same contract per epoch — model trainable state, optimizer state, scheduler state (if any), epoch index, RNG states. Local-only checkpoints are per-client (they're independent runs; checkpoint each client's progress so an interrupt mid-pool doesn't redo finished clients).
- Resume semantics: continue at round/epoch `k+1` with restored RNG states; the sequence of client selections and shuffles from a resumed run MUST match an uninterrupted run with the same seed (this is the new sanity check).
- Layout: `<save-dir>/checkpoints/{federated|centralized|local_only/<project>}/{latest,best,round_<k>}/`.
- Trainable-only states keep checkpoints small (same trick as client transfer); do NOT serialize the frozen backbone.
- `run_experiments.py`: forward the checkpoint flags; its existing per-seed resume logic (skip if results JSON exists) stays as the coarse layer, checkpoints are the fine-grained layer within a run.

**14. Leave-one-project-out onboarding experiment** ✅ Implemented (tag `task-4-done`)
`--holdout-project <name>` + `run_lopo.py` + `compare_lopo.py`. RUN_PLAN.md holds out Hyperledger_Sawtooth.
The external-client / cold-start evaluation — arguably the strongest experiment in the thesis:
- Train federated on 18 projects, hold one project out entirely (choose a mid-sized project, NOT Lsstcorp — it's the warm-start source and a size outlier).
- Evaluate on the held-out project's test split under increasing adaptation budgets: 0 labeled examples (zero-shot: shared-head model, or generic head in personalized mode), then head-only fine-tuning on 10 / 25 / 50 / 100 of its earliest issues.
- Compare shared-head vs personalized(+generic-head-init) conditions across budgets; plot MAE / macro-F1 vs adaptation size.
- Expected shape: shared head wins at budget 0; personalized+adaptation overtakes quickly; the crossover point is the practitioner-facing answer to "how much history does a new team need before joining pays off".
- Script: `run_lopo.py` (or a `--holdout-project` flag in the main script + a small adaptation script reusing head-only fine-tuning from the onboarding protocol).

---

## Things to fix / rename before the thesis appendix

- ✅ `StoryPointRegressor` renamed to `StoryPointClassifier` (`fl/model.py`, `train_federated_dl.py`, `predict_saved_model.py`) — the class does classification, not regression.
- ✅ The class docstring already says "classification head", not "regression model".
- ✅ `requirements.txt` already lists `peft>=0.19`.
- ✅ `coral-pytorch>=1.4.0` is in `requirements.txt` (gap #12).
- `MISSING_CATEGORY_TOKENS` in `data.py` is a non-empty set literal (`{"", "nan", ...}`) — it correctly constructs a `set`, not a `dict`. The earlier note about `{}` was stale; no action needed.

---

## Thesis framing

**Working title:** "Privacy-Preserving Story Point Estimation via Federated Learning with Parameter-Efficient Fine-Tuning"

**Core research questions (LOCKED — mirrored in [RUN_PLAN.md](RUN_PLAN.md), which maps each to its runs):**
- **RQ1 — Feasibility & the cost of privacy.** How much of the local-only → centralized gap does federated training recover, without sharing raw issue data across projects? Runs: local-only vs centralized vs federated, plus the classic within-project baselines that make "competitive" mean something.
- **RQ2 — Heterogeneity.** Does FedProx's proximal regularization improve over FedAvg under the non-IID project heterogeneity in TAWOS (RQ2.1), and how sensitive is that to μ ∈ {0, 0.001, 0.01, 0.1} (RQ2.2)?
- **RQ3 — Personalization.** Given the project-specific semantics of story points, does personalizing the estimation head (federating representations, not decision layers) improve federated SPE over a fully shared model?
- **RQ4 — Onboarding crossover.** How much of its own history does an external project need before joining pays off? Leave-one-project-out cold start, head-only adaptation over budgets {0, 10, 25, 50, 100}.
- **RQ5 — SOTA positioning.** Where does this land against the cheap comparators and the published deep SPE models? Classic baselines (TF-IDF+SVM, median-SP) are **run and statistically tested**; published deep SOTA (Deep-SE, GPT2SP, EGPT-SPE, Llama3SP) is a **citation table**, not re-run.

**Note on the reframing:** communication cost used to be an RQ of its own ("does FFA-LoRA cut cost without degrading accuracy?"). It is no longer — FFA-LoRA is a fixed method default, not a variable, and communication cost is **reported descriptively** (from `results/communication_cost.json`) in the method/results chapters. Do not re-promote it to an RQ. Personalization and onboarding, formerly one RQ, are now split (RQ3 / RQ4) because they need different runs and answer different questions.

**Central claim:** Federated training with FFA-LoRA and FedProx bridges a meaningful fraction of the gap between local-only and centralized training, while transmitting less than 1% of model parameters per round — and personalizing the classification head addresses the project-specific calibration of story points, making cross-project collaboration practical under data privacy constraints.

**Discussion chapter must engage with:** (a) Tawosi et al. (2023) — simple models matching deep ones; the argument is that federation+personalization is worthwhile even at modest absolute accuracy; (b) the within-project-only accuracy of prior SPE models as the motivation for personalization; (c) the planning-poker calibration analogy for the onboarding protocol; (d) confusion-matrix adjacency + Kappa/MAE as evidence the model learns ordinal structure even where macro-F1 is flat.

**Key papers to cite (minimum):**
- Choetkiertikul et al. (2018) — Deep-SE, the canonical deep SPE baseline (LSTM+RHN)
- Tawosi et al. (2022) — TAWOS dataset paper
- Tawosi et al. (2023) — Replication study questioning deep SPE baselines (important: read this carefully, it shows simple models can match deep ones — your framing must address it)
- McMahan et al. (2017) — FedAvg (Communication-Efficient Learning of Deep Networks)
- Li et al. (2020) — FedProx (Convergence of Federated Optimization)
- Sun et al. (2024) — FFA-LoRA (ICLR; the principled justification for freezing A)
- Hu et al. (2022) — LoRA original paper
- Shi et al. (2021) — CORN, rank-consistent ordinal regression (arXiv:2111.08851)
- Arivazhagan et al. (2019) — FedPer, federated learning with personalization layers
- Collins et al. (2021) — FedRep, shared representations for personalized FL
- Fu & Tantithamthavorn (2022) — GPT2SP (within-project accuracy point; regression-based comparator)

---

## Venue targets for publication

- **ESEM** (Empirical Software Engineering and Measurement) — best fit, empirical SE community
- **MSR** (Mining Software Repositories) — strong fit, data-driven SE
- **TOSEM / EMSE** — journal versions if extended
- **FSE / ICSE** — stretch, but not impossible for a strong empirical result

---

## Do NOT do these things

- Do not evaluate on the test set more than once per condition. Use val for model selection.
- Do not tune hyperparameters on the test set.
- Do not claim "privacy-preserving" without scoping it: raw data stays local, but gradient-inversion attacks on adapter weights are possible. Differential privacy and secure aggregation are future work. (If `--generic-head` is used, note that head weights are transmitted once — the privacy claim is about raw issue data, keep the scoping honest.)
- Do not average LoRA A and B matrices separately if FFA-LoRA is ever disabled — reconstruct ΔW = B·A first, then average, or you introduce aggregation error.
- Do not use a single seed for reported results. RUN_PLAN.md locks **3 seeds (42–44)** with statistical tests required; significance comes from the per-project paired tests (n=18), and the reduced seed count is stated as a thesis limitation.
- **Do not report a pooled "global" metric for a per-project model** (personalized-head, median-SP, TF-IDF+SVM). Pooling rewards a model for encoding project identity: the constant median predictor scores quadratic κ 0.0000 in all 18 projects but pools to 0.5006. Per-project + mean/median only.
- Do not re-promote communication cost to an RQ — FFA-LoRA is a fixed method default and cost is reported descriptively.
- Do not aggregate the head in personalized-head mode — that silently reverts to the shared-head condition. (One-way generic-head averaging for newcomer init is the only exception, and it must never be pushed back to participants.)
- Do not apply the FedProx proximal term to the local head in personalized mode — there is no global reference point for it.
- Do not report a pooled "global" test metric in personalized mode — it is undefined with per-client heads; report per-project + mean/median.
- Do not reuse CE communication-cost numbers for CORN runs (head sizes differ slightly); recompute.
- Do not interpret accuracy from bert-tiny runs; it is a plumbing model only.
- Do not resume a checkpoint across an incompatible config (different head_type / encoder / personalized_head) — `fl/checkpoint.py` must refuse.