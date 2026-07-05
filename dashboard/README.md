# FedScope — Federated Story Point Results Dashboard

A read-only web dashboard for exploring the JSON/CSV results produced by
`train_federated_dl.py` — config, per-project metrics, round-by-round
convergence, confusion matrices, and communication cost — across one or more
runs.

Stack: React 19 + Vite + TypeScript, Tailwind CSS, Recharts (charts),
React Router, PapaParse (CSV).

## Getting started

```bash
npm install
npm run mock-data   # generates a demo run under public/data/runs/demo
npm run dev         # start the dev server
```

Open the printed local URL (e.g. http://localhost:5173). The demo run lets you
click through every page before you have real results.

Other scripts: `npm run build` (type-check + production build),
`npm run preview` (serve the build), `npm run lint`.

## Loading your real results

After a training run finishes, its results live under `<save-dir>/results/`.
Copy them into the dashboard with:

```bash
npm run sync-data -- <path-to-results-dir> <run-id> ["Display label"]
```

Example:

```bash
npm run sync-data -- ../artifacts/results fedprox-corn "FedProx corn, r=8, mu=0.01"
```

This copies the known result files into `public/data/runs/<run-id>/` and
registers the run in `public/data/manifest.json`. Re-running with the same
`<run-id>` updates that run in place. Use the run selector in the top bar to
switch between runs.

Files copied (missing ones are skipped silently):

```
config.json                                summary.csv
summary.md                                 communication_cost.json
federated_per_project.json                 federated_round_history.json
federated_no_warmstart_per_project.json    federated_no_warmstart_round_history.json
centralized_per_project.json               local_only_per_project.json
```

## Pages

| Route | Page | Shows |
|---|---|---|
| `/` | Overview | Headline metrics (MAE, Cohen's κ, macro-F1, accuracy) per condition, plus run config summary. |
| `/rounds` | Round History | Per-round convergence: local loss and validation macro-F1 from `federated_round_history.json`. |
| `/projects` | Project Breakdown | Per-project metric table across conditions (from the `*_per_project.json` files + `summary.csv`). |
| `/conditions` | Condition Compare | Side-by-side comparison of Majority / Local-only / Centralized / FedAvg / FedProx. |
| `/confusion` | Confusion Matrix | Per-condition confusion matrix; check adjacency structure (near-diagonal = ordinal signal). |
| `/communication` | Communication Cost | FFA-LoRA upload cost vs. full fine-tuning and the reduction factor. |
| `/architecture` | Architecture | Static explainer of the FedSP-PEFT design (encoder + LoRA + heads + server). |

## Project structure

```
scripts/
  generate-mock-data.mjs   Writes a synthetic demo run into public/data/.
  sync-data.mjs            Copies a real results/ dir into public/data/ + updates manifest.json.
public/data/
  manifest.json            List of runs ({ id, label, path }); drives the run selector.
  runs/<run-id>/           One folder of copied result files per run.
src/
  types/results.ts         TypeScript types mirroring the training output schema.
  lib/
    loadResults.ts         Fetches manifest.json and a run's result files.
    conditions.ts          Condition labels / colors / ordering.
    metrics.ts             Metric keys, labels, and "higher/lower is better".
    format.ts              Number / byte / percent formatting.
  context/RunContext.tsx   Global selected-run state.
  components/
    Panel, MetricCard, ConditionBadge, PageHeader, StatusScreen   Reusable UI.
    charts/                Recharts wrappers.
    layout/                AppLayout, nav/top bar.
  pages/                   One file per page above.
  App.tsx / main.tsx       Router + entry point.
```

## Compatibility notes (dashboard is behind the training pipeline)

The result schema (`src/types/results.ts`), the sync file list, and the pages
currently target **shared-head runs**. Specifically they assume:

- a pooled **`"global"`** entry in each `*_per_project.json`, and
- round-history keys **`val_macro_f1` / `val_accuracy`**.

Newer training outputs are **not yet wired in** and will not render correctly
without updates:

- **Personalized-head runs** (`--personalized-head`, FedSP-PEFT-P): there is
  **no `"global"` entry** (per-project + mean only), and round history uses
  **`mean_val_macro_f1` / `weighted_val_macro_f1` / `mean_val_accuracy`**.
- **CORN head** (`--head-type corn`) and **generic head** (`--generic-head`):
  the `head_type` / `personalized_head` config fields and the per-project
  `heads/` artifacts are not surfaced.
- **Leave-one-project-out onboarding** (`lopo_*.json`,
  `lopo_comparison.csv` from `run_lopo.py` / `compare_lopo.py`): not synced or
  visualized.

To support these, extend `types/results.ts` (optional `global`, personalized
round-history keys, `head_type`), `loadResults.ts`, `sync-data.mjs` (add the
LOPO files), and the affected pages. These track CLAUDE.md gaps #11 (personalized),
#12 (CORN), and #14 (LOPO).