# FedScope — Federated Story Point Results Dashboard

A read-only dashboard for visualizing the JSON/CSV results produced by
`train_federated_dl.py` (config, per-project metrics, round history,
confusion matrices, communication cost).

## Getting started

```bash
npm install
npm run mock-data   # generates demo data under public/data/runs/demo
npm run dev
```

Open the printed local URL (e.g. http://localhost:5173).

## Loading your real results

After a training run finishes, its results live under
`<save-dir>/results/`. Copy them into the dashboard with:

```bash
npm run sync-data -- <path-to-results-dir> <run-id> ["Display label"]
```

Example:

```bash
npm run sync-data -- ../artifacts/results my-run "FedProx, r=8, mu=0.01"
```

This copies the known result files (`config.json`, `summary.csv`,
`*_per_project.json`, `*_round_history.json`, `communication_cost.json`)
into `public/data/runs/<run-id>/` and registers the run in
`public/data/manifest.json`. Re-running with the same `<run-id>` updates
that run in place. Use the run selector in the dashboard's top bar to
switch between runs.

## Project structure

- `src/types/results.ts` — TypeScript types mirroring the training output schema.
- `src/lib/loadResults.ts` — fetches `manifest.json` and per-run result files.
- `src/lib/conditions.ts`, `src/lib/metrics.ts`, `src/lib/format.ts` — shared helpers (condition colors/labels, metric lookups, number formatting).
- `src/context/RunContext.tsx` — global run-selection state.
- `src/components/` — reusable UI building blocks (`Panel`, `MetricCard`, `ConditionBadge`, `PageHeader`, charts).
- `src/pages/` — one page per dashboard view (Overview, Round History, Per-Project, Condition Compare, Confusion Matrix, Communication Cost).
- `scripts/generate-mock-data.mjs` / `scripts/sync-data.mjs` — populate `public/data/`.
