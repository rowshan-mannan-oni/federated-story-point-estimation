"""
Aggregate leave-one-project-out results into a tidy CSV for the crossover plot
(CLAUDE.md gap #14, step 4.3).

Reads the per-condition lopo_<project>_<tag>.json files written by run_lopo.py
(e.g. shared-head vs personalized+generic-head) and emits a tidy LONG CSV: one
row per (holdout_project, condition, head_init, budget, metric). This is the
shape a crossover plot wants — x=budget, y=value, series=condition/metric — and
directly answers "how much history does a new team need before joining pays
off" (the budget where personalized+adaptation overtakes the shared head).

Example:
    # After running run_lopo.py for both conditions into results/:
    python compare_lopo.py --results-dir results --out results/lopo_comparison.csv
"""
import argparse
import csv
import glob
import json
from pathlib import Path
from typing import List

METRIC_KEYS = ("accuracy", "macro_f1", "mae", "cohen_kappa")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate LOPO results into a tidy crossover CSV (gap #14)")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory holding lopo_*.json files")
    parser.add_argument("--glob", type=str, default="lopo_*.json", help="Glob for the LOPO JSON files")
    parser.add_argument("--metrics", type=str, default=",".join(METRIC_KEYS),
                        help="Comma-separated metrics to include as rows")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path (default <results-dir>/lopo_comparison.csv)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    files = sorted(glob.glob(str(results_dir / args.glob)))
    if not files:
        raise FileNotFoundError(f"No files matching {args.glob} in {results_dir}")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    rows: List[dict] = []
    conditions_seen = set()
    for path in files:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        tag = data.get("run_tag", data.get("condition", "unknown"))
        conditions_seen.add(tag)
        for budget in data["budgets"]:
            entry = data["results"][str(budget)]
            for metric in metrics:
                rows.append({
                    "holdout_project": data["holdout_project"],
                    "condition": data["condition"],
                    "run_tag": tag,
                    "head_init": data.get("head_init", ""),
                    "head_type": data.get("head_type", ""),
                    "budget": budget,
                    "n_adapt": entry.get("n_adapt", budget),
                    "metric": metric,
                    "value": entry[metric],
                })

    # Stable ordering: project, condition, metric, then budget — easy to eyeball crossovers.
    rows.sort(key=lambda r: (r["holdout_project"], r["run_tag"], r["metric"], r["budget"]))

    out_path = Path(args.out) if args.out else results_dir / "lopo_comparison.csv"
    fieldnames = ["holdout_project", "condition", "run_tag", "head_init", "head_type",
                  "budget", "n_adapt", "metric", "value"]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[compare_lopo] {len(files)} file(s), conditions={sorted(conditions_seen)}", flush=True)
    print(f"[compare_lopo] Wrote {len(rows)} rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
