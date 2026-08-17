"""
Cross-root statistical comparison (RUN_PLAN.md "Analysis / helpers" — the
cross-root helper needed by RQ2 and RQ3, since compute_statistics.py only
pairs conditions *within* one results-root).

compute_statistics.py compares conditions that live side-by-side under the
SAME --results-root (e.g. FedProx vs FedAvg vs Local-only vs Centralized, all
under experiments_temporal_shared/seed_*/{fedprox,fedavg}/results/). RQ2
(personalized vs shared head) and RQ3 (LoRA vs full fine-tuning) instead
compare the federated condition across TWO DIFFERENT roots, e.g.:

    experiments_temporal_shared/seed_*/fedprox/results/federated_per_project.json
  vs
    experiments_temporal_personalized/seed_*/fedprox/results/federated_per_project.json

This script loads one named JSON file from two roots, pairs values per
(seed, project) — dropping any project missing from either side — and runs
the same Wilcoxon signed-rank + Vargha-Delaney A12 the primary comparator
uses, so RQ2/RQ3 tables are computed the same way as RQ1.

Unit of analysis is per-project, always: "global" is ignored (and personalized
mode never writes it), matching CLAUDE.md's aggregation rule.

Example (RQ2 — personalized vs shared, temporal, primary metrics):
    python compare_conditions.py \\
        --root-a /mnt/Data/Oni/experiments_temporal_shared --label-a "Shared-head" \\
        --root-b /mnt/Data/Oni/experiments_temporal_personalized --label-b "P-head" \\
        --metric mae
    python compare_conditions.py \\
        --root-a /mnt/Data/Oni/experiments_temporal_shared --label-a "Shared-head" \\
        --root-b /mnt/Data/Oni/experiments_temporal_personalized --label-b "P-head" \\
        --metric cohen_kappa
    python compare_conditions.py \\
        --root-a /mnt/Data/Oni/experiments_temporal_shared --label-a "Shared-head" \\
        --root-b /mnt/Data/Oni/experiments_temporal_personalized --label-b "P-head" \\
        --metric macro_f1

Example (RQ3 — LoRA vs full fine-tuning):
    python compare_conditions.py \\
        --root-a experiments_temporal_shared --label-a "LoRA" \\
        --root-b experiments_temporal_fullft --label-b "Full-FT" \\
        --metric mae
"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

METRIC_DIRECTION = {
    "macro_f1": "higher",
    "accuracy": "higher",
    "cohen_kappa": "higher",
    "mae": "lower",
}


def parse_seeds(seeds_arg: str) -> List[int]:
    seeds: List[int] = []
    for part in seeds_arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-")
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    return seeds


def discover_seeds(root: Path, run_subdir: str, filename: str) -> List[int]:
    seeds = []
    for path in root.glob("seed_*"):
        if (path / run_subdir / "results" / filename).exists():
            seeds.append(int(path.name.split("_", 1)[1]))
    return sorted(seeds)


def load_root(root: Path, seeds: List[int], run_subdir: str, filename: str, metric: str) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        path = root / f"seed_{seed}" / run_subdir / "results" / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            per_project = json.load(fh)
        for project, metrics in per_project.items():
            if project == "global" or metric not in metrics:
                continue
            rows.append({"seed": seed, "project": project, metric: metrics[metric]})
    return pd.DataFrame(rows)


def vargha_delaney_a12(x: np.ndarray, y: np.ndarray) -> float:
    u_stat, _ = mannwhitneyu(x, y, alternative="two-sided")
    return float(u_stat / (len(x) * len(y)))


def effect_size_label(a12: float) -> str:
    delta = abs(a12 - 0.5) * 2.0
    if delta < 0.147:
        return "negligible"
    if delta < 0.33:
        return "small"
    if delta < 0.474:
        return "medium"
    return "large"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired Wilcoxon + Vargha-Delaney A12 between the SAME condition across two results-roots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--root-a", type=str, required=True)
    parser.add_argument("--root-b", type=str, required=True)
    parser.add_argument("--label-a", type=str, default="A")
    parser.add_argument("--label-b", type=str, default="B")
    parser.add_argument("--run-subdir", type=str, default="fedprox", help="Subdir under seed_<N>/ (default: fedprox).")
    parser.add_argument("--filename", type=str, default="federated_per_project.json")
    parser.add_argument("--metric", type=str, default="macro_f1", choices=list(METRIC_DIRECTION))
    parser.add_argument("--seeds", type=str, default=None, help="e.g. '42-44'. Default: auto-discover per root.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out-csv", type=str, default=None)
    args = parser.parse_args()

    root_a, root_b = Path(args.root_a), Path(args.root_b)
    seeds_a = parse_seeds(args.seeds) if args.seeds else discover_seeds(root_a, args.run_subdir, args.filename)
    seeds_b = parse_seeds(args.seeds) if args.seeds else discover_seeds(root_b, args.run_subdir, args.filename)
    seeds = sorted(set(seeds_a) & set(seeds_b))
    if not seeds:
        raise FileNotFoundError(f"No overlapping seeds with '{args.filename}' under both roots.")
    if set(seeds_a) != set(seeds_b):
        print(f"WARNING: seed mismatch — root A has {sorted(seeds_a)}, root B has {sorted(seeds_b)}; using intersection {seeds}.")

    df_a = load_root(root_a, seeds, args.run_subdir, args.filename, args.metric)
    df_b = load_root(root_b, seeds, args.run_subdir, args.filename, args.metric)
    if df_a.empty or df_b.empty:
        raise FileNotFoundError(f"No '{args.metric}' results found for seeds {seeds} in one of the two roots.")

    merged = pd.merge(df_a, df_b, on=["seed", "project"], suffixes=("_a", "_b"))
    dropped_a = set(zip(df_a["seed"], df_a["project"])) - set(zip(merged["seed"], merged["project"]))
    dropped_b = set(zip(df_b["seed"], df_b["project"])) - set(zip(merged["seed"], merged["project"]))
    if dropped_a or dropped_b:
        print(f"WARNING: dropped {len(dropped_a)} (seed, project) pairs only in {args.label_a}, "
              f"{len(dropped_b)} only in {args.label_b} — not present on both sides.")
    if merged.empty:
        raise ValueError("No (seed, project) pairs present on both sides after merge.")

    a_vals = merged[f"{args.metric}_a"].to_numpy()
    b_vals = merged[f"{args.metric}_b"].to_numpy()
    direction = METRIC_DIRECTION[args.metric]

    diff = a_vals - b_vals
    if np.allclose(diff, 0.0):
        stat, p_value = float("nan"), 1.0
    else:
        stat, p_value = wilcoxon(a_vals, b_vals)
    a12 = vargha_delaney_a12(a_vals, b_vals)
    prob_a_better = a12 if direction == "higher" else 1.0 - a12

    print(f"Metric: {args.metric} ({direction} is better)")
    print(f"Paired (seed, project) blocks: {len(merged)}  (seeds={seeds}, {merged['project'].nunique()} projects)")
    print(f"  {args.label_a:<14} mean={a_vals.mean():.4f}  std={a_vals.std():.4f}")
    print(f"  {args.label_b:<14} mean={b_vals.mean():.4f}  std={b_vals.std():.4f}")
    sig = "*" if p_value < args.alpha else " "
    print(f"\nWilcoxon signed-rank: stat={stat if stat==stat else float('nan'):.4f}  p={p_value:.4f}{sig} "
          f"({'significant' if p_value < args.alpha else 'not significant'} at alpha={args.alpha})")
    print(f"Vargha-Delaney A12={a12:.3f}  Cliff's delta={2*a12-1:.3f}  ({effect_size_label(a12)})")
    print(f"P({args.label_a} better)={prob_a_better:.3f}")

    n_a_wins = int(np.sum(diff > 0)) if direction == "higher" else int(np.sum(diff < 0))
    n_b_wins = int(np.sum(diff < 0)) if direction == "higher" else int(np.sum(diff > 0))
    n_ties = len(diff) - n_a_wins - n_b_wins
    print(f"\nPer-(seed,project) wins: {args.label_a}={n_a_wins}  {args.label_b}={n_b_wins}  ties={n_ties}")

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.rename(columns={f"{args.metric}_a": f"{args.metric}_{args.label_a}",
                                f"{args.metric}_b": f"{args.metric}_{args.label_b}"}).to_csv(out_path, index=False)
        print(f"\nPer-(seed,project) values saved to {out_path}")


if __name__ == "__main__":
    main()
