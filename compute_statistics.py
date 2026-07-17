"""
Statistical significance testing across multi-seed runs (CLAUDE.md gap #5).

Consumes the <results-root>/seed_<N>/{fedprox,fedavg}/results/*.json files
produced by run_experiments.py (gap #4) and runs:

  - Wilcoxon signed-rank test (paired, per (seed, project)): FedProx vs each
    of Median, TF-IDF+SVM, Local-only, Centralized, FedAvg.
  - Vargha-Delaney A12 / Cliff's delta effect size for the same pairs.
  - Friedman test across all available conditions.
  - Nemenyi post-hoc pairwise test on the Friedman ranks.

All tests are paired per (seed, project) — per-project is the unit of analysis, and
pooled "global" entries are ignored on load. That is not just a convention: pooling
rewards a model for encoding project identity (see fl/classic_baselines.py, where a
per-project constant predictor pools to kappa 0.50 while scoring 0.00 in every
project), so pooled numbers are not safe to compare across conditions.

Outputs (written to <out-dir>, default <results-root>/statistics/):
  - results_long.csv      per (seed, project, condition) metric values
  - pairwise_vs_fedprox.csv  Wilcoxon + A12 vs FedProx for each condition
  - friedman.json          Friedman statistic/p-value and average ranks
  - nemenyi.csv            pairwise Nemenyi p-value matrix
  - summary_table.tex      LaTeX-ready summary table

Example:
    python compute_statistics.py --experiments-root experiments --metric macro_f1
"""

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, mannwhitneyu, studentized_range, wilcoxon

# condition label -> (run subdirectory under seed_<N>/, per-project JSON file)
# Ordered weakest-comparator-first; this order drives the printouts and the LaTeX
# table rows. Median/TF-IDF+SVM are classic within-project baselines (not federated,
# no privacy constraint) — they answer "competitive with what?", which majority-class
# alone cannot. Any condition whose file is absent is skipped, so runs made with
# --skip-classic-baselines still analyse cleanly.
CONDITIONS: Dict[str, Tuple[str, str]] = {
    "Median": ("fedprox", "median_per_project.json"),
    "TF-IDF+SVM": ("fedprox", "tfidf_svm_per_project.json"),
    "Local-only": ("fedprox", "local_only_per_project.json"),
    "Centralized": ("fedprox", "centralized_per_project.json"),
    "FedAvg": ("fedavg", "federated_per_project.json"),
    "FedProx": ("fedprox", "federated_per_project.json"),
}
PRIMARY = "FedProx"

# Whether higher values of a metric are better.
METRIC_DIRECTION: Dict[str, str] = {
    "macro_f1": "higher",
    "accuracy": "higher",
    "cohen_kappa": "higher",
    "mae": "lower",
}


def parse_seeds(seeds_arg: str) -> List[int]:
    """Parse '42-51', '42,43,44', or '42' into a list of ints."""
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


def discover_seeds(experiments_root: Path) -> List[int]:
    seeds = []
    for path in experiments_root.glob("seed_*"):
        if path.is_dir():
            try:
                seeds.append(int(path.name.split("_", 1)[1]))
            except ValueError:
                continue
    return sorted(seeds)


def load_long_results(experiments_root: Path, seeds: List[int], metric: str) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        for condition, (run_subdir, filename) in CONDITIONS.items():
            path = experiments_root / f"seed_{seed}" / run_subdir / "results" / filename
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as fh:
                per_project = json.load(fh)
            for project, metrics in per_project.items():
                if project == "global" or metric not in metrics:
                    continue
                rows.append({"seed": seed, "project": project, "condition": condition, metric: metrics[metric]})
    return pd.DataFrame(rows)


def vargha_delaney_a12(x: np.ndarray, y: np.ndarray) -> float:
    """A12 = P(x > y) + 0.5 * P(x == y), via the Mann-Whitney U statistic for x."""
    u_stat, _ = mannwhitneyu(x, y, alternative="two-sided")
    return float(u_stat / (len(x) * len(y)))


def effect_size_label(a12: float) -> str:
    delta = abs(a12 - 0.5) * 2.0  # |Cliff's delta|
    if delta < 0.147:
        return "negligible"
    if delta < 0.33:
        return "small"
    if delta < 0.474:
        return "medium"
    return "large"


def pairwise_vs_primary(wide: pd.DataFrame, metric: str) -> pd.DataFrame:
    direction = METRIC_DIRECTION.get(metric, "higher")
    primary_vals = wide[PRIMARY].to_numpy()

    rows = []
    for condition in CONDITIONS:
        if condition == PRIMARY or condition not in wide.columns:
            continue
        other_vals = wide[condition].to_numpy()

        diff = primary_vals - other_vals
        if np.allclose(diff, 0.0):
            stat, p_value = float("nan"), 1.0
        else:
            stat, p_value = wilcoxon(primary_vals, other_vals)

        a12 = vargha_delaney_a12(primary_vals, other_vals)
        prob_primary_better = a12 if direction == "higher" else 1.0 - a12

        rows.append({
            "condition": condition,
            "n_pairs": len(wide),
            f"mean_{PRIMARY}": float(np.mean(primary_vals)),
            f"mean_{condition}": float(np.mean(other_vals)),
            "wilcoxon_stat": float(stat) if stat == stat else None,  # NaN check
            "wilcoxon_p": float(p_value),
            "a12": a12,
            "cliffs_delta": 2.0 * a12 - 1.0,
            "effect_size": effect_size_label(a12),
            f"p({PRIMARY}_better)": prob_primary_better,
        })
    return pd.DataFrame(rows)


def friedman_and_nemenyi(wide: pd.DataFrame, metric: str, alpha: float) -> Tuple[dict, pd.DataFrame, pd.Series]:
    direction = METRIC_DIRECTION.get(metric, "higher")
    present = [c for c in CONDITIONS if c in wide.columns]

    samples = [wide[c].to_numpy() for c in present]
    friedman_stat, friedman_p = friedmanchisquare(*samples)

    # Rank 1 = best per block (row). "higher" metrics -> rank descending.
    ranks = wide[present].rank(axis=1, ascending=(direction == "lower"))
    avg_ranks = ranks.mean(axis=0)

    k = len(present)
    n = len(wide)
    se = np.sqrt(k * (k + 1) / (6.0 * n))

    nemenyi = pd.DataFrame(np.ones((k, k)), index=present, columns=present)
    for a, b in itertools.combinations(present, 2):
        z = abs(avg_ranks[a] - avg_ranks[b]) / se
        p = float(studentized_range.sf(z * np.sqrt(2.0), k, np.inf))
        nemenyi.loc[a, b] = p
        nemenyi.loc[b, a] = p

    friedman_result = {
        "metric": metric,
        "conditions": present,
        "n_blocks": n,
        "statistic": float(friedman_stat),
        "p_value": float(friedman_p),
        "significant": bool(friedman_p < alpha),
        "average_ranks": {c: float(avg_ranks[c]) for c in present},
    }
    return friedman_result, nemenyi, avg_ranks


def write_latex_table(out_path: Path, wide: pd.DataFrame, pairwise: pd.DataFrame, metric: str, avg_ranks: pd.Series) -> None:
    present = [c for c in CONDITIONS if c in wide.columns]
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{{metric.replace('_', ' ').title()} across conditions (mean $\\pm$ std over seeds $\\times$ projects)}}",
        r"\begin{tabular}{l" + "c" * (len(present) * 0 + 3) + "}",
        r"\toprule",
        r"Condition & Mean $\pm$ Std & Avg. Rank & $p$ vs " + PRIMARY + r" \\",
        r"\midrule",
    ]
    p_lookup = {row["condition"]: row["wilcoxon_p"] for _, row in pairwise.iterrows()}
    for condition in present:
        values = wide[condition].to_numpy()
        mean_std = f"{values.mean():.4f} $\\pm$ {values.std():.4f}"
        rank = f"{avg_ranks[condition]:.2f}"
        p_str = "--" if condition == PRIMARY else f"{p_lookup.get(condition, float('nan')):.4f}"
        lines.append(f"{condition} & {mean_std} & {rank} & {p_str} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Wilcoxon/Friedman/Nemenyi significance tests across multi-seed runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--experiments-root", type=str, default="experiments")
    parser.add_argument("--seeds", type=str, default=None, help="Seed list/range, e.g. '42-51'. Default: auto-discover seed_* dirs.")
    parser.add_argument("--metric", type=str, default="macro_f1", choices=list(METRIC_DIRECTION))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out-dir", type=str, default=None, help="Default: <experiments-root>/statistics")
    args = parser.parse_args()

    experiments_root = Path(args.experiments_root)
    seeds = parse_seeds(args.seeds) if args.seeds else discover_seeds(experiments_root)

    if not seeds:
        raise FileNotFoundError(f"No seed_* directories found under {experiments_root}")

    out_dir = Path(args.out_dir) if args.out_dir else experiments_root / "statistics"
    out_dir.mkdir(parents=True, exist_ok=True)

    long_df = load_long_results(experiments_root, seeds, args.metric)
    if long_df.empty:
        raise FileNotFoundError(f"No '{args.metric}' results found for seeds {seeds} under {experiments_root}")
    long_df.to_csv(out_dir / "results_long.csv", index=False)

    wide = long_df.pivot_table(index=["seed", "project"], columns="condition", values=args.metric)
    if PRIMARY not in wide.columns:
        raise ValueError(f"'{PRIMARY}' results not found — cannot run statistics without the primary condition.")

    # dropna() keeps the blocks balanced, which the paired tests require — but a
    # condition present for only SOME seeds would silently delete every block from
    # the other seeds. Report it rather than quietly shrinking the analysis; a mixed
    # root (e.g. seeds run before the classic baselines existed) is the likely cause.
    missing_per_condition = {c: int(n) for c, n in (len(wide) - wide.count()).items() if n}
    complete = wide.dropna()
    if complete.empty:
        raise ValueError("No (seed, project) blocks have results for all conditions present.")
    if len(complete) < len(wide):
        print(
            f"WARNING: dropped {len(wide) - len(complete)} of {len(wide)} (seed, project) blocks "
            f"that lack at least one condition.\n"
            f"         Blocks missing, per condition: {missing_per_condition}\n"
            f"         Re-run the affected seeds, or the reported comparison silently rests on a subset.\n"
        )
    wide = complete

    pairwise = pairwise_vs_primary(wide, args.metric)
    pairwise.to_csv(out_dir / "pairwise_vs_fedprox.csv", index=False)

    friedman_result, nemenyi, avg_ranks = friedman_and_nemenyi(wide, args.metric, args.alpha)
    with (out_dir / "friedman.json").open("w", encoding="utf-8") as fh:
        json.dump(friedman_result, fh, indent=2)
    nemenyi.to_csv(out_dir / "nemenyi.csv")

    write_latex_table(out_dir / "summary_table.tex", wide, pairwise, args.metric, avg_ranks)

    print(f"Loaded {len(wide)} paired (seed, project) blocks across seeds {seeds} for metric '{args.metric}'.\n")

    print("Descriptive stats:")
    for condition in CONDITIONS:
        if condition in wide.columns:
            values = wide[condition]
            print(f"  {condition:<12} mean={values.mean():.4f}  std={values.std():.4f}  n={len(values)}")

    print(f"\nWilcoxon signed-rank + Vargha-Delaney A12 ({PRIMARY} vs other conditions):")
    for _, row in pairwise.iterrows():
        sig = "*" if row["wilcoxon_p"] < args.alpha else " "
        print(
            f"  {PRIMARY} vs {row['condition']:<12} p={row['wilcoxon_p']:.4f}{sig}  "
            f"A12={row['a12']:.3f} ({row['effect_size']})  "
            f"P({PRIMARY} better)={row[f'p({PRIMARY}_better)']:.3f}"
        )

    print(f"\nFriedman test across {friedman_result['conditions']}:")
    print(f"  chi2={friedman_result['statistic']:.4f}  p={friedman_result['p_value']:.4f}"
          f"  ({'significant' if friedman_result['significant'] else 'not significant'} at alpha={args.alpha})")
    print("  Average ranks (lower = better):")
    for condition, rank in friedman_result["average_ranks"].items():
        print(f"    {condition:<12} {rank:.2f}")

    print(f"\nNemenyi post-hoc p-values:\n{nemenyi.round(4)}")

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
