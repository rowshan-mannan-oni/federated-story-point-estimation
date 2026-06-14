"""
analyze_data.py
---------------
Exploratory data analysis for the TAWOS story-point dataset.

Run:
    python analyze_data.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")

DATA_DIR = "data_to_train_on"
LSSTCORP_PREFIX = "Lsstcorp"
SP_CLASSES = [1, 2, 3, 5, 8]
FIGURES_DIR = "analysis_figures"

os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load all projects
# ---------------------------------------------------------------------------

def load_all(data_dir: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    per_project: dict[str, pd.DataFrame] = {}
    frames = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".csv"):
            continue
        project = fname.replace(".csv", "")
        df = pd.read_csv(
            os.path.join(data_dir, fname),
            encoding="utf-8-sig",
            parse_dates=["Creation_Date"],
        )
        df["Story_Point"] = df["Story_Point"].astype(int)
        df["Project"] = project
        per_project[project] = df
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined, per_project


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def class_dist(df: pd.DataFrame) -> dict[int, int]:
    counts = df["Story_Point"].value_counts().reindex(SP_CLASSES, fill_value=0)
    return counts.to_dict()


def label_entropy(df: pd.DataFrame) -> float:
    probs = df["Story_Point"].value_counts(normalize=True).values
    return float(entropy(probs, base=2))


def imbalance_ratio(df: pd.DataFrame) -> float:
    counts = df["Story_Point"].value_counts().reindex(SP_CLASSES, fill_value=0)
    mn = counts[counts > 0].min()
    return float(counts.max() / mn) if mn > 0 else float("inf")


def text_token_approx(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(text.split())


# ---------------------------------------------------------------------------
# Section 1 — Dataset overview
# ---------------------------------------------------------------------------

def section_overview(combined: pd.DataFrame, per_project: dict) -> None:
    print("=" * 70)
    print("1. DATASET OVERVIEW")
    print("=" * 70)

    lss_projects = [p for p in per_project if p.startswith(LSSTCORP_PREFIX)]
    mask_lss = combined["Project"].str.startswith(LSSTCORP_PREFIX)

    total = len(combined)
    total_no_lss = int((~mask_lss).sum())
    total_lss = int(mask_lss.sum())

    print(f"  Total projects        : {len(per_project)}")
    print(f"  Total issues          : {total:,}")
    print(f"  Issues excl. Lsstcorp : {total_no_lss:,}")
    print(f"  Issues Lsstcorp only  : {total_lss:,}  ({lss_projects})")
    print()

    # Global class distribution
    print("  Global class distribution:")
    dist = class_dist(combined)
    for sp, cnt in dist.items():
        pct = 100 * cnt / total
        bar = "#" * int(pct / 2)
        print(f"    SP {sp:>2}: {cnt:>6,}  ({pct:5.1f}%)  {bar}")
    print()

    # Without Lsstcorp
    print("  Class distribution (excl. Lsstcorp):")
    dist_no = class_dist(combined[~mask_lss])
    for sp, cnt in dist_no.items():
        pct = 100 * cnt / total_no_lss if total_no_lss else 0
        print(f"    SP {sp:>2}: {cnt:>6,}  ({pct:5.1f}%)")
    print()


# ---------------------------------------------------------------------------
# Section 2 — Per-project statistics
# ---------------------------------------------------------------------------

def section_per_project(per_project: dict) -> pd.DataFrame:
    print("=" * 70)
    print("2. PER-PROJECT STATISTICS")
    print("=" * 70)

    rows = []
    for project, df in per_project.items():
        dist = class_dist(df)
        rows.append({
            "Project": project,
            "N": len(df),
            **{f"SP{sp}": dist[sp] for sp in SP_CLASSES},
            "Entropy(bits)": round(label_entropy(df), 3),
            "ImbalanceRatio": round(imbalance_ratio(df), 1),
            "MissingDesc": int(df["Description"].isna().sum()),
            "DateMin": df["Creation_Date"].min().date() if "Creation_Date" in df else "N/A",
            "DateMax": df["Creation_Date"].max().date() if "Creation_Date" in df else "N/A",
        })

    stats = pd.DataFrame(rows).sort_values("N", ascending=False)
    print(stats.to_string(index=False))
    print()
    return stats


# ---------------------------------------------------------------------------
# Section 3 — Text length analysis
# ---------------------------------------------------------------------------

def section_text_length(combined: pd.DataFrame) -> None:
    print("=" * 70)
    print("3. TEXT LENGTH ANALYSIS  (whitespace-tokenised approximation)")
    print("=" * 70)

    combined["title_words"] = combined["Title"].apply(text_token_approx)
    combined["desc_words"] = combined["Description"].apply(text_token_approx)
    combined["combined_words"] = combined["title_words"] + combined["desc_words"]

    for col, label in [
        ("title_words", "Title"),
        ("desc_words", "Description"),
        ("combined_words", "Title + Description"),
    ]:
        s = combined[col]
        print(f"  {label}:")
        print(f"    mean={s.mean():.1f}  median={s.median():.0f}  "
              f"p95={s.quantile(0.95):.0f}  max={s.max()}")
        pct_over_128 = 100 * (s > 128).mean()
        print(f"    > 128 tokens: {pct_over_128:.1f}%")
    print()

    # Truncation impact warning
    over = 100 * (combined["combined_words"] > 128).mean()
    if over > 10:
        print(f"  [!] {over:.1f}% of combined texts exceed 128 words — "
              f"consider increasing max_length.")
    print()


# ---------------------------------------------------------------------------
# Section 4 — Missing values & data quality
# ---------------------------------------------------------------------------

def section_data_quality(combined: pd.DataFrame) -> None:
    print("=" * 70)
    print("4. DATA QUALITY")
    print("=" * 70)

    total = len(combined)
    missing_title = combined["Title"].isna().sum()
    missing_desc = combined["Description"].isna().sum()
    empty_desc = (combined["Description"].fillna("").str.strip() == "").sum()

    print(f"  Missing Title       : {missing_title:,} ({100*missing_title/total:.1f}%)")
    print(f"  Missing Description : {missing_desc:,} ({100*missing_desc/total:.1f}%)")
    print(f"  Empty Description   : {empty_desc:,} ({100*empty_desc/total:.1f}%)")
    print()

    # Duplicate issues
    dupes = combined.duplicated(subset=["Issue_Key"]).sum()
    print(f"  Duplicate Issue_Key : {dupes:,}")
    print()


# ---------------------------------------------------------------------------
# Section 5 — Type and Priority distributions
# ---------------------------------------------------------------------------

def section_categorical(combined: pd.DataFrame) -> None:
    print("=" * 70)
    print("5. CATEGORICAL FEATURE DISTRIBUTIONS")
    print("=" * 70)

    for col in ["Type", "Priority"]:
        counts = combined[col].fillna("Unknown").value_counts()
        print(f"  {col} ({len(counts)} unique values):")
        for val, cnt in counts.head(10).items():
            pct = 100 * cnt / len(combined)
            print(f"    {val:<25} {cnt:>6,}  ({pct:.1f}%)")
        if len(counts) > 10:
            print(f"    ... and {len(counts)-10} more")
        print()


# ---------------------------------------------------------------------------
# Section 6 — Temporal coverage
# ---------------------------------------------------------------------------

def section_temporal(combined: pd.DataFrame, per_project: dict) -> None:
    if "Creation_Date" not in combined.columns:
        print("  [SKIP] No Creation_Date column found.")
        return

    print("=" * 70)
    print("6. TEMPORAL COVERAGE")
    print("=" * 70)

    print(f"  Overall date range : {combined['Creation_Date'].min().date()} "
          f"→ {combined['Creation_Date'].max().date()}")
    print()

    combined["year"] = combined["Creation_Date"].dt.year
    yearly = combined.groupby("year").size()
    print("  Issues per year:")
    for yr, cnt in yearly.items():
        bar = "#" * int(cnt / max(yearly) * 40)
        print(f"    {yr}: {cnt:>5,}  {bar}")
    print()


# ---------------------------------------------------------------------------
# Section 7 — Non-IID heterogeneity (Jensen-Shannon divergence)
# ---------------------------------------------------------------------------

def section_heterogeneity(per_project: dict) -> None:
    print("=" * 70)
    print("7. NON-IID HETEROGENEITY  (Jensen-Shannon divergence between projects)")
    print("=" * 70)

    def project_dist_vector(df):
        counts = df["Story_Point"].value_counts().reindex(SP_CLASSES, fill_value=0)
        total = counts.sum()
        return counts.values / total if total > 0 else np.ones(len(SP_CLASSES)) / len(SP_CLASSES)

    projects = list(per_project.keys())
    dists = {p: project_dist_vector(per_project[p]) for p in projects}

    # Pairwise JS divergence
    n = len(projects)
    jsd_matrix = np.zeros((n, n))
    for i, pi in enumerate(projects):
        for j, pj in enumerate(projects):
            jsd_matrix[i, j] = jensenshannon(dists[pi], dists[pj]) ** 2

    avg_jsd = jsd_matrix[np.triu_indices(n, k=1)].mean()
    max_jsd = jsd_matrix[np.triu_indices(n, k=1)].max()

    print(f"  Average pairwise JSD : {avg_jsd:.4f}  (0=identical, 1=maximally different)")
    print(f"  Max pairwise JSD     : {max_jsd:.4f}")
    print()

    # Most heterogeneous pairs
    idx = np.argsort(jsd_matrix[np.triu_indices(n, k=1)])[::-1]
    pairs = [(projects[i], projects[j])
             for i, j in zip(*np.triu_indices(n, k=1))]
    print("  Top-5 most heterogeneous project pairs:")
    for rank, i in enumerate(idx[:5], 1):
        p1, p2 = pairs[i]
        print(f"    {rank}. {p1} vs {p2}  JSD={jsd_matrix[np.triu_indices(n,k=1)][i]:.4f}")
    print()

    # Entropy per project (for thesis table)
    entropies = {p: label_entropy(per_project[p]) for p in projects}
    print(f"  Label entropy range : "
          f"{min(entropies.values()):.3f} – {max(entropies.values()):.3f} bits")
    print(f"  Max entropy possible: {np.log2(len(SP_CLASSES)):.3f} bits  "
          f"(uniform over {len(SP_CLASSES)} classes)")
    print()


# ---------------------------------------------------------------------------
# Section 8 — Figures
# ---------------------------------------------------------------------------

def section_figures(combined: pd.DataFrame, per_project: dict, stats: pd.DataFrame) -> None:
    print("=" * 70)
    print("8. SAVING FIGURES  →  " + FIGURES_DIR + "/")
    print("=" * 70)

    # Fig 1: Global class distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    dist = class_dist(combined)
    ax.bar([str(k) for k in dist], dist.values(), color="steelblue", edgecolor="white")
    ax.set_xlabel("Story Points")
    ax.set_ylabel("Issue count")
    ax.set_title("Global story-point class distribution")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "class_distribution_global.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    # Fig 2: Per-project issue counts
    fig, ax = plt.subplots(figsize=(10, 5))
    s = stats.set_index("Project")["N"].sort_values(ascending=True)
    colors = ["#e07070" if p.startswith(LSSTCORP_PREFIX) else "steelblue" for p in s.index]
    s.plot(kind="barh", ax=ax, color=colors)
    ax.set_xlabel("Number of issues")
    ax.set_title("Issues per project  (red = Lsstcorp)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "issues_per_project.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    # Fig 3: Heatmap of class distribution per project (normalised)
    sp_cols = [f"SP{sp}" for sp in SP_CLASSES]
    norm = stats.set_index("Project")[sp_cols].div(stats.set_index("Project")["N"], axis=0)
    fig, ax = plt.subplots(figsize=(8, max(6, len(per_project) * 0.35)))
    im = ax.imshow(norm.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(SP_CLASSES)))
    ax.set_xticklabels([str(sp) for sp in SP_CLASSES])
    ax.set_yticks(range(len(norm)))
    ax.set_yticklabels(norm.index, fontsize=7)
    ax.set_xlabel("Story Points")
    ax.set_title("Normalised class distribution per project")
    plt.colorbar(im, ax=ax, label="Fraction of issues")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "class_dist_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    # Fig 4: Label entropy per project
    entropies = stats.set_index("Project")["Entropy(bits)"].sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e07070" if p.startswith(LSSTCORP_PREFIX) else "steelblue"
              for p in entropies.index]
    entropies.plot(kind="barh", ax=ax, color=colors)
    ax.axvline(np.log2(len(SP_CLASSES)), color="red", linestyle="--",
               label=f"Max entropy ({np.log2(len(SP_CLASSES)):.2f} bits)")
    ax.set_xlabel("Shannon entropy (bits)")
    ax.set_title("Label entropy per project  (higher = more balanced)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "label_entropy_per_project.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    # Fig 5: Yearly issue volume
    if "Creation_Date" in combined.columns:
        combined["year"] = combined["Creation_Date"].dt.year
        yearly = combined.groupby("year").size()
        fig, ax = plt.subplots(figsize=(9, 4))
        yearly.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
        ax.set_xlabel("Year")
        ax.set_ylabel("Issue count")
        ax.set_title("Issues created per year (all projects)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, "issues_per_year.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\nLoading data from '{DATA_DIR}/' ...\n")
    combined, per_project = load_all(DATA_DIR)
    print(f"Loaded {len(per_project)} projects, {len(combined):,} issues total.\n")

    section_overview(combined, per_project)
    stats = section_per_project(per_project)
    section_text_length(combined)
    section_data_quality(combined)
    section_categorical(combined)
    section_temporal(combined, per_project)
    section_heterogeneity(per_project)
    section_figures(combined, per_project, stats)

    print("Done.")


if __name__ == "__main__":
    main()
