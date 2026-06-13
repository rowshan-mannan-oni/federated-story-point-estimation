"""
tests/test_statistics.py
-------------------------
Tests for the pure-function statistics utilities in compute_statistics.py
and the seed-list parser shared with run_experiments.py (gap #4/#5).
No FL training or GPU required.
"""
import json

import numpy as np
import pandas as pd
import pytest

from compute_statistics import (
    discover_seeds,
    effect_size_label,
    friedman_and_nemenyi,
    load_long_results,
    pairwise_vs_primary,
    parse_seeds,
    vargha_delaney_a12,
)
from run_experiments import parse_seeds as run_experiments_parse_seeds


# ---------------------------------------------------------------------------
# parse_seeds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("parser", [parse_seeds, run_experiments_parse_seeds])
def test_parse_seeds_range(parser):
    assert parser("42-45") == [42, 43, 44, 45]


@pytest.mark.parametrize("parser", [parse_seeds, run_experiments_parse_seeds])
def test_parse_seeds_list(parser):
    assert parser("42, 43, 50") == [42, 43, 50]


@pytest.mark.parametrize("parser", [parse_seeds, run_experiments_parse_seeds])
def test_parse_seeds_single(parser):
    assert parser("42") == [42]


# ---------------------------------------------------------------------------
# Vargha-Delaney A12 / Cliff's delta
# ---------------------------------------------------------------------------

def test_a12_x_always_greater():
    x = np.array([0.8, 0.85, 0.9])
    y = np.array([0.2, 0.25, 0.3])
    assert vargha_delaney_a12(x, y) == 1.0


def test_a12_identical_distributions():
    x = np.array([0.5, 0.6, 0.7, 0.8])
    y = np.array([0.5, 0.6, 0.7, 0.8])
    assert vargha_delaney_a12(x, y) == pytest.approx(0.5)


def test_effect_size_labels():
    assert effect_size_label(0.5) == "negligible"
    assert effect_size_label(0.6) == "small"
    assert effect_size_label(0.7) == "medium"
    assert effect_size_label(0.9) == "large"


# ---------------------------------------------------------------------------
# Friedman + Nemenyi on synthetic data
# ---------------------------------------------------------------------------

def _synthetic_wide(n_blocks: int = 12) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "FedProx": 0.70 + rng.normal(0, 0.01, n_blocks),
        "FedAvg": 0.65 + rng.normal(0, 0.01, n_blocks),
        "Local-only": 0.55 + rng.normal(0, 0.01, n_blocks),
        "Centralized": 0.75 + rng.normal(0, 0.01, n_blocks),
    })


def test_friedman_detects_systematic_difference():
    wide = _synthetic_wide()
    friedman_result, nemenyi, avg_ranks = friedman_and_nemenyi(wide, "macro_f1", alpha=0.05)

    assert friedman_result["significant"] is True
    assert set(friedman_result["conditions"]) == {"FedProx", "FedAvg", "Local-only", "Centralized"}
    # Centralized has the highest macro_f1 -> best (lowest) average rank.
    assert avg_ranks["Centralized"] < avg_ranks["Local-only"]
    assert nemenyi.shape == (4, 4)
    assert (np.diag(nemenyi.values) == 1.0).all()


def test_friedman_no_difference_for_identical_columns():
    n_blocks = 10
    base = np.linspace(0.5, 0.6, n_blocks)
    wide = pd.DataFrame({"FedProx": base, "FedAvg": base, "Local-only": base, "Centralized": base})
    friedman_result, _, avg_ranks = friedman_and_nemenyi(wide, "macro_f1", alpha=0.05)

    # All conditions tie on every block -> equal average ranks.
    ranks = list(avg_ranks.values)
    assert all(r == pytest.approx(ranks[0]) for r in ranks)


def test_pairwise_vs_primary_direction_for_mae():
    # Lower MAE is better. FedProx has lower MAE than the other condition.
    wide = pd.DataFrame({
        "FedProx": [0.5, 0.4, 0.6, 0.5],
        "FedAvg": [0.8, 0.9, 0.7, 0.85],
    })
    result = pairwise_vs_primary(wide, "mae")
    row = result.loc[result["condition"] == "FedAvg"].iloc[0]

    # FedProx values are all lower (better) -> a12 close to 0, but
    # P(FedProx better) should be close to 1 for a "lower is better" metric.
    assert row["a12"] < 0.5
    assert row[f"p(FedProx_better)"] > 0.5


# ---------------------------------------------------------------------------
# Loading results from disk
# ---------------------------------------------------------------------------

def _write_per_project_json(path, projects):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(projects, fh)


def test_discover_seeds_and_load_long_results(tmp_path):
    per_project = {
        "ProjectA": {"macro_f1": 0.6, "accuracy": 0.7},
        "ProjectB": {"macro_f1": 0.5, "accuracy": 0.6},
        "global": {"macro_f1": 0.55, "accuracy": 0.65},
    }

    for seed in (42, 43):
        seed_dir = tmp_path / f"seed_{seed}"
        _write_per_project_json(seed_dir / "fedprox" / "results" / "federated_per_project.json", per_project)
        _write_per_project_json(seed_dir / "fedprox" / "results" / "local_only_per_project.json", per_project)
        _write_per_project_json(seed_dir / "fedprox" / "results" / "centralized_per_project.json", per_project)
        _write_per_project_json(seed_dir / "fedavg" / "results" / "federated_per_project.json", per_project)

    assert discover_seeds(tmp_path) == [42, 43]

    long_df = load_long_results(tmp_path, [42, 43], "macro_f1")
    # 2 seeds x 4 conditions x 2 projects (global excluded) = 16 rows.
    assert len(long_df) == 16
    assert "global" not in long_df["project"].values
    assert set(long_df["condition"]) == {"Local-only", "Centralized", "FedAvg", "FedProx"}
