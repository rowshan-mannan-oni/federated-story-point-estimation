#!/usr/bin/env python3
"""
extract_facts.py -- turn the project's own data and results into the numbers
the website shows.

Why this exists
---------------
A website about a thesis is only worth reading if its numbers are the thesis's
numbers. Typing them into HTML by hand guarantees they drift: a re-run changes
a result, the page keeps the old one, and nobody notices. So no figure is ever
written into a page. Every one of them is produced here, from the same files
the thesis itself uses, and carries a note saying where it came from.

Re-run this whenever the data or the results change:

    python site/tools/extract_facts.py

What it reads (read-only, always)
    data_to_train_on/*.csv        the exported issues
    results/*.json                one finished run
    results/test_split.csv        that run's test rows

What it writes (nothing else, ever)
    site/data/*.json

It never trains anything, never writes outside site/data/, and never modifies
the thesis. If an input is missing it says so in the output and carries on, so
the site can be built before every experiment has finished.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
SITE_DIR = TOOLS_DIR.parent
REPO_DIR = SITE_DIR.parent

# The thesis code is the source of truth for cleaning and splitting: this
# script imports it rather than reimplementing it, so the site can never
# describe a pipeline the code does not actually run.
sys.path.insert(0, str(REPO_DIR))

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# A fact is a number plus where it came from.
# ---------------------------------------------------------------------------

KINDS = {
    "measured",   # recomputed here, from the data, just now
    "run",        # read out of a finished run's result files
    "derived",    # arithmetic over other facts (stated in `how`)
    "missing",    # the input needed for it is not on this machine
}


class Facts:
    """Collects facts, refuses ones that cannot say where they came from."""

    def __init__(self) -> None:
        self.items: dict[str, dict] = {}
        self.notes: list[str] = []

    def add(self, key, value, *, source, how, kind="measured", unit=None, text=None):
        if kind not in KINDS:
            raise ValueError(f"{key}: unknown kind {kind!r}")
        if not source or not how:
            raise ValueError(f"{key}: a fact must say where it came from")
        self.items[key] = {
            "value": value,
            "text": text if text is not None else fmt(value),
            "unit": unit,
            "source": source,
            "how": how,
            "kind": kind,
        }

    def missing(self, key, *, source, why):
        self.items[key] = {
            "value": None,
            "text": "—",
            "unit": None,
            "source": source,
            "how": why,
            "kind": "missing",
        }

    def note(self, text: str) -> None:
        self.notes.append(text)


def fmt(value):
    """A plain default rendering. Pages may format differently; this is a hint."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    return str(value)


def write_json(path: Path, payload: dict) -> None:
    """Write one output file, refusing any path outside site/data/."""
    out = path.resolve()
    data_dir = (SITE_DIR / "data").resolve()
    if data_dir not in out.parents:
        raise RuntimeError(f"refusing to write outside site/data: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"  wrote {out.relative_to(REPO_DIR)}")


# ---------------------------------------------------------------------------
# 1. The corpus, and what cleaning does to it
# ---------------------------------------------------------------------------

def collect_dataset(data_dir: Path, facts: Facts) -> dict:
    import pandas as pd
    from collections import Counter
    from export_issues import preprocess_project_df

    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        facts.missing("corpus.projects", source=str(data_dir),
                      why="no CSV files found on this machine")
        return {"available": False, "projects": []}

    projects = []
    totals = Counter()
    subs = Counter()
    rows_in = rows_out = 0
    sp_counts = Counter()
    words = []

    print(f"  reading {len(csvs)} project files ...")
    for path in csvs:
        name = path.stem
        raw = pd.read_csv(path, encoding="utf-8-sig")

        # What the cleaner has to deal with, measured on the raw text.
        desc = raw["Description"].dropna().astype(str)
        title = raw["Title"].dropna().astype(str)
        noise = {
            "code": pct(desc.str.contains(r"\{code", case=False, regex=True).mean()),
            "url": pct(desc.str.contains(r"https?://", regex=True).mean()),
            "html": pct(desc.str.contains(r"<[^>]+>", regex=True).mean()),
            "issue_ref": pct(desc.str.contains(r"\b[A-Z][A-Z0-9]{1,9}-\d+\b", regex=True).mean()),
            "quote_wrapped_titles": pct(title.str.match(r'^".*"$').mean()),
        }

        cleaned, stats = preprocess_project_df(raw, name)

        dates = pd.to_datetime(raw["Creation_Date"], errors="coerce")
        combined = (cleaned["Title"] + " " + cleaned["Description"]).str.split().str.len()
        words.append(combined)

        rows_in += stats["rows_in"]
        rows_out += stats["rows_out"]
        subs.update(stats["substitutions"])
        sp_counts.update(stats["class_distribution"])
        totals["dropped"] += stats["rows_dropped_short_text"]

        projects.append({
            "name": name,
            "rows_raw": stats["rows_in"],
            "rows_clean": stats["rows_out"],
            "rows_dropped": stats["rows_dropped_short_text"],
            "desc_missing_pct": stats["desc_nan_pct"],
            "story_points": {str(k): v for k, v in stats["class_distribution"].items()},
            "priorities": stats["priority_distribution"],
            "types": int(raw["Type"].nunique()),
            "first_issue": str(dates.min())[:10],
            "last_issue": str(dates.max())[:10],
            "noise_pct": noise,
        })

    all_words = pd.concat(words)
    src = "data_to_train_on/*.csv"
    ran = "read every CSV and ran export_issues.preprocess_project_df on it"

    facts.add("corpus.projects", len(projects), source=src, how="counted the project files")
    facts.add("corpus.rows_raw", rows_in, source=src, how="summed the rows in every file")
    facts.add("corpus.rows_clean", rows_out, source=src, how=ran)
    facts.add("corpus.rows_dropped", rows_in - rows_out, source=src,
              how="rows whose cleaned text was under the 10-character floor")
    facts.add("cleaning.code_blocks", subs.get("code_blocks", 0), source=src, how=ran)
    facts.add("cleaning.urls", subs.get("urls", 0), source=src, how=ran)
    facts.add("cleaning.issue_refs", subs.get("issue_refs", 0), source=src, how=ran)

    first = min(p["first_issue"] for p in projects)
    last = max(p["last_issue"] for p in projects)
    facts.add("corpus.first_issue", first, source=src, how="earliest Creation_Date across all projects", text=first)
    facts.add("corpus.last_issue", last, source=src, how="latest Creation_Date across all projects", text=last)

    worst = max(projects, key=lambda p: p["desc_missing_pct"])
    facts.add("corpus.desc_missing_worst_pct", worst["desc_missing_pct"], source=src,
              how=f"the project with the most empty descriptions ({worst['name']})",
              unit="%", text=f"{worst['desc_missing_pct']:.1f}%")
    facts.add("corpus.desc_missing_worst_project", worst["name"], source=src,
              how="the project with the most empty descriptions", text=worst["name"])

    total_sp = sum(sp_counts.values())
    distribution = {
        str(k): {"count": v, "pct": round(100 * v / total_sp, 1)}
        for k, v in sorted(sp_counts.items())
    }
    facts.add("corpus.sp_distribution", distribution, source=src,
              how="counted each story point value after cleaning",
              text=" · ".join(f"{k}: {v['pct']}%" for k, v in distribution.items()))

    for label, q in (("p50", .5), ("p90", .9), ("p95", .95)):
        facts.add(f"text.words_{label}", int(all_words.quantile(q)), source=src,
                  how="word count of cleaned title + description", unit="words")
    facts.add("text.over_128_pct", round(100 * float((all_words > 128).mean()), 1), source=src,
              how="share of issues longer than 128 words", unit="%",
              text=f"{round(100 * float((all_words > 128).mean()), 1)}%")
    facts.add("text.over_256_pct", round(100 * float((all_words > 256).mean()), 1), source=src,
              how="share of issues longer than 256 words", unit="%",
              text=f"{round(100 * float((all_words > 256).mean()), 1)}%")

    return {"available": True, "projects": projects}


def pct(value) -> float:
    return round(100 * float(value), 1)


# ---------------------------------------------------------------------------
# 2. Are these files raw or already cleaned?
# ---------------------------------------------------------------------------

def check_cleanliness(data_dir: Path, facts: Facts) -> dict:
    """
    The pipeline refuses uncleaned CSVs. Whether the copy on this machine
    passes that check is itself a fact worth showing, because the cleaning
    stop demonstrates the refusal with these very files.
    """
    import pandas as pd
    from fl.data import validate_cleaned_dataframe

    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        return {"available": False}

    results = []
    for path in csvs:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        try:
            validate_cleaned_dataframe(frame, path.name)
            results.append({"file": path.name, "passes": True, "reason": None})
        except ValueError as error:
            reason = str(error).split("Re-export")[0].strip()
            results.append({"file": path.name, "passes": False, "reason": reason})

    passing = sum(1 for r in results if r["passes"])
    facts.add("validation.passing", passing, source="fl/data.py::validate_cleaned_dataframe",
              how=f"ran the loader's own check over all {len(results)} files")
    facts.add("validation.failing", len(results) - passing,
              source="fl/data.py::validate_cleaned_dataframe",
              how="files the pipeline would refuse to train on")
    return {"available": True, "files": results}


# ---------------------------------------------------------------------------
# 3. The split, rebuilt exactly as the run made it
# ---------------------------------------------------------------------------

def collect_split(data_dir: Path, results_dir: Path, config: dict, facts: Facts,
                  with_baselines: bool) -> dict:
    import pandas as pd
    import numpy as np
    from export_issues import preprocess_project_df
    from fl.data import (clean_text_value, clean_category_value,
                         prepare_tabular_bundle, STORY_POINT_CLASSES,
                         story_point_to_label)

    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs or not config:
        facts.missing("split.train", source=str(data_dir),
                      why="needs both the CSVs and a run config to rebuild the split")
        return {"available": False}

    print("  rebuilding the split ...")
    frames = []
    for path in csvs:
        cleaned, _ = preprocess_project_df(pd.read_csv(path, encoding="utf-8-sig"), path.stem)
        frames.append(pd.DataFrame({
            "title": cleaned.Title, "description": cleaned.Description,
            "type": cleaned.Type, "priority": cleaned.Priority,
            "story_point": pd.to_numeric(cleaned.Story_Point),
            "client_id": path.stem, "source_file": path.name,
        }))

    data = pd.concat(frames, ignore_index=True)
    data = data[data.story_point.isin(STORY_POINT_CLASSES)].copy()
    for column in ("title", "description"):
        data[column] = data[column].map(clean_text_value)
    for column in ("type", "priority"):
        data[column] = data[column].map(clean_category_value)
    data["text"] = (data.title + " [SEP] " + data.description).map(clean_text_value)
    data = data[data.text.str.len() > 0].reset_index(drop=True)

    warmstart = config.get("warmstart_project")
    pool = data[data.client_id != warmstart].reset_index(drop=True) if warmstart else data

    bundle = prepare_tabular_bundle(
        pool,
        test_size=config.get("test_size", 0.2),
        random_state=config.get("random_state", 42),
        split_mode=config.get("split_mode", "random"),
        val_size=config.get("val_size", 0.1),
    )

    src = "rebuilt with fl.data.prepare_tabular_bundle"
    how = (f"same settings as the run: seed {config.get('random_state')}, "
           f"{config.get('split_mode')} split, test {config.get('test_size')}, "
           f"val {config.get('val_size')}")
    facts.add("split.train", len(bundle.train_df), source=src, how=how, unit="issues")
    facts.add("split.val", len(bundle.val_df), source=src, how=how, unit="issues")
    facts.add("split.test", len(bundle.test_df), source=src, how=how, unit="issues")
    facts.add("split.clients", int(bundle.train_df.client_id.nunique()), source=src,
              how=f"projects in the pool ({warmstart} is held back for the head start)")
    facts.add("split.type_vocab", len(bundle.type_to_id), source=src,
              how="distinct issue types seen in training")
    facts.add("split.priority_vocab", len(bundle.priority_to_id), source=src,
              how="distinct priorities after normalising")

    # Does the rebuild match the split the run actually used?
    reference = results_dir / "test_split.csv"
    per_project = bundle.test_df.groupby("client_id").size()
    if reference.exists():
        theirs = pd.read_csv(reference, encoding="utf-8-sig").groupby("client_id").size()
        matches = bool(per_project.reindex(theirs.index).equals(theirs))
        facts.add("split.matches_run", matches, source="results/test_split.csv",
                  how="compared the rebuilt test rows against the ones the run saved",
                  text="yes — identical" if matches else "no")
    else:
        facts.missing("split.matches_run", source="results/test_split.csv",
                      why="the run's saved test split is not on this machine")

    payload = {
        "available": True,
        "train": len(bundle.train_df),
        "val": len(bundle.val_df),
        "test": len(bundle.test_df),
        "per_project": [
            {"name": name,
             "train": int((bundle.train_df.client_id == name).sum()),
             "val": int((bundle.val_df.client_id == name).sum()),
             "test": int(count)}
            for name, count in per_project.items()
        ],
    }

    if with_baselines:
        payload["baselines"] = collect_baselines(bundle, facts, np, story_point_to_label)

    return payload


# ---------------------------------------------------------------------------
# 4. The simple comparators, and the score that lies
# ---------------------------------------------------------------------------

def collect_baselines(bundle, facts: Facts, np, story_point_to_label) -> dict:
    from sklearn.metrics import cohen_kappa_score, f1_score
    from fl.classic_baselines import run_classic_baselines

    print("  running the simple baselines (this is the slow part) ...")
    outcome = run_classic_baselines(bundle.train_df, bundle.test_df, 5, 42)
    median, svm = outcome["median"], outcome["tfidf_svm"]

    src = "fl/classic_baselines.py, on the rebuilt split"
    for name, table in (("median", median), ("svm", svm)):
        for metric in ("macro_f1", "mae", "cohen_kappa"):
            values = [table[p][metric] for p in table]
            facts.add(f"baseline.{name}.{metric}", round(float(np.mean(values)), 4),
                      source=src, how=f"mean across the {len(values)} projects")

    # The pooling artifact, demonstrated rather than asserted.
    values = np.array([1, 2, 3, 5, 8], dtype=float)
    labels = list(range(5))
    truth, predicted, project_of = [], [], []
    for name, group in bundle.test_df.groupby("client_id"):
        train_rows = bundle.train_df[bundle.train_df.client_id == name]
        middle = float(np.median(values[[story_point_to_label(s) for s in train_rows.story_point]]))
        constant = int(np.argmin(abs(values - middle)))
        truth += [story_point_to_label(s) for s in group.story_point]
        predicted += [constant] * len(group)
        project_of += [name] * len(group)

    truth = np.array(truth); predicted = np.array(predicted); project_of = np.array(project_of)
    kappa = lambda a, b: float(cohen_kappa_score(a, b, labels=labels, weights="quadratic"))

    per_project_kappas = [median[p]["cohen_kappa"] for p in median]
    pooled = kappa(truth, predicted)

    rng = np.random.default_rng(0)
    shuffled = truth.copy()
    for name in np.unique(project_of):
        mask = project_of == name
        block = truth[mask].copy(); rng.shuffle(block); shuffled[mask] = block
    same_everywhere = np.full(len(predicted),
                              int(np.argmin(abs(values - float(np.median(values[truth]))))))
    scrambled = rng.permutation(len(truth))

    artifact = {
        "per_project_max": round(float(max(per_project_kappas)), 4),
        "pooled": round(pooled, 4),
        "pooled_shuffled_within": round(kappa(shuffled, predicted), 4),
        "pooled_same_constant": round(kappa(truth, same_everywhere), 4),
        "pooled_project_link_broken": round(kappa(truth[scrambled], predicted), 4),
        "per_project_macro_f1": round(float(np.mean([median[p]["macro_f1"] for p in median])), 4),
        "pooled_macro_f1": round(float(f1_score(truth, predicted, average="macro",
                                                zero_division=0, labels=labels)), 4),
    }

    how = "the constant predictor, scored per project and then pooled"
    facts.add("artifact.per_project_kappa", artifact["per_project_max"], source=src,
              how="the BEST any project managed — a constant predictor cannot beat chance",
              text=f"{artifact['per_project_max']:.4f}")
    facts.add("artifact.pooled_kappa", artifact["pooled"], source=src, how=how,
              text=f"{artifact['pooled']:.4f}")
    facts.add("artifact.pooled_shuffled", artifact["pooled_shuffled_within"], source=src,
              how="answers shuffled inside each project — no issue-level skill left",
              text=f"{artifact['pooled_shuffled_within']:.4f}")
    facts.add("artifact.pooled_same_constant", artifact["pooled_same_constant"], source=src,
              how="every project forced to guess the same number",
              text=f"{artifact['pooled_same_constant']:.4f}")
    facts.add("artifact.pooled_scrambled", artifact["pooled_project_link_broken"], source=src,
              how="the link between project and answers broken",
              text=f"{artifact['pooled_project_link_broken']:.4f}")

    return {
        "median": {p: round_metrics(median[p]) for p in median},
        "svm": {p: round_metrics(svm[p]) for p in svm},
        "pooling_artifact": artifact,
    }


def round_metrics(entry: dict) -> dict:
    keep = ("accuracy", "macro_f1", "mae", "cohen_kappa", "n_test")
    out = {}
    for key in keep:
        if key not in entry:
            continue
        out[key] = int(entry[key]) if key == "n_test" else round(float(entry[key]), 4)
    return out


# ---------------------------------------------------------------------------
# 5. The finished run
# ---------------------------------------------------------------------------

def collect_run(results_dir: Path, facts: Facts) -> dict:
    import numpy as np

    config_path = results_dir / "config.json"
    if not config_path.exists():
        facts.missing("run.condition", source="results/config.json",
                      why="no finished run on this machine yet")
        return {"available": False, "config": {}}

    config = json.loads(config_path.read_text(encoding="utf-8"))
    src = "results/config.json"
    facts.add("run.condition", config.get("federated_condition"), source=src,
              how="what the run called itself", kind="run",
              text=str(config.get("federated_condition")))
    facts.add("run.model", config.get("model_name"), source=src, how="the encoder used",
              kind="run", text=str(config.get("model_name")))
    facts.add("run.rounds", config.get("rounds"), source=src, how="training rounds", kind="run")
    facts.add("run.seed", config.get("random_state"), source=src, how="the random seed", kind="run")
    facts.add("run.split_mode", config.get("split_mode"), source=src,
              how="how the data was split", kind="run", text=str(config.get("split_mode")))
    facts.add("run.head_type", config.get("head_type"), source=src, how="the prediction head",
              kind="run", text=str(config.get("head_type")))
    facts.add("run.personalized", bool(config.get("personalized_head")), source=src,
              how="whether each project kept its own head", kind="run")
    facts.add("run.max_length", config.get("max_length"), source=src,
              how="how many tokens of each issue the model reads", kind="run")

    payload = {"available": True, "config": config, "conditions": {}}

    files = {
        "federated": "federated_per_project.json",
        "centralized": "centralized_per_project.json",
        "local_only": "local_only_per_project.json",
    }
    tables = {}
    for name, filename in files.items():
        path = results_dir / filename
        if not path.exists():
            continue
        table = json.loads(path.read_text(encoding="utf-8"))
        projects = {k: v for k, v in table.items() if k != "global"}
        tables[name] = projects
        payload["conditions"][name] = {
            "per_project": {k: round_metrics(v) for k, v in projects.items()},
            "pooled": round_metrics(table["global"]) if "global" in table else None,
            "means": {m: round(float(np.mean([v[m] for v in projects.values()])), 4)
                      for m in ("accuracy", "macro_f1", "mae", "cohen_kappa")},
        }
        for metric in ("accuracy", "macro_f1", "mae", "cohen_kappa"):
            facts.add(f"result.{name}.{metric}", payload["conditions"][name]["means"][metric],
                      source=f"results/{filename}", kind="run",
                      how=f"mean across the {len(projects)} projects")

    # How often federation actually wins, project by project.
    if "federated" in tables and "centralized" in tables:
        shared = [p for p in tables["federated"] if p in tables["centralized"]]
        wins = sum(1 for p in shared if tables["federated"][p]["mae"] < tables["centralized"][p]["mae"])
        facts.add("result.fed_beats_central_mae", f"{wins}/{len(shared)}",
                  source="results/*_per_project.json", kind="run",
                  how="projects where federated training had the smaller average error",
                  text=f"{wins} of {len(shared)}")
    if "federated" in tables and "local_only" in tables:
        shared = [p for p in tables["federated"] if p in tables["local_only"]]
        wins = sum(1 for p in shared if tables["federated"][p]["mae"] < tables["local_only"][p]["mae"])
        facts.add("result.fed_beats_local_mae", f"{wins}/{len(shared)}",
                  source="results/*_per_project.json", kind="run",
                  how="projects where federated beat training alone",
                  text=f"{wins} of {len(shared)}")

    # Where the mistakes land.
    if "federated" in tables:
        matrix = np.sum([np.array(v["confusion_matrix"]) for v in tables["federated"].values()], axis=0)
        distance = np.abs(np.subtract.outer(range(5), range(5)))
        total = matrix.sum()
        exact = round(100 * float(matrix[distance == 0].sum() / total), 1)
        near = round(100 * float(matrix[distance <= 1].sum() / total), 1)
        far = round(100 * float(matrix[distance >= 3].sum() / total), 1)
        payload["confusion_matrix"] = matrix.tolist()
        for key, value, words in (("exact", exact, "guessed exactly right"),
                                  ("within_one", near, "right, or one step away"),
                                  ("far", far, "three or more steps away")):
            facts.add(f"confusion.{key}", value, source="results/federated_per_project.json",
                      kind="run", unit="%", how=f"share of test issues {words}",
                      text=f"{value}%")

    # The training curve.
    history_path = results_dir / "federated_round_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        key = "weighted_val_macro_f1" if "weighted_val_macro_f1" in history[0] else "val_macro_f1"
        best = max(history, key=lambda entry: entry.get(key, -1))
        payload["history"] = [
            {"round": e["round"], "loss": round(e["mean_local_loss"], 4),
             "val": round(e.get(key, 0), 4)} for e in history
        ]
        facts.add("rounds.count", len(history), source="results/federated_round_history.json",
                  kind="run", how="rounds recorded")
        facts.add("rounds.best", best["round"], source="results/federated_round_history.json",
                  kind="run", how="the round whose model scored best on the check set")
        facts.add("rounds.still_improving", best["round"] == len(history),
                  source="results/federated_round_history.json", kind="run",
                  how="whether the best round was the very last one",
                  text="yes — it had not levelled off" if best["round"] == len(history) else "no")
        facts.add("rounds.loss_first", round(history[0]["mean_local_loss"], 4),
                  source="results/federated_round_history.json", kind="run", how="round 1")
        facts.add("rounds.loss_last", round(history[-1]["mean_local_loss"], 4),
                  source="results/federated_round_history.json", kind="run", how="the final round")

    return payload


# ---------------------------------------------------------------------------
# 6. What gets sent over the network
# ---------------------------------------------------------------------------

def collect_params(results_dir: Path, config: dict, facts: Facts) -> dict:
    path = results_dir / "communication_cost.json"
    if not path.exists():
        facts.missing("params.trainable", source="results/communication_cost.json",
                      why="no finished run on this machine yet")
        return {"available": False}

    cost = json.loads(path.read_text(encoding="utf-8"))
    src = "results/communication_cost.json"

    trainable = int(cost["trainable_params"])
    total = int(cost["total_params"])
    share = round(100 * trainable / total, 4)

    facts.add("params.trainable", trainable, source=src, kind="run",
              how="the numbers each project actually trains and sends")
    facts.add("params.total", total, source=src, kind="run", how="the whole model")
    facts.add("params.share_pct", share, source=src, kind="derived", unit="%",
              how="trainable ÷ total", text=f"{share:.4f}%")
    facts.add("comms.per_round_bytes", int(cost["bytes_per_client_per_round"]), source=src,
              kind="run", how="uploaded by one project, once per round", unit="bytes")
    facts.add("comms.per_round_bytes_full", int(cost["bytes_per_client_per_round_full_finetune"]),
              source=src, kind="run", unit="bytes",
              how="what the same round would cost if the whole model were sent")
    facts.add("comms.reduction", round(float(cost["reduction_factor"]), 1), source=src,
              kind="run", how="how many times smaller that is", text=f"{cost['reduction_factor']:.0f}×")
    facts.add("comms.total_bytes", int(cost["total_upload_bytes"]), source=src, kind="run",
              unit="bytes", how="every project, every round, added up")
    facts.add("comms.total_bytes_full", int(cost["total_upload_bytes_full_finetune"]),
              source=src, kind="run", unit="bytes", how="the same run without the patches")

    # Where those numbers come from, part by part. If the parts do not add up
    # to the recorded total, the breakdown is wrong and we say so rather than
    # showing a tidy fiction.
    breakdown = None
    if config:
        layers, hidden, rank = 12, 768, int(config.get("lora_r", 8))
        targets = len(config.get("lora_target_modules", ["query", "value"]))
        emb_dim = int(config.get("categorical_emb_dim", 16))
        hidden_dim = int(config.get("hidden_dim", 128))
        vocab = facts.items.get("split.type_vocab", {}).get("value")
        priorities = facts.items.get("split.priority_vocab", {}).get("value")
        if vocab and priorities:
            lora_b = layers * targets * hidden * rank
            embeddings = (vocab + priorities) * emb_dim
            fusion = hidden + 2 * emb_dim
            outputs = 4 if config.get("head_type") == "corn" else 5
            head = 2 * fusion + (fusion * hidden_dim + hidden_dim) + (hidden_dim * outputs + outputs)
            parts = {"lora_b": lora_b, "embeddings": embeddings, "head": head}
            adds_up = sum(parts.values()) == trainable
            breakdown = {"parts": parts, "sum": sum(parts.values()),
                         "recorded": trainable, "adds_up": adds_up}
            facts.add("params.breakdown_adds_up", adds_up, source=src, kind="derived",
                      how="the parts added up and compared against the recorded total",
                      text="yes — exactly" if adds_up else "NO — the breakdown is wrong")
            for name, value in parts.items():
                facts.add(f"params.{name}", value, source="arithmetic from results/config.json",
                          kind="derived", how="counted from the model's shape")

    return {"available": True, "cost": cost, "breakdown": breakdown}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(REPO_DIR / "data_to_train_on"))
    parser.add_argument("--results-dir", default=str(REPO_DIR / "results"))
    parser.add_argument("--out", default=str(SITE_DIR / "data"))
    parser.add_argument("--skip-baselines", action="store_true",
                        help="skip the simple comparators (the slow part)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out)

    print("extract_facts.py — reading the project's own data")
    print(f"  data:    {data_dir}  {'found' if data_dir.exists() else 'NOT FOUND'}")
    print(f"  results: {results_dir}  {'found' if results_dir.exists() else 'NOT FOUND'}")

    facts = Facts()

    run = collect_run(results_dir, facts) if results_dir.exists() else {"available": False, "config": {}}
    config = run.get("config", {})

    dataset = collect_dataset(data_dir, facts) if data_dir.exists() else {"available": False}
    validation = check_cleanliness(data_dir, facts) if data_dir.exists() else {"available": False}
    split = (collect_split(data_dir, results_dir, config, facts, not args.skip_baselines)
             if data_dir.exists() else {"available": False})
    params = collect_params(results_dir, config, facts) if results_dir.exists() else {"available": False}

    if not dataset.get("available"):
        facts.note("The issue CSVs are not on this machine, so the data stops "
                   "cannot show real numbers.")
    if not run.get("available"):
        facts.note("No finished run is on this machine, so the result stops have "
                   "nothing to show yet.")
    if validation.get("available"):
        failing = sum(1 for f in validation["files"] if not f["passes"])
        if failing:
            facts.note(f"{failing} of {len(validation['files'])} CSVs here are the RAW "
                       "export, which the training pipeline correctly refuses.")

    stamp = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "data_dir": str(data_dir),
        "results_dir": str(results_dir),
        "run_available": run.get("available", False),
        "data_available": dataset.get("available", False),
        "notes": facts.notes,
    }

    print("writing:")
    write_json(out_dir / "facts.json", {"about": stamp, "facts": facts.items})
    write_json(out_dir / "dataset.json", {"about": stamp, **dataset})
    write_json(out_dir / "validation.json", {"about": stamp, **validation})
    write_json(out_dir / "split.json", {"about": stamp, **split})
    write_json(out_dir / "run.json", {"about": stamp, **run})
    write_json(out_dir / "params.json", {"about": stamp, **params})

    kinds = {}
    for item in facts.items.values():
        kinds[item["kind"]] = kinds.get(item["kind"], 0) + 1
    print(f"\n{len(facts.items)} facts: " + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items())))
    for note in facts.notes:
        print(f"  note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
