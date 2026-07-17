"""
Classic (non-deep) per-project baselines for empirical comparison (RQ5).

These are NOT federated conditions — each is trained *within* each project on
the SAME train/test split as the deep pipeline (they consume the prepared
`bundle.train_df` / `bundle.test_df`, so the per-project pairing is byte-identical
to the federated/centralized/local-only conditions). They contextualize the
federated numbers against the cheap comparators the SPE replication literature
(Tawosi et al.) shows are hard to beat:

  * tfidf_svm : per-project TF-IDF features + Linear SVM classifier. The within-
                project text-signal ceiling / canonical "simple model".
  * median    : per-project constant predictor = the class whose story-point
                value is nearest the project's median train story point. The
                trivial MAE floor (a deep model that can't beat it is a red flag).

Both write the standard per-project schema
(`{client_id: evaluate_classification(...) + n_test, ...}`) so the existing
statistics / summary tooling can consume them like any other condition.

NO pooled "global" entry is emitted, and that is deliberate. These are
*per-project* models: each fits its own project, so a pooled metric silently
rewards a predictor for encoding project identity. Measured on the real corpus
(seed 42, random split), the `median` predictor — a per-project CONSTANT — scores:

    per-project quadratic kappa : 0.0000 for all 18 projects (correct; a constant
                                  predictor is exactly chance, and kappa is
                                  chance-corrected)
    pooled "global" kappa       : 0.5006  (pure artifact)
    per-project macro-F1        : ~0.10   |  pooled macro-F1: 0.3023 (also inflated)

The pooled figure is high because chance agreement is computed from the *pooled*
marginals, so between-project differences in label distribution — which the
per-project constant encodes for free — read as skill. Reporting it next to a deep
model's pooled score would compare a trivial baseline against the real conditions
on an inflated scale. The unit of analysis is per-project throughout this project
(compute_statistics.py pairs per (seed, project)), so per-project entries plus a
mean across projects is both the honest and the consistent choice. This mirrors
personalized-head mode, which omits the pooled entry for the same reason.

Pure pandas/numpy/sklearn; `evaluate_classification` is imported lazily so this
module stays cheap to import.
"""
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from fl.data import STORY_POINT_CLASSES, story_point_to_label

# Story-point *values* indexed by class id (0..4) -> (1, 2, 3, 5, 8).
_SP_VALUES = np.array(STORY_POINT_CLASSES, dtype=float)


def _labels_for(frame: pd.DataFrame) -> np.ndarray:
    """Map a frame's story_point column to class indices (0..4)."""
    return np.array(
        [story_point_to_label(sp) for sp in frame["story_point"].tolist()],
        dtype=np.int64,
    )


def _median_class(train_labels: np.ndarray) -> int:
    """Class id whose story-point value is nearest the median train story point.

    Operates in story-point value space (so the median is meaningful on the
    Fibonacci scale, not on arbitrary class indices), then snaps to the nearest
    of {1,2,3,5,8}. Ties resolve to the lower class (argmin returns the first).
    """
    if len(train_labels) == 0:
        return 0
    train_values = _SP_VALUES[train_labels]
    med = float(np.median(train_values))
    return int(np.argmin(np.abs(_SP_VALUES - med)))


def _tfidf_svm_predict(
    texts_train: List[str],
    y_train: np.ndarray,
    texts_test: List[str],
    random_state: int,
) -> np.ndarray:
    """Per-project TF-IDF + LinearSVC predictions on the test texts.

    Falls back to predicting the majority train class when the project has fewer
    than two classes or too few rows to fit a classifier, or if the fit/predict
    raises for any degenerate reason. Never raises.
    """
    n_classes_present = len(np.unique(y_train))
    if n_classes_present < 2 or len(y_train) < 3:
        majority = int(np.bincount(y_train).argmax()) if len(y_train) else 0
        return np.full(len(texts_test), majority, dtype=np.int64)

    # Imported here so the module is importable without sklearn's heavier
    # submodules loaded, and so a sklearn hiccup can't break module import.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    try:
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2))),
                ("svm", LinearSVC(random_state=random_state)),
            ]
        )
        pipeline.fit(texts_train, y_train)
        return pipeline.predict(texts_test).astype(np.int64)
    except Exception:
        majority = int(np.bincount(y_train).argmax())
        return np.full(len(texts_test), majority, dtype=np.int64)


def _assemble(
    per_project_true: Dict[str, np.ndarray],
    per_project_pred: Dict[str, np.ndarray],
    num_classes: int,
) -> Dict[str, Dict[str, Any]]:
    """Build the per-project results dict.

    Per-project entries only — no pooled "global" key. See the module docstring:
    pooling inflates these per-project models' scores (a constant predictor pools
    to kappa 0.50 despite scoring exactly 0.0 in every project).
    """
    from fl.metrics import evaluate_classification

    results: Dict[str, Dict[str, Any]] = {}
    for cid in per_project_true:
        true = per_project_true[cid]
        metrics = evaluate_classification(true, per_project_pred[cid], num_classes)
        metrics["n_test"] = int(len(true))
        results[cid] = metrics
    return results


def run_classic_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_classes: int,
    random_state: int,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Fit the per-project classic baselines and return their per-project results.

    Returns {"tfidf_svm": {<per-project>}, "median": {<per-project>}} in the same
    schema as evaluate_per_project / evaluate_local_only, minus the pooled "global"
    key (see the module docstring for why these must not be pooled).

    train_df / test_df must be the SAME frames used by the deep pipeline
    (bundle.train_df / bundle.test_df) so splits are identical. Only projects that
    appear in test_df are scored.
    """
    svm_true: Dict[str, np.ndarray] = {}
    svm_pred: Dict[str, np.ndarray] = {}
    med_true: Dict[str, np.ndarray] = {}
    med_pred: Dict[str, np.ndarray] = {}

    train_by_client = {str(cid): g for cid, g in train_df.groupby("client_id")}

    for cid, test_g in test_df.groupby("client_id"):
        cid = str(cid)
        test_g = test_g.reset_index(drop=True)
        y_test = _labels_for(test_g)

        train_g = train_by_client.get(cid, test_g.iloc[0:0])
        y_train = _labels_for(train_g)

        # TF-IDF + Linear SVM (within-project).
        texts_train = train_g["text"].astype(str).tolist()
        texts_test = test_g["text"].astype(str).tolist()
        svm_true[cid] = y_test
        svm_pred[cid] = _tfidf_svm_predict(texts_train, y_train, texts_test, random_state)

        # Median-story-point constant predictor.
        med_true[cid] = y_test
        med_pred[cid] = np.full(len(y_test), _median_class(y_train), dtype=np.int64)

    return {
        "tfidf_svm": _assemble(svm_true, svm_pred, num_classes),
        "median": _assemble(med_true, med_pred, num_classes),
    }
