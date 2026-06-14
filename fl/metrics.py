from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5,
) -> Dict[str, Any]:
    """
    Primary metric: macro-F1 (robust to class imbalance).
    Also returns accuracy, per-class F1, and confusion matrix.
    """
    labels = list(range(num_classes))
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels))
    per_class_f1: List[float] = f1_score(
        y_true, y_pred, average=None, zero_division=0, labels=labels
    ).tolist()
    cm: List[List[int]] = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
    }


def format_metrics(prefix: str, metrics: Dict[str, Any]) -> str:
    per_class = "  ".join(f"C{i}:{v:.3f}" for i, v in enumerate(metrics["per_class_f1"]))
    return (
        f"{prefix:<20} | "
        f"Acc: {metrics['accuracy']:.4f} | "
        f"Macro-F1: {metrics['macro_f1']:.4f} | "
        f"Per-class F1: [{per_class}]"
    )
