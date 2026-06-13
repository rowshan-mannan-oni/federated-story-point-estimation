from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from fl.data import INV_LABEL_MAP


def run_prediction(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Return predicted class indices (argmax over logits)."""
    model.eval()
    preds: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = model(batch).detach().cpu()
            preds.append(logits.argmax(dim=1).numpy())

    return np.concatenate(preds, axis=0)


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5,
) -> Dict[str, Any]:
    """
    Primary metric: macro-F1 (robust to class imbalance).
    Also returns accuracy, per-class F1, confusion matrix, MAE on the
    underlying story-point values (bridge to the regression literature), and
    quadratic-weighted Cohen's Kappa (chance-corrected agreement that
    penalizes larger misses more — appropriate for the ordinal Fibonacci
    classes).
    """
    labels = list(range(num_classes))
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels))
    per_class_f1: List[float] = f1_score(
        y_true, y_pred, average=None, zero_division=0, labels=labels
    ).tolist()
    cm: List[List[int]] = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    inv_map = np.array([INV_LABEL_MAP.get(i, i) for i in range(num_classes)])
    true_sp = inv_map[y_true]
    pred_sp = inv_map[y_pred]
    mae = float(np.mean(np.abs(true_sp - pred_sp)))

    kappa = float(cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic"))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
        "mae": mae,
        "cohen_kappa": kappa,
    }


def format_metrics(prefix: str, metrics: Dict[str, Any]) -> str:
    per_class = "  ".join(f"C{i}:{v:.3f}" for i, v in enumerate(metrics["per_class_f1"]))
    return (
        f"{prefix:<20} | "
        f"Acc: {metrics['accuracy']:.4f} | "
        f"Macro-F1: {metrics['macro_f1']:.4f} | "
        f"MAE: {metrics['mae']:.4f} | "
        f"Kappa: {metrics['cohen_kappa']:.4f} | "
        f"Per-class F1: [{per_class}]"
    )


def compute_communication_cost(
    trainable_params: int,
    total_params: int,
    rounds: int,
    num_clients: int,
) -> Dict[str, Any]:
    """
    Per-round, per-client upload cost (float32), assuming only trainable
    parameters are transmitted (FFA-LoRA excludes the frozen backbone and
    frozen LoRA-A matrices). Compared against full fine-tuning, where every
    parameter would be transmitted, as the communication-cost baseline.
    """
    bytes_per_param = 4  # float32

    bytes_per_round_trainable = trainable_params * bytes_per_param
    bytes_per_round_full = total_params * bytes_per_param

    total_upload_bytes = bytes_per_round_trainable * rounds * num_clients
    total_upload_bytes_full_finetune = bytes_per_round_full * rounds * num_clients

    return {
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "bytes_per_client_per_round": int(bytes_per_round_trainable),
        "bytes_per_client_per_round_full_finetune": int(bytes_per_round_full),
        "rounds": int(rounds),
        "num_clients": int(num_clients),
        "total_upload_bytes": int(total_upload_bytes),
        "total_upload_bytes_full_finetune": int(total_upload_bytes_full_finetune),
        "reduction_factor": float(total_upload_bytes_full_finetune / max(total_upload_bytes, 1)),
    }
