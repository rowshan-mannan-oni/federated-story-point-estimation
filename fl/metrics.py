from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression metrics for story point estimation."""
    y_pred = np.clip(y_pred, a_min=0.0, a_max=None)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    denominator = np.maximum(np.abs(y_true), 1e-6)
    mape = np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape),
    }


def format_metrics(prefix: str, metrics: Dict[str, float]) -> str:
    return (
        f"{prefix:<20} | "
        f"MAE: {metrics['mae']:.4f} | "
        f"RMSE: {metrics['rmse']:.4f} | "
        f"R2: {metrics['r2']:.4f} | "
        f"MAPE: {metrics['mape']:.2f}%"
    )
