"""Regression metrics."""

from __future__ import annotations

import numpy as np


def _check(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and have the same shape")
    return y_true, y_pred


def mse(y_true, y_pred) -> float:
    y_true, y_pred = _check(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _check(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    """Share of the variance explained by the model (1.0 = perfect)."""
    y_true, y_pred = _check(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def report(y_true, y_pred) -> dict[str, float]:
    return {"r2": r2(y_true, y_pred), "rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred)}
