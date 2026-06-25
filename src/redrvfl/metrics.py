from __future__ import annotations

import numpy as np


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""

    truth = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.sqrt(np.mean((truth - pred) ** 2)))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean absolute percentage error with zero-safe denominator."""

    truth = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()
    denom = np.maximum(np.abs(truth), epsilon)
    return float(np.mean(np.abs((truth - pred) / denom)))
