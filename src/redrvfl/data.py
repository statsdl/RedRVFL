from __future__ import annotations

from typing import Iterable

import numpy as np


def make_forecasting_frame(
    series: Iterable[float] | np.ndarray,
    order: int = 1,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a univariate or multivariate series into supervised samples."""

    values = np.asarray(series, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError("series must be a 1D or 2D array.")
    if order <= 0:
        raise ValueError("order must be positive.")
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    n_samples = values.shape[0] - order - horizon + 1
    if n_samples <= 0:
        raise ValueError("series is too short for the requested order and horizon.")

    X = np.zeros((n_samples, values.shape[1] * order))
    y = np.zeros((n_samples, values.shape[1]))
    for i in range(n_samples):
        X[i] = values[i : i + order].ravel()
        y[i] = values[i + order + horizon - 1]
    return X, y.ravel() if y.shape[1] == 1 else y
