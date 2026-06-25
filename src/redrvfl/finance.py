from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf
from hyperopt import hp

from .data import make_forecasting_frame
from .metrics import mean_absolute_percentage_error, root_mean_squared_error
from .models import EDRVFLRegressor, REDRVFLRegressor, RVFLRegressor
from .tuning import layerwise_tune_edrvfl, layerwise_tune_redrvfl, tune_rvfl


@dataclass(frozen=True)
class FinanceRunResult:
    dataset: str
    seed: int
    model: str
    rmse: float
    mae: float
    mape: float
    tuning_seconds: float
    training_seconds: float
    testing_seconds: float
    best_params: dict


def download_dji(
    output_path: str | Path = "datasets/DJI.csv",
    start: str = "2013-01-01",
    end: str = "2023-01-01",
) -> Path:
    """Download the paper's DJI Yahoo Finance dataset to a CSV file."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = yf.download("^DJI", start=start, end=end, progress=False, auto_adjust=False)
    if data.empty:
        raise RuntimeError("Yahoo Finance returned no rows for ^DJI.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [column[0] for column in data.columns]
    data = data.reset_index()
    keep = [column for column in ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"] if column in data]
    data[keep].to_csv(output, index=False)
    return output


def load_dji_close(path: str | Path = "datasets/DJI.csv") -> np.ndarray:
    """Load DJI close prices from the local dataset CSV."""

    frame = pd.read_csv(path)
    close_column = "Close" if "Close" in frame.columns else "Adj Close"
    return frame[close_column].ffill().bfill().to_numpy(dtype=float)


def run_dji_paper_experiment(
    dataset_path: str | Path = "datasets/DJI.csv",
    seeds: Iterable[int] = (0,),
    horizon: int = 20,
    look_ahead: int = 1,
    n_layers: int = 10,
    max_evals: int = 100,
) -> list[FinanceRunResult]:
    """Run the financial time series DJI experiment without writing result artifacts.

    The split follows the paper: 70% train, 10% validation, 20% test in
    chronological order, then train+validation are combined for final testing.
    """

    close = load_dji_close(dataset_path)
    X, y = make_forecasting_frame(close, order=horizon, horizon=look_ahead)
    train_idx, val_idx, full_train_idx, test_idx = chronological_indices(len(X))
    results: list[FinanceRunResult] = []

    for seed in seeds:
        train_scaled, val_scaled, full_scaled, test_scaled, inverse_y = _scaled_splits(
            X, y, train_idx, val_idx, full_train_idx, test_idx
        )

        rvfl_result = _run_tuned_model(
            "RVFL",
            tune_rvfl,
            RVFLRegressor,
            train_scaled,
            val_scaled,
            full_scaled,
            test_scaled,
            _rvfl_space(seed),
            max_evals=max_evals,
            seed=seed,
            inverse_y=inverse_y,
        )
        results.append(rvfl_result)

        ed_result = _run_layerwise_model(
            "EDRVFL",
            layerwise_tune_edrvfl,
            EDRVFLRegressor,
            train_scaled,
            val_scaled,
            full_scaled,
            test_scaled,
            n_layers=n_layers,
            max_evals=max_evals,
            seed=seed,
            inverse_y=inverse_y,
        )
        results.append(ed_result)

        red_result = _run_layerwise_model(
            "REDRVFL",
            layerwise_tune_redrvfl,
            REDRVFLRegressor,
            train_scaled,
            val_scaled,
            full_scaled,
            test_scaled,
            n_layers=n_layers,
            max_evals=max_evals,
            seed=seed,
            fixed_params={"recurrent_scale": 0.1, "random_state": seed},
            inverse_y=inverse_y,
        )
        results.append(red_result)

    return results


def chronological_indices(n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    test_len = int(0.2 * n_samples)
    val_len = int(0.1 * n_samples)
    train_len = n_samples - val_len - test_len
    train_idx = np.arange(train_len)
    val_idx = np.arange(train_len, train_len + val_len)
    full_train_idx = np.arange(train_len + val_len)
    test_idx = np.arange(train_len + val_len, n_samples)
    return train_idx, val_idx, full_train_idx, test_idx


def _run_tuned_model(
    name,
    tuner,
    estimator_cls,
    train_scaled,
    val_scaled,
    full_scaled,
    test_scaled,
    space,
    max_evals,
    seed,
    inverse_y,
):
    X_train, y_train = train_scaled
    X_val, y_val = val_scaled
    X_full, y_full = full_scaled
    X_test, y_test = test_scaled
    tune_X = np.vstack([X_train, X_val])
    tune_y = np.concatenate([y_train, y_val])

    start = perf_counter()
    tuned = tuner(
        tune_X,
        tune_y,
        space=space,
        validation_fraction=len(y_val) / len(tune_y),
        refit=False,
        max_evals=max_evals,
        random_state=seed,
    )
    tuning_seconds = perf_counter() - start

    start = perf_counter()
    model = estimator_cls(**tuned.best_params).fit(X_full, y_full)
    training_seconds = perf_counter() - start

    start = perf_counter()
    pred = model.predict(X_test)
    testing_seconds = perf_counter() - start
    return _finance_result(
        name,
        seed,
        inverse_y(y_test),
        inverse_y(pred),
        tuning_seconds,
        training_seconds,
        testing_seconds,
        tuned.best_params,
    )


def _run_layerwise_model(
    name,
    tuner,
    estimator_cls,
    train_scaled,
    val_scaled,
    full_scaled,
    test_scaled,
    n_layers,
    max_evals,
    seed,
    fixed_params=None,
    inverse_y=None,
):
    X_train, y_train = train_scaled
    X_val, y_val = val_scaled
    X_full, y_full = full_scaled
    X_test, y_test = test_scaled
    fixed_params = dict(fixed_params or {"random_state": seed})
    tune_X = np.vstack([X_train, X_val])
    tune_y = np.concatenate([y_train, y_val])

    start = perf_counter()
    tuned = tuner(
        tune_X,
        tune_y,
        n_layers=n_layers,
        layer_space=_layer_space(),
        fixed_params=fixed_params,
        validation_fraction=len(y_val) / len(tune_y),
        refit=False,
        max_evals=max_evals,
        random_state=seed,
    )
    tuning_seconds = perf_counter() - start

    model_params = dict(tuned.best_params)
    layer_params = model_params.pop("layer_params")
    start = perf_counter()
    model = estimator_cls(layer_params=layer_params, **model_params).fit(X_full, y_full)
    training_seconds = perf_counter() - start

    start = perf_counter()
    pred = model.predict(X_test)
    testing_seconds = perf_counter() - start
    return _finance_result(
        name,
        seed,
        inverse_y(y_test),
        inverse_y(pred),
        tuning_seconds,
        training_seconds,
        testing_seconds,
        tuned.best_params,
    )


def _scaled_splits(X, y, train_idx, val_idx, full_train_idx, test_idx):
    x_min = X[train_idx].min(axis=0)
    x_range = np.maximum(X[train_idx].max(axis=0) - x_min, 1e-12)
    y_min = y[train_idx].min()
    y_range = max(float(y[train_idx].max() - y_min), 1e-12)

    def scale_x(values):
        return (values - x_min) / x_range

    def scale_y(values):
        return (values - y_min) / y_range

    def inverse_y(values):
        return np.asarray(values, dtype=float) * y_range + y_min

    return (
        (scale_x(X[train_idx]), scale_y(y[train_idx])),
        (scale_x(X[val_idx]), scale_y(y[val_idx])),
        (scale_x(X[full_train_idx]), scale_y(y[full_train_idx])),
        (scale_x(X[test_idx]), scale_y(y[test_idx])),
        inverse_y,
    )


def _rvfl_space(seed):
    return {
        "n_hidden": hp.quniform("n_hidden", 20, 200, 1),
        "regularization": hp.uniform("regularization", 0, 1),
        "input_scale": hp.uniform("input_scale", 0, 1),
        "random_state": seed,
    }


def _layer_space():
    return {
        "n_hidden": hp.quniform("n_hidden", 20, 200, 1),
        "regularization": hp.uniform("regularization", 0, 1),
        "input_scale": hp.uniform("input_scale", 0, 1),
    }


def _finance_result(name, seed, y_true, y_pred, tuning_seconds, training_seconds, testing_seconds, best_params):
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return FinanceRunResult(
        dataset="DJI",
        seed=seed,
        model=name,
        rmse=root_mean_squared_error(truth, pred),
        mae=float(np.mean(np.abs(truth - pred))),
        mape=mean_absolute_percentage_error(truth, pred),
        tuning_seconds=tuning_seconds,
        training_seconds=training_seconds,
        testing_seconds=testing_seconds,
        best_params=best_params,
    )
