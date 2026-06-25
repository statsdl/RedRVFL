from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, tpe

from .metrics import root_mean_squared_error
from .models import EDRVFLRegressor, LayerParams, REDRVFLRegressor, RVFLRegressor

Scorer = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class TuningResult:
    """Best model and validation score returned by a tuning helper."""

    model: Any
    best_params: dict[str, Any]
    best_score: float
    history: list[dict[str, Any]]


def _split(
    X: np.ndarray,
    y: np.ndarray,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    n_validation = max(1, int(round(X.shape[0] * validation_fraction)))
    n_train = X.shape[0] - n_validation
    if n_train <= 0:
        raise ValueError("Not enough samples for the requested validation split.")
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


def _tune(
    estimator_cls: type,
    X: np.ndarray,
    y: np.ndarray,
    space: dict[str, Any],
    scorer: Scorer,
    validation_fraction: float,
    refit: bool,
    max_evals: int,
    random_state: int,
) -> TuningResult:
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    X_train, X_val, y_train, y_val = _split(X_arr, y_arr, validation_fraction)
    history = []

    def objective(params: dict[str, Any]) -> dict[str, Any]:
        clean_params = _clean_params(params)
        model = estimator_cls(**clean_params).fit(X_train, y_train)
        predictions = model.predict(X_val)
        score = float(scorer(y_val, predictions))
        record = {"score": score, **clean_params}
        history.append(record)
        return {"loss": score, "status": STATUS_OK}

    if max_evals <= 0:
        raise ValueError("max_evals must be positive.")
    trials = Trials()
    fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(random_state),
        show_progressbar=False,
    )
    best_record = min(history, key=lambda item: item["score"])
    best_params = {key: value for key, value in best_record.items() if key != "score"}
    final_X, final_y = (X_arr, y_arr) if refit else (X_train, y_train)
    return TuningResult(
        model=estimator_cls(**best_params).fit(final_X, final_y),
        best_params=best_params,
        best_score=float(best_record["score"]),
        history=history,
    )


def tune_rvfl(
    X: np.ndarray,
    y: np.ndarray,
    space: dict[str, Any],
    scorer: Scorer = root_mean_squared_error,
    validation_fraction: float = 0.2,
    refit: bool = True,
    max_evals: int = 50,
    random_state: int = 0,
) -> TuningResult:
    """Tune a single RVFL model with Hyperopt/TPE and chronological validation."""

    return _tune(RVFLRegressor, X, y, space, scorer, validation_fraction, refit, max_evals, random_state)


def tune_edrvfl(
    X: np.ndarray,
    y: np.ndarray,
    space: dict[str, Any],
    scorer: Scorer = root_mean_squared_error,
    validation_fraction: float = 0.2,
    refit: bool = True,
    max_evals: int = 50,
    random_state: int = 0,
) -> TuningResult:
    """Tune an EDRVFL model with shared hyperparameters across layers."""

    return _tune(EDRVFLRegressor, X, y, space, scorer, validation_fraction, refit, max_evals, random_state)


def tune_redrvfl(
    X: np.ndarray,
    y: np.ndarray,
    space: dict[str, Any],
    scorer: Scorer = root_mean_squared_error,
    validation_fraction: float = 0.2,
    refit: bool = True,
    max_evals: int = 50,
    random_state: int = 0,
) -> TuningResult:
    """Tune a REDRVFL model with shared hyperparameters across layers."""

    return _tune(REDRVFLRegressor, X, y, space, scorer, validation_fraction, refit, max_evals, random_state)


def layerwise_tune_edrvfl(
    X: np.ndarray,
    y: np.ndarray,
    n_layers: int,
    layer_space: dict[str, Any],
    fixed_params: dict[str, Any] | None = None,
    scorer: Scorer = root_mean_squared_error,
    validation_fraction: float = 0.2,
    refit: bool = True,
    max_evals: int = 50,
    random_state: int = 0,
) -> TuningResult:
    """Tune EDRVFL one layer at a time.

    The already selected layer hyperparameters are frozen while the next layer
    is searched. This follows the layerwise workflow used in many Deep RVFL
    experiments with Hyperopt/TPE.
    """

    return _layerwise_tune(
        EDRVFLRegressor,
        X,
        y,
        n_layers,
        layer_space,
        fixed_params,
        scorer,
        validation_fraction,
        refit,
        max_evals,
        random_state,
    )


def layerwise_tune_redrvfl(
    X: np.ndarray,
    y: np.ndarray,
    n_layers: int,
    layer_space: dict[str, Any],
    fixed_params: dict[str, Any] | None = None,
    scorer: Scorer = root_mean_squared_error,
    validation_fraction: float = 0.2,
    refit: bool = True,
    max_evals: int = 50,
    random_state: int = 0,
) -> TuningResult:
    """Tune REDRVFL one layer at a time."""

    return _layerwise_tune(
        REDRVFLRegressor,
        X,
        y,
        n_layers,
        layer_space,
        fixed_params,
        scorer,
        validation_fraction,
        refit,
        max_evals,
        random_state,
    )


def _layerwise_tune(
    estimator_cls: type,
    X: np.ndarray,
    y: np.ndarray,
    n_layers: int,
    layer_space: dict[str, Any],
    fixed_params: dict[str, Any] | None,
    scorer: Scorer,
    validation_fraction: float,
    refit: bool,
    max_evals: int,
    random_state: int,
) -> TuningResult:
    if n_layers <= 0:
        raise ValueError("n_layers must be positive.")
    if max_evals <= 0:
        raise ValueError("max_evals must be positive.")
    fixed_params = dict(fixed_params or {})
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    X_train, X_val, y_train, y_val = _split(X_arr, y_arr, validation_fraction)
    selected: list[LayerParams] = []
    history = []

    for layer_index in range(n_layers):
        layer_history = []

        def objective(params: dict[str, Any]) -> dict[str, Any]:
            clean_params = _clean_params(params)
            candidate_layers = selected + [LayerParams(**clean_params)]
            model = estimator_cls(layer_params=candidate_layers, **fixed_params).fit(X_train, y_train)
            score = float(scorer(y_val, model.predict(X_val)))
            record = {"layer": layer_index + 1, "score": score, **clean_params}
            layer_history.append(record)
            history.append(record)
            return {"loss": score, "status": STATUS_OK}

        fmin(
            fn=objective,
            space=layer_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=Trials(),
            rstate=np.random.default_rng(random_state + layer_index),
            show_progressbar=False,
        )
        best_record = min(layer_history, key=lambda item: item["score"])
        selected.append(
            LayerParams(
                n_hidden=int(best_record["n_hidden"]),
                regularization=float(best_record["regularization"]),
                input_scale=float(best_record["input_scale"]),
            )
        )

    final_X, final_y = (X_arr, y_arr) if refit else (X_train, y_train)
    best_params = {"layer_params": [layer.__dict__ for layer in selected], **fixed_params}
    return TuningResult(
        model=estimator_cls(layer_params=selected, **fixed_params).fit(final_X, final_y),
        best_params=best_params,
        best_score=min(record["score"] for record in history if record["layer"] == n_layers),
        history=history,
    )


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    clean = dict(params)
    for key in ("n_hidden", "n_layers"):
        if key in clean:
            clean[key] = int(clean[key])
    for key in ("regularization", "input_scale", "recurrent_scale"):
        if key in clean:
            clean[key] = float(clean[key])
    return clean
