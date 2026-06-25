from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

Aggregation = Literal["mean", "median"]
Activation = Literal["sigmoid", "tanh", "relu", "sin"]


def _as_2d_array(values: np.ndarray | Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def _activation(values: np.ndarray, kind: Activation) -> np.ndarray:
    if kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))
    if kind == "tanh":
        return np.tanh(values)
    if kind == "relu":
        return np.maximum(values, 0.0)
    if kind == "sin":
        return np.sin(values)
    raise ValueError(f"Unsupported activation: {kind!r}")


def _ridge_solve(design: np.ndarray, target: np.ndarray, regularization: float) -> np.ndarray:
    if regularization < 0:
        raise ValueError("regularization must be non-negative.")
    n_features = design.shape[1]
    penalty = float(regularization) * np.eye(n_features)
    penalty[-1, -1] = 0.0
    left = design.T @ design + penalty
    right = design.T @ target
    try:
        return np.linalg.solve(left, right)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(left) @ right


@dataclass(frozen=True)
class LayerParams:
    """Hyperparameters for one hidden layer."""

    n_hidden: int = 50
    regularization: float = 1e-3
    input_scale: float = 0.1


class RVFLRegressor:
    """Single-layer Random Vector Functional Link regressor.

    The readout is trained with ridge regression over hidden random features,
    direct input links, and an intercept term.
    """

    def __init__(
        self,
        n_hidden: int = 50,
        regularization: float = 1e-3,
        input_scale: float = 0.1,
        activation: Activation = "sigmoid",
        include_direct_link: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.n_hidden = n_hidden
        self.regularization = regularization
        self.input_scale = input_scale
        self.activation = activation
        self.include_direct_link = include_direct_link
        self.random_state = random_state

    def fit(self, X: np.ndarray | Iterable[float], y: np.ndarray | Iterable[float]) -> "RVFLRegressor":
        X_arr = _as_2d_array(X, "X")
        y_arr = _as_2d_array(y, "y")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if self.n_hidden <= 0:
            raise ValueError("n_hidden must be positive.")

        rng = np.random.default_rng(self.random_state)
        self.input_weights_ = rng.uniform(
            -self.input_scale, self.input_scale, size=(X_arr.shape[1], int(self.n_hidden))
        )
        self.bias_ = rng.uniform(-self.input_scale, self.input_scale, size=(int(self.n_hidden),))
        hidden = self._hidden(X_arr)
        design = self._design(X_arr, hidden)
        self.coef_ = _ridge_solve(design, y_arr, self.regularization)
        self.n_features_in_ = X_arr.shape[1]
        self.n_outputs_ = y_arr.shape[1]
        return self

    def predict(self, X: np.ndarray | Iterable[float]) -> np.ndarray:
        self._check_fitted()
        X_arr = _as_2d_array(X, "X")
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X_arr.shape[1]}.")
        predictions = self._design(X_arr, self._hidden(X_arr)) @ self.coef_
        return predictions.ravel() if self.n_outputs_ == 1 else predictions

    def _hidden(self, X: np.ndarray) -> np.ndarray:
        return _activation(X @ self.input_weights_ + self.bias_, self.activation)

    def _design(self, X: np.ndarray, hidden: np.ndarray) -> np.ndarray:
        parts = [hidden]
        if self.include_direct_link:
            parts.append(X)
        parts.append(np.ones((X.shape[0], 1)))
        return np.hstack(parts)

    def _check_fitted(self) -> None:
        if not hasattr(self, "coef_"):
            raise RuntimeError("The model must be fitted before prediction.")


class EDRVFLRegressor:
    """Ensemble Deep RVFL regressor.

    Each layer receives the previous layer state plus optional direct input
    links. Layer readouts are trained independently, then aggregated.
    """

    def __init__(
        self,
        n_layers: int = 3,
        n_hidden: int = 50,
        regularization: float = 1e-3,
        input_scale: float = 0.1,
        activation: Activation = "sigmoid",
        aggregation: Aggregation = "median",
        random_state: int | None = None,
        layer_params: list[LayerParams | dict] | None = None,
    ) -> None:
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.regularization = regularization
        self.input_scale = input_scale
        self.activation = activation
        self.aggregation = aggregation
        self.random_state = random_state
        self.layer_params = layer_params

    def fit(self, X: np.ndarray | Iterable[float], y: np.ndarray | Iterable[float]) -> "EDRVFLRegressor":
        X_arr = _as_2d_array(X, "X")
        y_arr = _as_2d_array(y, "y")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        params = self._resolved_layer_params()
        rng = np.random.default_rng(self.random_state)

        self.layers_ = []
        state = X_arr
        for layer_param in params:
            weights = rng.uniform(
                -layer_param.input_scale,
                layer_param.input_scale,
                size=(state.shape[1], int(layer_param.n_hidden)),
            )
            bias = rng.uniform(
                -layer_param.input_scale, layer_param.input_scale, size=(int(layer_param.n_hidden),)
            )
            hidden = _activation(state @ weights + bias, self.activation)
            design = np.hstack([hidden, X_arr, np.ones((X_arr.shape[0], 1))])
            coef = _ridge_solve(design, y_arr, layer_param.regularization)
            self.layers_.append(
                {
                    "weights": weights,
                    "bias": bias,
                    "coef": coef,
                    "params": layer_param,
                }
            )
            state = hidden

        self.n_features_in_ = X_arr.shape[1]
        self.n_outputs_ = y_arr.shape[1]
        return self

    def predict(self, X: np.ndarray | Iterable[float], return_layers: bool = False) -> np.ndarray:
        self._check_fitted()
        X_arr = _as_2d_array(X, "X")
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X_arr.shape[1]}.")

        state = X_arr
        outputs = []
        for layer in self.layers_:
            hidden = _activation(state @ layer["weights"] + layer["bias"], self.activation)
            design = np.hstack([hidden, X_arr, np.ones((X_arr.shape[0], 1))])
            outputs.append(design @ layer["coef"])
            state = hidden

        stacked = np.stack(outputs, axis=0)
        if return_layers:
            result = np.moveaxis(stacked, 0, 1)
        elif self.aggregation == "mean":
            result = np.mean(stacked, axis=0)
        elif self.aggregation == "median":
            result = np.median(stacked, axis=0)
        else:
            raise ValueError("aggregation must be 'mean' or 'median'.")
        return result.ravel() if self.n_outputs_ == 1 and result.ndim == 2 else result

    def _resolved_layer_params(self) -> list[LayerParams]:
        if self.layer_params is None:
            if self.n_layers <= 0:
                raise ValueError("n_layers must be positive.")
            return [
                LayerParams(self.n_hidden, self.regularization, self.input_scale)
                for _ in range(int(self.n_layers))
            ]
        params = []
        for item in self.layer_params:
            params.append(item if isinstance(item, LayerParams) else LayerParams(**item))
        if not params:
            raise ValueError("layer_params cannot be empty.")
        return params

    def _check_fitted(self) -> None:
        if not hasattr(self, "layers_"):
            raise RuntimeError("The model must be fitted before prediction.")


class REDRVFLRegressor(EDRVFLRegressor):
    """Recurrent Ensemble Deep RVFL regressor for ordered samples.

    Hidden states are computed recurrently across the sample axis, which makes
    this estimator suitable for supervised time-series frames.
    """

    def __init__(
        self,
        n_layers: int = 3,
        n_hidden: int = 50,
        regularization: float = 1e-3,
        input_scale: float = 0.1,
        recurrent_scale: float = 0.1,
        activation: Activation = "sigmoid",
        aggregation: Aggregation = "median",
        random_state: int | None = None,
        layer_params: list[LayerParams | dict] | None = None,
    ) -> None:
        super().__init__(
            n_layers=n_layers,
            n_hidden=n_hidden,
            regularization=regularization,
            input_scale=input_scale,
            activation=activation,
            aggregation=aggregation,
            random_state=random_state,
            layer_params=layer_params,
        )
        self.recurrent_scale = recurrent_scale

    def fit(self, X: np.ndarray | Iterable[float], y: np.ndarray | Iterable[float]) -> "REDRVFLRegressor":
        X_arr = _as_2d_array(X, "X")
        y_arr = _as_2d_array(y, "y")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        params = self._resolved_layer_params()
        rng = np.random.default_rng(self.random_state)

        self.layers_ = []
        state = X_arr
        for layer_param in params:
            input_weights = rng.uniform(
                -layer_param.input_scale,
                layer_param.input_scale,
                size=(state.shape[1], int(layer_param.n_hidden)),
            )
            recurrent_weights = rng.uniform(
                -self.recurrent_scale,
                self.recurrent_scale,
                size=(int(layer_param.n_hidden), int(layer_param.n_hidden)),
            )
            bias = rng.uniform(
                -layer_param.input_scale, layer_param.input_scale, size=(int(layer_param.n_hidden),)
            )
            hidden = self._recurrent_hidden(state, input_weights, recurrent_weights, bias)
            design = np.hstack([hidden, X_arr, np.ones((X_arr.shape[0], 1))])
            coef = _ridge_solve(design, y_arr, layer_param.regularization)
            self.layers_.append(
                {
                    "input_weights": input_weights,
                    "recurrent_weights": recurrent_weights,
                    "bias": bias,
                    "coef": coef,
                    "params": layer_param,
                }
            )
            state = hidden

        self.n_features_in_ = X_arr.shape[1]
        self.n_outputs_ = y_arr.shape[1]
        return self

    def predict(self, X: np.ndarray | Iterable[float], return_layers: bool = False) -> np.ndarray:
        self._check_fitted()
        X_arr = _as_2d_array(X, "X")
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X_arr.shape[1]}.")

        state = X_arr
        outputs = []
        for layer in self.layers_:
            hidden = self._recurrent_hidden(
                state, layer["input_weights"], layer["recurrent_weights"], layer["bias"]
            )
            design = np.hstack([hidden, X_arr, np.ones((X_arr.shape[0], 1))])
            outputs.append(design @ layer["coef"])
            state = hidden

        stacked = np.stack(outputs, axis=0)
        if return_layers:
            result = np.moveaxis(stacked, 0, 1)
        elif self.aggregation == "mean":
            result = np.mean(stacked, axis=0)
        elif self.aggregation == "median":
            result = np.median(stacked, axis=0)
        else:
            raise ValueError("aggregation must be 'mean' or 'median'.")
        return result.ravel() if self.n_outputs_ == 1 and result.ndim == 2 else result

    def _recurrent_hidden(
        self,
        X: np.ndarray,
        input_weights: np.ndarray,
        recurrent_weights: np.ndarray,
        bias: np.ndarray,
    ) -> np.ndarray:
        hidden = np.zeros((X.shape[0], input_weights.shape[1]))
        previous = np.zeros(input_weights.shape[1])
        for i, row in enumerate(X):
            previous = _activation(row @ input_weights + previous @ recurrent_weights + bias, self.activation)
            hidden[i] = previous
        return hidden
