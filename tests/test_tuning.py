import numpy as np
from hyperopt import hp

from redrvfl.tuning import layerwise_tune_redrvfl, tune_rvfl


def test_tune_rvfl_returns_best_model():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(60, 3))
    y = X[:, 0] + 0.2 * X[:, 1]

    result = tune_rvfl(
        X,
        y,
        {
            "n_hidden": hp.choice("n_hidden", [5, 10]),
            "regularization": hp.choice("regularization", [1e-3]),
            "input_scale": hp.choice("input_scale", [0.2]),
            "random_state": 10,
        },
        validation_fraction=0.25,
        max_evals=2,
        random_state=10,
    )

    assert "n_hidden" in result.best_params
    assert len(result.history) == 2
    assert result.model.predict(X[:3]).shape == (3,)


def test_layerwise_tune_redrvfl_selects_each_layer():
    series = np.cos(np.linspace(0, 5, 70))
    X = np.column_stack([series[:-1], np.roll(series, 1)[:-1]])
    y = series[1:]

    result = layerwise_tune_redrvfl(
        X,
        y,
        n_layers=2,
        layer_space={
            "n_hidden": hp.choice("layer_n_hidden", [5, 8]),
            "regularization": hp.choice("layer_regularization", [1e-3]),
            "input_scale": hp.choice("layer_input_scale", [0.1]),
        },
        fixed_params={"random_state": 11, "recurrent_scale": 0.05},
        validation_fraction=0.2,
        max_evals=2,
        random_state=11,
    )

    assert len(result.best_params["layer_params"]) == 2
    assert len(result.history) == 4
