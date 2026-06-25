import numpy as np

from redrvfl import EDRVFLRegressor, REDRVFLRegressor, RVFLRegressor, make_forecasting_frame


def test_rvfl_predicts_expected_shape():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 3))
    y = X[:, 0] - 0.5 * X[:, 1]

    model = RVFLRegressor(n_hidden=12, random_state=2).fit(X, y)
    predictions = model.predict(X[:5])

    assert predictions.shape == (5,)
    assert np.all(np.isfinite(predictions))


def test_edrvfl_layer_predictions():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 4))
    y = np.sin(X[:, 0]) + X[:, 2]

    model = EDRVFLRegressor(n_layers=3, n_hidden=10, random_state=4).fit(X, y)
    predictions = model.predict(X[:7])
    layer_predictions = model.predict(X[:7], return_layers=True)

    assert predictions.shape == (7,)
    assert layer_predictions.shape == (7, 3, 1)


def test_redrvfl_forecasting_frame():
    series = np.sin(np.linspace(0, 4, 80))
    X, y = make_forecasting_frame(series, order=5)

    model = REDRVFLRegressor(n_layers=2, n_hidden=8, random_state=5).fit(X[:60], y[:60])
    predictions = model.predict(X[60:])

    assert predictions.shape == y[60:].shape
    assert np.all(np.isfinite(predictions))
