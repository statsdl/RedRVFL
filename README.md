# RedRVFL

RedRVFL is a lightweight Python package for financial time-series forecasting with
Random Vector Functional Link models:

- `RVFLRegressor`: single hidden-layer RVFL with ridge readout.
- `EDRVFLRegressor`: ensemble deep RVFL with independent layer readouts.
- `REDRVFLRegressor`: recurrent ensemble deep RVFL for ordered time-series frames.
- Hyperopt/TPE tuning for RVFL.
- Layerwise Hyperopt/TPE tuning for EDRVFL and REDRVFL.

## Installation

Core install:

```bash
git clone https://github.com/statsdl/RedRVFL.git
cd RedRVFL
pip install .
```

Finance experiment dependencies:

```bash
pip install ".[finance]"
```

For development:

```bash
pip install -e ".[dev]"
pytest
```

## Financial Time Series Forecasting Example

```python
from redrvfl.finance import download_dji, run_dji_paper_experiment

download_dji("datasets/DJI.csv")
results = run_dji_paper_experiment(
    dataset_path="datasets/DJI.csv",
    seeds=(0,),
    horizon=20,
    look_ahead=1,
    n_layers=10,
    max_evals=100,
)
```

The split follows the paper: 70% training, 10% validation, and 20% test in
chronological order. Hyperparameters are selected on validation data, then the
model is fitted on train+validation and evaluated on the final test segment.

Command-line usage:

```bash
python examples/run_finance_forecasting.py --download --seeds 0 --horizon 20 --look-ahead 1
python examples/run_finance_forecasting.py --seeds 0,1,2
```

## Hyperopt Tuning

```python
from hyperopt import hp
from redrvfl.tuning import layerwise_tune_redrvfl

result = layerwise_tune_redrvfl(
    X,
    y,
    n_layers=10,
    layer_space={
        "n_hidden": hp.quniform("n_hidden", 20, 200, 1),
        "regularization": hp.uniform("regularization", 0, 1),
        "input_scale": hp.uniform("input_scale", 0, 1),
    },
    fixed_params={
        "recurrent_scale": 0.1,
        "random_state": 0,
    },
    validation_fraction=0.1 / 0.8,
    max_evals=100,
)
```

## API

Supported package code lives in `src/redrvfl`.

All estimators expose:

- `fit(X, y)`: train readout weights.
- `predict(X)`: return predictions.
- `predict(X, return_layers=True)`: for EDRVFL/REDRVFL, return each layer's prediction.

`make_forecasting_frame(series, order, horizon)` converts a time series into supervised lagged samples.

Finance and tuning utilities are intentionally imported from submodules:

- `redrvfl.tuning`
- `redrvfl.finance`

This keeps `import redrvfl` lightweight and avoids importing optional Hyperopt
and Yahoo Finance dependencies unless they are needed.

## Repository Notes

The `legacy/`, `utils/`, `RecRVFL_/`, and `ForecastLib.py` files are retained
for traceability to earlier research scripts. They are not included in the
published wheel and are not the supported package API.

## PyPI Release

The publish workflow uses PyPI Trusted Publishing. Configure the PyPI trusted
publisher with:

- owner: `statsdl`
- repository: `RedRVFL`
- workflow: `publish.yml`
- environment: `pypi`

## License

MIT

## Reference

If you use RedRVFL in your work, please cite:

```bibtex
@article{bhambu2024recurrent,
  title={Recurrent ensemble random vector functional link neural network for financial time series forecasting},
  author={Bhambu, Aryan and Gao, Ruobin and Suganthan, Ponnuthurai Nagaratnam},
  journal={Applied Soft Computing},
  volume={161},
  pages={111759},
  year={2024},
  publisher={Elsevier}
}
```
