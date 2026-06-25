"""Random Vector Functional Link models for regression and forecasting."""

from .data import make_forecasting_frame
from .metrics import mean_absolute_percentage_error, root_mean_squared_error
from .models import EDRVFLRegressor, REDRVFLRegressor, RVFLRegressor

__all__ = [
    "EDRVFLRegressor",
    "REDRVFLRegressor",
    "RVFLRegressor",
    "make_forecasting_frame",
    "mean_absolute_percentage_error",
    "root_mean_squared_error",
]

__version__ = "0.1.1"
