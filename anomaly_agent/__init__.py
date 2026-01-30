"""Anomaly detection agent package for time series data analysis.

This package provides tools for detecting anomalies in time series data using
various statistical and machine learning methods.
"""

from .agent import Anomaly, AnomalyAgent, AnomalyList
from .plot import plot_df
from .utils import make_anomaly_config, make_df

__version__ = "0.10.0"

__all__ = [
    "AnomalyAgent",
    "Anomaly",
    "AnomalyList",
    "plot_df",
    "make_df",
    "make_anomaly_config",
]
