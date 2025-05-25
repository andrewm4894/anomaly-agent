"""
Utility functions for generating and manipulating time series data with anomalies.

This module provides functions for generating synthetic time series data and injecting
controlled anomalies for testing and demonstration purposes.
"""

from typing import Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd


def make_df(
    num_rows: int,
    n_variables: int,
    start_date: str = "2020-01-01",
    freq: str = "D",
    anomaly_config: Optional[Dict[str, Union[bool, float, List[str]]]] = None,
    col_prefix: str = "var",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Generate a DataFrame with random variables and optional anomalies.

    Args:
        num_rows: Number of rows (timestamps) in the DataFrame.
        n_variables: Number of random variable columns to generate.
        start_date: The start date for the timestamp series.
        freq: Frequency string for the timestamps.
        anomaly_config: Configuration dictionary for injecting anomalies.
            Keys:
                - enabled (bool): Whether to inject anomalies.
                - fraction (float): Fraction of data points to modify.
                - methods (list): List of anomaly methods to apply.
                - spike_factor (float): Factor to multiply value in 'spike' method.
                - shift_value (float): Value to add in 'shift' method.
                - noise_std (float): Standard deviation for noise in 'noise' method.
        col_prefix: Prefix for variable column names.
        timestamp_col: Name of the timestamp column.

    Returns:
        DataFrame with timestamp column and random variable columns.
    """
    # Create a timestamp series
    timestamps = pd.date_range(start=start_date, periods=num_rows, freq=freq)
    data = {timestamp_col: timestamps}

    # Generate random data for each variable column
    for i in range(1, n_variables + 1):
        col_name = f"{col_prefix}{i}"
        data[col_name] = np.random.random(num_rows)

    # Create the DataFrame
    df = pd.DataFrame(data)

    # Inject anomalies if an anomaly configuration is provided and enabled
    if anomaly_config is not None and anomaly_config.get("enabled", True):
        # Get anomaly configuration parameters or use defaults
        fraction = cast(float, anomaly_config.get("fraction", 0.05))
        methods = cast(
            List[str],
            anomaly_config.get(
                "methods", ["spike", "drop", "shift", "noise"]
            ),  # noqa: E501
        )

        # For each variable column, select random indices to modify
        for i in range(1, n_variables + 1):
            col = f"{col_prefix}{i}"
            n_anomalies = int(num_rows * fraction)
            if n_anomalies > 0:
                anomaly_indices = np.random.choice(
                    num_rows,
                    n_anomalies,
                    replace=False,
                )
                for idx in anomaly_indices:
                    # Randomly choose an anomaly method for this data point
                    method = np.random.choice(methods)
                    original_value = df.loc[idx, col]

                    if method == "spike":
                        # Multiply the original value by spike_factor
                        spike_factor = cast(
                            float,
                            anomaly_config.get("spike_factor", 10),  # noqa: E501
                        )
                        df.loc[idx, col] = original_value * spike_factor
                    elif method == "drop":
                        # Replace the value with NaN to simulate a missing value
                        df.loc[idx, col] = np.nan
                    elif method == "shift":
                        # Add a constant shift to the original value
                        shift_value = cast(
                            float,
                            anomaly_config.get("shift_value", 5),  # noqa: E501
                        )
                        df.loc[idx, col] = original_value + shift_value
                    elif method == "noise":
                        # Add normally distributed noise to the original value
                        noise_std = cast(
                            float,
                            anomaly_config.get("noise_std", 0.5),  # noqa: E501
                        )
                        df.loc[idx, col] = original_value + np.random.normal(
                            0, noise_std  # noqa: E501
                        )
                    else:
                        # If the method is not recognized, leave the value unchanged
                        pass
    return df


def make_anomaly_config(
    enabled: bool = True,
    fraction: float = 0.02,
    methods: Optional[List[str]] = None,
    spike_factor: float = 10,
    shift_value: float = 3,
    noise_std: float = 0.2,
) -> Dict[str, Union[bool, float, List[str]]]:
    """Create a configuration dictionary for injecting anomalies.

    Args:
        enabled: Whether to enable anomaly injection.
        fraction: Fraction of data points per variable column to modify.
        methods: List of anomaly methods to apply.
            Options: 'spike', 'drop', 'shift', 'noise'.
        spike_factor: Factor to multiply value in 'spike' method.
        shift_value: Value to add in 'shift' method.
        noise_std: Standard deviation for noise in 'noise' method.

    Returns:
        Configuration dictionary for injecting anomalies.
    """
    if methods is None:
        methods = ["spike", "drop", "shift", "noise"]

    return {
        "enabled": enabled,
        "fraction": fraction,
        "methods": methods,
        "spike_factor": spike_factor,
        "shift_value": shift_value,
        "noise_std": noise_std,
    }
