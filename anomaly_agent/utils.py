import pandas as pd
import numpy as np


def make_df(
    num_rows: int,
    n_variables: int,
    start_date: str = "2020-01-01",
    freq: str = "D",
    anomaly_config: dict = None,
    col_prefix: str = "var",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Generate a DataFrame with a timestamp column and n random variable columns,
    and optionally inject anomalies into the random data.

    Parameters:
        num_rows (int): Number of rows (timestamps) in the DataFrame.
        n_variables (int): Number of random variable columns to generate.
        start_date (str): The start date for the timestamp series (default '2020-01-01').
        freq (str): Frequency string for the timestamps (default 'D' for daily).
        anomaly_config (dict, optional): Configuration dictionary for injecting anomalies.
            Keys:
                - enabled (bool): Whether to inject anomalies (default True if provided).
                - fraction (float): Fraction of data points per variable column to modify (default 0.05).
                - methods (list of str): List of anomaly methods to apply. Options: 'spike', 'drop', 'shift', 'noise'.
                - spike_factor (float): Factor to multiply value in 'spike' method (default 10).
                - shift_value (float): Value to add in 'shift' method (default 5).
                - noise_std (float): Standard deviation for noise in 'noise' method (default 0.5).

    Returns:
        pd.DataFrame: A DataFrame with a 'timestamp' column and n random variable columns,
                      with anomalies injected if configured.
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
        fraction = anomaly_config.get("fraction", 0.05)
        methods = anomaly_config.get("methods", ["spike", "drop", "shift", "noise"])

        # For each variable column, select random indices to modify
        for i in range(1, n_variables + 1):
            col = f"{col_prefix}{i}"
            n_anomalies = int(num_rows * fraction)
            if n_anomalies > 0:
                anomaly_indices = np.random.choice(num_rows, n_anomalies, replace=False)
                for idx in anomaly_indices:
                    # Randomly choose an anomaly method for this data point
                    method = np.random.choice(methods)
                    original_value = df.loc[idx, col]

                    if method == "spike":
                        # Multiply the original value by spike_factor
                        spike_factor = anomaly_config.get("spike_factor", 10)
                        df.loc[idx, col] = original_value * spike_factor
                    elif method == "drop":
                        # Replace the value with NaN to simulate a missing value
                        df.loc[idx, col] = np.nan
                    elif method == "shift":
                        # Add a constant shift to the original value
                        shift_value = anomaly_config.get("shift_value", 5)
                        df.loc[idx, col] = original_value + shift_value
                    elif method == "noise":
                        # Add normally distributed noise to the original value
                        noise_std = anomaly_config.get("noise_std", 0.5)
                        df.loc[idx, col] = original_value + np.random.normal(
                            0, noise_std
                        )
                    else:
                        # If the method is not recognized, leave the value unchanged.
                        pass
    return df


def make_anomaly_config(
    enabled: bool = True,
    fraction: float = 0.05,
    methods: list[str] = ["spike", "drop", "shift", "noise"],
    spike_factor: float = 10,
    shift_value: float = 3,
    noise_std: float = 0.2,
) -> dict:
    """
    Create a configuration dictionary for injecting anomalies into a DataFrame.

    Parameters:
        enabled (bool): Whether to enable anomaly injection (default True).
        fraction (float): Fraction of data points per variable column to modify (default 0.1).
        methods (list of str): List of anomaly methods to apply. Options: 'spike', 'drop', 'shift', 'noise'.
        spike_factor (float): Factor to multiply value in 'spike' method (default 10).
        shift_value (float): Value to add in 'shift' method (default 3).
        noise_std (float): Standard deviation for noise in 'noise' method (default 0.2).

    Returns:
        dict: A configuration dictionary for injecting anomalies into a DataFrame.
    """
    anomaly_cfg = {
        "enabled": enabled,
        "fraction": fraction,
        "methods": methods,
        "spike_factor": spike_factor,
        "shift_value": shift_value,
        "noise_std": noise_std,
    }

    return anomaly_cfg
