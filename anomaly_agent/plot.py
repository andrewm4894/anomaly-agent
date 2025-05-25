"""Plotting utilities for visualizing time series data and anomalies."""

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_df(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    show_anomalies: bool = True,
    anomaly_suffix: str = "_anomaly_flag",
    title: str = "",
    return_fig: bool = False,
) -> Optional[go.Figure]:
    """Plot time series data with optional anomaly markers using Plotly.

    Args:
        df: DataFrame containing timestamp and variable columns.
        timestamp_col: Name of the timestamp column.
        show_anomalies: Whether to plot anomaly flags.
        anomaly_suffix: Suffix used for anomaly flag columns.
        title: Title for the overall figure.
        return_fig: If True, returns the figure object instead of displaying it.

    Returns:
        Plotly figure object if return_fig is True, otherwise None.
    """
    # Identify variable columns (all columns except timestamp and anomaly flags)
    variable_columns = [
        col
        for col in df.columns
        if col != timestamp_col and not col.endswith(anomaly_suffix)
    ]  # noqa: E501
    n_plots = len(variable_columns)

    # Create subplots with a shared x-axis for better alignment
    fig = make_subplots(
        rows=n_plots,
        cols=1,
        shared_xaxes=True,
        subplot_titles=variable_columns,
        vertical_spacing=0.05,
    )  # noqa: E501

    # Add each time series as a separate trace in its corresponding subplot
    for i, col in enumerate(variable_columns):
        # Plot original time series
        fig.add_trace(
            go.Scatter(
                x=df[timestamp_col],
                y=df[col],
                mode="lines+markers",
                name=col,
                showlegend=False,
            ),
            row=i + 1,
            col=1,
        )  # noqa: E501

        # Plot anomaly points if they exist and show_anomalies is True
        anomaly_col = f"{col}{anomaly_suffix}"
        if show_anomalies and anomaly_col in df.columns:
            # Get timestamps and values where anomalies occur
            anomaly_df = df[df[anomaly_col].notna()]
            if not anomaly_df.empty:
                marker_config = {
                    "symbol": "x",
                    "size": 10,
                    "color": "red",
                }
                fig.add_trace(
                    go.Scatter(
                        x=anomaly_df[timestamp_col],
                        y=anomaly_df[col],
                        mode="markers",
                        name=f"{col} Anomalies",
                        marker=marker_config,
                        showlegend=False,
                    ),
                    row=i + 1,
                    col=1,
                )  # noqa: E501

    # Update layout settings
    fig.update_layout(
        height=300 * n_plots,
        width=800,
        title_text=title,
    )  # noqa: E501

    # Either return the figure or display it
    if return_fig:
        return fig

    fig.show()
    return None


def plot_df_matplotlib(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    show_anomalies: bool = True,
    anomaly_suffix: str = "_anomaly_flag",
    title: str = "",
) -> None:
    """Plot time series data with optional anomaly markers using Matplotlib.

    Args:
        df: DataFrame containing timestamp and variable columns.
        timestamp_col: Name of the timestamp column.
        show_anomalies: Whether to plot anomaly flags.
        anomaly_suffix: Suffix used for anomaly flag columns.
        title: Title for the overall figure.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    # Ensure timestamp column is datetime
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Identify variable columns (all columns except timestamp and anomaly flags)
    # Only include numeric columns
    variable_columns = [
        col
        for col in df.columns
        if col != timestamp_col
        and not col.endswith(anomaly_suffix)
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not variable_columns:
        raise ValueError("No numeric columns found to plot")

    n_plots = len(variable_columns)

    # Create figure and subplots
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
    fig.suptitle(title)

    # Handle case where there's only one subplot
    if n_plots == 1:
        axes = [axes]

    # Add each time series as a separate subplot
    for ax, col in zip(axes, variable_columns):
        # Plot original time series
        ax.plot(
            df[timestamp_col],
            df[col],
            "o-",
            label=col,
            markersize=4,
        )

        # Plot anomaly points if they exist and show_anomalies is True
        anomaly_col = f"{col}{anomaly_suffix}"
        if show_anomalies and anomaly_col in df.columns:
            # Get timestamps and values where anomalies occur
            anomaly_df = df[df[anomaly_col].notna()]
            if not anomaly_df.empty:
                ax.plot(
                    anomaly_df[timestamp_col],
                    anomaly_df[col],
                    "rx",
                    label=f"{col} Anomalies",
                    markersize=10,
                )

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        ax.set_title(col)
        ax.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
