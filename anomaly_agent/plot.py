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
) -> go.Figure | None:
    """

    Plot each time series column from the DataFrame on separate subplots using Plotly.
    If show_anomalies is True, it will also plot corresponding anomaly flags as 'X' markers.

    Parameters:
        df (pd.DataFrame): DataFrame containing a timestamp column and one or more variable columns.
        timestamp_col (str): Name of the timestamp column. Default is 'timestamp'.
        show_anomalies (bool): Whether to plot anomaly flags. Default is True.
            For each column 'col', looks for 'col{anomaly_suffix}' in the DataFrame.
        anomaly_suffix (str): Suffix used for anomaly flag columns. Default is '_anomaly_flag'.
            For a column 'temperature', it will look for 'temperature_anomaly_flag'.
        return_fig (bool): If True, returns the figure object instead of displaying it.
            Default is False.

    The function creates a subplot for each variable column (all columns except the timestamp),
    with the x-axis representing the timestamps.

    Returns:
        go.Figure | None: If return_fig is True, returns the Plotly figure object.
            Otherwise returns None.
    """
    # Identify variable columns (all columns except the timestamp column and anomaly flags)
    variable_columns = [
        col
        for col in df.columns
        if col != timestamp_col and not col.endswith(anomaly_suffix)
    ]
    n_plots = len(variable_columns)

    # Create subplots with a shared x-axis for better alignment
    fig = make_subplots(
        rows=n_plots,
        cols=1,
        shared_xaxes=True,
        subplot_titles=variable_columns,
        vertical_spacing=0.05,
    )

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
        )

        # Plot anomaly points if they exist and show_anomalies is True
        anomaly_col = f"{col}{anomaly_suffix}"
        if show_anomalies and anomaly_col in df.columns:
            # Get timestamps and values where anomalies occur
            anomaly_df = df[df[anomaly_col].notna()]
            if not anomaly_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=anomaly_df[timestamp_col],
                        y=anomaly_df[col],
                        mode="markers",
                        name=f"{col} Anomalies",
                        marker=dict(symbol="x", size=10, color="red"),
                        showlegend=False,
                    ),
                    row=i + 1,
                    col=1,
                )

    # Update layout settings: adjust the overall height based on the number of plots
    fig.update_layout(height=300 * n_plots, width=800, title_text=title)

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
    """
    Plot each time series column from the DataFrame on separate subplots using Matplotlib.
    If show_anomalies is True, it will also plot corresponding anomaly flags as 'X' markers.

    Parameters:
        df (pd.DataFrame): DataFrame containing a timestamp column and one or more variable columns.
        timestamp_col (str): Name of the timestamp column. Default is 'timestamp'.
        show_anomalies (bool): Whether to plot anomaly flags. Default is True.
            For each column 'col', looks for 'col{anomaly_suffix}' in the DataFrame.
        anomaly_suffix (str): Suffix used for anomaly flag columns. Default is '_anomaly_flag'.
        title (str): Title for the overall figure. Default is empty string.
    """
    import matplotlib.pyplot as plt

    # Identify variable columns (all columns except the timestamp column and anomaly flags)
    variable_columns = [
        col
        for col in df.columns
        if col != timestamp_col and not col.endswith(anomaly_suffix)
    ]
    n_plots = len(variable_columns)

    # Create figure and subplots
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4*n_plots))
    fig.suptitle(title)
    
    # Handle case where there's only one subplot
    if n_plots == 1:
        axes = [axes]

    # Add each time series as a separate subplot
    for ax, col in zip(axes, variable_columns):
        # Plot original time series
        ax.plot(df[timestamp_col], df[col], 'o-', label=col, markersize=4)
        
        # Plot anomaly points if they exist and show_anomalies is True
        anomaly_col = f"{col}{anomaly_suffix}"
        if show_anomalies and anomaly_col in df.columns:
            # Get timestamps and values where anomalies occur
            anomaly_df = df[df[anomaly_col].notna()]
            if not anomaly_df.empty:
                ax.plot(
                    anomaly_df[timestamp_col],
                    anomaly_df[col],
                    'rx',
                    label=f"{col} Anomalies",
                    markersize=10,
                )

        ax.set_title(col)
        ax.grid(True)
        
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    
    

