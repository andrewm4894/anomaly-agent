#!/usr/bin/env python3

"""Example scripts demonstrating the usage of the anomaly detection agent.

This module contains various examples showing how to use the anomaly detection
agent with different types of data and scenarios.
"""

import argparse
import os
import uuid
from typing import Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from anomaly_agent.agent import AnomalyAgent
from anomaly_agent.plot import plot_df
from anomaly_agent.utils import make_anomaly_config, make_df


def example_basic_usage() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Demonstrate basic usage of the anomaly agent with dummy data.

    Returns:
        Tuple containing the input DataFrame and the anomalies DataFrame.
    """
    print("\n=== Basic Usage Example ===")

    # Generate dummy data
    anomaly_cfg = make_anomaly_config()
    df = make_df(100, 3, anomaly_config=anomaly_cfg)

    # Create agent and detect anomalies
    agent = AnomalyAgent()
    anomalies = agent.detect_anomalies(df)

    # Convert to DataFrame for easier viewing
    df_anomalies = agent.get_anomalies_df(anomalies)
    print("\nDetected anomalies:")
    print(df_anomalies.head())

    return df, df_anomalies


def example_custom_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Demonstrate usage with custom time series data.

    Returns:
        Tuple containing the input DataFrame and the anomalies DataFrame.
    """
    print("\n=== Custom Data Example ===")

    # Create custom time series with known anomalies
    dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
    x = np.linspace(0, 4 * np.pi, 50)
    sin_wave = np.sin(x)
    noise = np.random.normal(0, 0.1, 50)
    values = sin_wave + noise

    # Add some obvious anomalies
    values[10] = 5.0  # Spike
    values[25] = -3.0  # Dip
    values[40] = np.nan  # Missing value

    # Create DataFrame
    df = pd.DataFrame({"timestamp": dates, "temperature": values})

    # Create agent and detect anomalies
    agent = AnomalyAgent(timestamp_col="timestamp")
    anomalies = agent.detect_anomalies(df)

    # Convert to DataFrame for easier viewing
    df_anomalies = agent.get_anomalies_df(anomalies)
    print("\nDetected anomalies in custom data:")
    print(df_anomalies)

    return df, df_anomalies


def example_multiple_variables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Demonstrate handling of multiple variables.

    Returns:
        Tuple containing the input DataFrame and the anomalies DataFrame.
    """
    print("\n=== Multiple Variables Example ===")

    # Create time series with multiple variables
    dates = pd.date_range(start="2024-01-01", periods=30, freq="D")

    # Create three variables with different patterns
    x = np.linspace(0, 2 * np.pi, 30)
    temp = np.sin(x) + np.random.normal(0, 0.1, 30)
    humid = np.cos(x) + np.random.normal(0, 0.1, 30)
    press = np.linspace(1000, 1020, 30) + np.random.normal(0, 0.5, 30)

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "temperature": temp,
            "humidity": humid,
            "pressure": press,
        }
    )

    # Add anomalies to each variable
    df.loc[10, "temperature"] = 5.0  # Temperature spike
    df.loc[15, "humidity"] = -2.0  # Humidity dip
    df.loc[20, "pressure"] = np.nan  # Missing pressure value

    # Create agent and detect anomalies
    agent = AnomalyAgent(timestamp_col="timestamp")
    anomalies = agent.detect_anomalies(df)

    # Convert to DataFrame for easier viewing
    df_anomalies = agent.get_anomalies_df(anomalies)
    print("\nDetected anomalies across multiple variables:")
    print(df_anomalies)

    return df, df_anomalies


def example_real_world_scenario() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a real-world scenario with sensor data.

    Returns:
        Tuple containing the input DataFrame and the anomalies DataFrame.
    """
    print("\n=== Real-world Scenario Example ===")

    # Create time series with realistic patterns
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")

    # Create base patterns
    x_temp = np.linspace(0, 4 * np.pi, 100)
    x_power = np.linspace(0, 2 * np.pi, 100)
    base_temp = 20 + 5 * np.sin(x_temp)  # Daily temp
    base_power = 1000 + 200 * np.sin(x_power)  # Power

    # Add noise
    temp = base_temp + np.random.normal(0, 0.5, 100)
    power = base_power + np.random.normal(0, 50, 100)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "timestamp": dates.strftime("%Y-%m-%d"),
            "temperature": temp,
            "power_consumption": power,
        }
    )

    # Add realistic anomalies
    df.loc[30:32, "temperature"] = 35  # Heat wave
    df.loc[50:52, "power_consumption"] = 2000  # Power surge
    df.loc[70, "temperature"] = np.nan  # Sensor failure

    # Create agent and detect anomalies
    agent = AnomalyAgent(timestamp_col="timestamp")
    anomalies = agent.detect_anomalies(df)

    # Convert to DataFrame for easier viewing
    df_anomalies = agent.get_anomalies_df(anomalies)
    print("\nDetected anomalies in sensor data:")
    print(df_anomalies)

    return df, df_anomalies


def main() -> None:
    """Run anomaly detection examples based on command line arguments."""
    parser = argparse.ArgumentParser(description="Run anomaly detection examples")
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (if not set in environment)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use",
    )
    parser.add_argument(
        "--example",
        choices=["basic", "custom", "multiple", "real-world", "all"],
        default="all",
        help="Which example to run",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the results",
    )
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Set OpenAI API key if provided
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    # Set up PostHog session tracking if enabled
    posthog_enabled = os.getenv("POSTHOG_ENABLED", "false").lower() == "true"
    if posthog_enabled and not os.getenv("POSTHOG_AI_SESSION_ID"):
        session_id = str(uuid.uuid4())
        os.environ["POSTHOG_AI_SESSION_ID"] = session_id
        print(f"🔗 PostHog Session ID: {session_id}")
        print("All traces in this session will be grouped together in PostHog.\n")

    # Run selected example(s)
    if args.example == "all":
        examples = [
            example_basic_usage,
            example_custom_data,
            example_multiple_variables,
            example_real_world_scenario,
        ]
    else:
        example_map = {
            "basic": [example_basic_usage],
            "custom": [example_custom_data],
            "multiple": [example_multiple_variables],
            "real-world": [example_real_world_scenario],
        }
        examples = example_map[args.example]

    for example in examples:
        df, df_anomalies = example()
        if args.plot:
            plot_df(df, show_anomalies=True)


if __name__ == "__main__":
    main()
