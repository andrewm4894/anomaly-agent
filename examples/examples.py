#!/usr/bin/env python3

"""Example scripts demonstrating the usage of the anomaly detection agent.

This module contains various examples showing how to use the anomaly detection
agent with different types of data and scenarios.
"""

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from anomaly_agent.agent import AnomalyAgent
from anomaly_agent.plot import plot_df
from anomaly_agent.utils import make_anomaly_config, make_df

# Load environment variables from .env file
load_dotenv()


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


def example_parallel_processing() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Demonstrate parallel processing for faster execution with multiple variables.

    Returns:
        Tuple containing the input DataFrame and the anomalies DataFrame.
    """
    print("\n=== Parallel Processing Example ===")

    # Create multi-variable time series data
    dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
    
    # Create 5 different sensor variables
    temp = [20.0 + i * 0.1 + (15.0 if i == 25 else 0) for i in range(50)]  # Temperature with spike
    pressure = [1013.0 + i * 0.2 + (-50.0 if i == 35 else 0) for i in range(50)]  # Pressure with drop
    humidity = [50.0 + i * 0.3 + (25.0 if i == 15 else 0) for i in range(50)]  # Humidity with spike
    wind = [10.0 + i * 0.05 + (-8.0 if i == 40 else 0) for i in range(50)]  # Wind with drop
    vibration = [0.1 + i * 0.001 + (2.0 if i == 30 else 0) for i in range(50)]  # Vibration with spike
    
    df = pd.DataFrame({
        "timestamp": dates,
        "temperature": temp,
        "pressure": pressure,  
        "humidity": humidity,
        "wind_speed": wind,
        "vibration": vibration
    })

    print(f"Processing {len(df.columns)-1} sensor variables with {len(df)} time points")
    
    # Compare sequential vs parallel processing
    agent = AnomalyAgent()
    
    # Sequential processing (default)
    print("\n1. Sequential processing:")
    import time
    start_time = time.time()
    anomalies_sequential = agent.detect_anomalies(df)
    sequential_time = time.time() - start_time
    sequential_total = sum(len(al.anomalies) for al in anomalies_sequential.values())
    print(f"   Found {sequential_total} anomalies in {sequential_time:.2f} seconds")
    
    # Parallel processing
    print("\n2. Parallel processing (parallel=True):")
    start_time = time.time()
    anomalies_parallel = agent.detect_anomalies(df, parallel=True, max_concurrent=3)
    parallel_time = time.time() - start_time
    parallel_total = sum(len(al.anomalies) for al in anomalies_parallel.values())
    print(f"   Found {parallel_total} anomalies in {parallel_time:.2f} seconds")
    
    # Performance comparison
    if sequential_time > 0:
        speedup = sequential_time / parallel_time
        print(f"\n3. Performance improvement: {speedup:.2f}x faster with parallel processing")
    
    # Convert to DataFrame for easier viewing
    df_anomalies = agent.get_anomalies_df(anomalies_parallel)
    print("\nDetected anomalies with parallel processing:")
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
        choices=["basic", "custom", "multiple", "real-world", "parallel", "all"],
        default="all",
        help="Which example to run",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the results",
    )
    args = parser.parse_args()

    # Set OpenAI API key if provided
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    # Run selected example(s)
    if args.example == "all":
        examples = [
            example_basic_usage,
            example_custom_data,
            example_multiple_variables,
            example_real_world_scenario,
            example_parallel_processing,
        ]
    else:
        example_map = {
            "basic": [example_basic_usage],
            "custom": [example_custom_data],
            "multiple": [example_multiple_variables],
            "real-world": [example_real_world_scenario],
            "parallel": [example_parallel_processing],
        }
        examples = example_map[args.example]

    for example in examples:
        df, df_anomalies = example()
        if args.plot:
            plot_df(df, show_anomalies=True)


if __name__ == "__main__":
    main()
