"""Test suite for the anomaly detection agent.

This module contains tests for the AnomalyAgent class and its components,
including anomaly detection, validation, and data handling.
"""

import numpy as np
import pandas as pd
import pytest

from anomaly_agent.agent import Anomaly, AnomalyAgent, AnomalyList


@pytest.fixture  # type: ignore
def single_variable_df() -> pd.DataFrame:
    """Create a DataFrame with a single variable and known anomalies."""
    dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
    x = np.linspace(0, 4 * np.pi, 50)
    values = np.sin(x) + np.random.normal(0, 0.1, 50)

    # Add some obvious anomalies
    values[10] = 5.0  # Spike
    values[25] = -3.0  # Dip
    values[40] = np.nan  # Missing value

    return pd.DataFrame({"timestamp": dates, "temperature": values})


@pytest.fixture  # type: ignore
def multi_variable_df() -> pd.DataFrame:
    """Create a DataFrame with multiple variables and known anomalies."""
    dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
    x = np.linspace(0, 2 * np.pi, 30)

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "temperature": np.sin(x) + np.random.normal(0, 0.1, 30),
            "humidity": np.cos(x) + np.random.normal(0, 0.1, 30),
            "pressure": np.linspace(1000, 1020, 30) + np.random.normal(0, 0.5, 30),
        }
    )

    # Add anomalies to each variable
    df.loc[10, "temperature"] = 5.0  # Temperature spike
    df.loc[15, "humidity"] = -2.0  # Humidity dip
    df.loc[20, "pressure"] = np.nan  # Missing pressure value

    return df


def test_agent_initialization() -> None:
    """Test basic initialization of AnomalyAgent."""
    agent = AnomalyAgent()
    assert agent.timestamp_col == "timestamp"
    assert agent.llm is not None


def test_agent_custom_initialization() -> None:
    """Test initialization with custom parameters."""
    agent = AnomalyAgent(model_name="gpt-4o-mini", timestamp_col="time")
    assert agent.timestamp_col == "time"
    assert agent.llm is not None


def test_detect_anomalies_single_variable(
    single_variable_df: pd.DataFrame,
) -> None:
    """Test anomaly detection with a single variable."""
    agent = AnomalyAgent()
    anomalies = agent.detect_anomalies(single_variable_df)

    assert isinstance(anomalies, dict)
    assert "temperature" in anomalies
    assert isinstance(anomalies["temperature"], AnomalyList)
    assert len(anomalies["temperature"].anomalies) > 0


def test_detect_anomalies_multiple_variables(
    multi_variable_df: pd.DataFrame,
) -> None:
    """Test anomaly detection with multiple variables."""
    agent = AnomalyAgent()
    anomalies = agent.detect_anomalies(multi_variable_df)

    assert isinstance(anomalies, dict)
    assert all(col in anomalies for col in ["temperature", "humidity", "pressure"])
    assert all(
        isinstance(anomaly_list, AnomalyList) for anomaly_list in anomalies.values()
    )


def test_get_anomalies_df_long_format(
    single_variable_df: pd.DataFrame,
) -> None:
    """Test conversion of anomalies to DataFrame in long format."""
    agent = AnomalyAgent()
    anomalies = agent.detect_anomalies(single_variable_df)
    df_anomalies = agent.get_anomalies_df(anomalies, format="long")

    assert isinstance(df_anomalies, pd.DataFrame)
    expected_cols = [
        "timestamp",
        "variable_name",
        "value",
        "anomaly_description",
    ]
    assert all(col in df_anomalies.columns for col in expected_cols)
    assert len(df_anomalies) > 0


def test_get_anomalies_df_wide_format(
    single_variable_df: pd.DataFrame,
) -> None:
    """Test conversion of anomalies to DataFrame in wide format."""
    agent = AnomalyAgent()
    anomalies = agent.detect_anomalies(single_variable_df)
    df_anomalies = agent.get_anomalies_df(anomalies, format="wide")

    assert isinstance(df_anomalies, pd.DataFrame)
    assert "timestamp" in df_anomalies.columns
    assert "temperature" in df_anomalies.columns
    assert len(df_anomalies) > 0


def test_invalid_format(single_variable_df: pd.DataFrame) -> None:
    """Test error handling for invalid format parameter."""
    agent = AnomalyAgent()
    anomalies = agent.detect_anomalies(single_variable_df)

    with pytest.raises(ValueError):
        agent.get_anomalies_df(anomalies, format="invalid")


def test_empty_dataframe() -> None:
    """Test handling of empty DataFrame."""
    agent = AnomalyAgent()
    empty_df = pd.DataFrame(columns=["timestamp", "value"])

    anomalies = agent.detect_anomalies(empty_df)
    assert isinstance(anomalies, dict)
    assert "value" in anomalies
    assert isinstance(anomalies["value"], AnomalyList)
    assert len(anomalies["value"].anomalies) == 0


def test_custom_timestamp_column() -> None:
    """Test using a custom timestamp column name."""
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({"time": dates, "value": np.random.random(10)})

    agent = AnomalyAgent(timestamp_col="time")
    anomalies = agent.detect_anomalies(df)

    assert isinstance(anomalies, dict)
    assert "value" in anomalies


def test_missing_timestamp_column() -> None:
    """Test handling of missing timestamp column."""
    df = pd.DataFrame({"value": np.random.random(10)})

    agent = AnomalyAgent()
    with pytest.raises(KeyError):
        agent.detect_anomalies(df)


def test_non_numeric_columns() -> None:
    """Test handling of non-numeric columns."""
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "value": np.random.random(10),
            "category": ["A"] * 10,
            "text": ["test"] * 10,
            "date": pd.date_range(start="2024-01-01", periods=10, freq="D"),
        }
    )

    agent = AnomalyAgent()
    anomalies = agent.detect_anomalies(df)

    assert isinstance(anomalies, dict)
    assert "value" in anomalies
    # Check that only numeric columns are included
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    assert all(col in anomalies for col in numeric_cols)
    assert all(
        col not in anomalies for col in df.select_dtypes(exclude=[np.number]).columns
    )


def test_anomaly_model_validation() -> None:
    """Test validation of Anomaly model."""
    with pytest.raises(ValueError):
        Anomaly(
            timestamp="invalid",
            variable_value="not a number",
            anomaly_description="test",
        )


def test_anomaly_list_validation() -> None:
    """Test validation of AnomalyList model."""
    with pytest.raises(ValueError):
        AnomalyList(anomalies="not a list")


def test_anomaly_timestamp_validation() -> None:
    """Test validation of anomaly timestamp format."""
    with pytest.raises(ValueError):
        Anomaly(
            timestamp="not-a-date",
            variable_value=1.0,
            anomaly_description="test",
        )


def test_anomaly_value_validation() -> None:
    """Test validation of anomaly value type."""
    with pytest.raises(ValueError):
        Anomaly(
            timestamp="2024-01-01",
            variable_value="not-a-number",
            anomaly_description="test",
        )


def test_anomaly_description_validation() -> None:
    """Test validation of anomaly description."""
    with pytest.raises(ValueError):
        Anomaly(
            timestamp="2024-01-01", variable_value=1.0, anomaly_description=123
        )  # Should be string
