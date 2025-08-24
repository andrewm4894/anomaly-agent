"""Test suite for the anomaly detection agent.

This module contains tests for the AnomalyAgent class and its components,
including anomaly detection, validation, and data handling.
"""

import numpy as np
import pandas as pd
import pytest

from anomaly_agent.agent import Anomaly, AnomalyAgent, AnomalyList
from anomaly_agent.prompt import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_VERIFY_SYSTEM_PROMPT,
    get_detection_prompt,
    get_verification_prompt,
)


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


@pytest.fixture  # type: ignore
def irregular_time_series_df() -> pd.DataFrame:
    """Create a DataFrame with irregular timestamps and known anomalies."""
    # Create irregular timestamps with varying gaps
    timestamps = [
        "2024-01-01 10:00:00",
        "2024-01-01 10:15:00",  # 15 min gap
        "2024-01-01 11:30:00",  # 1h 15min gap
        "2024-01-01 12:00:00",  # 30 min gap
        "2024-01-02 09:00:00",  # 21h gap
        "2024-01-02 09:05:00",  # 5 min gap
        "2024-01-02 09:10:00",  # 5 min gap
        "2024-01-03 15:00:00",  # ~30h gap
    ]

    # Create some sample data with anomalies
    values = np.random.normal(0, 1, len(timestamps))
    values[2] = 5.0  # Spike
    values[5] = -4.0  # Dip
    values[7] = np.nan  # Missing value

    return pd.DataFrame({"timestamp": pd.to_datetime(timestamps), "value": values})


@pytest.fixture  # type: ignore
def sub_second_time_series_df() -> pd.DataFrame:
    """Create a DataFrame with sub-second timestamps and known anomalies."""
    # Create timestamps with sub-second precision
    timestamps = [
        "2024-01-01 10:00:00.123",
        "2024-01-01 10:00:00.456",
        "2024-01-01 10:00:00.789",
        "2024-01-01 10:00:01.123",
        "2024-01-01 10:00:01.456",
    ]

    # Create some sample data with anomalies
    values = np.random.normal(0, 1, len(timestamps))
    values[1] = 5.0  # Spike
    values[3] = -4.0  # Dip

    return pd.DataFrame({"timestamp": pd.to_datetime(timestamps), "value": values})


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
    # Test valid timestamps
    valid_timestamps = [
        "2024-01-01 10:00:00",
        "2024-01-01 10:00:00.123",
        "2024-01-01 23:59:59",
    ]

    for ts in valid_timestamps:
        anomaly = Anomaly(
            timestamp=ts,
            variable_value=1.0,
            anomaly_description="test",
        )
        # Convert both to datetime for comparison to handle format differences
        input_dt = pd.to_datetime(ts)
        output_dt = pd.to_datetime(anomaly.timestamp)
        assert input_dt == output_dt


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


def test_irregular_time_series(irregular_time_series_df: pd.DataFrame) -> None:
    """Test anomaly detection with irregular time series."""
    agent = AnomalyAgent()
    anomalies = agent.detect_anomalies(irregular_time_series_df)

    assert isinstance(anomalies, dict)
    assert "value" in anomalies
    assert isinstance(anomalies["value"], AnomalyList)
    assert len(anomalies["value"].anomalies) > 0

    # Verify that timestamps are preserved correctly
    df_anomalies = agent.get_anomalies_df(anomalies, format="long")
    # Convert both to datetime for comparison to handle format differences
    original_timestamps = pd.to_datetime(irregular_time_series_df["timestamp"])
    anomaly_timestamps = pd.to_datetime(df_anomalies["timestamp"])

    # Check that all anomaly timestamps exist in the original data
    assert all(ts in original_timestamps.values for ts in anomaly_timestamps)


def test_sub_second_timestamps(sub_second_time_series_df: pd.DataFrame) -> None:
    """Test anomaly detection with sub-second timestamps."""
    agent = AnomalyAgent()
    anomalies = agent.detect_anomalies(sub_second_time_series_df)

    assert isinstance(anomalies, dict)
    assert "value" in anomalies
    assert isinstance(anomalies["value"], AnomalyList)
    assert len(anomalies["value"].anomalies) > 0

    # Verify that sub-second precision is preserved
    df_anomalies = agent.get_anomalies_df(anomalies, format="long")
    original_timestamps = sub_second_time_series_df["timestamp"].dt.strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )
    anomaly_timestamps = pd.to_datetime(df_anomalies["timestamp"]).dt.strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )

    # Check that all anomaly timestamps exist in the original data
    assert all(ts in original_timestamps.values for ts in anomaly_timestamps)


def test_agent_without_verification(single_variable_df: pd.DataFrame) -> None:
    """Test anomaly detection when verification is disabled."""
    # Test instance-level verification control
    agent = AnomalyAgent(verify_anomalies=False)
    assert agent.verify_anomalies is False

    # Detect anomalies with instance default
    anomalies = agent.detect_anomalies(single_variable_df)
    assert isinstance(anomalies, dict)
    assert "temperature" in anomalies
    assert isinstance(anomalies["temperature"], AnomalyList)
    assert len(anomalies["temperature"].anomalies) > 0

    # Test method-level verification control
    # Create agent with default verification enabled
    agent = AnomalyAgent(verify_anomalies=True)
    assert agent.verify_anomalies is True

    # Override verification at method level
    anomalies = agent.detect_anomalies(single_variable_df, verify_anomalies=False)
    assert isinstance(anomalies, dict)
    assert "temperature" in anomalies
    assert isinstance(anomalies["temperature"], AnomalyList)
    assert len(anomalies["temperature"].anomalies) > 0

    # Convert to DataFrame to check the results
    df_anomalies = agent.get_anomalies_df(anomalies, format="long")
    assert isinstance(df_anomalies, pd.DataFrame)
    assert len(df_anomalies) > 0

    # Verify that the timestamps of detected anomalies exist in the original data
    original_timestamps = pd.to_datetime(single_variable_df["timestamp"])
    anomaly_timestamps = pd.to_datetime(df_anomalies["timestamp"])
    assert all(ts in original_timestamps.values for ts in anomaly_timestamps)


# =============================================================================
# Prompt Customization Tests
# =============================================================================


def test_agent_with_default_prompts(single_variable_df: pd.DataFrame) -> None:
    """Test that agent works correctly with default prompts."""
    agent = AnomalyAgent()

    # Verify default prompts are stored
    assert agent.detection_prompt == DEFAULT_SYSTEM_PROMPT
    assert agent.verification_prompt == DEFAULT_VERIFY_SYSTEM_PROMPT

    # Test detection still works
    anomalies = agent.detect_anomalies(single_variable_df)
    assert isinstance(anomalies, dict)
    assert "temperature" in anomalies
    assert isinstance(anomalies["temperature"], AnomalyList)


def test_agent_with_custom_detection_prompt(single_variable_df: pd.DataFrame) -> None:
    """Test agent with custom detection prompt."""
    custom_detection = """
    You are a temperature sensor analyst. Look for temperature spikes above 3.0 degrees.
    """

    agent = AnomalyAgent(detection_prompt=custom_detection)

    # Verify custom prompt is stored
    assert agent.detection_prompt == custom_detection
    assert (
        agent.verification_prompt == DEFAULT_VERIFY_SYSTEM_PROMPT
    )  # Should use default

    # Test detection still works
    anomalies = agent.detect_anomalies(single_variable_df)
    assert isinstance(anomalies, dict)
    assert "temperature" in anomalies
    assert isinstance(anomalies["temperature"], AnomalyList)


def test_agent_with_custom_verification_prompt(
    single_variable_df: pd.DataFrame,
) -> None:
    """Test agent with custom verification prompt."""
    custom_verification = """
    You are extremely strict. Only confirm anomalies that are above 4.0 in absolute value.
    """

    agent = AnomalyAgent(verification_prompt=custom_verification)

    # Verify custom prompt is stored
    assert agent.detection_prompt == DEFAULT_SYSTEM_PROMPT  # Should use default
    assert agent.verification_prompt == custom_verification

    # Test detection still works
    anomalies = agent.detect_anomalies(single_variable_df)
    assert isinstance(anomalies, dict)
    assert "temperature" in anomalies
    assert isinstance(anomalies["temperature"], AnomalyList)


def test_agent_with_both_custom_prompts(single_variable_df: pd.DataFrame) -> None:
    """Test agent with both custom detection and verification prompts."""
    custom_detection = """
    You are a sensitive anomaly detector. Flag any value that seems unusual.
    """
    custom_verification = """
    You are a conservative verifier. Only confirm clear, obvious anomalies.
    """

    agent = AnomalyAgent(
        detection_prompt=custom_detection, verification_prompt=custom_verification
    )

    # Verify both custom prompts are stored
    assert agent.detection_prompt == custom_detection
    assert agent.verification_prompt == custom_verification

    # Test detection still works
    anomalies = agent.detect_anomalies(single_variable_df)
    assert isinstance(anomalies, dict)
    assert "temperature" in anomalies
    assert isinstance(anomalies["temperature"], AnomalyList)


def test_prompt_functions_with_defaults() -> None:
    """Test prompt functions with default parameters."""
    # Test detection prompt function
    detection_template = get_detection_prompt()
    assert detection_template is not None

    # Test verification prompt function
    verification_template = get_verification_prompt()
    assert verification_template is not None


def test_prompt_functions_with_custom_prompts() -> None:
    """Test prompt functions with custom prompts."""
    custom_detection = "Custom detection prompt for testing."
    custom_verification = "Custom verification prompt for testing."

    # Test detection prompt function with custom prompt
    detection_template = get_detection_prompt(custom_detection)
    assert detection_template is not None

    # Test verification prompt function with custom prompt
    verification_template = get_verification_prompt(custom_verification)
    assert verification_template is not None


def test_prompt_consistency_across_calls(single_variable_df: pd.DataFrame) -> None:
    """Test that custom prompts are consistently used across multiple detection calls."""
    custom_detection = "Look for values above 2.0."
    custom_verification = "Be very strict in verification."

    agent = AnomalyAgent(
        detection_prompt=custom_detection, verification_prompt=custom_verification
    )

    # Run detection multiple times
    anomalies1 = agent.detect_anomalies(single_variable_df)
    anomalies2 = agent.detect_anomalies(single_variable_df)

    # Verify prompts haven't changed
    assert agent.detection_prompt == custom_detection
    assert agent.verification_prompt == custom_verification

    # Both calls should return results
    assert isinstance(anomalies1, dict)
    assert isinstance(anomalies2, dict)


def test_mixed_initialization_parameters() -> None:
    """Test agent initialization with custom prompts and other parameters."""
    custom_detection = "Custom detection for testing."

    agent = AnomalyAgent(
        model_name="gpt-4o-mini",
        timestamp_col="time",
        verify_anomalies=False,
        detection_prompt=custom_detection,
    )

    # Verify all parameters are set correctly
    assert agent.timestamp_col == "time"
    assert agent.verify_anomalies is False
    assert agent.detection_prompt == custom_detection
    assert agent.verification_prompt == DEFAULT_VERIFY_SYSTEM_PROMPT
