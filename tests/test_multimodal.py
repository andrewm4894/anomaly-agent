"""Test suite for multimodal anomaly detection functionality.

This module contains tests for the multimodal (image + text) anomaly detection
capabilities of the AnomalyAgent class.
"""

import numpy as np
import pandas as pd
import pytest

from anomaly_agent.agent import AnomalyAgent, AnomalyList
from anomaly_agent.prompt import (
    DEFAULT_SYSTEM_PROMPT_WITH_IMAGE,
    build_multimodal_detection_messages,
)


@pytest.fixture
def sample_df_with_anomaly() -> pd.DataFrame:
    """Create a sample DataFrame with a known anomaly for testing."""
    dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
    x = np.linspace(0, 4 * np.pi, 50)
    values = np.sin(x) + np.random.normal(0, 0.1, 50)

    # Add an obvious anomaly
    values[25] = 5.0  # Large spike

    return pd.DataFrame({"timestamp": dates, "temperature": values})


@pytest.fixture
def multi_variable_df_with_anomalies() -> pd.DataFrame:
    """Create a DataFrame with multiple variables and known anomalies."""
    dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
    x = np.linspace(0, 2 * np.pi, 30)

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "temperature": np.sin(x) + np.random.normal(0, 0.1, 30),
            "humidity": np.cos(x) + np.random.normal(0, 0.1, 30),
        }
    )

    # Add anomalies
    df.loc[10, "temperature"] = 5.0  # Temperature spike
    df.loc[15, "humidity"] = -3.0  # Humidity dip

    return df


# =============================================================================
# Initialization Tests
# =============================================================================


def test_agent_initialization_with_include_plot_false() -> None:
    """Test agent initialization with include_plot=False (default)."""
    agent = AnomalyAgent()
    assert agent.include_plot is False


def test_agent_initialization_with_include_plot_true() -> None:
    """Test agent initialization with include_plot=True."""
    agent = AnomalyAgent(include_plot=True)
    assert agent.include_plot is True


def test_agent_initialization_mixed_parameters() -> None:
    """Test agent initialization with include_plot and other parameters."""
    agent = AnomalyAgent(
        model_name="gpt-4o-mini",
        timestamp_col="time",
        verify_anomalies=False,
        include_plot=True,
    )
    assert agent.include_plot is True
    assert agent.timestamp_col == "time"
    assert agent.verify_anomalies is False


# =============================================================================
# Plot Generation Tests
# =============================================================================


def test_generate_plot_base64(sample_df_with_anomaly: pd.DataFrame) -> None:
    """Test that _generate_plot_base64 produces valid base64 output."""
    agent = AnomalyAgent(include_plot=True)

    # Generate plot
    base64_str = agent._generate_plot_base64(
        sample_df_with_anomaly, "timestamp", "temperature"
    )

    # Verify it's a non-empty base64 string
    assert isinstance(base64_str, str)
    assert len(base64_str) > 0

    # Verify it can be decoded as base64
    import base64

    try:
        decoded = base64.b64decode(base64_str)
        # PNG files start with these bytes
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"
    except Exception as e:
        pytest.fail(f"Failed to decode base64 string: {e}")


def test_generate_plot_base64_different_columns(
    multi_variable_df_with_anomalies: pd.DataFrame,
) -> None:
    """Test plot generation for different columns."""
    agent = AnomalyAgent(include_plot=True)

    # Generate plots for each variable
    temp_plot = agent._generate_plot_base64(
        multi_variable_df_with_anomalies, "timestamp", "temperature"
    )
    humidity_plot = agent._generate_plot_base64(
        multi_variable_df_with_anomalies, "timestamp", "humidity"
    )

    # Both should be valid, non-empty base64 strings
    assert isinstance(temp_plot, str)
    assert len(temp_plot) > 0
    assert isinstance(humidity_plot, str)
    assert len(humidity_plot) > 0

    # They should be different (different data)
    assert temp_plot != humidity_plot


# =============================================================================
# Multimodal Message Building Tests
# =============================================================================


def test_build_multimodal_detection_messages() -> None:
    """Test that multimodal messages are built correctly."""
    messages = build_multimodal_detection_messages(
        variable_name="temperature",
        time_series="timestamp, temperature\n2024-01-01, 10.5\n2024-01-02, 11.0",
        plot_image_base64="dGVzdA==",  # "test" in base64
    )

    # Should have two messages: system and human
    assert len(messages) == 2

    # First message should be system message
    assert messages[0].type == "system"
    assert DEFAULT_SYSTEM_PROMPT_WITH_IMAGE in messages[0].content

    # Second message should be human message with multimodal content
    assert messages[1].type == "human"
    content = messages[1].content
    assert isinstance(content, list)
    assert len(content) == 2

    # First part should be text
    assert content[0]["type"] == "text"
    assert "temperature" in content[0]["text"]

    # Second part should be image
    assert content[1]["type"] == "image_url"
    assert "data:image/png;base64," in content[1]["image_url"]["url"]


def test_build_multimodal_detection_messages_custom_prompt() -> None:
    """Test multimodal message building with custom system prompt."""
    custom_prompt = "Custom system prompt for testing."

    messages = build_multimodal_detection_messages(
        variable_name="test_var",
        time_series="data here",
        plot_image_base64="dGVzdA==",
        system_prompt=custom_prompt,
    )

    assert messages[0].content == custom_prompt


# =============================================================================
# Detection with Plot Tests
# =============================================================================


def test_detection_with_plot_enabled(sample_df_with_anomaly: pd.DataFrame) -> None:
    """Test anomaly detection with plot image included."""
    agent = AnomalyAgent(include_plot=True)
    anomalies = agent.detect_anomalies(sample_df_with_anomaly)

    assert isinstance(anomalies, dict)
    assert "temperature" in anomalies
    assert isinstance(anomalies["temperature"], AnomalyList)
    # Should detect the spike at index 25
    assert len(anomalies["temperature"].anomalies) > 0


def test_detection_without_plot_still_works(
    sample_df_with_anomaly: pd.DataFrame,
) -> None:
    """Test that detection without plot (default) still works correctly."""
    agent = AnomalyAgent(include_plot=False)
    anomalies = agent.detect_anomalies(sample_df_with_anomaly)

    assert isinstance(anomalies, dict)
    assert "temperature" in anomalies
    assert isinstance(anomalies["temperature"], AnomalyList)
    # Should still detect anomalies
    assert len(anomalies["temperature"].anomalies) > 0


def test_detection_with_plot_multiple_variables(
    multi_variable_df_with_anomalies: pd.DataFrame,
) -> None:
    """Test multimodal detection with multiple variables."""
    agent = AnomalyAgent(include_plot=True)
    anomalies = agent.detect_anomalies(multi_variable_df_with_anomalies)

    assert isinstance(anomalies, dict)
    assert "temperature" in anomalies
    assert "humidity" in anomalies
    assert all(
        isinstance(anomaly_list, AnomalyList) for anomaly_list in anomalies.values()
    )


def test_detection_with_plot_and_verification_disabled(
    sample_df_with_anomaly: pd.DataFrame,
) -> None:
    """Test multimodal detection with verification disabled."""
    agent = AnomalyAgent(include_plot=True, verify_anomalies=False)
    anomalies = agent.detect_anomalies(sample_df_with_anomaly)

    assert isinstance(anomalies, dict)
    assert "temperature" in anomalies
    assert isinstance(anomalies["temperature"], AnomalyList)


def test_detection_with_plot_debug_mode(
    sample_df_with_anomaly: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that debug logging works with plot generation."""
    import logging

    caplog.set_level(logging.DEBUG)

    agent = AnomalyAgent(include_plot=True, debug=True)
    anomalies = agent.detect_anomalies(sample_df_with_anomaly)

    assert isinstance(anomalies, dict)
    # Debug logging should have been called (check logs contain plot generation info)
    # Note: This test mainly verifies no errors occur with debug=True


# =============================================================================
# Edge Cases
# =============================================================================


def test_detection_with_plot_empty_dataframe() -> None:
    """Test multimodal detection with empty DataFrame."""
    agent = AnomalyAgent(include_plot=True)
    empty_df = pd.DataFrame(columns=["timestamp", "value"])

    anomalies = agent.detect_anomalies(empty_df)
    assert isinstance(anomalies, dict)
    assert "value" in anomalies
    assert isinstance(anomalies["value"], AnomalyList)
    assert len(anomalies["value"].anomalies) == 0


def test_detection_with_plot_single_row() -> None:
    """Test multimodal detection with single row DataFrame."""
    agent = AnomalyAgent(include_plot=True)
    single_row_df = pd.DataFrame(
        {"timestamp": [pd.Timestamp("2024-01-01")], "value": [10.0]}
    )

    anomalies = agent.detect_anomalies(single_row_df)
    assert isinstance(anomalies, dict)
    assert "value" in anomalies


def test_default_system_prompt_with_image_content() -> None:
    """Test that the image-aware system prompt contains expected content."""
    assert "visualization" in DEFAULT_SYSTEM_PROMPT_WITH_IMAGE.lower()
    assert "plot" in DEFAULT_SYSTEM_PROMPT_WITH_IMAGE.lower()
    assert "visual" in DEFAULT_SYSTEM_PROMPT_WITH_IMAGE.lower()
    assert "numeric" in DEFAULT_SYSTEM_PROMPT_WITH_IMAGE.lower()
