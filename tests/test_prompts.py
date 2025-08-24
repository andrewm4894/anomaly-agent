"""Test suite for prompt functionality.

This module contains tests for the prompt functions and custom prompt features
of the anomaly detection agent.
"""

import numpy as np
import pandas as pd
import pytest
from langchain_core.prompts import ChatPromptTemplate

from anomaly_agent.agent import AnomalyAgent, AnomalyList
from anomaly_agent.prompt import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_VERIFY_SYSTEM_PROMPT,
    get_detection_prompt,
    get_verification_prompt,
)


@pytest.fixture  # type: ignore
def sample_df() -> pd.DataFrame:
    """Create a simple DataFrame for testing."""
    dates = pd.date_range(start="2024-01-01", periods=20, freq="D")
    values = np.random.normal(0, 1, 20)
    values[10] = 5.0  # Add an obvious anomaly
    return pd.DataFrame({"timestamp": dates, "value": values})


# =============================================================================
# Prompt Function Tests
# =============================================================================


def test_default_prompts_exist() -> None:
    """Test that default prompts are defined and non-empty."""
    assert isinstance(DEFAULT_SYSTEM_PROMPT, str)
    assert len(DEFAULT_SYSTEM_PROMPT.strip()) > 0
    assert "anomaly" in DEFAULT_SYSTEM_PROMPT.lower()

    assert isinstance(DEFAULT_VERIFY_SYSTEM_PROMPT, str)
    assert len(DEFAULT_VERIFY_SYSTEM_PROMPT.strip()) > 0
    # Check for verification-related keywords
    verification_content = DEFAULT_VERIFY_SYSTEM_PROMPT.lower()
    assert any(
        word in verification_content
        for word in ["verify", "verification", "confirm", "review"]
    )


def test_get_detection_prompt_default() -> None:
    """Test detection prompt function with default parameters."""
    prompt_template = get_detection_prompt()

    assert isinstance(prompt_template, ChatPromptTemplate)

    # Test that it can be formatted
    formatted = prompt_template.format(
        variable_name="temperature", time_series="2024-01-01,20.5\n2024-01-02,21.0"
    )
    assert isinstance(formatted, str)
    assert "temperature" in formatted
    assert "2024-01-01" in formatted


def test_get_detection_prompt_custom() -> None:
    """Test detection prompt function with custom prompt."""
    custom_prompt = "You are a custom anomaly detector specialized in financial data."
    prompt_template = get_detection_prompt(custom_prompt)

    assert isinstance(prompt_template, ChatPromptTemplate)

    # Test that it uses the custom prompt
    formatted = prompt_template.format(
        variable_name="price", time_series="2024-01-01,100\n2024-01-02,105"
    )
    assert isinstance(formatted, str)
    assert "financial data" in formatted
    assert "price" in formatted


def test_get_verification_prompt_default() -> None:
    """Test verification prompt function with default parameters."""
    prompt_template = get_verification_prompt()

    assert isinstance(prompt_template, ChatPromptTemplate)

    # Test that it can be formatted
    formatted = prompt_template.format(
        variable_name="temperature",
        time_series="2024-01-01,20.5\n2024-01-02,21.0",
        detected_anomalies="timestamp: 2024-01-01, value: 20.5, Description: spike",
    )
    assert isinstance(formatted, str)
    assert "temperature" in formatted
    assert "2024-01-01" in formatted
    assert "spike" in formatted


def test_get_verification_prompt_custom() -> None:
    """Test verification prompt function with custom prompt."""
    custom_prompt = "You are a strict verifier. Only confirm extreme anomalies."
    prompt_template = get_verification_prompt(custom_prompt)

    assert isinstance(prompt_template, ChatPromptTemplate)

    # Test that it uses the custom prompt
    formatted = prompt_template.format(
        variable_name="value",
        time_series="2024-01-01,1\n2024-01-02,100",
        detected_anomalies="timestamp: 2024-01-02, value: 100, Description: spike",
    )
    assert isinstance(formatted, str)
    assert "strict verifier" in formatted
    assert "extreme anomalies" in formatted


# =============================================================================
# Agent Integration Tests
# =============================================================================


def test_agent_default_prompts(sample_df: pd.DataFrame) -> None:
    """Test that agent correctly uses default prompts."""
    agent = AnomalyAgent()

    # Verify default prompts are stored
    assert agent.detection_prompt == DEFAULT_SYSTEM_PROMPT
    assert agent.verification_prompt == DEFAULT_VERIFY_SYSTEM_PROMPT

    # Test that detection works
    anomalies = agent.detect_anomalies(sample_df)
    assert isinstance(anomalies, dict)
    assert "value" in anomalies
    assert isinstance(anomalies["value"], AnomalyList)


def test_agent_custom_detection_prompt(sample_df: pd.DataFrame) -> None:
    """Test agent with custom detection prompt."""
    custom_detection = """
    You are a highly sensitive anomaly detector. Mark any value that deviates
    from the mean by more than 1 standard deviation as an anomaly.
    """

    agent = AnomalyAgent(detection_prompt=custom_detection)

    # Verify custom prompt is stored correctly
    assert agent.detection_prompt == custom_detection
    assert agent.verification_prompt == DEFAULT_VERIFY_SYSTEM_PROMPT

    # Test that detection works with custom prompt
    anomalies = agent.detect_anomalies(sample_df)
    assert isinstance(anomalies, dict)
    assert "value" in anomalies
    assert isinstance(anomalies["value"], AnomalyList)


def test_agent_custom_verification_prompt(sample_df: pd.DataFrame) -> None:
    """Test agent with custom verification prompt."""
    custom_verification = """
    You are extremely conservative. Only confirm anomalies that are more than
    3 standard deviations from the mean and represent clear operational issues.
    """

    agent = AnomalyAgent(verification_prompt=custom_verification)

    # Verify custom prompt is stored correctly
    assert agent.detection_prompt == DEFAULT_SYSTEM_PROMPT
    assert agent.verification_prompt == custom_verification

    # Test that detection works with custom verification
    anomalies = agent.detect_anomalies(sample_df)
    assert isinstance(anomalies, dict)
    assert "value" in anomalies
    assert isinstance(anomalies["value"], AnomalyList)


def test_agent_both_custom_prompts(sample_df: pd.DataFrame) -> None:
    """Test agent with both custom detection and verification prompts."""
    custom_detection = "You are a sensitive detector. Flag unusual patterns."
    custom_verification = "You are a strict verifier. Be conservative."

    agent = AnomalyAgent(
        detection_prompt=custom_detection, verification_prompt=custom_verification
    )

    # Verify both custom prompts are stored
    assert agent.detection_prompt == custom_detection
    assert agent.verification_prompt == custom_verification

    # Test that detection works with both custom prompts
    anomalies = agent.detect_anomalies(sample_df)
    assert isinstance(anomalies, dict)
    assert "value" in anomalies
    assert isinstance(anomalies["value"], AnomalyList)


def test_prompt_persistence_across_calls(sample_df: pd.DataFrame) -> None:
    """Test that custom prompts persist across multiple detection calls."""
    custom_detection = "Custom detection prompt for persistence test."
    custom_verification = "Custom verification prompt for persistence test."

    agent = AnomalyAgent(
        detection_prompt=custom_detection, verification_prompt=custom_verification
    )

    # Run detection multiple times
    anomalies1 = agent.detect_anomalies(sample_df)
    anomalies2 = agent.detect_anomalies(sample_df)

    # Verify prompts haven't changed after multiple calls
    assert agent.detection_prompt == custom_detection
    assert agent.verification_prompt == custom_verification

    # Both calls should return valid results
    assert isinstance(anomalies1, dict)
    assert isinstance(anomalies2, dict)
    assert "value" in anomalies1
    assert "value" in anomalies2


def test_mixed_parameters_with_prompts() -> None:
    """Test agent initialization with custom prompts and other parameters."""
    custom_detection = "Financial anomaly detector."

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


def test_empty_custom_prompt() -> None:
    """Test behavior with empty custom prompts (should still work)."""
    # Empty strings should still work (though not recommended)
    agent = AnomalyAgent(detection_prompt="", verification_prompt="")

    assert agent.detection_prompt == ""
    assert agent.verification_prompt == ""


def test_prompt_content_isolation() -> None:
    """Test that different agents with different prompts don't interfere."""
    agent1 = AnomalyAgent(detection_prompt="Agent 1 detection prompt")
    agent2 = AnomalyAgent(detection_prompt="Agent 2 detection prompt")

    assert agent1.detection_prompt != agent2.detection_prompt
    assert "Agent 1" in agent1.detection_prompt
    assert "Agent 2" in agent2.detection_prompt
