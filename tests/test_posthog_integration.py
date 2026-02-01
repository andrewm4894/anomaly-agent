"""
Tests for PostHog LLM analytics integration.

This module tests the optional PostHog integration functionality.
"""

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from anomaly_agent.agent import AnomalyAgent


class TestPostHogIntegration:
    """Test PostHog integration functionality."""

    def test_posthog_disabled_by_default(self):
        """Test that PostHog is disabled when not configured."""
        with patch.dict(os.environ, {}, clear=True):
            agent = AnomalyAgent()
            assert agent.posthog_client is None
            assert agent.posthog_callback_handler is None

    def test_posthog_disabled_when_not_enabled(self):
        """Test that PostHog is disabled when POSTHOG_ENABLED is false."""
        with patch.dict(
            os.environ, {"POSTHOG_ENABLED": "false", "POSTHOG_API_KEY": "test-key"}
        ):
            agent = AnomalyAgent()
            assert agent.posthog_client is None
            assert agent.posthog_callback_handler is None

    def test_posthog_initialization_with_env_vars(self):
        """Test PostHog initialization with environment variables."""
        # Skip if PostHog package with LangChain support is not available
        from anomaly_agent.agent import POSTHOG_AVAILABLE

        if not POSTHOG_AVAILABLE:
            pytest.skip("PostHog package not available")

        with patch.dict(
            os.environ,
            {
                "POSTHOG_ENABLED": "true",
                "POSTHOG_API_KEY": "test-api-key",
                "POSTHOG_HOST": "http://localhost:8010",
                "POSTHOG_DISTINCT_ID": "test-user",
                "POSTHOG_AI_SESSION_ID": "test-session",
                "POSTHOG_PRIVACY_MODE": "true",
            },
        ):
            agent = AnomalyAgent()

            # Verify PostHog was initialized
            assert agent.posthog_client is not None
            assert agent.posthog_callback_handler is not None

    def test_posthog_without_optional_params(self):
        """Test PostHog initialization without optional parameters."""
        # Skip if PostHog package with LangChain support is not available
        from anomaly_agent.agent import POSTHOG_AVAILABLE

        if not POSTHOG_AVAILABLE:
            pytest.skip("PostHog package not available")

        with patch.dict(
            os.environ,
            {
                "POSTHOG_ENABLED": "true",
                "POSTHOG_API_KEY": "test-api-key",
            },
            clear=True,
        ):
            agent = AnomalyAgent()

            # Verify PostHog was initialized with minimal parameters
            assert agent.posthog_client is not None
            assert agent.posthog_callback_handler is not None

    @patch("anomaly_agent.agent.POSTHOG_AVAILABLE", False)
    def test_posthog_not_available_warning(self, caplog):
        """Test warning when PostHog is enabled but not installed."""
        with patch.dict(
            os.environ, {"POSTHOG_ENABLED": "true", "POSTHOG_API_KEY": "test-key"}
        ):
            agent = AnomalyAgent(debug=True)
            assert agent.posthog_client is None
            assert agent.posthog_callback_handler is None

    @patch("anomaly_agent.agent.POSTHOG_AVAILABLE", True)
    def test_posthog_missing_api_key_warning(self, caplog):
        """Test warning when PostHog is enabled but API key is missing."""
        with patch.dict(os.environ, {"POSTHOG_ENABLED": "true"}, clear=True):
            agent = AnomalyAgent(debug=True)
            assert agent.posthog_client is None
            assert agent.posthog_callback_handler is None

    def test_posthog_callback_used_in_detection(self):
        """Test that PostHog callback is configured during agent initialization."""
        # Skip if PostHog package with LangChain support is not available
        from anomaly_agent.agent import POSTHOG_AVAILABLE

        if not POSTHOG_AVAILABLE:
            pytest.skip("PostHog package not available")

        with patch.dict(
            os.environ,
            {"POSTHOG_ENABLED": "true", "POSTHOG_API_KEY": "test-api-key"},
            clear=True,
        ):
            agent = AnomalyAgent()
            assert agent.posthog_callback_handler is not None

    def test_posthog_initialization_error_handling(self):
        """Test error handling when PostHog initialization fails."""
        # Mock Posthog to raise an error during initialization
        with patch("anomaly_agent.agent.POSTHOG_AVAILABLE", True):
            with patch("anomaly_agent.agent.Posthog", side_effect=Exception("Init failed")):
                with patch.dict(
                    os.environ,
                    {"POSTHOG_ENABLED": "true", "POSTHOG_API_KEY": "test-api-key"},
                    clear=True,
                ):
                    # Agent should still be created even if PostHog fails
                    agent = AnomalyAgent(debug=True)
                    assert agent.posthog_client is None
                    assert agent.posthog_callback_handler is None

    def test_posthog_with_custom_metadata(self):
        """Test PostHog initialization with custom metadata."""
        from anomaly_agent.agent import POSTHOG_AVAILABLE

        if not POSTHOG_AVAILABLE:
            pytest.skip("PostHog package not available")

        mock_posthog = MagicMock()
        mock_callback = MagicMock()

        with patch("anomaly_agent.agent.Posthog", return_value=mock_posthog) as posthog_cls:
            with patch(
                "anomaly_agent.agent.PostHogCallbackHandler", return_value=mock_callback
            ):
                with patch.dict(
                    os.environ,
                    {"POSTHOG_ENABLED": "true", "POSTHOG_API_KEY": "test-api-key"},
                    clear=True,
                ):
                    custom_metadata = {
                        "metric_batch": "test-batch",
                        "metric_name": "test-metric",
                        "dagster_run_id": "abc123",
                    }
                    agent = AnomalyAgent(posthog_metadata=custom_metadata)

                    # Verify PostHog was initialized with super_properties containing metadata
                    assert agent.posthog_client is not None
                    call_kwargs = posthog_cls.call_args[1]
                    assert "super_properties" in call_kwargs
                    assert call_kwargs["super_properties"]["metric_batch"] == "test-batch"
                    assert call_kwargs["super_properties"]["metric_name"] == "test-metric"
                    assert call_kwargs["super_properties"]["dagster_run_id"] == "abc123"

    def test_posthog_metadata_merged_with_session_id(self):
        """Test that custom metadata is merged with session ID."""
        from anomaly_agent.agent import POSTHOG_AVAILABLE

        if not POSTHOG_AVAILABLE:
            pytest.skip("PostHog package not available")

        mock_posthog = MagicMock()
        mock_callback = MagicMock()

        with patch("anomaly_agent.agent.Posthog", return_value=mock_posthog) as posthog_cls:
            with patch(
                "anomaly_agent.agent.PostHogCallbackHandler", return_value=mock_callback
            ):
                with patch.dict(
                    os.environ,
                    {
                        "POSTHOG_ENABLED": "true",
                        "POSTHOG_API_KEY": "test-api-key",
                        "POSTHOG_AI_SESSION_ID": "test-session-123",
                    },
                    clear=True,
                ):
                    custom_metadata = {"metric_batch": "my-batch"}
                    agent = AnomalyAgent(posthog_metadata=custom_metadata)

                    # Verify both metadata and session ID are in super_properties
                    call_kwargs = posthog_cls.call_args[1]
                    assert "super_properties" in call_kwargs
                    assert call_kwargs["super_properties"]["metric_batch"] == "my-batch"
                    assert call_kwargs["super_properties"]["$ai_session_id"] == "test-session-123"
