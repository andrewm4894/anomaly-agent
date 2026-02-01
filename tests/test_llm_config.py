"""
Tests for LLM configuration including OpenRouter support.
"""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestLLMConfiguration:
    """Test LLM configuration and provider support."""

    def test_default_openai_configuration(self):
        """Test that default configuration uses OpenAI."""
        with patch("anomaly_agent.agent.ChatOpenAI") as mock_chat:
            mock_chat.return_value = MagicMock()
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
                from anomaly_agent.agent import AnomalyAgent

                agent = AnomalyAgent()

                # Verify ChatOpenAI was called with just the model
                call_kwargs = mock_chat.call_args[1]
                assert call_kwargs["model"] == "gpt-4o-mini"
                assert "base_url" not in call_kwargs

    def test_openrouter_via_env_var(self):
        """Test OpenRouter configuration via OPENROUTER_BASE_URL env var."""
        with patch("anomaly_agent.agent.ChatOpenAI") as mock_chat:
            mock_chat.return_value = MagicMock()
            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "test-openrouter-key",
                    "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
                },
                clear=True,
            ):
                from anomaly_agent.agent import AnomalyAgent

                agent = AnomalyAgent(model_name="anthropic/claude-3.5-sonnet")

                call_kwargs = mock_chat.call_args[1]
                assert call_kwargs["model"] == "anthropic/claude-3.5-sonnet"
                assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"

    def test_openrouter_via_parameter(self):
        """Test OpenRouter configuration via base_url parameter."""
        with patch("anomaly_agent.agent.ChatOpenAI") as mock_chat:
            mock_chat.return_value = MagicMock()
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
                from anomaly_agent.agent import AnomalyAgent

                agent = AnomalyAgent(
                    model_name="google/gemini-pro-1.5",
                    base_url="https://openrouter.ai/api/v1",
                    api_key="my-openrouter-key",
                )

                call_kwargs = mock_chat.call_args[1]
                assert call_kwargs["model"] == "google/gemini-pro-1.5"
                assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"
                assert call_kwargs["api_key"] == "my-openrouter-key"

    def test_parameter_overrides_env_var(self):
        """Test that parameter base_url overrides environment variable."""
        with patch("anomaly_agent.agent.ChatOpenAI") as mock_chat:
            mock_chat.return_value = MagicMock()
            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "test-key",
                    "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
                },
                clear=True,
            ):
                from anomaly_agent.agent import AnomalyAgent

                agent = AnomalyAgent(
                    model_name="custom-model",
                    base_url="https://my-custom-api.com/v1",
                )

                call_kwargs = mock_chat.call_args[1]
                assert call_kwargs["base_url"] == "https://my-custom-api.com/v1"

    def test_openai_base_url_takes_precedence(self):
        """Test that OPENAI_BASE_URL takes precedence over OPENROUTER_BASE_URL."""
        with patch("anomaly_agent.agent.ChatOpenAI") as mock_chat:
            mock_chat.return_value = MagicMock()
            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "test-key",
                    "OPENAI_BASE_URL": "https://custom-openai.com/v1",
                    "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
                },
                clear=True,
            ):
                from anomaly_agent.agent import AnomalyAgent

                agent = AnomalyAgent()

                call_kwargs = mock_chat.call_args[1]
                assert call_kwargs["base_url"] == "https://custom-openai.com/v1"
