"""Test suite for Anthropic Claude model integration.

This module contains tests specifically for Claude model functionality,
including model selection, API integration, and cross-model compatibility.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

from anomaly_agent.agent import AnomalyAgent, _create_llm
from anomaly_agent.constants import ANTHROPIC_MODELS, OPENAI_MODELS
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    dates = pd.date_range(start="2024-01-01", periods=20, freq="D")
    values = np.sin(np.linspace(0, 2 * np.pi, 20)) + np.random.normal(0, 0.1, 20)
    values[10] = 5.0  # Add anomaly
    
    return pd.DataFrame({"timestamp": dates, "temperature": values})


class TestLLMCreation:
    """Test LLM instance creation for different model types."""
    
    def test_create_anthropic_llm(self):
        """Test creation of Anthropic LLM instances."""
        for model in ANTHROPIC_MODELS:
            llm = _create_llm(model)
            assert isinstance(llm, ChatAnthropic)
            assert llm.model == model
    
    def test_create_openai_llm(self):
        """Test creation of OpenAI LLM instances.""" 
        for model in OPENAI_MODELS:
            llm = _create_llm(model)
            assert isinstance(llm, ChatOpenAI)
            assert llm.model_name == model
    
    def test_create_unknown_model_defaults_to_openai(self):
        """Test that unknown models default to OpenAI."""
        unknown_model = "unknown-model-123"
        llm = _create_llm(unknown_model)
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == unknown_model


class TestAnomalyAgentModelSelection:
    """Test AnomalyAgent initialization with different models."""
    
    def test_agent_with_claude_model(self):
        """Test agent initialization with Claude models."""
        model_name = "claude-3-5-haiku-20241022"
        agent = AnomalyAgent(model_name=model_name)
        assert isinstance(agent.llm, ChatAnthropic)
        assert agent.llm.model == model_name
    
    def test_agent_with_openai_model(self):
        """Test agent initialization with OpenAI models."""
        model_name = "gpt-4o-mini"
        agent = AnomalyAgent(model_name=model_name)
        assert isinstance(agent.llm, ChatOpenAI)
        assert agent.llm.model_name == model_name
    
    def test_agent_configuration_preserved(self):
        """Test that other agent configuration is preserved with different models."""
        model_name = "claude-3-5-sonnet-20241022"
        custom_prompt = "Custom detection prompt"
        
        agent = AnomalyAgent(
            model_name=model_name,
            timestamp_col="custom_timestamp",
            verify_anomalies=False,
            detection_prompt=custom_prompt
        )
        
        assert isinstance(agent.llm, ChatAnthropic)
        assert agent.timestamp_col == "custom_timestamp"
        assert agent.verify_anomalies == False
        assert agent.detection_prompt == custom_prompt


class TestClaudeModelConstants:
    """Test model constants and validation."""
    
    def test_anthropic_models_list_not_empty(self):
        """Test that ANTHROPIC_MODELS contains expected models."""
        assert len(ANTHROPIC_MODELS) > 0
        assert "claude-3-5-sonnet-20241022" in ANTHROPIC_MODELS
        assert "claude-3-5-haiku-20241022" in ANTHROPIC_MODELS
    
    def test_openai_models_list_not_empty(self):
        """Test that OPENAI_MODELS contains expected models."""
        assert len(OPENAI_MODELS) > 0
        assert "gpt-4o-mini" in OPENAI_MODELS
        assert "gpt-4o" in OPENAI_MODELS
    
    def test_no_model_overlap(self):
        """Test that there's no overlap between model lists."""
        overlap = set(ANTHROPIC_MODELS) & set(OPENAI_MODELS)
        assert len(overlap) == 0


@pytest.mark.integration
class TestClaudeIntegration:
    """Integration tests for Claude models (requires API key)."""
    
    @pytest.fixture(autouse=True)
    def check_api_key(self):
        """Check for Anthropic API key before running integration tests."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not found in environment")
    
    def test_claude_haiku_detection(self, sample_df):
        """Test anomaly detection with Claude Haiku."""
        agent = AnomalyAgent(model_name="claude-3-5-haiku-20241022")
        
        try:
            anomalies = agent.detect_anomalies(sample_df)
            assert isinstance(anomalies, dict)
            assert "temperature" in anomalies
            
            # Should detect at least the obvious anomaly
            temp_anomalies = anomalies["temperature"]
            assert len(temp_anomalies.anomalies) >= 1
            
        except Exception as e:
            pytest.fail(f"Claude Haiku detection failed: {e}")
    
    def test_claude_sonnet_detection(self, sample_df):
        """Test anomaly detection with Claude Sonnet."""
        agent = AnomalyAgent(model_name="claude-3-5-sonnet-20241022")
        
        try:
            anomalies = agent.detect_anomalies(sample_df)
            assert isinstance(anomalies, dict)
            assert "temperature" in anomalies
            
            # Should detect at least the obvious anomaly
            temp_anomalies = anomalies["temperature"]
            assert len(temp_anomalies.anomalies) >= 1
            
        except Exception as e:
            pytest.fail(f"Claude Sonnet detection failed: {e}")
    
    def test_claude_vs_openai_consistency(self, sample_df):
        """Test that Claude and OpenAI models produce reasonable results."""
        # Skip if no OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not found for comparison")
        
        # Test both models
        agent_claude = AnomalyAgent(model_name="claude-3-5-haiku-20241022")
        agent_openai = AnomalyAgent(model_name="gpt-4o-mini")
        
        try:
            claude_anomalies = agent_claude.detect_anomalies(sample_df)
            openai_anomalies = agent_openai.detect_anomalies(sample_df)
            
            # Both should detect anomalies in the same variables
            assert set(claude_anomalies.keys()) == set(openai_anomalies.keys())
            
            # Both should detect at least one anomaly
            claude_count = sum(len(al.anomalies) for al in claude_anomalies.values())
            openai_count = sum(len(al.anomalies) for al in openai_anomalies.values())
            
            assert claude_count >= 1
            assert openai_count >= 1
            
        except Exception as e:
            pytest.fail(f"Cross-model comparison failed: {e}")


class TestClaudeModelMocking:
    """Test Claude functionality with mocked responses."""
    
    @patch('anomaly_agent.agent.ChatAnthropic')
    def test_mocked_claude_detection(self, mock_claude_class, sample_df):
        """Test Claude detection with mocked LLM response."""
        # Create mock instance
        mock_llm = MagicMock()
        mock_claude_class.return_value = mock_llm
        
        # Mock structured output
        from anomaly_agent.agent import AnomalyList, Anomaly
        mock_response = AnomalyList(anomalies=[
            Anomaly(
                timestamp="2024-01-11 00:00:00.000000",
                variable_value=5.0,
                anomaly_description="Significant spike in temperature"
            )
        ])
        
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response
        mock_llm.with_structured_output.return_value = mock_chain
        
        # Test agent
        agent = AnomalyAgent(model_name="claude-3-5-haiku-20241022")
        anomalies = agent.detect_anomalies(sample_df)
        
        # Verify mock was called correctly
        mock_claude_class.assert_called_once_with(model="claude-3-5-haiku-20241022")
        assert "temperature" in anomalies
        assert len(anomalies["temperature"].anomalies) == 1
        assert anomalies["temperature"].anomalies[0].variable_value == 5.0


if __name__ == "__main__":
    pytest.main([__file__])