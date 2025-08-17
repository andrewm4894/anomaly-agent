"""
Test suite for multiple verification steps functionality.

This module tests the n_verify_steps parameter and associated behavior including:
- Configuration validation
- Graph creation for different verification step counts
- State management across multiple verification steps
- Runtime parameter overrides
- Metadata tracking across steps
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from anomaly_agent import (
    AnomalyAgent, 
    AgentConfig,
    GraphManager,
    Anomaly, 
    AnomalyList
)


class TestNVerifyStepsConfiguration:
    """Test configuration and validation of n_verify_steps parameter."""
    
    def test_default_n_verify_steps(self) -> None:
        """Test that default n_verify_steps is 1."""
        agent = AnomalyAgent()
        assert agent.n_verify_steps == 1
        assert agent.config.n_verify_steps == 1
    
    def test_custom_n_verify_steps(self) -> None:
        """Test setting custom n_verify_steps values."""
        for n in [1, 2, 3, 4, 5]:
            agent = AnomalyAgent(n_verify_steps=n)
            assert agent.n_verify_steps == n
            assert agent.config.n_verify_steps == n
    
    def test_n_verify_steps_validation_minimum(self) -> None:
        """Test that n_verify_steps must be >= 1."""
        with pytest.raises(Exception) as exc_info:
            AnomalyAgent(n_verify_steps=0)
        assert "greater than or equal to 1" in str(exc_info.value)
    
    def test_n_verify_steps_validation_maximum(self) -> None:
        """Test that n_verify_steps must be <= 5."""
        with pytest.raises(Exception) as exc_info:
            AnomalyAgent(n_verify_steps=6)
        assert "less than or equal to 5" in str(exc_info.value)
    
    def test_n_verify_steps_validation_type(self) -> None:
        """Test that n_verify_steps must be an integer."""
        with pytest.raises(Exception):
            AnomalyAgent(n_verify_steps=2.5)
        # Note: String "3" is automatically converted to int by Pydantic, so it doesn't fail
        # This is expected behavior for Pydantic v2


class TestGraphManagerWithMultipleVerification:
    """Test GraphManager behavior with multiple verification steps."""
    
    def test_graph_caching_with_different_n_verify_steps(self) -> None:
        """Test that different n_verify_steps values create different cached graphs."""
        manager = GraphManager()
        
        config1 = AgentConfig(n_verify_steps=1)
        config2 = AgentConfig(n_verify_steps=2)
        config3 = AgentConfig(n_verify_steps=3)
        
        llm = Mock()
        
        # Create graphs for different configurations
        graph1 = manager.get_or_create_graph(config1, llm)
        graph2 = manager.get_or_create_graph(config2, llm) 
        graph3 = manager.get_or_create_graph(config3, llm)
        
        # Should be different graphs
        assert graph1 is not graph2
        assert graph2 is not graph3
        assert graph1 is not graph3
        
        # But same configs should return same graphs
        graph1_again = manager.get_or_create_graph(config1, llm)
        assert graph1 is graph1_again
    
    def test_cache_key_includes_n_verify_steps(self) -> None:
        """Test that cache keys properly include n_verify_steps."""
        manager = GraphManager()
        
        config1 = AgentConfig(verify_anomalies=True, n_verify_steps=1, max_retries=3)
        config2 = AgentConfig(verify_anomalies=True, n_verify_steps=2, max_retries=3)
        
        llm = Mock()
        
        # Create graphs and verify they're cached separately
        manager.get_or_create_graph(config1, llm)
        manager.get_or_create_graph(config2, llm)
        
        # Should have 2 different cached graphs
        assert len(manager._compiled_graphs) == 2
        
        # Cache keys should be different
        keys = list(manager._compiled_graphs.keys())
        assert keys[0] != keys[1]
        assert "steps_1" in keys[0] or "steps_2" in keys[0]
        assert "steps_1" in keys[1] or "steps_2" in keys[1]


class TestMultipleVerificationStepsExecution:
    """Test execution behavior with multiple verification steps."""
    
    def test_runtime_n_verify_steps_override(self) -> None:
        """Test runtime override of n_verify_steps parameter."""
        # Create mock data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5),
            'value': [1, 2, 10, 2, 1]
        })
        
        agent = AnomalyAgent(n_verify_steps=1)
        
        # Mock the graph manager and LLM calls
        with patch.object(agent._graph_manager, 'get_or_create_graph') as mock_get_graph:
            mock_app = Mock()
            mock_app.invoke.return_value = {"verified_anomalies": AnomalyList(anomalies=[])}
            mock_get_graph.return_value = mock_app
            
            # Test runtime override
            agent.detect_anomalies(df, n_verify_steps=3)
            
            # Verify that get_or_create_graph was called with the overridden config
            call_args = mock_get_graph.call_args[0]
            config = call_args[0]
            assert config.n_verify_steps == 3  # Should be overridden value, not default 1
    
    def test_no_verification_with_n_verify_steps(self) -> None:
        """Test that n_verify_steps is properly handled when verify_anomalies=False."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=3),
            'value': [1, 10, 1]
        })
        
        agent = AnomalyAgent(verify_anomalies=False, n_verify_steps=3)
        
        # Mock the app directly since no graph recreation happens when verify_anomalies=False
        with patch.object(agent, 'app') as mock_app:
            mock_app.invoke.return_value = {"detected_anomalies": AnomalyList(anomalies=[])}
            
            result = agent.detect_anomalies(df)
            
            # Should complete successfully and return detected_anomalies (not verified_anomalies)
            assert isinstance(result, dict)
            mock_app.invoke.assert_called_once()
            
            # Verify the agent configuration
            assert agent.config.verify_anomalies is False
            assert agent.config.n_verify_steps == 3  # Still stored but not used


class TestMultipleVerificationMetadata:
    """Test metadata tracking across multiple verification steps."""
    
    def test_state_metadata_includes_n_verify_steps(self) -> None:
        """Test that state metadata includes n_verify_steps for nodes to access."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=3),
            'value': [1, 10, 1]
        })
        
        agent = AnomalyAgent(n_verify_steps=3, debug=True)
        
        # Mock the app.invoke to capture the state
        captured_state = None
        def capture_state(state):
            nonlocal captured_state
            captured_state = state
            return {"verified_anomalies": AnomalyList(anomalies=[])}
        
        with patch.object(agent, 'app') as mock_app:
            mock_app.invoke = capture_state
            agent.detect_anomalies(df)
        
        # Verify state contains n_verify_steps
        assert captured_state is not None
        assert captured_state.processing_metadata["n_verify_steps"] == 3
        assert captured_state.processing_metadata["verification_enabled"] is True
    
    def test_verification_step_metadata_tracking(self) -> None:
        """Test that verification steps are properly tracked in metadata."""
        from anomaly_agent.nodes import VerificationNode
        from anomaly_agent import AgentState
        from datetime import datetime
        
        # Create a verification node
        llm_mock = Mock()
        verification_node = VerificationNode(llm_mock)
        
        # Mock the chain to return empty results
        def mock_get_chain(prompt):
            chain_mock = Mock()
            chain_mock.invoke.return_value = AnomalyList(anomalies=[])
            return chain_mock
        
        verification_node._get_chain = mock_get_chain
        
        # Create initial state with anomalies to verify
        initial_anomalies = AnomalyList(anomalies=[
            Anomaly(timestamp="2024-01-01", variable_value=10.0, anomaly_description="Test")
        ])
        
        state = AgentState(
            time_series="test data",
            variable_name="test_var",
            detected_anomalies=initial_anomalies,
            processing_metadata={
                "n_verify_steps": 3,
                "verification_prompt": "test prompt",
                "debug": False
            }
        )
        
        # Run verification step
        result = verification_node(state)
        
        # Check that metadata was updated correctly
        metadata = result["processing_metadata"]
        assert "verification_1_completed" in metadata
        assert metadata["anomalies_after_verification_1"] == 0
        assert metadata["verification_node_calls"] == 1
        assert metadata["anomalies_before_current_verification"] == 1


class TestMultipleVerificationIntegration:
    """Integration tests for multiple verification steps."""
    
    def test_agent_configuration_consistency(self) -> None:
        """Test that agent configuration remains consistent across operations."""
        agent = AnomalyAgent(
            model_name="gpt-5-nano",
            verify_anomalies=True,
            n_verify_steps=2,
            max_retries=3
        )
        
        # Check initial configuration
        assert agent.config.n_verify_steps == 2
        assert agent.n_verify_steps == 2
        
        # Configuration should be immutable
        original_config = agent.config
        
        # Mock a detection call
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=3),
            'value': [1, 10, 1]
        })
        
        with patch.object(agent, 'app') as mock_app:
            mock_app.invoke.return_value = {"verified_anomalies": AnomalyList(anomalies=[])}
            agent.detect_anomalies(df)
        
        # Configuration should remain unchanged
        assert agent.config is original_config
        assert agent.n_verify_steps == 2
    
    def test_graph_manager_reuse_with_multiple_agents(self) -> None:
        """Test that multiple agents properly share the GraphManager."""
        agent1 = AnomalyAgent(n_verify_steps=2)
        agent2 = AnomalyAgent(n_verify_steps=2)  # Same config
        agent3 = AnomalyAgent(n_verify_steps=3)  # Different config
        
        # Should share the same GraphManager instance
        assert agent1._graph_manager is agent2._graph_manager
        assert agent2._graph_manager is agent3._graph_manager
        
        # Should reuse graphs for same configuration
        assert agent1.app is agent2.app
        
        # Should create different graphs for different configurations
        assert agent1.app is not agent3.app
    
    @pytest.mark.parametrize("n_steps", [1, 2, 3, 4, 5])
    def test_all_valid_n_verify_steps_values(self, n_steps: int) -> None:
        """Test that all valid n_verify_steps values work correctly."""
        agent = AnomalyAgent(n_verify_steps=n_steps)
        
        # Should create successfully
        assert agent.n_verify_steps == n_steps
        assert agent.config.n_verify_steps == n_steps
        
        # Should have a compiled graph
        assert agent.app is not None
        
        # Graph should be cached
        assert len(agent._graph_manager._compiled_graphs) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])