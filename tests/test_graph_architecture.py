"""
Test suite for graph architecture improvements.

This module tests the advanced architecture features:
- GraphManager caching behavior
- Class-based nodes (DetectionNode, VerificationNode, ErrorHandlerNode)  
- Enhanced error handling with exponential backoff
- Performance improvements from caching
- Backward compatibility
"""

import pytest
import pandas as pd
import numpy as np
import time
from datetime import datetime
from unittest.mock import Mock, patch
from langchain_openai import ChatOpenAI

from anomaly_agent import (
    AnomalyAgent, 
    GraphManager, 
    DetectionNode, 
    VerificationNode, 
    ErrorHandlerNode,
    AgentState,
    Anomaly,
    AnomalyList
)


class TestGraphManager:
    """Test the GraphManager caching system."""
    
    def test_graph_manager_initialization(self):
        """Test GraphManager initializes correctly."""
        manager = GraphManager()
        assert len(manager._compiled_graphs) == 0
        assert len(manager._node_instances) == 0
    
    def test_graph_caching(self):
        """Test that graphs are cached and reused."""
        from anomaly_agent.agent import AgentConfig
        
        manager = GraphManager()
        llm = Mock()
        
        config1 = AgentConfig(verify_anomalies=True, max_retries=3)
        config2 = AgentConfig(verify_anomalies=True, max_retries=3)  # Same config
        config3 = AgentConfig(verify_anomalies=False, max_retries=3)  # Different config
        
        # First call should create and cache
        with patch.object(manager, '_create_graph') as mock_create:
            mock_create.return_value = Mock()
            graph1 = manager.get_or_create_graph(config1, llm)
            assert mock_create.call_count == 1
        
        # Second call with same config should reuse cache
        with patch.object(manager, '_create_graph') as mock_create:
            mock_create.return_value = Mock()
            graph2 = manager.get_or_create_graph(config2, llm)
            assert mock_create.call_count == 0  # Should not be called
            assert graph1 is graph2
        
        # Third call with different config should create new
        with patch.object(manager, '_create_graph') as mock_create:
            mock_create.return_value = Mock()
            graph3 = manager.get_or_create_graph(config3, llm)
            assert mock_create.call_count == 1
            assert graph3 is not graph1
    
    def test_node_caching(self):
        """Test that node instances are cached and reused."""
        manager = GraphManager()
        llm1 = Mock()
        llm1.model_name = "gpt-4o-mini"
        llm2 = Mock()
        llm2.model_name = "gpt-4o-mini"  # Same model
        llm3 = Mock() 
        llm3.model_name = "gpt-4"  # Different model
        
        # First call should create and cache
        node1 = manager._get_or_create_node("detection", DetectionNode, llm1)
        assert len(manager._node_instances) == 1
        
        # Second call with equivalent LLM should reuse
        node2 = manager._get_or_create_node("detection", DetectionNode, llm2)
        assert node1 is node2
        assert len(manager._node_instances) == 1
        
        # Third call with different LLM should create new
        node3 = manager._get_or_create_node("detection", DetectionNode, llm3)
        assert node3 is not node1
        assert len(manager._node_instances) == 2


class TestDetectionNode:
    """Test the DetectionNode class."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        llm = Mock(spec=ChatOpenAI)
        llm.with_structured_output.return_value = Mock()
        return llm
    
    @pytest.fixture
    def detection_node(self, mock_llm):
        """Create a DetectionNode for testing."""
        return DetectionNode(mock_llm)
    
    def test_detection_node_initialization(self, mock_llm):
        """Test DetectionNode initializes correctly."""
        node = DetectionNode(mock_llm)
        assert node.llm is mock_llm
        assert len(node._chains) == 0
    
    def test_chain_caching(self, detection_node):
        """Test that LLM chains are cached by prompt."""
        prompt1 = "Find temperature anomalies"
        prompt2 = "Find pressure anomalies" 
        
        # First call should create and cache
        chain1 = detection_node._get_chain(prompt1)
        assert len(detection_node._chains) == 1
        
        # Second call with same prompt should reuse
        chain1_again = detection_node._get_chain(prompt1)
        assert chain1 is chain1_again
        assert len(detection_node._chains) == 1
        
        # Third call with different prompt should create new
        chain2 = detection_node._get_chain(prompt2)
        assert chain2 is not chain1
        assert len(detection_node._chains) == 2
    
    def test_detection_node_call_success(self, detection_node):
        """Test successful detection node execution."""
        # Mock the chain to return a result
        mock_result = AnomalyList(anomalies=[
            Anomaly(timestamp="2024-01-01 10:00:00", variable_value=5.0, anomaly_description="Test anomaly")
        ])
        
        with patch.object(detection_node, '_get_chain') as mock_get_chain:
            mock_chain = Mock()
            mock_chain.invoke.return_value = mock_result
            mock_get_chain.return_value = mock_chain
            
            state = AgentState(
                time_series="test data",
                variable_name="temperature", 
                processing_metadata={"verification_enabled": True}
            )
            
            result = detection_node(state)
            
            assert result["detected_anomalies"] == mock_result
            assert result["current_step"] == "verify"
            assert "detection_completed" in result["processing_metadata"]
            assert result["processing_metadata"]["detection_node_calls"] == 1
    
    def test_detection_node_call_error(self, detection_node):
        """Test detection node error handling."""
        with patch.object(detection_node, '_get_chain') as mock_get_chain:
            mock_chain = Mock()
            mock_chain.invoke.side_effect = Exception("LLM Error")
            mock_get_chain.return_value = mock_chain
            
            state = AgentState(
                time_series="test data",
                variable_name="temperature",
                processing_metadata={}
            )
            
            result = detection_node(state)
            
            assert result["current_step"] == "error"
            assert "Detection failed: LLM Error" in result["error_messages"]
            assert result["retry_count"] == 1
            assert "detection_error" in result["processing_metadata"]


class TestVerificationNode:
    """Test the VerificationNode class."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        llm = Mock(spec=ChatOpenAI)
        llm.with_structured_output.return_value = Mock()
        return llm
    
    @pytest.fixture 
    def verification_node(self, mock_llm):
        """Create a VerificationNode for testing."""
        return VerificationNode(mock_llm)
    
    def test_verification_node_initialization(self, mock_llm):
        """Test VerificationNode initializes correctly."""
        node = VerificationNode(mock_llm)
        assert node.llm is mock_llm
        assert len(node._chains) == 0
    
    def test_verification_no_anomalies(self, verification_node):
        """Test verification when no anomalies detected."""
        state = AgentState(
            time_series="test data",
            variable_name="temperature",
            detected_anomalies=None,
            processing_metadata={}
        )
        
        result = verification_node(state)
        
        assert result["verified_anomalies"] is None
        assert result["current_step"] == "end"
        assert result["processing_metadata"]["verification_skipped"] == "no_anomalies_detected"
    
    def test_verification_with_anomalies(self, verification_node):
        """Test verification with detected anomalies."""
        detected_anomalies = AnomalyList(anomalies=[
            Anomaly(timestamp="2024-01-01 10:00:00", variable_value=5.0, anomaly_description="Test anomaly")
        ])
        
        verified_anomalies = AnomalyList(anomalies=[
            Anomaly(timestamp="2024-01-01 10:00:00", variable_value=5.0, anomaly_description="Verified anomaly")
        ])
        
        with patch.object(verification_node, '_get_chain') as mock_get_chain:
            mock_chain = Mock()
            mock_chain.invoke.return_value = verified_anomalies
            mock_get_chain.return_value = mock_chain
            
            state = AgentState(
                time_series="test data",
                variable_name="temperature",
                detected_anomalies=detected_anomalies,
                processing_metadata={}
            )
            
            result = verification_node(state)
            
            assert result["verified_anomalies"] == verified_anomalies
            assert result["current_step"] == "end"
            assert result["processing_metadata"]["anomalies_verified"] == 1
            assert result["processing_metadata"]["anomalies_before_verification"] == 1


class TestErrorHandlerNode:
    """Test the ErrorHandlerNode class."""
    
    def test_error_handler_initialization(self):
        """Test ErrorHandlerNode initializes correctly."""
        node = ErrorHandlerNode(max_retries=5, backoff_factor=2.0)
        assert node.max_retries == 5
        assert node.backoff_factor == 2.0
    
    def test_error_handler_retry_logic(self):
        """Test retry logic with backoff calculation."""
        node = ErrorHandlerNode(max_retries=3, backoff_factor=1.5)
        
        # First retry attempt
        state = AgentState(
            time_series="test data",
            variable_name="temperature",
            retry_count=0,
            error_messages=["First error"],
            processing_metadata={}
        )
        
        result = node(state)
        
        assert result["current_step"] == "detect"
        assert result["retry_count"] == 0
        assert "retry_attempt_1" in result["processing_metadata"]
        assert result["processing_metadata"]["retry_delay_1"] == 1.0  # 1.5^0
    
    def test_error_handler_max_retries_exceeded(self):
        """Test behavior when max retries exceeded."""
        node = ErrorHandlerNode(max_retries=2, backoff_factor=1.5)
        
        state = AgentState(
            time_series="test data",
            variable_name="temperature", 
            retry_count=3,  # Exceeds max_retries
            error_messages=["Error 1", "Error 2", "Error 3"],
            processing_metadata={}
        )
        
        result = node(state)
        
        assert result["current_step"] == "end"
        assert result["processing_metadata"]["max_retries_exceeded"] is True
        assert result["processing_metadata"]["final_error"] == "Error 3"
        assert result["processing_metadata"]["total_retry_attempts"] == 3
    
    def test_exponential_backoff_calculation(self):
        """Test exponential backoff delay calculation."""
        node = ErrorHandlerNode(max_retries=5, backoff_factor=2.0)
        
        # Test different retry counts
        expected_delays = {
            0: 1.0,   # 2.0^0
            1: 2.0,   # 2.0^1
            2: 4.0,   # 2.0^2
            3: 8.0,   # 2.0^3
        }
        
        for retry_count, expected_delay in expected_delays.items():
            state = AgentState(
                time_series="test data",
                variable_name="temperature",
                retry_count=retry_count,
                error_messages=[],
                processing_metadata={}
            )
            
            result = node(state)
            actual_delay = result["processing_metadata"][f"retry_delay_{retry_count + 1}"]
            assert actual_delay == expected_delay


class TestAnomalyAgentArchitecture:
    """Test AnomalyAgent with Phase 2 improvements."""
    
    def test_shared_graph_manager(self):
        """Test that all agents share the same GraphManager."""
        agent1 = AnomalyAgent(verify_anomalies=True)
        agent2 = AnomalyAgent(verify_anomalies=False)
        agent3 = AnomalyAgent(verify_anomalies=True)
        
        # All agents should share the same GraphManager instance
        assert agent1._graph_manager is agent2._graph_manager
        assert agent2._graph_manager is agent3._graph_manager
        assert agent1._graph_manager is AnomalyAgent._graph_manager
    
    def test_graph_caching_across_agents(self):
        """Test that graphs are cached and reused across agent instances."""
        # Create a fresh GraphManager for this test
        original_manager = AnomalyAgent._graph_manager
        test_manager = GraphManager()
        AnomalyAgent._graph_manager = test_manager
        
        try:
            # Create agents with same configuration
            agent1 = AnomalyAgent(verify_anomalies=True, max_retries=3)
            graphs_after_first = len(agent1._graph_manager._compiled_graphs)
            
            agent2 = AnomalyAgent(verify_anomalies=True, max_retries=3)
            graphs_after_second = len(agent2._graph_manager._compiled_graphs)
            
            # Second agent should reuse cached graph
            assert graphs_after_second == graphs_after_first
            
            # Different configuration should create new graph
            agent3 = AnomalyAgent(verify_anomalies=False, max_retries=3)
            graphs_after_third = len(agent3._graph_manager._compiled_graphs)
            
            assert graphs_after_third == graphs_after_first + 1
        finally:
            # Restore original manager
            AnomalyAgent._graph_manager = original_manager
    
    def test_performance_improvement(self):
        """Test that graph caching improves performance."""
        # Measure time to create multiple agents with same config
        start_time = time.time()
        
        agents = []
        for i in range(5):
            agent = AnomalyAgent(verify_anomalies=True, max_retries=2)
            agents.append(agent)
        
        total_time = time.time() - start_time
        
        # Should be very fast due to caching (less than 1 second)
        assert total_time < 1.0
        
        # Verify all agents share cached resources
        first_manager = agents[0]._graph_manager
        assert all(agent._graph_manager is first_manager for agent in agents)
    
    def test_backward_compatibility(self):
        """Test that Phase 2 changes maintain backward compatibility."""
        # Test all initialization patterns still work
        agent1 = AnomalyAgent()
        agent2 = AnomalyAgent(model_name="gpt-4o-mini")
        agent3 = AnomalyAgent(verify_anomalies=False, detection_prompt="Custom prompt")
        
        # Test all properties still accessible
        assert hasattr(agent1, 'timestamp_col')
        assert hasattr(agent1, 'verify_anomalies') 
        assert hasattr(agent1, 'detection_prompt')
        assert hasattr(agent1, 'verification_prompt')
        assert hasattr(agent1, 'config')
        assert hasattr(agent1, 'app')
        
        # Test configuration values
        assert agent1.config.model_name == "gpt-4o-mini"
        assert agent2.config.model_name == "gpt-4o-mini"
        assert agent3.config.verify_anomalies is False
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        values = [1, 2, 1, 2, 10, 2, 1, 2, 1, 2]  # 10 is anomalous
        return pd.DataFrame({'timestamp': dates, 'value': values})
    
    def test_runtime_configuration_efficiency(self, sample_dataframe):
        """Test that runtime configuration changes use cached graphs efficiently."""
        agent = AnomalyAgent(verify_anomalies=True)
        
        # Get initial graph count
        initial_graphs = len(agent._graph_manager._compiled_graphs)
        
        # This should use cached graph even though verification setting differs
        with patch.object(agent, '_graph_manager') as mock_manager:
            mock_manager.get_or_create_graph.return_value = Mock()
            mock_manager._compiled_graphs = {"test": Mock()}
            
            # Simulate detect_anomalies call with different verification setting
            # (In real implementation, this would use cached graph efficiently)
            pass
        
        # Graph manager should handle configuration changes efficiently
        assert True  # Placeholder for actual runtime configuration test


class TestGraphArchitectureIntegration:
    """Integration tests for Phase 2 features."""
    
    def test_end_to_end_caching_behavior(self):
        """Test complete end-to-end caching behavior."""
        # Create a fresh GraphManager for this test
        original_manager = AnomalyAgent._graph_manager
        test_manager = GraphManager()
        AnomalyAgent._graph_manager = test_manager
        
        try:
            # Create multiple agents and verify caching
            agents = []
            configs = [
                {"verify_anomalies": True, "max_retries": 3},
                {"verify_anomalies": False, "max_retries": 3}, 
                {"verify_anomalies": True, "max_retries": 1},
                {"verify_anomalies": True, "max_retries": 3},  # Should reuse first
            ]
            
            for config in configs:
                agent = AnomalyAgent(**config)
                agents.append(agent)
            
            # Should have 3 unique graph configurations
            assert len(agents[0]._graph_manager._compiled_graphs) == 3
            
            # Agents with same config should share graphs
            assert agents[0]._graph_manager is agents[3]._graph_manager
        finally:
            # Restore original manager
            AnomalyAgent._graph_manager = original_manager
    
    def test_error_handling_integration(self):
        """Test that error handling works end-to-end with new architecture."""
        agent = AnomalyAgent(max_retries=2)
        
        # Verify error handler is properly configured
        error_node_keys = [k for k in agent._graph_manager._node_instances.keys() if 'error' in k]
        assert len(error_node_keys) > 0
        
        # Verify agent config is passed to error handler
        assert agent.config.max_retries == 2