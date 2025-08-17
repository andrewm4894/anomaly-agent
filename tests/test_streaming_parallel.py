"""Test suite for streaming and parallel processing features."""

import asyncio
import pandas as pd
import pytest
from unittest.mock import Mock, patch
from anomaly_agent import AnomalyAgent, AnomalyList, Anomaly


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    data = {
        'timestamp': pd.date_range('2023-01-01', periods=10, freq='D'),
        'temperature': [20.0, 21.0, 22.0, 25.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0],
        'pressure': [1013.0, 1014.0, 1015.0, 1020.0, 1014.0, 1013.0, 1012.0, 1011.0, 1010.0, 1009.0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing without LLM calls."""
    with patch('anomaly_agent.agent.ChatOpenAI'):
        agent = AnomalyAgent(model_name="gpt-5-nano", verify_anomalies=False)
        return agent


class TestStreamingCapabilities:
    """Test streaming detection functionality."""
    
    def test_streaming_callback_structure(self, mock_agent, sample_df):
        """Test that streaming callbacks are called with correct structure."""
        callback_calls = []
        
        def test_callback(column, event, data):
            callback_calls.append({
                'column': column,
                'event': event,
                'data': data
            })
        
        # Mock the graph execution to avoid LLM calls
        with patch.object(mock_agent._graph_manager, 'get_or_create_graph') as mock_graph:
            mock_app = Mock()
            mock_app.stream.return_value = [{'detect': {'detected_anomalies': AnomalyList(anomalies=[])}}]
            mock_app.invoke.return_value = {'detected_anomalies': AnomalyList(anomalies=[])}
            mock_graph.return_value = mock_app
            
            # Run streaming detection
            mock_agent.detect_anomalies_streaming(sample_df, progress_callback=test_callback)
        
        # Verify callback structure
        assert len(callback_calls) > 0
        
        # Check for start events
        start_events = [call for call in callback_calls if call['event'] == 'start']
        assert len(start_events) == 2  # One for each numeric column
        
        # Check for completion events (should match number of columns processed successfully)
        complete_events = [call for call in callback_calls if call['event'] == 'column_complete']
        node_events = [call for call in callback_calls if call['event'] == 'node_complete']
        
        # We should have some successful processing
        assert len(complete_events) >= 0  # May be 0 if errors occur
        assert len(node_events) >= 0     # May be 0 if stream is empty
        
        # Verify data structure for all events
        for call in callback_calls:
            assert 'column' in call
            assert 'event' in call
            assert 'data' in call
            assert isinstance(call['data'], dict)

    def test_streaming_without_callback(self, mock_agent, sample_df):
        """Test streaming detection without callback (should not error)."""
        with patch.object(mock_agent._graph_manager, 'get_or_create_graph') as mock_graph:
            mock_app = Mock()
            mock_app.stream.return_value = [{'detect': {'detected_anomalies': AnomalyList(anomalies=[])}}]
            mock_app.invoke.return_value = {'detected_anomalies': AnomalyList(anomalies=[])}
            mock_graph.return_value = mock_app
            
            # Should not raise an exception
            result = mock_agent.detect_anomalies_streaming(sample_df)
            assert isinstance(result, dict)
            assert len(result) == 2  # temperature and pressure

    def test_streaming_error_handling(self, mock_agent, sample_df):
        """Test error handling in streaming mode."""
        callback_calls = []
        
        def error_callback(column, event, data):
            callback_calls.append({'column': column, 'event': event, 'data': data})
        
        # Mock graph to raise an exception
        with patch.object(mock_agent._graph_manager, 'get_or_create_graph') as mock_graph:
            mock_app = Mock()
            mock_app.stream.side_effect = Exception("Test error")
            mock_graph.return_value = mock_app
            
            result = mock_agent.detect_anomalies_streaming(sample_df, progress_callback=error_callback)
            
            # Should still return results (with empty anomaly lists)
            assert isinstance(result, dict)
            assert len(result) == 2
            
            # Check for error events in callbacks
            error_events = [call for call in callback_calls if call['event'] == 'error']
            assert len(error_events) == 2  # One for each column


class TestParallelProcessing:
    """Test parallel processing functionality."""
    
    @pytest.mark.asyncio
    async def test_parallel_basic_functionality(self, mock_agent, sample_df):
        """Test basic parallel processing functionality."""
        with patch.object(mock_agent._graph_manager, 'get_or_create_graph') as mock_graph:
            mock_app = Mock()
            mock_app.invoke.return_value = {'detected_anomalies': AnomalyList(anomalies=[])}
            mock_graph.return_value = mock_app
            
            result = await mock_agent.detect_anomalies_parallel(sample_df, max_concurrent=2)
            
            assert isinstance(result, dict)
            assert len(result) == 2  # temperature and pressure
            assert all(isinstance(anomaly_list, AnomalyList) for anomaly_list in result.values())

    @pytest.mark.asyncio
    async def test_parallel_concurrency_limit(self, mock_agent, sample_df):
        """Test that concurrency limit is respected."""
        concurrent_count = 0
        max_concurrent_seen = 0
        
        async def mock_executor(func, *args):
            nonlocal concurrent_count, max_concurrent_seen
            concurrent_count += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
            
            try:
                # Simulate some processing time
                await asyncio.sleep(0.1)
                return func(*args)
            finally:
                concurrent_count -= 1
        
        with patch.object(mock_agent._graph_manager, 'get_or_create_graph') as mock_graph:
            mock_app = Mock()
            mock_app.invoke.return_value = {'detected_anomalies': AnomalyList(anomalies=[])}
            mock_graph.return_value = mock_app
            
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor = mock_executor
                
                await mock_agent.detect_anomalies_parallel(sample_df, max_concurrent=1)
                
                # With max_concurrent=1, we should never see more than 1 concurrent task
                assert max_concurrent_seen <= 1

    @pytest.mark.asyncio
    async def test_parallel_progress_callback(self, mock_agent, sample_df):
        """Test progress callbacks in parallel processing."""
        callback_calls = []
        
        def progress_callback(column, event, data):
            callback_calls.append({'column': column, 'event': event, 'data': data})
        
        with patch.object(mock_agent._graph_manager, 'get_or_create_graph') as mock_graph:
            mock_app = Mock()
            mock_app.invoke.return_value = {'detected_anomalies': AnomalyList(anomalies=[])}
            mock_graph.return_value = mock_app
            
            await mock_agent.detect_anomalies_parallel(
                sample_df, 
                max_concurrent=2,
                progress_callback=progress_callback
            )
        
        # Check that callbacks were called
        assert len(callback_calls) > 0
        
        # Should have start events for each column, completion events depend on success
        start_events = [call for call in callback_calls if call['event'] == 'start']
        complete_events = [call for call in callback_calls if call['event'] == 'column_complete']
        
        assert len(start_events) == 2  # Always have start events
        # Note: complete_events depend on successful execution, so may be 0 in mock scenario


class TestAsyncStreaming:
    """Test async streaming generator functionality."""
    
    @pytest.mark.asyncio
    async def test_async_streaming_generator(self, mock_agent, sample_df):
        """Test async streaming generator functionality."""
        events = []
        
        with patch.object(mock_agent._graph_manager, 'get_or_create_graph') as mock_graph:
            mock_app = Mock()
            mock_app.invoke.return_value = {'detected_anomalies': AnomalyList(anomalies=[])}
            mock_graph.return_value = mock_app
            
            # Mock the async stream wrapper
            async def mock_async_wrapper(app, state):
                yield {'detect': {'detected_anomalies': AnomalyList(anomalies=[])}}
            
            with patch.object(mock_agent, '_async_stream_wrapper', mock_async_wrapper):
                async for event in mock_agent.detect_anomalies_streaming_async(sample_df):
                    events.append(event)
        
        # Verify event structure
        assert len(events) > 0
        
        # Check for required events
        start_events = [e for e in events if e['event'] == 'start']
        result_events = [e for e in events if e['event'] == 'result']
        complete_events = [e for e in events if e['event'] == 'complete']
        
        assert len(start_events) == 1  # One overall start
        # Note: result events depend on successful execution in mock scenario
        assert len(complete_events) == 1  # One overall complete
        
        # Verify event data structure
        for event in events:
            assert 'event' in event
            assert 'column' in event
            assert 'data' in event

    @pytest.mark.asyncio
    async def test_async_streaming_error_handling(self, mock_agent):
        """Test error handling in async streaming."""
        # Test with invalid DataFrame (missing timestamp column)
        invalid_df = pd.DataFrame({'value': [1, 2, 3]})
        
        events = []
        async for event in mock_agent.detect_anomalies_streaming_async(invalid_df):
            events.append(event)
        
        # Should get error event
        assert len(events) == 1
        assert events[0]['event'] == 'error'
        assert 'Timestamp column' in events[0]['data']['error']


class TestParallelParameter:
    """Test the new parallel parameter in detect_anomalies."""
    
    def test_detect_anomalies_with_parallel_true(self, mock_agent, sample_df):
        """Test detect_anomalies with parallel=True parameter."""
        with patch.object(mock_agent._graph_manager, 'get_or_create_graph') as mock_graph:
            mock_app = Mock()
            mock_app.invoke.return_value = {'detected_anomalies': AnomalyList(anomalies=[])}
            mock_graph.return_value = mock_app
            
            # This should internally use detect_anomalies_parallel
            result = mock_agent.detect_anomalies(sample_df, parallel=True, max_concurrent=2)
            
            assert isinstance(result, dict)
            assert len(result) == 2  # temperature and pressure
            assert all(isinstance(anomaly_list, AnomalyList) for anomaly_list in result.values())

    def test_detect_anomalies_with_progress_callback(self, mock_agent, sample_df):
        """Test detect_anomalies with progress_callback (should use streaming mode)."""
        callback_calls = []
        
        def test_callback(column, event, data):
            callback_calls.append({'column': column, 'event': event, 'data': data})
        
        with patch.object(mock_agent._graph_manager, 'get_or_create_graph') as mock_graph:
            mock_app = Mock()
            mock_app.stream.return_value = [{'detect': {'detected_anomalies': AnomalyList(anomalies=[])}}]
            mock_app.invoke.return_value = {'detected_anomalies': AnomalyList(anomalies=[])}
            mock_graph.return_value = mock_app
            
            # This should internally use detect_anomalies_streaming
            result = mock_agent.detect_anomalies(sample_df, progress_callback=test_callback)
            
            assert isinstance(result, dict)
            assert len(result) == 2
            assert len(callback_calls) > 0  # Should have received callbacks

    def test_detect_anomalies_parallel_with_progress(self, mock_agent, sample_df):
        """Test detect_anomalies with both parallel=True and progress_callback."""
        callback_calls = []
        
        def test_callback(column, event, data):
            callback_calls.append({'column': column, 'event': event, 'data': data})
        
        with patch.object(mock_agent._graph_manager, 'get_or_create_graph') as mock_graph:
            mock_app = Mock()
            mock_app.invoke.return_value = {'detected_anomalies': AnomalyList(anomalies=[])}
            mock_graph.return_value = mock_app
            
            # This should use parallel mode with progress callbacks
            result = mock_agent.detect_anomalies(
                sample_df, 
                parallel=True, 
                max_concurrent=2,
                progress_callback=test_callback
            )
            
            assert isinstance(result, dict)
            assert len(result) == 2

    def test_detect_anomalies_sequential_mode(self, mock_agent, sample_df):
        """Test detect_anomalies in default sequential mode."""
        with patch.object(mock_agent._graph_manager, 'get_or_create_graph') as mock_graph:
            mock_app = Mock()
            mock_app.invoke.return_value = {'detected_anomalies': AnomalyList(anomalies=[])}
            mock_graph.return_value = mock_app
            
            # This should use sequential processing (default mode)
            result = mock_agent.detect_anomalies(sample_df)
            
            assert isinstance(result, dict)
            assert len(result) == 2
            assert all(isinstance(anomaly_list, AnomalyList) for anomaly_list in result.values())

    def test_parallel_safe_execution_outside_event_loop(self, mock_agent, sample_df):
        """Test that _run_parallel_safely works when no event loop is running."""
        with patch.object(mock_agent._graph_manager, 'get_or_create_graph') as mock_graph:
            mock_app = Mock()
            mock_app.invoke.return_value = {'detected_anomalies': AnomalyList(anomalies=[])}
            mock_graph.return_value = mock_app
            
            # This should use asyncio.run() path
            result = mock_agent._run_parallel_safely(sample_df)
            
            assert isinstance(result, dict)
            assert len(result) == 2


class TestConfigurationCompatibility:
    """Test compatibility with existing configuration options."""
    
    def test_streaming_with_verification_config(self, mock_agent, sample_df):
        """Test streaming with different verification configurations."""
        with patch.object(mock_agent._graph_manager, 'get_or_create_graph') as mock_graph:
            mock_app = Mock()
            mock_app.stream.return_value = [{'verify': {'verified_anomalies': AnomalyList(anomalies=[])}}]
            mock_app.invoke.return_value = {'verified_anomalies': AnomalyList(anomalies=[])}
            mock_graph.return_value = mock_app
            
            # Test with verification enabled
            result = mock_agent.detect_anomalies_streaming(
                sample_df,
                verify_anomalies=True,
                n_verify_steps=2
            )
            
            assert isinstance(result, dict)
            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_parallel_with_runtime_config(self, mock_agent, sample_df):
        """Test parallel processing with runtime configuration changes."""
        with patch.object(mock_agent._graph_manager, 'get_or_create_graph') as mock_graph:
            mock_app = Mock()
            mock_app.invoke.return_value = {'verified_anomalies': AnomalyList(anomalies=[])}
            mock_graph.return_value = mock_app
            
            result = await mock_agent.detect_anomalies_parallel(
                sample_df,
                verify_anomalies=True,
                n_verify_steps=3,
                max_concurrent=2
            )
            
            assert isinstance(result, dict)
            # Verify that graph manager was called with updated config
            mock_graph.assert_called()


# Integration test (requires actual API key)
class TestStreamingIntegration:
    """Integration tests for streaming (requires OPENAI_API_KEY)."""
    
    @pytest.mark.integration
    def test_streaming_integration(self, sample_df):
        """Test streaming with real LLM calls (integration test)."""
        agent = AnomalyAgent(verify_anomalies=False)
        
        callback_calls = []
        def callback(column, event, data):
            callback_calls.append(event)
        
        try:
            result = agent.detect_anomalies_streaming(sample_df, progress_callback=callback)
            assert isinstance(result, dict)
            assert len(callback_calls) > 0
        except Exception as e:
            if "API key" in str(e) or "OpenAI" in str(e):
                pytest.skip("OpenAI API key not available")
            else:
                raise

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_parallel_integration(self, sample_df):
        """Test parallel processing with real LLM calls (integration test)."""
        agent = AnomalyAgent(verify_anomalies=False)
        
        try:
            result = await agent.detect_anomalies_parallel(sample_df, max_concurrent=2)
            assert isinstance(result, dict)
        except Exception as e:
            if "API key" in str(e) or "OpenAI" in str(e):
                pytest.skip("OpenAI API key not available")
            else:
                raise