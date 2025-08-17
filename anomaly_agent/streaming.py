"""
Streaming and parallel processing capabilities for the AnomalyAgent.

This module provides enhanced execution patterns including real-time progress updates,
concurrent processing, and async streaming generators following LangGraph best practices.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional, Callable, AsyncGenerator, Any, List, Tuple, TYPE_CHECKING

import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from .agent import AgentState, AnomalyList


class StreamingMixin:
    """Mixin class providing streaming and parallel processing capabilities."""
    
    def detect_anomalies_streaming(
        self,
        df: pd.DataFrame,
        progress_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
        timestamp_col: Optional[str] = None,
        verify_anomalies: Optional[bool] = None,
        n_verify_steps: Optional[int] = None,
    ) -> Dict[str, "AnomalyList"]:
        """Detect anomalies with real-time streaming progress updates.
        
        Args:
            df: DataFrame containing time series data
            progress_callback: Optional callback function that receives:
                - column_name (str): Current column being processed
                - event_type (str): Type of event ("start", "node_complete", "column_complete", "error")
                - data (dict): Additional event data
            timestamp_col: Name of the timestamp column (optional)
            verify_anomalies: Whether to verify detected anomalies (optional)
            n_verify_steps: Number of verification steps to run (optional)

        Returns:
            Dictionary mapping column names to their respective AnomalyList

        Example:
            def progress_handler(column, event, data):
                if event == "start":
                    print(f"🔍 Starting analysis for {column}")
                elif event == "node_complete":
                    print(f"✓ {data['node']} completed for {column}")
                elif event == "column_complete":
                    print(f"🎯 Found {data['anomaly_count']} anomalies in {column}")

            anomalies = agent.detect_anomalies_streaming(df, progress_handler)
        """
        # Import at runtime to avoid circular imports
        from .agent import AgentState, AnomalyList
        
        # Handle dynamic configuration
        current_timestamp_col = timestamp_col or self.config.timestamp_col
        current_verify = verify_anomalies if verify_anomalies is not None else self.config.verify_anomalies
        current_n_verify = n_verify_steps if n_verify_steps is not None else self.config.n_verify_steps
        
        # Get appropriate compiled graph
        if current_verify != self.config.verify_anomalies or current_n_verify != self.config.n_verify_steps:
            temp_config = self.config.model_copy(update={
                "verify_anomalies": current_verify,
                "n_verify_steps": current_n_verify
            })
            app = self._graph_manager.get_or_create_graph(temp_config, self.llm)
        else:
            app = self.app

        # Validate inputs
        if current_timestamp_col not in df.columns:
            raise KeyError(f"Timestamp column '{current_timestamp_col}' not found in DataFrame")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {col: AnomalyList(anomalies=[]) for col in df.columns if col != current_timestamp_col}

        df_str = df.to_string(index=False)
        results = {}

        # Process each column with streaming updates
        for col in numeric_cols:
            if progress_callback:
                progress_callback(col, "start", {"total_columns": len(numeric_cols)})

            # Create state for this column
            state = AgentState(
                time_series=df_str,
                variable_name=col,
                current_step="detect",
                processing_start_time=datetime.now(),
                processing_metadata={
                    "column": col,
                    "total_rows": len(df),
                    "verification_enabled": current_verify,
                    "n_verify_steps": current_n_verify,
                    "detection_prompt": self.config.detection_prompt,
                    "verification_prompt": self.config.verification_prompt,
                    "max_retries": self.config.max_retries,
                    "timestamp_col": current_timestamp_col,
                    "debug": self.debug
                }
            )

            # Stream the graph execution
            try:
                for chunk in app.stream(state):
                    # Extract node information from the chunk
                    if isinstance(chunk, dict):
                        for node_name, node_data in chunk.items():
                            if progress_callback and node_name != "__end__":
                                progress_callback(col, "node_complete", {
                                    "node": node_name,
                                    "step_data": node_data
                                })

                # Get final result
                result = app.invoke(state)
                
                # Extract anomalies
                if current_verify:
                    anomalies = result.get("verified_anomalies") or AnomalyList(anomalies=[])
                else:
                    anomalies = result.get("detected_anomalies") or AnomalyList(anomalies=[])
                
                results[col] = anomalies

                if progress_callback:
                    progress_callback(col, "column_complete", {
                        "anomaly_count": len(anomalies.anomalies),
                        "processing_time": (datetime.now() - state.processing_start_time).total_seconds()
                    })

            except Exception as e:
                if progress_callback:
                    progress_callback(col, "error", {"error": str(e)})
                # Still add empty result to maintain consistency
                results[col] = AnomalyList(anomalies=[])

        return results

    async def detect_anomalies_parallel(
        self,
        df: pd.DataFrame,
        max_concurrent: int = 3,
        progress_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
        timestamp_col: Optional[str] = None,
        verify_anomalies: Optional[bool] = None,
        n_verify_steps: Optional[int] = None,
    ) -> Dict[str, "AnomalyList"]:
        """Detect anomalies using parallel processing for multiple time series variables.

        Args:
            df: DataFrame containing time series data
            max_concurrent: Maximum number of concurrent processing tasks (default: 3)
            progress_callback: Optional callback for progress updates
            timestamp_col: Name of the timestamp column (optional)
            verify_anomalies: Whether to verify detected anomalies (optional)
            n_verify_steps: Number of verification steps to run (optional)

        Returns:
            Dictionary mapping column names to their respective AnomalyList

        Example:
            async def main():
                agent = AnomalyAgent()
                
                def progress_handler(column, event, data):
                    print(f"[{column}] {event}: {data}")
                
                anomalies = await agent.detect_anomalies_parallel(
                    df, 
                    max_concurrent=5,
                    progress_callback=progress_handler
                )
                return anomalies
                
            # Run with: asyncio.run(main())
        """
        # Import at runtime to avoid circular imports
        from .agent import AgentState, AnomalyList
        
        # Handle dynamic configuration
        current_timestamp_col = timestamp_col or self.config.timestamp_col
        current_verify = verify_anomalies if verify_anomalies is not None else self.config.verify_anomalies
        current_n_verify = n_verify_steps if n_verify_steps is not None else self.config.n_verify_steps
        
        # Validate inputs
        if current_timestamp_col not in df.columns:
            raise KeyError(f"Timestamp column '{current_timestamp_col}' not found in DataFrame")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {col: AnomalyList(anomalies=[]) for col in df.columns if col != current_timestamp_col}

        df_str = df.to_string(index=False)

        async def process_column(col: str) -> Tuple[str, "AnomalyList"]:
            """Process a single column asynchronously."""
            # Get appropriate compiled graph (each task needs its own)
            if current_verify != self.config.verify_anomalies or current_n_verify != self.config.n_verify_steps:
                temp_config = self.config.model_copy(update={
                    "verify_anomalies": current_verify,
                    "n_verify_steps": current_n_verify
                })
                app = self._graph_manager.get_or_create_graph(temp_config, self.llm)
            else:
                app = self.app

            if progress_callback:
                progress_callback(col, "start", {"total_columns": len(numeric_cols)})

            state = AgentState(
                time_series=df_str,
                variable_name=col,
                current_step="detect",
                processing_start_time=datetime.now(),
                processing_metadata={
                    "column": col,
                    "total_rows": len(df),
                    "verification_enabled": current_verify,
                    "n_verify_steps": current_n_verify,
                    "detection_prompt": self.config.detection_prompt,
                    "verification_prompt": self.config.verification_prompt,
                    "max_retries": self.config.max_retries,
                    "timestamp_col": current_timestamp_col,
                    "debug": self.debug
                }
            )

            try:
                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, app.invoke, state)
                
                # Extract anomalies
                if current_verify:
                    anomalies = result.get("verified_anomalies") or AnomalyList(anomalies=[])
                else:
                    anomalies = result.get("detected_anomalies") or AnomalyList(anomalies=[])

                if progress_callback:
                    progress_callback(col, "column_complete", {
                        "anomaly_count": len(anomalies.anomalies),
                        "processing_time": (datetime.now() - state.processing_start_time).total_seconds()
                    })

                return col, anomalies

            except Exception as e:
                if progress_callback:
                    progress_callback(col, "error", {"error": str(e)})
                return col, AnomalyList(anomalies=[])

        # Create semaphore to limit concurrent tasks
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(col: str):
            async with semaphore:
                return await process_column(col)

        # Process columns concurrently
        tasks = [process_with_semaphore(col) for col in numeric_cols]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert results to dictionary
        results = {}
        for result in results_list:
            if isinstance(result, Exception):
                self.logger.error(f"Error processing column: {result}")
                continue
            col, anomalies = result
            results[col] = anomalies

        return results

    async def detect_anomalies_streaming_async(
        self,
        df: pd.DataFrame,
        timestamp_col: Optional[str] = None,
        verify_anomalies: Optional[bool] = None,
        n_verify_steps: Optional[int] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Async generator that streams anomaly detection progress and results.

        Args:
            df: DataFrame containing time series data
            timestamp_col: Name of the timestamp column (optional)
            verify_anomalies: Whether to verify detected anomalies (optional)
            n_verify_steps: Number of verification steps to run (optional)

        Yields:
            Dictionary with event information:
            {
                "event": "start" | "progress" | "result" | "complete" | "error",
                "column": str,
                "data": Any
            }

        Example:
            async for event in agent.detect_anomalies_streaming_async(df):
                if event["event"] == "progress":
                    print(f"Processing {event['column']}: {event['data']['node']}")
                elif event["event"] == "result":
                    print(f"Found {len(event['data']['anomalies'])} anomalies in {event['column']}")
        """
        # Import at runtime to avoid circular imports
        from .agent import AgentState, AnomalyList
        
        # Handle dynamic configuration
        current_timestamp_col = timestamp_col or self.config.timestamp_col
        current_verify = verify_anomalies if verify_anomalies is not None else self.config.verify_anomalies
        current_n_verify = n_verify_steps if n_verify_steps is not None else self.config.n_verify_steps

        # Validate inputs
        if current_timestamp_col not in df.columns:
            yield {"event": "error", "column": None, "data": {"error": f"Timestamp column '{current_timestamp_col}' not found"}}
            return

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            yield {"event": "complete", "column": None, "data": {"results": {}}}
            return

        yield {"event": "start", "column": None, "data": {"total_columns": len(numeric_cols), "columns": list(numeric_cols)}}

        df_str = df.to_string(index=False)
        results = {}

        for col in numeric_cols:
            yield {"event": "progress", "column": col, "data": {"status": "starting"}}

            try:
                # Get appropriate graph for this column
                if current_verify != self.config.verify_anomalies or current_n_verify != self.config.n_verify_steps:
                    temp_config = self.config.model_copy(update={
                        "verify_anomalies": current_verify,
                        "n_verify_steps": current_n_verify
                    })
                    app = self._graph_manager.get_or_create_graph(temp_config, self.llm)
                else:
                    app = self.app

                state = AgentState(
                    time_series=df_str,
                    variable_name=col,
                    current_step="detect",
                    processing_start_time=datetime.now(),
                    processing_metadata={
                        "column": col,
                        "total_rows": len(df),
                        "verification_enabled": current_verify,
                        "n_verify_steps": current_n_verify,
                        "detection_prompt": self.config.detection_prompt,
                        "verification_prompt": self.config.verification_prompt,
                        "max_retries": self.config.max_retries,
                        "timestamp_col": current_timestamp_col,
                        "debug": self.debug
                    }
                )

                # Stream execution in executor to avoid blocking
                loop = asyncio.get_event_loop()
                
                # Stream each step
                async for chunk in self._async_stream_wrapper(app, state):
                    yield {"event": "progress", "column": col, "data": {"step": chunk}}

                # Get final result
                result = await loop.run_in_executor(None, app.invoke, state)
                
                # Extract anomalies
                if current_verify:
                    anomalies = result.get("verified_anomalies") or AnomalyList(anomalies=[])
                else:
                    anomalies = result.get("detected_anomalies") or AnomalyList(anomalies=[])
                
                results[col] = anomalies

                yield {"event": "result", "column": col, "data": {
                    "anomalies": anomalies.anomalies,
                    "anomaly_count": len(anomalies.anomalies),
                    "processing_time": (datetime.now() - state.processing_start_time).total_seconds()
                }}

            except Exception as e:
                yield {"event": "error", "column": col, "data": {"error": str(e)}}
                results[col] = AnomalyList(anomalies=[])

        yield {"event": "complete", "column": None, "data": {"results": results}}

    async def _async_stream_wrapper(self, app, state):
        """Wrapper to make synchronous streaming async."""
        loop = asyncio.get_event_loop()
        
        def run_stream():
            return list(app.stream(state))
        
        chunks = await loop.run_in_executor(None, run_stream)
        for chunk in chunks:
            yield chunk


class StreamingProgressHandler:
    """Helper class for standardized progress handling."""
    
    def __init__(self, verbose: bool = True, use_emojis: bool = True):
        """Initialize progress handler.
        
        Args:
            verbose: Whether to print detailed progress messages
            use_emojis: Whether to use emoji indicators in output
        """
        self.verbose = verbose
        self.use_emojis = use_emojis
        self.start_time = None
        self.column_counts = {}
    
    def __call__(self, column: str, event: str, data: Dict[str, Any]) -> None:
        """Handle progress callback events."""
        if not self.verbose:
            return
            
        icons = {
            "start": "🔍" if self.use_emojis else "[START]",
            "node_complete": "✓" if self.use_emojis else "[NODE]",
            "column_complete": "🎯" if self.use_emojis else "[DONE]",
            "error": "❌" if self.use_emojis else "[ERROR]"
        }
        
        if event == "start":
            if self.start_time is None:
                self.start_time = datetime.now()
            total = data.get('total_columns', '?')
            print(f"{icons['start']} Starting analysis for '{column}' (column {len(self.column_counts)+1}/{total})")
            
        elif event == "node_complete":
            node = data.get('node', 'unknown')
            if node == "detect":
                print(f"   {icons['node_complete']} Detection completed for '{column}'")
            elif node.startswith("verify"):
                step = node.split('_')[1] if '_' in node else "1"
                print(f"   {icons['node_complete']} Verification step {step} completed for '{column}'")
                
        elif event == "column_complete":
            count = data.get('anomaly_count', 0)
            time_taken = data.get('processing_time', 0)
            self.column_counts[column] = count
            print(f"{icons['column_complete']} Found {count} anomalies in '{column}' (took {time_taken:.2f}s)")
            
        elif event == "error":
            error_msg = data.get('error', 'Unknown error')
            print(f"{icons['error']} Error processing '{column}': {error_msg}")
    
    def summary(self) -> None:
        """Print summary of processing results."""
        if not self.verbose or not self.column_counts:
            return
            
        total_anomalies = sum(self.column_counts.values())
        total_time = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        icon = "📊" if self.use_emojis else "[SUMMARY]"
        print(f"\n{icon} Processing Summary:")
        print(f"   Total anomalies found: {total_anomalies}")
        print(f"   Columns processed: {len(self.column_counts)}")
        print(f"   Total time: {total_time:.2f}s")
        
        if self.column_counts:
            print("   Per-column results:")
            for column, count in self.column_counts.items():
                print(f"     - {column}: {count} anomalies")