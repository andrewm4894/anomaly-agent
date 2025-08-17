"""
Core anomaly detection agent implementation.

This module contains the main AnomalyAgent class and Pydantic models for
configuration and state management.
"""

import asyncio
import logging
import threading
from datetime import datetime
from typing import Dict, List, Literal, Optional, Any, Annotated, Callable
from operator import add

import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator, ConfigDict

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional - continue without it if not installed
    pass

from .constants import DEFAULT_MODEL_NAME, DEFAULT_TIMESTAMP_COL, TIMESTAMP_FORMAT
from .prompt import DEFAULT_SYSTEM_PROMPT, DEFAULT_VERIFY_SYSTEM_PROMPT
from .graph import GraphManager
from .streaming import StreamingMixin


class Anomaly(BaseModel):
    """Represents a single anomaly in the time series data."""

    timestamp: str = Field(description="The timestamp of the anomaly")
    variable_value: float = Field(
        description="The value of the variable at the anomaly timestamp"
    )
    anomaly_description: str = Field(description="A description of the anomaly")

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """Validate that the timestamp is in a valid format."""
        try:
            # Try parsing with our custom format first
            datetime.strptime(v, TIMESTAMP_FORMAT)
            return v
        except ValueError:
            try:
                # Try parsing as ISO format
                datetime.fromisoformat(v.replace("Z", "+00:00"))
                return v
            except ValueError:
                try:
                    # Try parsing as YYYY-MM-DD format
                    datetime.strptime(v, "%Y-%m-%d")
                    return v
                except ValueError:
                    try:
                        # Try parsing as YYYY-MM-DD HH:MM:SS format
                        datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
                        return v
                    except ValueError:
                        raise ValueError(
                            f"Invalid timestamp format: {v}. "
                            f"timestamp must be in {TIMESTAMP_FORMAT} format, "
                            "ISO format, or YYYY-MM-DD format"
                        )

    @field_validator("variable_value")
    @classmethod
    def validate_variable_value(cls, v: float) -> float:
        """Validate that the variable value is a number."""
        if not isinstance(v, (int, float)):
            raise ValueError("variable_value must be a number")
        return float(v)

    @field_validator("anomaly_description")
    @classmethod
    def validate_anomaly_description(cls, v: str) -> str:
        """Validate that the anomaly description is a string."""
        if not isinstance(v, str):
            raise ValueError("anomaly_description must be a string")
        return v


class AnomalyList(BaseModel):
    """Represents a list of anomalies."""

    anomalies: List[Anomaly] = Field(description="The list of anomalies")

    @field_validator("anomalies")
    @classmethod
    def validate_anomalies(cls, v: List[Anomaly]) -> List[Anomaly]:
        """Validate that anomalies is a list."""
        if not isinstance(v, list):
            raise ValueError("anomalies must be a list")
        return v


class AgentConfig(BaseModel):
    """Configuration for the anomaly detection agent."""
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=True
    )
    
    model_name: str = Field(default=DEFAULT_MODEL_NAME, description="OpenAI model name")
    timestamp_col: str = Field(default=DEFAULT_TIMESTAMP_COL, description="Timestamp column name")
    verify_anomalies: bool = Field(default=True, description="Whether to verify detected anomalies")
    n_verify_steps: int = Field(default=1, ge=1, le=5, description="Number of verification steps to run")
    detection_prompt: str = Field(default="", description="Custom detection prompt")
    verification_prompt: str = Field(default="", description="Custom verification prompt")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    timeout_seconds: int = Field(default=300, ge=30, le=3600, description="Operation timeout")


class AgentState(BaseModel):
    """Enhanced state for the anomaly detection agent with proper validation."""
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True
    )
    
    # Core data
    time_series: str = Field(description="Time series data as string")
    variable_name: str = Field(description="Name of the variable being analyzed")
    
    # Results with accumulation support
    detected_anomalies: Optional[AnomalyList] = Field(default=None, description="Initially detected anomalies")
    verified_anomalies: Optional[AnomalyList] = Field(default=None, description="Verified anomalies after review")
    
    # Execution tracking
    current_step: str = Field(default="detect", description="Current processing step")
    error_messages: Annotated[List[str], add] = Field(default_factory=list, description="Accumulated error messages")
    retry_count: int = Field(default=0, ge=0, description="Current retry attempt")
    
    # Metadata
    processing_start_time: Optional[datetime] = Field(default=None, description="When processing started")
    processing_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional processing metadata")
    
    @field_validator("variable_name")
    @classmethod
    def validate_variable_name(cls, v: str) -> str:
        """Validate variable name is not empty."""
        if not v or not v.strip():
            raise ValueError("variable_name cannot be empty")
        return v.strip()
    
    @field_validator("time_series")
    @classmethod
    def validate_time_series(cls, v: str) -> str:
        """Validate time series data is not empty."""
        if not v or not v.strip():
            raise ValueError("time_series data cannot be empty")
        return v
    
    @field_validator("current_step")
    @classmethod
    def validate_current_step(cls, v: str) -> str:
        """Validate current step is valid."""
        valid_steps = {"detect", "verify", "end", "error"}
        if v not in valid_steps:
            raise ValueError(f"current_step must be one of {valid_steps}")
        return v


class AnomalyAgent(StreamingMixin):
    """Enhanced agent for detecting and verifying anomalies in time series data."""
    
    # Shared graph manager for reusability across instances
    _graph_manager = GraphManager()

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        timestamp_col: str = DEFAULT_TIMESTAMP_COL,
        verify_anomalies: bool = True,
        n_verify_steps: int = 1,
        detection_prompt: str = DEFAULT_SYSTEM_PROMPT,
        verification_prompt: str = DEFAULT_VERIFY_SYSTEM_PROMPT,
        max_retries: int = 3,
        timeout_seconds: int = 300,
        debug: bool = False,
    ):
        """Initialize the AnomalyAgent with enhanced configuration.

        Args:
            model_name: The name of the OpenAI model to use
            timestamp_col: The name of the timestamp column
            verify_anomalies: Whether to verify detected anomalies (default: True)
            n_verify_steps: Number of verification steps to run (default: 1)
            detection_prompt: System prompt for anomaly detection.
                Defaults to the standard detection prompt.
            verification_prompt: System prompt for anomaly verification.
                Defaults to the standard verification prompt.
            max_retries: Maximum retry attempts for failed operations
            timeout_seconds: Operation timeout in seconds
            debug: Enable debug mode with verbose logging
        """
        # Create configuration with validation
        self.config = AgentConfig(
            model_name=model_name,
            timestamp_col=timestamp_col,
            verify_anomalies=verify_anomalies,
            n_verify_steps=n_verify_steps,
            detection_prompt=detection_prompt or DEFAULT_SYSTEM_PROMPT,
            verification_prompt=verification_prompt or DEFAULT_VERIFY_SYSTEM_PROMPT,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
        )
        
        # Initialize LLM
        self.llm = ChatOpenAI(model=self.config.model_name)
        
        # Store debug mode and setup logging
        self.debug = debug
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if debug:
            self.logger.setLevel(logging.DEBUG)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
                self.logger.addHandler(handler)
        
        # Expose commonly used config as properties for backward compatibility
        self.timestamp_col = self.config.timestamp_col
        self.verify_anomalies = self.config.verify_anomalies
        self.n_verify_steps = self.config.n_verify_steps
        self.detection_prompt = self.config.detection_prompt
        self.verification_prompt = self.config.verification_prompt

        # Get reusable compiled graph
        self.app = self._graph_manager.get_or_create_graph(self.config, self.llm)

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        timestamp_col: Optional[str] = None,
        verify_anomalies: Optional[bool] = None,
        n_verify_steps: Optional[int] = None,
        parallel: bool = False,
        max_concurrent: int = 3,
        progress_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, AnomalyList]:
        """Detect anomalies in the given time series data.

        This method supports multiple execution modes:
        1. Sequential (default): Process columns one by one
        2. Streaming: Process with real-time progress updates (when progress_callback provided)
        3. Parallel: Process multiple columns concurrently (when parallel=True)

        Args:
            df: DataFrame containing the time series data
            timestamp_col: Name of the timestamp column (optional)
            verify_anomalies: Whether to verify detected anomalies. If None, uses the
                instance default (default: None)
            n_verify_steps: Number of verification steps to run. If None, uses the
                instance default (default: None)
            parallel: Whether to use parallel processing for multiple columns (default: False)
            max_concurrent: Maximum number of concurrent tasks when parallel=True (default: 3)
            progress_callback: Optional callback for progress updates. If provided without
                parallel=True, uses streaming mode for real-time updates (default: None)

        Returns:
            Dictionary mapping column names to their respective AnomalyList
            
        Examples:
            # Sequential processing (default)
            anomalies = agent.detect_anomalies(df)
            
            # Parallel processing
            anomalies = agent.detect_anomalies(df, parallel=True, max_concurrent=5)
            
            # Streaming with progress updates
            def progress(col, event, data):
                print(f"[{col}] {event}: {data}")
            anomalies = agent.detect_anomalies(df, progress_callback=progress)
            
            # Parallel with progress updates
            anomalies = agent.detect_anomalies(df, parallel=True, progress_callback=progress)
        """
        # Use parallel processing if requested
        if parallel:
            return self._run_parallel_safely(
                df=df,
                timestamp_col=timestamp_col,
                verify_anomalies=verify_anomalies,
                n_verify_steps=n_verify_steps,
                max_concurrent=max_concurrent,
                progress_callback=progress_callback,
            )
        
        # Use streaming processing if progress_callback is provided
        if progress_callback:
            return self.detect_anomalies_streaming(
                df=df,
                timestamp_col=timestamp_col,
                verify_anomalies=verify_anomalies,
                n_verify_steps=n_verify_steps,
                progress_callback=progress_callback,
            )

        # Handle dynamic configuration efficiently with graph manager
        current_timestamp_col = timestamp_col or self.config.timestamp_col
        current_verify = verify_anomalies if verify_anomalies is not None else self.config.verify_anomalies
        current_n_verify = n_verify_steps if n_verify_steps is not None else self.config.n_verify_steps
        
        # Use graph manager to get appropriate compiled graph (cached and reusable)
        if current_verify != self.config.verify_anomalies or current_n_verify != self.config.n_verify_steps:
            # Create temporary config for different verification settings
            temp_config = self.config.model_copy(update={
                "verify_anomalies": current_verify,
                "n_verify_steps": current_n_verify
            })
            app = self._graph_manager.get_or_create_graph(temp_config, self.llm)
        else:
            app = self.app

        # Check if timestamp column exists
        if current_timestamp_col not in df.columns:
            raise KeyError(
                f"Timestamp column '{current_timestamp_col}' not found in DataFrame"
            )

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # If no numeric columns found, return empty results for all columns
        if len(numeric_cols) == 0:
            return {
                col: AnomalyList(anomalies=[])
                for col in df.columns
                if col != current_timestamp_col
            }

        # Convert DataFrame to string format
        df_str = df.to_string(index=False)

        # Process each numeric column sequentially (standard mode)
        results = {}
        for col in numeric_cols:
            self.logger.debug(f"Processing column: {col}")
                
            # Create enhanced state for this column using Pydantic model
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

            self.logger.debug(f"Created state for {col}, starting graph execution...")
                
            # Run the graph
            result = app.invoke(state)
            
            self.logger.debug(f"Graph execution completed for {col}")
            if hasattr(result, 'processing_metadata'):
                metadata = result.processing_metadata
                self.logger.debug(f"Processing metadata: {metadata}")
            
            # Extract results based on verification setting
            if current_verify:
                results[col] = result.get("verified_anomalies") or AnomalyList(anomalies=[])
                self.logger.debug(f"Found {len(results[col].anomalies)} verified anomalies for {col}")
            else:
                results[col] = result.get("detected_anomalies") or AnomalyList(anomalies=[])
                self.logger.debug(f"Found {len(results[col].anomalies)} detected anomalies for {col}")

        return results

    def get_processing_metadata(self, result_state: Any) -> Dict[str, Any]:
        """Extract processing metadata from the final state.
        
        Args:
            result_state: Final state from graph execution
            
        Returns:
            Dictionary containing processing metadata
        """
        if hasattr(result_state, 'processing_metadata'):
            return result_state.processing_metadata
        elif isinstance(result_state, dict):
            return result_state.get('processing_metadata', {})
        else:
            return {}

    def get_anomalies_df(
        self, anomalies: Dict[str, AnomalyList], format: str = "long"
    ) -> pd.DataFrame:
        """Convert anomalies to a DataFrame.

        Args:
            anomalies: Dictionary mapping column names to their respective
                AnomalyList
            format: Output format, either "long" or "wide"

        Returns:
            DataFrame containing the anomalies
        """
        if format not in ["long", "wide"]:
            raise ValueError("format must be either 'long' or 'wide'")

        all_anomalies = []
        for variable_name, anomaly_list in anomalies.items():
            for anomaly in anomaly_list.anomalies:
                all_anomalies.append(
                    {
                        "timestamp": anomaly.timestamp,
                        "variable_name": variable_name,
                        "value": anomaly.variable_value,
                        "anomaly_description": anomaly.anomaly_description,
                    }
                )

        df = pd.DataFrame(all_anomalies)
        
        if len(df) == 0:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=["timestamp", "variable_name", "value", "anomaly_description"])
        
        # Convert timestamp to datetime for proper merging
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        if format == "wide":
            # Pivot to wide format - create separate DataFrames for values and descriptions
            df_values = df.pivot_table(
                index="timestamp", 
                columns="variable_name", 
                values="value", 
                aggfunc='first'
            )
            df_descriptions = df.pivot_table(
                index="timestamp", 
                columns="variable_name", 
                values="anomaly_description", 
                aggfunc='first'
            )
            
            # Combine the two pivot tables
            df_wide = df_values.copy()
            for col in df_descriptions.columns:
                df_wide[f"{col}_description"] = df_descriptions[col]
            
            df_wide = df_wide.reset_index()
            return df_wide

        return df

    def _run_parallel_safely(
        self,
        df: pd.DataFrame,
        timestamp_col: Optional[str] = None,
        verify_anomalies: Optional[bool] = None,
        n_verify_steps: Optional[int] = None,
        max_concurrent: int = 3,
        progress_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, AnomalyList]:
        """Safely run parallel processing, handling existing event loops.
        
        This method detects if we're already in an event loop (like Jupyter notebooks)
        and handles the async execution appropriately.
        
        Args:
            df: DataFrame containing time series data
            timestamp_col: Name of the timestamp column (optional)
            verify_anomalies: Whether to verify detected anomalies (optional)
            n_verify_steps: Number of verification steps to run (optional)
            max_concurrent: Maximum number of concurrent tasks (default: 3)
            progress_callback: Optional callback for progress updates (default: None)
            
        Returns:
            Dictionary mapping column names to their respective AnomalyList
        """
        try:
            # Check if we're already in an event loop
            asyncio.get_running_loop()
            # If we get here, we're in an event loop (like Jupyter)
            # We need to use a different approach
            
            # Create a result container
            result = None
            exception = None
            
            def run_in_thread():
                nonlocal result, exception
                try:
                    # Create a new event loop for this thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(self.detect_anomalies_parallel(
                            df=df,
                            timestamp_col=timestamp_col,
                            verify_anomalies=verify_anomalies,
                            n_verify_steps=n_verify_steps,
                            max_concurrent=max_concurrent,
                            progress_callback=progress_callback,
                        ))
                    finally:
                        loop.close()
                except Exception as e:
                    exception = e
            
            # Run the async code in a separate thread
            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()
            
            if exception:
                raise exception
            return result
            
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            return asyncio.run(self.detect_anomalies_parallel(
                df=df,
                timestamp_col=timestamp_col,
                verify_anomalies=verify_anomalies,
                n_verify_steps=n_verify_steps,
                max_concurrent=max_concurrent,
                progress_callback=progress_callback,
            ))