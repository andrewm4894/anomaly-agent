"""
Anomaly detection agent using LLMs to identify and verify anomalies in time series data.

This module provides functionality for detecting and verifying anomalies in time series
data using language models.
"""

from datetime import datetime
from typing import Dict, List, Literal, Optional, Any, Annotated
from operator import add

import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, field_validator, ConfigDict

from .constants import DEFAULT_MODEL_NAME, DEFAULT_TIMESTAMP_COL, TIMESTAMP_FORMAT
from .prompt import get_detection_prompt, get_verification_prompt, DEFAULT_SYSTEM_PROMPT, DEFAULT_VERIFY_SYSTEM_PROMPT


class Anomaly(BaseModel):
    """Represents a single anomaly in a time series."""

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
                dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
                # If input had microseconds, preserve them
                if "." in v:
                    return dt.strftime(TIMESTAMP_FORMAT)
                # Otherwise use second precision
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    # Try parsing as date only (add time component)
                    dt = datetime.strptime(v, "%Y-%m-%d")
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    try:
                        # Try parsing without microseconds
                        dt = datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
                        return v  # Return original format
                    except ValueError:
                        raise ValueError(
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


def create_detection_node(llm: ChatOpenAI, detection_prompt: str = DEFAULT_SYSTEM_PROMPT, verify_anomalies: bool = True):
    """Create the detection node for the graph."""
    chain = get_detection_prompt(detection_prompt) | llm.with_structured_output(AnomalyList)

    def detection_node(state: AgentState) -> Dict[str, Any]:
        """Process the state and detect anomalies."""
        try:
            result = chain.invoke(
                {
                    "time_series": state.time_series,
                    "variable_name": state.variable_name,
                }
            )
            next_step = "verify" if verify_anomalies else "end"
            return {
                "detected_anomalies": result, 
                "current_step": next_step,
                "processing_metadata": {
                    **state.processing_metadata,
                    "detection_completed": datetime.now().isoformat()
                }
            }
        except Exception as e:
            return {
                "current_step": "error",
                "error_messages": [f"Detection failed: {str(e)}"],
                "retry_count": state.retry_count + 1
            }

    return detection_node


def create_verification_node(llm: ChatOpenAI, verification_prompt: str = DEFAULT_VERIFY_SYSTEM_PROMPT):
    """Create the verification node for the graph."""
    chain = get_verification_prompt(verification_prompt) | llm.with_structured_output(AnomalyList)

    def verification_node(state: AgentState) -> Dict[str, Any]:
        """Process the state and verify anomalies."""
        try:
            if state.detected_anomalies is None:
                return {
                    "verified_anomalies": None, 
                    "current_step": "end",
                    "processing_metadata": {
                        **state.processing_metadata,
                        "verification_skipped": "no_anomalies_detected"
                    }
                }

            detected_str = "\n".join(
                [
                    (
                        f"timestamp: {a.timestamp}, "
                        f"value: {a.variable_value}, "
                        f"Description: {a.anomaly_description}"
                    )
                    for a in state.detected_anomalies.anomalies
                ]
            )

            result = chain.invoke(
                {
                    "time_series": state.time_series,
                    "variable_name": state.variable_name,
                    "detected_anomalies": detected_str,
                }
            )
            return {
                "verified_anomalies": result, 
                "current_step": "end",
                "processing_metadata": {
                    **state.processing_metadata,
                    "verification_completed": datetime.now().isoformat(),
                    "anomalies_verified": len(result.anomalies) if result else 0
                }
            }
        except Exception as e:
            return {
                "current_step": "error",
                "error_messages": [f"Verification failed: {str(e)}"],
                "retry_count": state.retry_count + 1
            }

    return verification_node


def should_verify(state: AgentState) -> Literal["verify", "end", "error"]:
    """Determine if we should proceed to verification."""
    if state.current_step == "error":
        return "error"
    return "verify" if state.current_step == "verify" else "end"


def create_error_handler_node():
    """Create an error handling node for failed operations."""
    
    def error_handler_node(state: AgentState) -> Dict[str, Any]:
        """Handle errors and determine retry logic."""
        max_retries = 3  # Could be configurable
        
        if state.retry_count < max_retries:
            return {
                "current_step": "detect",
                "retry_count": state.retry_count,
                "processing_metadata": {
                    **state.processing_metadata,
                    f"retry_attempt_{state.retry_count}": datetime.now().isoformat()
                }
            }
        else:
            return {
                "current_step": "end",
                "processing_metadata": {
                    **state.processing_metadata,
                    "max_retries_exceeded": True,
                    "final_error": state.error_messages[-1] if state.error_messages else "Unknown error"
                }
            }
    
    return error_handler_node


class AnomalyAgent:
    """Enhanced agent for detecting and verifying anomalies in time series data."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        timestamp_col: str = DEFAULT_TIMESTAMP_COL,
        verify_anomalies: bool = True,
        detection_prompt: str = DEFAULT_SYSTEM_PROMPT,
        verification_prompt: str = DEFAULT_VERIFY_SYSTEM_PROMPT,
        max_retries: int = 3,
        timeout_seconds: int = 300,
    ):
        """Initialize the AnomalyAgent with enhanced configuration.

        Args:
            model_name: The name of the OpenAI model to use
            timestamp_col: The name of the timestamp column
            verify_anomalies: Whether to verify detected anomalies (default: True)
            detection_prompt: System prompt for anomaly detection.
                Defaults to the standard detection prompt.
            verification_prompt: System prompt for anomaly verification.
                Defaults to the standard verification prompt.
            max_retries: Maximum retry attempts for failed operations
            timeout_seconds: Operation timeout in seconds
        """
        # Create configuration with validation
        self.config = AgentConfig(
            model_name=model_name,
            timestamp_col=timestamp_col,
            verify_anomalies=verify_anomalies,
            detection_prompt=detection_prompt or DEFAULT_SYSTEM_PROMPT,
            verification_prompt=verification_prompt or DEFAULT_VERIFY_SYSTEM_PROMPT,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
        )
        
        # Initialize LLM
        self.llm = ChatOpenAI(model=self.config.model_name)
        
        # Expose commonly used config as properties for backward compatibility
        self.timestamp_col = self.config.timestamp_col
        self.verify_anomalies = self.config.verify_anomalies
        self.detection_prompt = self.config.detection_prompt
        self.verification_prompt = self.config.verification_prompt

        # Create the graph
        self._create_graph()

    def _create_graph(self) -> None:
        """Create and compile the processing graph."""
        # Create graph with proper state schema
        self.graph = StateGraph(AgentState)

        # Add nodes
        self.graph.add_node("detect", create_detection_node(self.llm, self.config.detection_prompt, self.config.verify_anomalies))
        self.graph.add_node("error", create_error_handler_node())
        
        if self.config.verify_anomalies:
            self.graph.add_node("verify", create_verification_node(self.llm, self.config.verification_prompt))

        # Add edges with proper routing based on verification setting
        if self.config.verify_anomalies:
            self.graph.add_conditional_edges(
                "detect", 
                should_verify, 
                {"verify": "verify", "end": END, "error": "error"}
            )
            self.graph.add_edge("verify", END)
            self.graph.add_conditional_edges(
                "error",
                lambda state: "detect" if state.retry_count < self.config.max_retries else "end",
                {"detect": "detect", "end": END}
            )
        else:
            # Without verification, go directly to end or error
            self.graph.add_conditional_edges(
                "detect", 
                lambda state: "error" if state.current_step == "error" else "end",
                {"end": END, "error": "error"}
            )
            self.graph.add_conditional_edges(
                "error",
                lambda state: "detect" if state.retry_count < self.config.max_retries else "end",
                {"detect": "detect", "end": END}
            )

        # Set entry point
        self.graph.set_entry_point("detect")

        # Compile the graph
        self.app = self.graph.compile()

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        timestamp_col: Optional[str] = None,
        verify_anomalies: Optional[bool] = None,
    ) -> Dict[str, AnomalyList]:
        """Detect anomalies in the given time series data.

        Args:
            df: DataFrame containing the time series data
            timestamp_col: Name of the timestamp column (optional)
            verify_anomalies: Whether to verify detected anomalies. If None, uses the
                instance default (default: None)

        Returns:
            Dictionary mapping column names to their respective AnomalyList
        """
        # Update configuration if needed
        current_timestamp_col = timestamp_col or self.config.timestamp_col
        current_verify = verify_anomalies if verify_anomalies is not None else self.config.verify_anomalies
        
        # Recreate graph if verification setting changed (avoid this in future with reusable graph)
        if current_verify != self.config.verify_anomalies:
            temp_config = self.config.model_copy(update={"verify_anomalies": current_verify})
            temp_agent = AnomalyAgent(
                model_name=temp_config.model_name,
                timestamp_col=current_timestamp_col,
                verify_anomalies=current_verify,
                detection_prompt=temp_config.detection_prompt,
                verification_prompt=temp_config.verification_prompt,
                max_retries=temp_config.max_retries,
                timeout_seconds=temp_config.timeout_seconds
            )
            app = temp_agent.app
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

        # Process each numeric column
        results = {}
        for col in numeric_cols:
            # Create enhanced state for this column using Pydantic model
            state = AgentState(
                time_series=df_str,
                variable_name=col,
                current_step="detect",
                processing_start_time=datetime.now(),
                processing_metadata={
                    "column": col,
                    "total_rows": len(df),
                    "verification_enabled": current_verify
                }
            )

            # Run the graph
            result = app.invoke(state)
            
            # Extract results based on verification setting
            if current_verify:
                results[col] = result.get("verified_anomalies") or AnomalyList(anomalies=[])
            else:
                results[col] = result.get("detected_anomalies") or AnomalyList(anomalies=[])

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

        if format == "long":
            # Create long format DataFrame
            rows = []
            for col, anomaly_list in anomalies.items():
                for anomaly in anomaly_list.anomalies:
                    rows.append(
                        {
                            "timestamp": pd.to_datetime(anomaly.timestamp),
                            "variable_name": col,
                            "value": anomaly.variable_value,
                            "anomaly_description": anomaly.anomaly_description,
                        }
                    )
            return pd.DataFrame(rows)

        # Create wide format DataFrame
        rows = []
        for col, anomaly_list in anomalies.items():
            for anomaly in anomaly_list.anomalies:
                rows.append(
                    {
                        "timestamp": pd.to_datetime(anomaly.timestamp),
                        col: anomaly.variable_value,
                        f"{col}_description": anomaly.anomaly_description,
                    }
                )
        return pd.DataFrame(rows)
