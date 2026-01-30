"""
Anomaly detection agent using LLMs to identify and verify anomalies in time series data.

This module provides functionality for detecting and verifying anomalies in time series
data using language models.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Literal, Optional, TypedDict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, validator

from .constants import DEFAULT_MODEL_NAME, DEFAULT_TIMESTAMP_COL, TIMESTAMP_FORMAT
from .prompt import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_SYSTEM_PROMPT_WITH_IMAGE,
    DEFAULT_VERIFY_SYSTEM_PROMPT,
    build_multimodal_detection_messages,
    get_detection_prompt,
    get_verification_prompt,
)

# Optional PostHog integration
try:
    from posthog import Posthog
    from posthog.ai.langchain import CallbackHandler as PostHogCallbackHandler

    POSTHOG_AVAILABLE = True
except ImportError:
    POSTHOG_AVAILABLE = False


class Anomaly(BaseModel):
    """Represents a single anomaly in a time series."""

    timestamp: str = Field(description="The timestamp of the anomaly")
    variable_value: float = Field(
        description="The value of the variable at the anomaly timestamp"
    )
    anomaly_description: str = Field(description="A description of the anomaly")

    @validator("timestamp")  # type: ignore
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

    @validator("variable_value")  # type: ignore
    def validate_variable_value(cls, v: float) -> float:
        """Validate that the variable value is a number."""
        if not isinstance(v, (int, float)):
            raise ValueError("variable_value must be a number")
        return float(v)

    @validator("anomaly_description")  # type: ignore
    def validate_anomaly_description(cls, v: str) -> str:
        """Validate that the anomaly description is a string."""
        if not isinstance(v, str):
            raise ValueError("anomaly_description must be a string")
        return v


class AnomalyList(BaseModel):
    """Represents a list of anomalies."""

    anomalies: List[Anomaly] = Field(description="The list of anomalies")

    @validator("anomalies")  # type: ignore
    def validate_anomalies(cls, v: List[Anomaly]) -> List[Anomaly]:
        """Validate that anomalies is a list."""
        if not isinstance(v, list):
            raise ValueError("anomalies must be a list")
        return v


class AgentState(TypedDict, total=False):
    """State for the anomaly detection agent."""

    time_series: str
    plot_image_base64: Optional[str]
    variable_name: str
    detected_anomalies: Optional[AnomalyList]
    verified_anomalies: Optional[AnomalyList]
    current_step: str


def create_detection_node(
    llm: ChatOpenAI,
    detection_prompt: str = DEFAULT_SYSTEM_PROMPT,
    include_plot: bool = False,
) -> ToolNode:
    """Create the detection node for the graph.

    Args:
        llm: The ChatOpenAI language model to use.
        detection_prompt: System prompt for anomaly detection.
        include_plot: Whether to use multimodal detection with plot images.

    Returns:
        Detection node function for the LangGraph.
    """
    # For text-only detection, use the standard chain
    chain = get_detection_prompt(detection_prompt) | llm.with_structured_output(
        AnomalyList
    )

    def detection_node(state: AgentState) -> AgentState:
        """Process the state and detect anomalies."""
        # Check if we should use multimodal detection
        if include_plot and state.get("plot_image_base64"):
            # Use multimodal detection with image
            messages = build_multimodal_detection_messages(
                variable_name=state["variable_name"],
                time_series=state["time_series"],
                plot_image_base64=state["plot_image_base64"],
                system_prompt=DEFAULT_SYSTEM_PROMPT_WITH_IMAGE,
            )
            result = llm.with_structured_output(AnomalyList).invoke(messages)
        else:
            # Use standard text-only detection
            result = chain.invoke(
                {
                    "time_series": state["time_series"],
                    "variable_name": state["variable_name"],
                }
            )
        return {"detected_anomalies": result, "current_step": "verify"}

    return detection_node


def create_verification_node(
    llm: ChatOpenAI, verification_prompt: str = DEFAULT_VERIFY_SYSTEM_PROMPT
) -> ToolNode:
    """Create the verification node for the graph."""
    chain = get_verification_prompt(verification_prompt) | llm.with_structured_output(
        AnomalyList
    )

    def verification_node(state: AgentState) -> AgentState:
        """Process the state and verify anomalies."""
        if state["detected_anomalies"] is None:
            return {"verified_anomalies": None, "current_step": "end"}

        detected_str = "\n".join(
            [
                (
                    f"timestamp: {a.timestamp}, "
                    f"value: {a.variable_value}, "  # noqa: E501
                    f"Description: {a.anomaly_description}"  # noqa: E501
                )
                for a in state["detected_anomalies"].anomalies
            ]
        )

        result = chain.invoke(
            {
                "time_series": state["time_series"],
                "variable_name": state["variable_name"],
                "detected_anomalies": detected_str,  # noqa: E501
            }
        )
        return {"verified_anomalies": result, "current_step": "end"}

    return verification_node


def should_verify(state: AgentState) -> Literal["verify", "end"]:
    """Determine if we should proceed to verification."""
    return "verify" if state["current_step"] == "verify" else "end"


class AnomalyAgent:
    """Agent for detecting and verifying anomalies in time series data."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        timestamp_col: str = DEFAULT_TIMESTAMP_COL,
        verify_anomalies: bool = True,
        detection_prompt: str = DEFAULT_SYSTEM_PROMPT,
        verification_prompt: str = DEFAULT_VERIFY_SYSTEM_PROMPT,
        debug: bool = False,
        include_plot: bool = False,
    ):
        """Initialize the AnomalyAgent with a specific model.

        Args:
            model_name: The name of the OpenAI model to use
            timestamp_col: The name of the timestamp column
            verify_anomalies: Whether to verify detected anomalies (default: True)
            detection_prompt: System prompt for anomaly detection.
                Defaults to the standard detection prompt.
            verification_prompt: System prompt for anomaly verification.
                Defaults to the standard verification prompt.
            debug: Enable debug logging (default: False)
            include_plot: Whether to include a time series plot image in the
                detection prompt for multimodal analysis (default: False).
                Requires kaleido package for image generation.
        """
        # Load .env if present
        load_dotenv()

        self.debug = debug or os.getenv("ANOMALY_AGENT_DEBUG") == "1"

        # Configure logger
        self._logger = logging.getLogger("anomaly_agent")
        if self.debug:
            self._logger.setLevel(logging.DEBUG)
            if not self._logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S",
                    )
                )
                self._logger.addHandler(handler)

        # Initialize PostHog for LLM analytics (optional)
        self.posthog_client = None
        self.posthog_callback_handler = None
        posthog_enabled = os.getenv("POSTHOG_ENABLED", "false").lower() == "true"

        if posthog_enabled:
            if not POSTHOG_AVAILABLE:
                self._logger.warning(
                    "PostHog is enabled but the posthog package is not installed. "
                    "Install it with: pip install posthog"
                )
            else:
                posthog_api_key = os.getenv("POSTHOG_API_KEY")
                posthog_host = os.getenv("POSTHOG_HOST", "https://app.posthog.com")

                if not posthog_api_key:
                    self._logger.warning(
                        "POSTHOG_ENABLED is true but POSTHOG_API_KEY is not set. "
                        "PostHog tracking will be disabled."
                    )
                else:
                    try:
                        # Build super_properties for session and span tracking
                        super_properties = {"$ai_span_name": "anomaly_agent"}

                        # Add session ID from environment if provided
                        ai_session_id = os.getenv("POSTHOG_AI_SESSION_ID")
                        if ai_session_id:
                            super_properties["$ai_session_id"] = ai_session_id

                        # Initialize PostHog client with super_properties
                        self.posthog_client = Posthog(
                            posthog_api_key,
                            host=posthog_host,
                            super_properties=super_properties,
                        )

                        # Build callback handler config
                        callback_config = {"client": self.posthog_client}

                        # Add optional distinct_id
                        distinct_id = os.getenv("POSTHOG_DISTINCT_ID")
                        if distinct_id:
                            callback_config["distinct_id"] = distinct_id

                        # Add privacy mode setting
                        privacy_mode = (
                            os.getenv("POSTHOG_PRIVACY_MODE", "false").lower() == "true"
                        )
                        callback_config["privacy_mode"] = privacy_mode

                        self.posthog_callback_handler = PostHogCallbackHandler(
                            **callback_config
                        )

                        if self.debug:
                            session_info = (
                                f"session_id={ai_session_id}"
                                if ai_session_id
                                else "no session"
                            )
                            self._logger.debug(
                                f"PostHog LLM analytics initialized (host={posthog_host}, "
                                f"distinct_id={distinct_id or 'anonymous'}, "
                                f"privacy_mode={privacy_mode}, {session_info})"
                            )
                    except Exception as e:
                        self._logger.error(f"Failed to initialize PostHog: {e}")
                        self.posthog_client = None
                        self.posthog_callback_handler = None

        self.llm = ChatOpenAI(model=model_name)
        self.timestamp_col = timestamp_col
        self.verify_anomalies = verify_anomalies
        self.detection_prompt = detection_prompt
        self.verification_prompt = verification_prompt
        self.include_plot = include_plot

        # Create the graph
        self.graph = StateGraph(AgentState)

        # Add nodes
        self.graph.add_node(
            "detect",
            create_detection_node(self.llm, detection_prompt, include_plot=include_plot),
        )
        if self.verify_anomalies:
            self.graph.add_node(
                "verify", create_verification_node(self.llm, verification_prompt)
            )

        # Add edges with proper routing
        if self.verify_anomalies:
            self.graph.add_conditional_edges(
                "detect", should_verify, {"verify": "verify", "end": END}
            )
            self.graph.add_edge("verify", END)
        else:
            self.graph.add_edge("detect", END)

        # Set entry point
        self.graph.set_entry_point("detect")

        # Compile the graph
        self.app = self.graph.compile()

    def _generate_plot_base64(
        self, df: pd.DataFrame, timestamp_col: str, value_col: str
    ) -> str:
        """Generate a base64-encoded PNG plot of the time series.

        Args:
            df: DataFrame containing the time series data.
            timestamp_col: Name of the timestamp column.
            value_col: Name of the value column to plot.

        Returns:
            Base64-encoded PNG image string.
        """
        import base64

        import plotly.io as pio

        from .plot import plot_df

        # Create a subset DataFrame with just the timestamp and value columns
        plot_data = df[[timestamp_col, value_col]].copy()

        # Generate the plot figure
        fig = plot_df(
            plot_data,
            timestamp_col=timestamp_col,
            show_anomalies=False,
            return_fig=True,
        )

        # Convert to PNG bytes
        png_bytes = pio.to_image(fig, format="png", width=800, height=400)

        # Encode to base64
        return base64.b64encode(png_bytes).decode("utf-8")

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
        if timestamp_col is not None:
            self.timestamp_col = timestamp_col

        # Use instance default if verify_anomalies not specified
        verify_anomalies = (
            self.verify_anomalies if verify_anomalies is None else verify_anomalies
        )

        # Create a new graph for this detection run
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node(
            "detect",
            create_detection_node(
                self.llm, self.detection_prompt, include_plot=self.include_plot
            ),
        )
        if verify_anomalies:
            graph.add_node(
                "verify", create_verification_node(self.llm, self.verification_prompt)
            )

        # Add edges with proper routing
        if verify_anomalies:
            graph.add_conditional_edges(
                "detect", should_verify, {"verify": "verify", "end": END}
            )
            graph.add_edge("verify", END)
        else:
            graph.add_edge("detect", END)

        # Set entry point
        graph.set_entry_point("detect")

        # Compile the graph
        app = graph.compile()

        # Check if timestamp column exists
        if self.timestamp_col not in df.columns:
            raise KeyError(
                f"Timestamp column '{self.timestamp_col}' not found in DataFrame"
            )

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # If no numeric columns found, return empty results for all columns
        if len(numeric_cols) == 0:
            return {
                col: AnomalyList(anomalies=[])
                for col in df.columns
                if col != self.timestamp_col
            }

        # Convert DataFrame to string format
        df_str = df.to_string(index=False)

        # Process each numeric column
        results = {}
        for col in numeric_cols:
            # Generate plot image if enabled
            plot_base64 = None
            if self.include_plot:
                try:
                    plot_base64 = self._generate_plot_base64(
                        df, self.timestamp_col, col
                    )
                    if self.debug:
                        self._logger.debug(
                            f"Generated plot image for column '{col}' "
                            f"({len(plot_base64)} base64 chars)"
                        )
                except Exception as e:
                    self._logger.warning(
                        f"Failed to generate plot for column '{col}': {e}. "
                        "Falling back to text-only detection."
                    )
                    plot_base64 = None

            # Create state for this column
            state = {
                "time_series": df_str,
                "plot_image_base64": plot_base64,
                "variable_name": col,
                "current_step": "detect",
            }

            # Build config with optional PostHog callback
            invoke_config = {}
            if self.posthog_callback_handler:
                invoke_config["callbacks"] = [self.posthog_callback_handler]

            # Run the graph
            result = app.invoke(state, config=invoke_config)
            if verify_anomalies:
                results[col] = result["verified_anomalies"] or AnomalyList(anomalies=[])
            else:
                results[col] = result["detected_anomalies"] or AnomalyList(anomalies=[])

        return results

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
