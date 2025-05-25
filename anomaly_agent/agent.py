"""
Anomaly detection agent using LLMs to identify and verify anomalies in time series data.

This module provides functionality for detecting and verifying anomalies in time series
data using language models.
"""

from datetime import datetime
from typing import Dict, List, Literal, Optional, TypedDict

import numpy as np
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, validator

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert anomaly detection agent. "
    "You are given a time series and you need to identify the anomalies."
)

DEFAULT_VERIFY_SYSTEM_PROMPT = (
    "You are an expert at verifying anomaly detections. "
    "Review the time series and the detected anomalies to confirm if they are "
    "genuine anomalies."
)


class Anomaly(BaseModel):
    """Represents a single anomaly in a time series."""

    timestamp: str = Field(description="The timestamp of the anomaly")
    variable_value: float = Field(
        description="The value of the variable at the anomaly timestamp"
    )
    anomaly_description: str = Field(description="A description of the anomaly")

    @validator("timestamp")  # type: ignore
    def validate_timestamp(cls, v: str) -> str:
        """Validate that the timestamp is in YYYY-MM-DD format."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("timestamp must be in YYYY-MM-DD format")

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
    variable_name: str
    detected_anomalies: Optional[AnomalyList]
    verified_anomalies: Optional[AnomalyList]
    current_step: str


def create_detection_node(llm: ChatOpenAI) -> ToolNode:
    """Create the detection node for the graph."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", DEFAULT_SYSTEM_PROMPT),
            (
                "human",
                "Variable name: {variable_name}\nTime series: \n\n {time_series} \n\n"
                "IMPORTANT: Return timestamps in YYYY-MM-DD format only.",
            ),
        ]
    )
    chain = prompt | llm.with_structured_output(AnomalyList)

    def detection_node(state: AgentState) -> AgentState:
        """Process the state and detect anomalies."""
        result = chain.invoke(
            {
                "time_series": state["time_series"],
                "variable_name": state["variable_name"],
            }
        )
        return {"detected_anomalies": result, "current_step": "verify"}

    return detection_node


def create_verification_node(llm: ChatOpenAI) -> ToolNode:
    """Create the verification node for the graph."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", DEFAULT_VERIFY_SYSTEM_PROMPT),
            (
                "human",
                "Variable name: {variable_name}\nTime series:\n{time_series}\n\n"
                "Detected anomalies:\n{detected_anomalies}\n\n"
                "Please verify these anomalies and return only the confirmed ones.",  # noqa: E501
            ),
        ]
    )
    chain = prompt | llm.with_structured_output(AnomalyList)

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
        model_name: str = "gpt-4o-mini",
        timestamp_col: str = "timestamp",
    ):
        """Initialize the AnomalyAgent with a specific model.

        Args:
            model_name: The name of the OpenAI model to use
            timestamp_col: The name of the timestamp column
        """
        self.llm = ChatOpenAI(model=model_name)
        self.timestamp_col = timestamp_col

        # Create the graph
        self.graph = StateGraph(AgentState)

        # Add nodes
        self.graph.add_node("detect", create_detection_node(self.llm))
        self.graph.add_node("verify", create_verification_node(self.llm))

        # Add edges with proper routing
        self.graph.add_conditional_edges(
            "detect", should_verify, {"verify": "verify", "end": END}
        )
        self.graph.add_edge("verify", END)

        # Set entry point
        self.graph.set_entry_point("detect")

        # Compile the graph
        self.app = self.graph.compile()

    def detect_anomalies(
        self, df: pd.DataFrame, timestamp_col: Optional[str] = None
    ) -> Dict[str, AnomalyList]:
        """Detect anomalies in the given time series data.

        Args:
            df: DataFrame containing the time series data
            timestamp_col: Name of the timestamp column (optional)

        Returns:
            Dictionary mapping column names to their respective AnomalyList
        """
        if timestamp_col is not None:
            self.timestamp_col = timestamp_col

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
            # Create state for this column
            state = {
                "time_series": df_str,
                "variable_name": col,
                "current_step": "detect",
            }

            # Run the graph
            result = self.app.invoke(state)
            results[col] = result["verified_anomalies"] or AnomalyList(anomalies=[])

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
                            "timestamp": anomaly.timestamp,
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
                        "timestamp": anomaly.timestamp,
                        col: anomaly.variable_value,
                        f"{col}_description": anomaly.anomaly_description,
                    }
                )
        return pd.DataFrame(rows)
