"""High-level anomaly detection agent API."""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI

from .constants import DEFAULT_MODEL_NAME, DEFAULT_TIMESTAMP_COL
from .graph import create_graph
from .models import AnomalyList
from .nodes import AgentState
from .prompt import DEFAULT_SYSTEM_PROMPT, DEFAULT_VERIFY_SYSTEM_PROMPT


class AnomalyAgent:
    """Agent for detecting and verifying anomalies in time series data."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        timestamp_col: str = DEFAULT_TIMESTAMP_COL,
        verify_anomalies: bool = True,
        detection_prompt: str = DEFAULT_SYSTEM_PROMPT,
        verification_prompt: str = DEFAULT_VERIFY_SYSTEM_PROMPT,
    ):
        """Initialize the AnomalyAgent with a specific model."""
        self.llm = ChatOpenAI(model=model_name)
        self.timestamp_col = timestamp_col
        self.verify_anomalies = verify_anomalies
        self.detection_prompt = detection_prompt
        self.verification_prompt = verification_prompt

        self.graph = create_graph(
            self.llm, verify_anomalies, detection_prompt, verification_prompt
        )
        self.app = self.graph.compile()

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        timestamp_col: Optional[str] = None,
        verify_anomalies: Optional[bool] = None,
    ) -> Dict[str, AnomalyList]:
        """Detect anomalies in the given time series data."""
        if timestamp_col is not None:
            self.timestamp_col = timestamp_col

        verify_anomalies = (
            self.verify_anomalies if verify_anomalies is None else verify_anomalies
        )

        graph = create_graph(
            self.llm, verify_anomalies, self.detection_prompt, self.verification_prompt
        )
        app = graph.compile()

        if self.timestamp_col not in df.columns:
            raise KeyError(
                f"Timestamp column '{self.timestamp_col}' not found in DataFrame"
            )

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {
                col: AnomalyList(anomalies=[])
                for col in df.columns
                if col != self.timestamp_col
            }

        df_str = df.to_string(index=False)

        results = {}
        for col in numeric_cols:
            state: AgentState = {
                "time_series": df_str,
                "variable_name": col,
                "current_step": "detect",
            }
            result = app.invoke(state)
            if verify_anomalies:
                results[col] = result["verified_anomalies"] or AnomalyList(anomalies=[])
            else:
                results[col] = result["detected_anomalies"] or AnomalyList(anomalies=[])

        return results

    def get_anomalies_df(
        self, anomalies: Dict[str, AnomalyList], format: str = "long"
    ) -> pd.DataFrame:
        """Convert anomalies to a DataFrame."""
        if format not in ["long", "wide"]:
            raise ValueError("format must be either 'long' or 'wide'")

        if format == "long":
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
