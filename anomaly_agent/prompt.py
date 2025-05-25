"""Prompt templates and system prompts for the anomaly detection agent."""

from langchain_core.prompts import ChatPromptTemplate

DEFAULT_SYSTEM_PROMPT = """
You are an expert anomaly detection agent.
You are given a time series and you need to identify the anomalies.
"""

DEFAULT_VERIFY_SYSTEM_PROMPT = """
You are an expert at verifying anomaly detections.
Review the time series and the detected anomalies to confirm if they are
genuine anomalies.
"""


def get_detection_prompt() -> ChatPromptTemplate:
    """Get the detection prompt template."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", DEFAULT_SYSTEM_PROMPT),
            (
                "human",
                "Variable name: {variable_name}\nTime series: \n\n {time_series} \n\n",
            ),
        ]
    )


def get_verification_prompt() -> ChatPromptTemplate:
    """Get the verification prompt template."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", DEFAULT_VERIFY_SYSTEM_PROMPT),
            (
                "human",
                "Variable name: {variable_name}\nTime series:\n{time_series}\n\n"
                "Detected anomalies:\n{detected_anomalies}\n\n"
                "Please verify these anomalies and return only the confirmed ones.",
            ),
        ]
    )
