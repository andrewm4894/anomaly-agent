"""Graph nodes for anomaly detection and verification."""

from typing import Callable, Literal, Optional, TypedDict

from langchain_openai import ChatOpenAI

from .models import AnomalyList
from .tools import create_detection_chain, create_verification_chain


class AgentState(TypedDict, total=False):
    """State for the anomaly detection agent."""

    time_series: str
    variable_name: str
    detected_anomalies: Optional[AnomalyList]
    verified_anomalies: Optional[AnomalyList]
    current_step: str


def create_detection_node(
    llm: ChatOpenAI, detection_prompt: str
) -> Callable[[AgentState], AgentState]:
    """Create the detection node for the graph."""
    chain = create_detection_chain(llm, detection_prompt)

    def detection_node(state: AgentState) -> AgentState:
        result = chain.invoke(
            {
                "time_series": state["time_series"],
                "variable_name": state["variable_name"],
            }
        )
        return {"detected_anomalies": result, "current_step": "verify"}

    return detection_node


def create_verification_node(
    llm: ChatOpenAI, verification_prompt: str
) -> Callable[[AgentState], AgentState]:
    """Create the verification node for the graph."""
    chain = create_verification_chain(llm, verification_prompt)

    def verification_node(state: AgentState) -> AgentState:
        if state["detected_anomalies"] is None:
            return {"verified_anomalies": None, "current_step": "end"}

        detected_str = "\n".join(
            [
                (
                    f"timestamp: {a.timestamp}, "
                    f"value: {a.variable_value}, "
                    f"Description: {a.anomaly_description}"
                )
                for a in state["detected_anomalies"].anomalies
            ]
        )

        result = chain.invoke(
            {
                "time_series": state["time_series"],
                "variable_name": state["variable_name"],
                "detected_anomalies": detected_str,
            }
        )
        return {"verified_anomalies": result, "current_step": "end"}

    return verification_node


def should_verify(state: AgentState) -> Literal["verify", "end"]:
    """Determine if we should proceed to verification."""
    return "verify" if state["current_step"] == "verify" else "end"
