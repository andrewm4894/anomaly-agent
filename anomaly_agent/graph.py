"""Graph construction utilities for the anomaly agent."""
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from .nodes import AgentState, create_detection_node, create_verification_node, should_verify


def create_graph(
    llm: ChatOpenAI,
    verify_anomalies: bool,
    detection_prompt: str,
    verification_prompt: str,
) -> StateGraph:
    """Create the LangGraph state graph for anomaly detection."""
    graph = StateGraph(AgentState)
    graph.add_node("detect", create_detection_node(llm, detection_prompt))
    if verify_anomalies:
        graph.add_node("verify", create_verification_node(llm, verification_prompt))
        graph.add_conditional_edges(
            "detect", should_verify, {"verify": "verify", "end": END}
        )
        graph.add_edge("verify", END)
    else:
        graph.add_edge("detect", END)
    graph.set_entry_point("detect")
    return graph
