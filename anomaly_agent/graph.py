"""
Graph management and routing for the anomaly detection agent.

This module contains the GraphManager for efficient graph caching and reuse,
as well as routing logic for conditional graph execution.
"""

from typing import Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from .nodes import DetectionNode, VerificationNode, ErrorHandlerNode


def should_verify(state) -> Literal["verify", "end", "error"]:
    """Determine if we should proceed to verification."""
    if state.current_step == "error":
        return "error"
    return "verify" if state.current_step == "verify" else "end"


class GraphManager:
    """Manages reusable compiled graphs for different configurations."""
    
    def __init__(self):
        """Initialize graph manager with caching."""
        self._compiled_graphs = {}  # Cache for compiled graphs
        self._node_instances = {}   # Cache for node instances
    
    def get_or_create_graph(self, config, llm: ChatOpenAI):
        """Get or create a compiled graph for the given configuration."""
        # Create a cache key based on verification setting (main differentiator)
        cache_key = f"verify_{config.verify_anomalies}_steps_{config.n_verify_steps}_retries_{config.max_retries}"
        
        if cache_key not in self._compiled_graphs:
            self._compiled_graphs[cache_key] = self._create_graph(config, llm)
        
        return self._compiled_graphs[cache_key]
    
    def _create_graph(self, config, llm: ChatOpenAI):
        """Create and compile a graph for the given configuration."""
        from .agent import AgentState
        graph = StateGraph(AgentState)
        
        # Get or create node instances (cached for reuse)
        detection_node = self._get_or_create_node("detection", DetectionNode, llm)
        error_node = self._get_or_create_node("error", ErrorHandlerNode, config.max_retries)
        
        # Add core nodes
        graph.add_node("detect", detection_node)
        graph.add_node("error", error_node)
        
        # Add verification nodes if needed
        verify_node_names = []
        if config.verify_anomalies:
            verification_node = self._get_or_create_node("verification", VerificationNode, llm)
            for i in range(config.n_verify_steps):
                node_name = f"verify_{i+1}" if config.n_verify_steps > 1 else "verify"
                graph.add_node(node_name, verification_node)
                verify_node_names.append(node_name)
        
        # Add edges based on configuration
        if config.verify_anomalies:
            # Connect detection to first verification step
            first_verify = verify_node_names[0]
            graph.add_conditional_edges(
                "detect", 
                should_verify, 
                {"verify": first_verify, "end": END, "error": "error"}
            )
            
            # Chain verification steps together
            for i in range(len(verify_node_names)):
                current_verify = verify_node_names[i]
                if i < len(verify_node_names) - 1:
                    # More verification steps to go
                    next_verify = verify_node_names[i + 1]
                    graph.add_edge(current_verify, next_verify)
                else:
                    # Last verification step, go to end
                    graph.add_edge(current_verify, END)
            
            # Error handling
            graph.add_conditional_edges(
                "error",
                lambda state: "detect" if state.retry_count < config.max_retries else "end",
                {"detect": "detect", "end": END}
            )
        else:
            # Without verification, go directly to end or error
            graph.add_conditional_edges(
                "detect", 
                lambda state: "error" if state.current_step == "error" else "end",
                {"end": END, "error": "error"}
            )
            graph.add_conditional_edges(
                "error",
                lambda state: "detect" if state.retry_count < config.max_retries else "end",
                {"detect": "detect", "end": END}
            )
        
        # Set entry point
        graph.set_entry_point("detect")
        
        return graph.compile()
    
    def _get_or_create_node(self, node_type: str, node_class, *args):
        """Get or create a node instance."""
        # Create a more robust cache key for different argument types
        cache_parts = [node_type]
        for arg in args:
            if hasattr(arg, 'model_name'):  # ChatOpenAI case
                cache_parts.append(f"llm_{arg.model_name}")
            else:
                cache_parts.append(str(arg))
        cache_key = "_".join(cache_parts)
        
        if cache_key not in self._node_instances:
            self._node_instances[cache_key] = node_class(*args)
        return self._node_instances[cache_key]