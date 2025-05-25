from typing import Dict, List, Optional, TypedDict, Annotated, Sequence
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
import pandas as pd
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

DEFAULT_SYSTEM_PROMPT = """
You are an expert anomaly detection agent. You are given a time series and you need to identify the anomalies.
"""

DEFAULT_VERIFY_SYSTEM_PROMPT = """
You are an expert at verifying anomaly detections. Review the time series and the detected anomalies to confirm if they are genuine anomalies.
"""

class Anomaly(BaseModel):
    timestamp: str = Field(description="The timestamp of the anomaly")
    variable_value: float = Field(description="The value of the variable at the anomaly timestamp")
    anomaly_description: str = Field(description="A description of the anomaly")

class AnomalyList(BaseModel):
    anomalies: List[Anomaly] = Field(description="The list of anomalies")

class AgentState(TypedDict):
    """State for the anomaly detection agent."""
    time_series: str
    variable_name: str
    detected_anomalies: Optional[AnomalyList]
    verified_anomalies: Optional[AnomalyList]
    current_step: str

def create_detection_node(llm: ChatOpenAI) -> ToolNode:
    """Create the detection node for the graph."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", DEFAULT_SYSTEM_PROMPT),
        ("human", "Variable name: {variable_name}\nTime series: \n\n {time_series} \n\n")
    ])
    chain = prompt | llm.with_structured_output(AnomalyList)
    
    def detection_node(state: AgentState) -> AgentState:
        result = chain.invoke({
            "time_series": state["time_series"],
            "variable_name": state["variable_name"]
        })
        return {"detected_anomalies": result, "current_step": "verify"}
    
    return ToolNode(detection_node)

def create_verification_node(llm: ChatOpenAI) -> ToolNode:
    """Create the verification node for the graph."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", DEFAULT_VERIFY_SYSTEM_PROMPT),
        ("human", "Variable name: {variable_name}\nTime series:\n{time_series}\n\nDetected anomalies:\n{detected_anomalies}\n\nPlease verify these anomalies and return only the confirmed ones.")
    ])
    chain = prompt | llm.with_structured_output(AnomalyList)
    
    def verification_node(state: AgentState) -> AgentState:
        detected_str = "\n".join([
            f"timestamp: {a.timestamp}, value: {a.variable_value}, Description: {a.anomaly_description}"
            for a in state["detected_anomalies"].anomalies
        ])
        
        result = chain.invoke({
            "time_series": state["time_series"],
            "variable_name": state["variable_name"],
            "detected_anomalies": detected_str
        })
        return {"verified_anomalies": result, "current_step": "end"}
    
    return ToolNode(verification_node)

def should_verify(state: AgentState) -> str:
    """Determine if we should proceed to verification."""
    return "verify" if state["current_step"] == "verify" else "end"

class AnomalyAgentGraph:
    def __init__(
        self,
        model_name: str = "gpt-4-turbo-preview",
        timestamp_col: str = "timestamp",
    ):
        """Initialize the AnomalyAgentGraph with a specific model.
        
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
        
        # Add edges
        self.graph.add_edge("detect", should_verify)
        self.graph.add_edge("verify", END)
        
        # Set entry point
        self.graph.set_entry_point("detect")
        
        # Compile the graph
        self.app = self.graph.compile()

    def detect_anomalies(
        self, df: pd.DataFrame, timestamp_col: Optional[str] = None
    ) -> Dict[str, AnomalyList]:
        """Detect anomalies in the given time series data for all numeric columns except timestamp.
        
        Args:
            df: DataFrame containing the time series data
            timestamp_col: Name of the timestamp column (optional, uses instance default if not provided)
            
        Returns:
            Dictionary mapping column names to their respective AnomalyList
        """
        if timestamp_col is not None:
            self.timestamp_col = timestamp_col
            
        # Get all columns except timestamp
        value_cols = [col for col in df.columns if col != self.timestamp_col]
        anomalies: Dict[str, AnomalyList] = {}
        
        # Process each column
        for value_col in value_cols:
            time_series = df[[self.timestamp_col, value_col]].to_string()
            
            # Run the graph
            result = self.app.invoke({
                "time_series": time_series,
                "variable_name": value_col,
                "current_step": "detect"
            })
            
            anomalies[value_col] = result["verified_anomalies"]
            
        return anomalies

    def get_anomalies_df(
        self, anomalies: Dict[str, AnomalyList], format: str = "long"
    ) -> pd.DataFrame:
        """Create a DataFrame from the detected anomalies.
        
        Args:
            anomalies: Dictionary of anomalies returned by detect_anomalies
            format: Either 'long' or 'wide'. Long format has one row per anomaly with variable_name column.
                   Wide format has one row per timestamp with a column for each variable.
                   
        Returns:
            DataFrame containing the anomalous data points
        """
        if format.lower() not in ["long", "wide"]:
            raise ValueError("format must be either 'long' or 'wide'")
            
        if format.lower() == "long":
            rows = []
            for variable_name, anomaly_list in anomalies.items():
                for anomaly in anomaly_list.anomalies:
                    rows.append({
                        self.timestamp_col: pd.to_datetime(anomaly.timestamp),
                        "variable_name": variable_name,
                        "value": anomaly.variable_value,
                        "anomaly_description": anomaly.anomaly_description,
                    })
            return pd.DataFrame(rows)
        else:
            # Create a dictionary to store values for each variable at each timestamp
            wide_data = {}
            for variable_name, anomaly_list in anomalies.items():
                for anomaly in anomaly_list.anomalies:
                    timestamp = pd.to_datetime(anomaly.timestamp)
                    if timestamp not in wide_data:
                        wide_data[timestamp] = {}
                    wide_data[timestamp][variable_name] = anomaly.variable_value
            
            # Convert to DataFrame
            wide_df = pd.DataFrame.from_dict(wide_data, orient='index')
            wide_df.index.name = self.timestamp_col
            return wide_df.reset_index() 