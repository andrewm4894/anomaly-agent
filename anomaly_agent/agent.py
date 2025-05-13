from typing import Optional, Dict
from pydantic import BaseModel, Field, model_validator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd


DEFAULT_SYSTEM_PROMPT = """
You are an expert anomaly detection agent. You are given a time series and you need to identify the anomalies.
"""

DEFAULT_VERIFY_SYSTEM_PROMPT = """
You are an expert at verifying anomaly detections. Review the time series and the detected anomalies to confirm if they are genuine anomalies.
"""


class Anomaly(BaseModel):
    timestamp: str = Field(description="The timestamp of the anomaly")
    variable_value: float = Field(
        description="The value of the variable at the anomaly timestamp"
    )
    anomaly_description: str = Field(description="A description of the anomaly")


class AnomalyList(BaseModel):
    anomalies: list[Anomaly] = Field(description="The list of anomalies")


class AnomalyAgent:
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        timestamp_col: str = "timestamp",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        verify_system_prompt: str = DEFAULT_VERIFY_SYSTEM_PROMPT,
        detection_prompt_template: str = "Variable name: {variable_name}\nTime series: \n\n {time_series} \n\n",
        verification_prompt_template: str = "Variable name: {variable_name}\nTime series:\n{time_series}\n\nDetected anomalies:\n{detected_anomalies}\n\nPlease verify these anomalies and return only the confirmed ones."
    ):
        """Initialize the AnomalyAgent with a specific model and prompts.

        Args:
            model_name: The name of the OpenAI model to use
            timestamp_col: The name of the timestamp column
            system_prompt: The system prompt for anomaly detection
            verify_system_prompt: The system prompt for anomaly verification
            detection_prompt_template: The template for the detection prompt
            verification_prompt_template: The template for the verification prompt
        """
        self.llm = ChatOpenAI(model=model_name).with_structured_output(AnomalyList)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", detection_prompt_template),
            ]
        )
        self.chain = self.prompt | self.llm
        self.timestamp_col = timestamp_col

        # Add verification chain
        self.verify_prompt = ChatPromptTemplate.from_messages([
            ("system", verify_system_prompt),
            ("human", verification_prompt_template)
        ])
        self.verify_chain = self.verify_prompt | self.llm

    def detect_anomalies(
        self, df: pd.DataFrame, timestamp_col: str = "timestamp", verify: bool = True
    ) -> Dict[str, AnomalyList]:
        """Detect anomalies in the given time series data for all numeric columns except timestamp.

        Args:
            df: DataFrame containing the time series data
            timestamp_col: Name of the timestamp column
            verify: Whether to verify the detected anomalies using a second pass

        Returns:
            Dictionary mapping column names to their respective AnomalyList
        """
        # Set the timestamp column
        self.timestamp_col = timestamp_col
        
        # Get all columns except timestamp
        value_cols = [col for col in df.columns if col != timestamp_col]
        anomalies: Dict[str, AnomalyList] = {}

        # Process each column
        for value_col in value_cols:
            time_series = df[[timestamp_col, value_col]].to_string()
            result = self.chain.invoke(
                {"time_series": time_series, "variable_name": value_col}
            )
            
            if verify:
                # Convert detected anomalies to string format for verification
                detected_str = "\n".join([
                    f"{self.timestamp_col}: {a.timestamp}, {value_col}: {a.variable_value}, Description: {a.anomaly_description}"
                    for a in result.anomalies
                ])
                
                # Verify the anomalies
                verified_result = self.verify_chain.invoke({
                    "time_series": time_series,
                    "variable_name": value_col,
                    "detected_anomalies": detected_str
                })
                result = verified_result

            anomalies[value_col] = result

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
            DataFrame containing the anomalous data points. For long format, includes columns for timestamp,
            variable name, value, and description. For wide format, includes timestamp and one column per variable.
            The timestamp column is converted to datetime type.
        """
        if format.lower() not in ["long", "wide"]:
            raise ValueError("format must be either 'long' or 'wide'")

        if format.lower() == "long":
            rows = []
            for variable_name, anomaly_list in anomalies.items():
                for anomaly in anomaly_list.anomalies:
                    rows.append(
                        {
                            self.timestamp_col : pd.to_datetime(anomaly.timestamp),
                            "variable_name": variable_name,
                            "value": anomaly.variable_value,
                            "anomaly_description": anomaly.anomaly_description,
                        }
                    )
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
