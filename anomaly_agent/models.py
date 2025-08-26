"""Pydantic models used by the anomaly agent."""

from datetime import datetime
from typing import List

from pydantic import BaseModel, Field, validator

from .constants import TIMESTAMP_FORMAT


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
            datetime.strptime(v, TIMESTAMP_FORMAT)
            return v
        except ValueError:
            try:
                dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
                if "." in v:
                    return dt.strftime(TIMESTAMP_FORMAT)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    dt = datetime.strptime(v, "%Y-%m-%d")
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    try:
                        datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
                        return v
                    except ValueError:
                        raise ValueError(
                            f"timestamp must be in {TIMESTAMP_FORMAT} format, ISO format, or YYYY-MM-DD format"
                        )

    @validator("variable_value")  # type: ignore
    def validate_variable_value(cls, v: float) -> float:
        """Validate that the variable value is numeric."""
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
        """Validate that anomalies is a list of Anomaly objects."""
        if not isinstance(v, list):
            raise ValueError("anomalies must be a list")
        return v
