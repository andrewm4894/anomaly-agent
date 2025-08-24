"""Constants used throughout the anomaly detection agent."""

# Timestamp format used for validation and parsing
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

# Default model name for the LLM
DEFAULT_MODEL_NAME = "gpt-4o-mini"

# Supported model providers
OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini", 
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-turbo"
]

ANTHROPIC_MODELS = [
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022", 
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307"
]

# Default timestamp column name in DataFrames
DEFAULT_TIMESTAMP_COL = "timestamp"
