"""Utility chains used by anomaly detection nodes."""
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from .models import AnomalyList
from .prompt import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_VERIFY_SYSTEM_PROMPT,
    get_detection_prompt,
    get_verification_prompt,
)


def create_detection_chain(
    llm: ChatOpenAI, detection_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> Runnable:
    """Create the LLM chain for anomaly detection."""
    return get_detection_prompt(detection_prompt) | llm.with_structured_output(
        AnomalyList
    )


def create_verification_chain(
    llm: ChatOpenAI, verification_prompt: str = DEFAULT_VERIFY_SYSTEM_PROMPT
) -> Runnable:
    """Create the LLM chain for anomaly verification."""
    return get_verification_prompt(
        verification_prompt
    ) | llm.with_structured_output(AnomalyList)
