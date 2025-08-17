"""
Node implementations for the anomaly detection agent.

This module contains class-based node implementations that provide better
separation of concerns and reusability compared to factory functions.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from .prompt import get_detection_prompt, get_verification_prompt, DEFAULT_SYSTEM_PROMPT, DEFAULT_VERIFY_SYSTEM_PROMPT


class DetectionNode:
    """Class-based detection node with dynamic configuration support."""
    
    def __init__(self, llm: ChatOpenAI):
        """Initialize with LLM instance."""
        self.llm = llm
        self._chains = {}  # Cache compiled chains by prompt
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _get_chain(self, detection_prompt: str):
        """Get or create a chain for the given prompt."""
        if detection_prompt not in self._chains:
            from .agent import AnomalyList
            self._chains[detection_prompt] = (
                get_detection_prompt(detection_prompt) | 
                self.llm.with_structured_output(AnomalyList)
            )
        return self._chains[detection_prompt]
    
    def __call__(self, state, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Process the state and detect anomalies."""
        # Extract configuration from state metadata or use defaults
        runtime_config = (config or {}).get("configurable", {}) if config else {}
        detection_prompt = (
            runtime_config.get("detection_prompt") or 
            state.processing_metadata.get("detection_prompt", DEFAULT_SYSTEM_PROMPT)
        )
        verify_enabled = state.processing_metadata.get("verification_enabled", True)
        
        # Check for debug mode and setup logging
        debug = state.processing_metadata.get("debug", False)
        if debug and not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)
        
        self.logger.debug(f"DetectionNode: Processing {state.variable_name}")
        
        try:
            chain = self._get_chain(detection_prompt)
            result = chain.invoke(
                {
                    "time_series": state.time_series,
                    "variable_name": state.variable_name,
                }
            )
            
            # Determine next step based on verification setting
            next_step = "verify" if verify_enabled else "end"
            
            self.logger.debug(f"DetectionNode: Found {len(result.anomalies)} anomalies, next step: {next_step}")
            
            return {
                "detected_anomalies": result, 
                "current_step": next_step,
                "processing_metadata": {
                    **state.processing_metadata,
                    "detection_completed": datetime.now().isoformat(),
                    "detection_node_calls": state.processing_metadata.get("detection_node_calls", 0) + 1
                }
            }
        except Exception as e:
            return {
                "current_step": "error",
                "error_messages": [f"Detection failed: {str(e)}"],
                "retry_count": state.retry_count + 1,
                "processing_metadata": {
                    **state.processing_metadata,
                    "detection_error": str(e),
                    "detection_error_time": datetime.now().isoformat()
                }
            }


class VerificationNode:
    """Class-based verification node with dynamic configuration support."""
    
    def __init__(self, llm: ChatOpenAI):
        """Initialize with LLM instance."""
        self.llm = llm
        self._chains = {}  # Cache compiled chains by prompt
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _get_chain(self, verification_prompt: str):
        """Get or create a chain for the given prompt."""
        if verification_prompt not in self._chains:
            from .agent import AnomalyList
            self._chains[verification_prompt] = (
                get_verification_prompt(verification_prompt) | 
                self.llm.with_structured_output(AnomalyList)
            )
        return self._chains[verification_prompt]
    
    def __call__(self, state, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Process the state and verify anomalies."""
        runtime_config = (config or {}).get("configurable", {}) if config else {}
        verification_prompt = (
            runtime_config.get("verification_prompt") or
            state.processing_metadata.get("verification_prompt", DEFAULT_VERIFY_SYSTEM_PROMPT)
        )
        
        try:
            if state.detected_anomalies is None:
                return {
                    "verified_anomalies": None, 
                    "current_step": "end",
                    "processing_metadata": {
                        **state.processing_metadata,
                        "verification_skipped": "no_anomalies_detected",
                        "verification_node_calls": state.processing_metadata.get("verification_node_calls", 0) + 1
                    }
                }

            detected_str = "\n".join(
                [
                    (
                        f"timestamp: {a.timestamp}, "
                        f"value: {a.variable_value}, "
                        f"Description: {a.anomaly_description}"
                    )
                    for a in state.detected_anomalies.anomalies
                ]
            )

            chain = self._get_chain(verification_prompt)
            result = chain.invoke(
                {
                    "time_series": state.time_series,
                    "variable_name": state.variable_name,
                    "detected_anomalies": detected_str,
                }
            )
            
            return {
                "verified_anomalies": result, 
                "current_step": "end",
                "processing_metadata": {
                    **state.processing_metadata,
                    "verification_completed": datetime.now().isoformat(),
                    "anomalies_verified": len(result.anomalies) if result else 0,
                    "verification_node_calls": state.processing_metadata.get("verification_node_calls", 0) + 1,
                    "anomalies_before_verification": len(state.detected_anomalies.anomalies)
                }
            }
        except Exception as e:
            return {
                "current_step": "error",
                "error_messages": [f"Verification failed: {str(e)}"],
                "retry_count": state.retry_count + 1,
                "processing_metadata": {
                    **state.processing_metadata,
                    "verification_error": str(e),
                    "verification_error_time": datetime.now().isoformat()
                }
            }


class ErrorHandlerNode:
    """Class-based error handler with configurable retry strategies."""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.5):
        """Initialize error handler with retry configuration."""
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def __call__(self, state, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Handle errors and determine retry logic."""
        runtime_config = (config or {}).get("configurable", {}) if config else {}
        max_retries = (
            runtime_config.get("max_retries") or 
            state.processing_metadata.get("max_retries", self.max_retries)
        )
        
        # Calculate backoff delay (exponential backoff)
        delay = self.backoff_factor ** state.retry_count
        
        if state.retry_count < max_retries:
            return {
                "current_step": "detect",
                "retry_count": state.retry_count,  # Will be incremented by next detection failure
                "processing_metadata": {
                    **state.processing_metadata,
                    f"retry_attempt_{state.retry_count + 1}": datetime.now().isoformat(),
                    f"retry_delay_{state.retry_count + 1}": delay,
                    "total_errors": len(state.error_messages),
                    "error_handler_calls": state.processing_metadata.get("error_handler_calls", 0) + 1
                }
            }
        else:
            return {
                "current_step": "end",
                "processing_metadata": {
                    **state.processing_metadata,
                    "max_retries_exceeded": True,
                    "final_error": state.error_messages[-1] if state.error_messages else "Unknown error",
                    "total_retry_attempts": state.retry_count,
                    "failure_reason": "max_retries_exceeded"
                }
            }


# Factory functions for backward compatibility
def create_detection_node(llm: ChatOpenAI, detection_prompt: str = DEFAULT_SYSTEM_PROMPT, verify_anomalies: bool = True):
    """Factory function for backward compatibility."""
    node = DetectionNode(llm)
    
    def detection_wrapper(state) -> Dict[str, Any]:
        config = {"detection_prompt": detection_prompt}
        # Update state to include verification setting
        state.processing_metadata["verification_enabled"] = verify_anomalies
        return node(state, config)
    
    return detection_wrapper


def create_verification_node(llm: ChatOpenAI, verification_prompt: str = DEFAULT_VERIFY_SYSTEM_PROMPT):
    """Factory function for backward compatibility."""
    node = VerificationNode(llm)
    
    def verification_wrapper(state) -> Dict[str, Any]:
        config = {"verification_prompt": verification_prompt}
        return node(state, config)
    
    return verification_wrapper


def create_error_handler_node(max_retries: int = 3):
    """Factory function for error handler node."""
    node = ErrorHandlerNode(max_retries)
    return node