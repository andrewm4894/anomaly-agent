#!/usr/bin/env python3

"""Example demonstrating the usage of Anthropic Claude models with the anomaly detection agent.

This example shows how to use different Claude models for anomaly detection and compares
their performance and results.
"""

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from anomaly_agent.agent import AnomalyAgent, AnomalyList
from anomaly_agent.constants import ANTHROPIC_MODELS, OPENAI_MODELS
from anomaly_agent.utils import make_anomaly_config, make_df


def create_test_data() -> pd.DataFrame:
    """Create test data with known anomalies."""
    # Create time series with multiple variables
    dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
    
    # Create realistic patterns with noise
    x = np.linspace(0, 4 * np.pi, 50)
    temp = 20 + 5 * np.sin(x) + np.random.normal(0, 0.5, 50)
    cpu_usage = 50 + 20 * np.cos(x) + np.random.normal(0, 2, 50)
    
    # Add clear anomalies
    temp[15] = 45.0  # Temperature spike
    temp[30] = 5.0   # Temperature drop
    cpu_usage[20] = 95.0  # CPU spike
    cpu_usage[35] = np.nan  # Missing data
    
    df = pd.DataFrame({
        "timestamp": dates,
        "temperature": temp,
        "cpu_usage": cpu_usage
    })
    
    return df


def compare_models() -> None:
    """Compare detection results across different Claude and OpenAI models."""
    print("\n=== Model Comparison Example ===")
    
    df = create_test_data()
    
    # Test models (mix of Claude and OpenAI)
    test_models = [
        "gpt-4o-mini",  # OpenAI baseline
        "claude-3-5-haiku-20241022",  # Fast Claude model
        "claude-3-5-sonnet-20241022", # Best Claude model
    ]
    
    results = {}
    
    for model_name in test_models:
        print(f"\nTesting {model_name}...")
        
        try:
            agent = AnomalyAgent(model_name=model_name)
            anomalies = agent.detect_anomalies(df)
            df_anomalies = agent.get_anomalies_df(anomalies)
            
            # Count anomalies per variable
            anomaly_counts = {}
            for var, anomaly_list in anomalies.items():
                anomaly_counts[var] = len(anomaly_list.anomalies)
            
            results[model_name] = {
                'anomalies': anomalies,
                'counts': anomaly_counts,
                'total': sum(anomaly_counts.values())
            }
            
            print(f"  Anomalies detected: {anomaly_counts}")
            print(f"  Total: {results[model_name]['total']}")
            
        except Exception as e:
            print(f"  Error with {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    # Summary comparison
    print("\n=== Summary ===")
    for model, result in results.items():
        if 'error' not in result:
            print(f"{model}: {result['total']} total anomalies")
        else:
            print(f"{model}: Failed - {result['error']}")


def example_claude_with_custom_prompts() -> None:
    """Demonstrate Claude models with custom prompts optimized for Claude."""
    print("\n=== Claude with Custom Prompts Example ===")
    
    df = create_test_data()
    
    # Custom prompt that leverages Claude's strengths
    claude_detection_prompt = """You are an expert data analyst specializing in time series anomaly detection. 

Analyze the provided time series data and identify anomalies based on these criteria:
- Statistical outliers (values beyond 2-3 standard deviations)
- Sudden spikes or drops that break normal patterns
- Missing or invalid values (NaN, null)
- Values that violate expected ranges for the variable type

Consider the temporal context and seasonal patterns when available. Be precise but not overly sensitive to minor variations.

For each anomaly found, provide:
1. The exact timestamp
2. The anomalous value
3. A clear, technical description of why it's anomalous

Focus on significant anomalies that would require attention in a monitoring system."""
    
    agent = AnomalyAgent(
        model_name="claude-3-5-sonnet-20241022",
        detection_prompt=claude_detection_prompt
    )
    
    anomalies = agent.detect_anomalies(df)
    df_anomalies = agent.get_anomalies_df(anomalies)
    
    print("\nAnomaliies detected with Claude and custom prompt:")
    print(df_anomalies)


def example_model_specific_features() -> None:
    """Show model-specific configuration and features."""
    print("\n=== Model-Specific Features Example ===")
    
    df = create_test_data()
    
    # Claude model with verification disabled (faster)
    print("\nClaude Haiku (fast, no verification):")
    agent_fast = AnomalyAgent(
        model_name="claude-3-5-haiku-20241022",
        verify_anomalies=False
    )
    anomalies_fast = agent_fast.detect_anomalies(df)
    total_fast = sum(len(al.anomalies) for al in anomalies_fast.values())
    print(f"  Fast detection: {total_fast} anomalies")
    
    # Claude Sonnet with verification (more accurate)
    print("\nClaude Sonnet (thorough, with verification):")
    agent_thorough = AnomalyAgent(
        model_name="claude-3-5-sonnet-20241022",
        verify_anomalies=True
    )
    anomalies_thorough = agent_thorough.detect_anomalies(df)
    total_thorough = sum(len(al.anomalies) for al in anomalies_thorough.values())
    print(f"  Thorough detection: {total_thorough} anomalies")


def main() -> None:
    """Run Claude model examples based on command line arguments."""
    parser = argparse.ArgumentParser(description="Run Claude model examples")
    parser.add_argument(
        "--anthropic-api-key",
        help="Anthropic API key (if not set in environment)",
    )
    parser.add_argument(
        "--openai-api-key", 
        help="OpenAI API key (if not set in environment)",
    )
    parser.add_argument(
        "--example",
        choices=["compare", "custom-prompts", "features", "all"],
        default="all",
        help="Which example to run",
    )
    args = parser.parse_args()
    
    # Set API keys if provided
    if args.anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = args.anthropic_api_key
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    
    # Check for required API keys
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("Warning: No ANTHROPIC_API_KEY found. Claude models will not work.")
        print("Set it in your .env file or pass --anthropic-api-key")
    
    if "OPENAI_API_KEY" not in os.environ:
        print("Warning: No OPENAI_API_KEY found. OpenAI models will not work.")
        print("Set it in your .env file or pass --openai-api-key")
    
    # Run selected example(s)
    if args.example == "all":
        compare_models()
        example_claude_with_custom_prompts() 
        example_model_specific_features()
    elif args.example == "compare":
        compare_models()
    elif args.example == "custom-prompts":
        example_claude_with_custom_prompts()
    elif args.example == "features":
        example_model_specific_features()


if __name__ == "__main__":
    main()