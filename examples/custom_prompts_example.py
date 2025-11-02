"""
Example demonstrating how to use custom prompts with the AnomalyAgent.

This example shows how to provide your own detection and verification prompts
to customize the behavior of the anomaly detection agent.
"""

import os
import uuid

import pandas as pd
from dotenv import load_dotenv

from anomaly_agent import AnomalyAgent

# Load environment variables and set up PostHog session if enabled
load_dotenv()
posthog_enabled = os.getenv("POSTHOG_ENABLED", "false").lower() == "true"
if posthog_enabled and not os.getenv("POSTHOG_AI_SESSION_ID"):
    session_id = str(uuid.uuid4())
    os.environ["POSTHOG_AI_SESSION_ID"] = session_id
    print(f"🔗 PostHog Session ID: {session_id}")
    print("All traces in this session will be grouped together in PostHog.\n")

# Create sample time series data
data = {
    "timestamp": pd.date_range("2023-01-01", periods=100, freq="H"),
    "temperature": [20 + i * 0.1 + (5 if i == 50 else 0) for i in range(100)],
    "pressure": [1013 + i * 0.2 + (50 if i == 75 else 0) for i in range(100)],
}
df = pd.DataFrame(data)

# Example 1: Using improved default prompts
print("=== Example 1: Improved Default Prompts ===")
print("The default prompts have been significantly improved to provide:")
print("- Clear statistical criteria (2-3 standard deviations)")
print("- Domain context awareness")
print("- Pattern recognition guidelines")
print("- Conservative false positive reduction")
print()

agent_default = AnomalyAgent()
anomalies_default = agent_default.detect_anomalies(df)
print(f"Detected anomalies with improved default prompts: {len(anomalies_default)}")

# Example 2: Custom detection prompt for temperature-specific analysis
print("\n=== Example 2: Custom Detection Prompt ===")
temperature_detection_prompt = """
You are an expert in temperature anomaly detection for industrial systems.
You specialize in identifying temperature spikes, drops, and unusual patterns that could
indicate equipment malfunction or environmental changes. Focus on:
1. Sudden temperature increases that could indicate overheating
2. Rapid temperature drops that might suggest cooling system failure
3. Temperature oscillations that are outside normal operating ranges
4. Any temperature readings that deviate significantly from the expected trend

Consider the operational context and be conservative in your detections to minimize false alarms.
"""

agent_custom_detection = AnomalyAgent(detection_prompt=temperature_detection_prompt)
anomalies_custom_detection = agent_custom_detection.detect_anomalies(df)
print(
    f"Detected anomalies with custom detection prompt: {len(anomalies_custom_detection)}"
)

# Example 3: Custom verification prompt for stricter verification
print("\n=== Example 3: Custom Verification Prompt ===")
strict_verification_prompt = """
You are a senior data scientist specializing in anomaly verification.
Your role is to be extremely conservative and only confirm anomalies that meet strict criteria:

1. The anomaly must be statistically significant (>3 standard deviations from normal)
2. The anomaly must persist for more than a single data point
3. The anomaly must not be explainable by normal operational variations
4. The anomaly should have potential business or operational impact

Reject any anomalies that could be:
- Normal operational fluctuations
- Minor statistical variations
- Single-point outliers without context
- Gradual trends that are within acceptable ranges

Only return anomalies that require immediate attention or investigation.
"""

agent_custom_verification = AnomalyAgent(verification_prompt=strict_verification_prompt)
anomalies_custom_verification = agent_custom_verification.detect_anomalies(df)
print(
    f"Detected anomalies with custom verification prompt: {len(anomalies_custom_verification)}"
)

# Example 4: Both custom detection and verification prompts
print("\n=== Example 4: Both Custom Prompts ===")
agent_both_custom = AnomalyAgent(
    detection_prompt=temperature_detection_prompt,
    verification_prompt=strict_verification_prompt,
)
anomalies_both_custom = agent_both_custom.detect_anomalies(df)
print(f"Detected anomalies with both custom prompts: {len(anomalies_both_custom)}")

# Example 5: Domain-specific prompts for financial data
print("\n=== Example 5: Financial Domain Prompts ===")
financial_detection_prompt = """
You are a financial risk analyst specializing in market anomaly detection.
Focus on detecting:
1. Price movements that exceed normal volatility ranges
2. Volume spikes that could indicate unusual trading activity
3. Sudden trend reversals that might signal market shifts
4. Patterns that could indicate market manipulation or insider trading

Consider market hours, economic events, and seasonal patterns in your analysis.
"""

financial_verification_prompt = """
You are a senior financial risk manager reviewing potential market anomalies.
Only confirm anomalies that:
1. Represent significant financial risk or opportunity
2. Require immediate risk management attention
3. Could impact regulatory compliance
4. Indicate possible market abuse or manipulation

Filter out normal market movements and minor fluctuations.
"""

# This would work with financial time series data
agent_financial = AnomalyAgent(
    detection_prompt=financial_detection_prompt,
    verification_prompt=financial_verification_prompt,
)

print("\n=== Key Improvements in Default Prompts ===")
print("✅ Statistical criteria: Clear thresholds (2-3 standard deviations)")
print("✅ Domain awareness: Considers variable context and domain knowledge")
print("✅ Pattern recognition: Identifies trend breaks, level shifts, and seasonality")
print("✅ False positive reduction: Conservative approach with strict verification")
print("✅ Actionable focus: Emphasizes anomalies that require investigation")
print("✅ Clear guidance: Specific do's and don'ts for better consistency")

print(
    "\nExample complete! You can now customize prompts for your specific domain and use case."
)
print(
    "The new API is cleaner - just use detection_prompt and verification_prompt parameters!"
)
