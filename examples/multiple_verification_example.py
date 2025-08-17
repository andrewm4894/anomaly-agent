"""
Example demonstrating multiple verification steps for improved anomaly detection accuracy.

This example shows how to use the n_verify_steps parameter to run multiple rounds of
verification, helping to reduce false positives through the stochastic nature of LLMs.
"""

import pandas as pd
from anomaly_agent import AnomalyAgent

# Create test data with clear anomalies
data = {
    'timestamp': pd.date_range('2023-01-01', periods=20, freq='D'),
    'temperature': [20.0] * 9 + [45.0] + [20.0] * 10,  # Clear spike at day 10
    'pressure': [1013.0] * 15 + [900.0] + [1013.0] * 4   # Clear drop at day 16
}
df = pd.DataFrame(data)

print("=== Multiple Verification Steps Example ===")
print("This example demonstrates how multiple verification steps can improve accuracy")
print("by running verification multiple times due to the stochastic nature of LLMs.\n")

print("Sample data:")
print(df.head(10))
print("...")
print(df.tail(10))
print()

# Example 1: Default single verification
print("=== Example 1: Single Verification (Default) ===")
agent_single = AnomalyAgent(
    model_name="gpt-5-nano", 
    verify_anomalies=True,
    n_verify_steps=1,  # Default
    debug=True
)

print(f"Agent configuration: verify_anomalies={agent_single.verify_anomalies}, n_verify_steps={agent_single.n_verify_steps}")
print("Running detection with single verification step...")

anomalies_single = agent_single.detect_anomalies(df)
print(f"Single verification result:")
for col, anomaly_list in anomalies_single.items():
    print(f"  - {col}: {len(anomaly_list.anomalies)} anomalies detected")

print("This runs: detect → verify → end\n")

# Example 2: Double verification for more confidence
print("=== Example 2: Double Verification ===")
agent_double = AnomalyAgent(
    model_name="gpt-5-nano",
    verify_anomalies=True, 
    n_verify_steps=2,
    debug=True
)

print(f"Agent configuration: verify_anomalies={agent_double.verify_anomalies}, n_verify_steps={agent_double.n_verify_steps}")
print("Running detection with double verification steps...")

anomalies_double = agent_double.detect_anomalies(df)
print(f"Double verification result:")
for col, anomaly_list in anomalies_double.items():
    print(f"  - {col}: {len(anomaly_list.anomalies)} anomalies detected")

print("This runs: detect → verify_1 → verify_2 → end\n")

# Example 3: Triple verification for maximum confidence
print("=== Example 3: Triple Verification ===")
agent_triple = AnomalyAgent(
    model_name="gpt-5-nano",
    verify_anomalies=True,
    n_verify_steps=3,
    debug=True
)

print(f"Agent configuration: verify_anomalies={agent_triple.verify_anomalies}, n_verify_steps={agent_triple.n_verify_steps}")
print("Running detection with triple verification steps...")

anomalies_triple = agent_triple.detect_anomalies(df)
print(f"Triple verification result:")
for col, anomaly_list in anomalies_triple.items():
    print(f"  - {col}: {len(anomaly_list.anomalies)} anomalies detected")

print("This runs: detect → verify_1 → verify_2 → verify_3 → end\n")

# Example 4: Runtime override of verification steps
print("=== Example 4: Runtime Override ===")
agent_configurable = AnomalyAgent(n_verify_steps=1, debug=True)  # Default to 1
print(f"Default agent configuration: n_verify_steps={agent_configurable.n_verify_steps}")

# Override at runtime
print("Overriding to 4 verification steps at runtime:")

anomalies_override = agent_configurable.detect_anomalies(df, n_verify_steps=4)
print(f"Runtime override result:")
for col, anomaly_list in anomalies_override.items():
    print(f"  - {col}: {len(anomaly_list.anomalies)} anomalies detected")

print("This runs: detect → verify_1 → verify_2 → verify_3 → verify_4 → end\n")

# Example 5: Comparison of results
print("=== Example 5: Results Comparison ===")
print("Comparing anomaly counts across different verification step counts:")
print(f"  Single verification (n=1):  temperature={len(anomalies_single['temperature'].anomalies)}, pressure={len(anomalies_single['pressure'].anomalies)}")
print(f"  Double verification (n=2):   temperature={len(anomalies_double['temperature'].anomalies)}, pressure={len(anomalies_double['pressure'].anomalies)}")  
print(f"  Triple verification (n=3):   temperature={len(anomalies_triple['temperature'].anomalies)}, pressure={len(anomalies_triple['pressure'].anomalies)}")
print(f"  Runtime override (n=4):      temperature={len(anomalies_override['temperature'].anomalies)}, pressure={len(anomalies_override['pressure'].anomalies)}")

print("\nObservation: More verification steps typically result in fewer final anomalies")
print("due to the filtering effect of multiple verification rounds.\n")

# Example 6: Configuration validation
print("=== Example 6: Configuration Validation ===")
print("The n_verify_steps parameter has validation constraints:")

try:
    invalid_agent = AnomalyAgent(n_verify_steps=0)
except Exception as e:
    print(f"✅ n_verify_steps=0 rejected: {e}")

try:
    invalid_agent = AnomalyAgent(n_verify_steps=6)
except Exception as e:
    print(f"✅ n_verify_steps=6 rejected: {e}")

print("✅ Valid range: 1-5 verification steps")

# Example 7: Cost and performance considerations
print("\n=== Example 7: Cost and Performance Considerations ===")
print("Multiple verification steps trade cost/time for accuracy:")
print()
print("n_verify_steps=1:")
print("  - Fastest execution")
print("  - Lowest cost") 
print("  - Standard accuracy")
print()
print("n_verify_steps=2:")
print("  - 2x verification cost/time")
print("  - Reduced false positives")
print("  - Good balance for production")
print()
print("n_verify_steps=3:")
print("  - 3x verification cost/time") 
print("  - Maximum confidence")
print("  - Best for critical applications")
print()

# Example 8: Combining with other parameters
print("=== Example 8: Advanced Configuration ===")
advanced_agent = AnomalyAgent(
    model_name="gpt-5-nano",  # Cost-optimized
    verify_anomalies=True,
    n_verify_steps=2,         # Double verification
    max_retries=3,            # Error recovery
    debug=True                # Detailed logging
)

print("Advanced configuration combines multiple verification with:")
print(f"  - Model: {advanced_agent.config.model_name}")
print(f"  - Verification steps: {advanced_agent.config.n_verify_steps}")
print(f"  - Max retries: {advanced_agent.config.max_retries}")
print(f"  - Debug mode: {'enabled' if advanced_agent.debug else 'disabled'}")

print("\n=== Key Benefits of Multiple Verification ===")
print("🎯 Reduced false positives through multiple LLM evaluations")
print("🔍 Better consistency in anomaly detection results")
print("⚖️ Configurable trade-off between accuracy and cost")
print("🔧 Runtime configurability for different use cases")
print("📊 Detailed metadata tracking for each verification step")

print("\n=== Recommendations ===")
print("• Use n_verify_steps=1 for fast, cost-effective detection")
print("• Use n_verify_steps=2 for production systems (good balance)")  
print("• Use n_verify_steps=3+ for critical applications requiring high confidence")
print("• Enable debug=True to observe verification step filtering in action")
print("• Consider cost implications: each step multiplies verification costs")

if __name__ == "__main__":
    print("\n🚀 Multiple verification steps ready for use!")
    print("Set OPENAI_API_KEY environment variable to test with real data.")