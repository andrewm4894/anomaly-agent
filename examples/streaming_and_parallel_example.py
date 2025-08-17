"""
Example demonstrating streaming and parallel processing capabilities in the AnomalyAgent.

This example shows how to use the new streaming and parallel processing features
for improved user experience and performance with multiple time series variables.
"""

import asyncio
import pandas as pd
import time
from anomaly_agent import AnomalyAgent
from anomaly_agent.streaming import StreamingProgressHandler

# Create sample multi-variable time series data
print("=== Creating Sample Data ===")
data = {
    'timestamp': pd.date_range('2023-01-01', periods=50, freq='D'),
    'temperature': [20.0 + i * 0.1 + (15.0 if i == 25 else 0) for i in range(50)],  # Spike at day 25
    'pressure': [1013.0 + i * 0.2 + (-50.0 if i == 35 else 0) for i in range(50)],  # Drop at day 35
    'humidity': [50.0 + i * 0.3 + (25.0 if i == 15 else 0) for i in range(50)],     # Spike at day 15
    'wind_speed': [10.0 + i * 0.05 + (-8.0 if i == 40 else 0) for i in range(50)]  # Drop at day 40
}
df = pd.DataFrame(data)

print(f"Created DataFrame with {len(df.columns)-1} variables and {len(df)} time points")
print("Variables:", [col for col in df.columns if col != 'timestamp'])
print()

# Example 1: Streaming with Progress Callbacks
print("=== Example 1: Streaming Detection with Progress Callbacks ===")

# Use the built-in progress handler for clean output
progress_handler = StreamingProgressHandler(verbose=True, use_emojis=True)

# Create agent with multiple verification steps for better accuracy
agent = AnomalyAgent(
    model_name="gpt-5-nano",
    verify_anomalies=True,
    n_verify_steps=2,  # Use 2 verification steps
    debug=True
)

print("Running streaming detection...")
start_time = time.time()

# Run streaming detection
anomalies_streaming = agent.detect_anomalies_streaming(
    df, 
    progress_callback=progress_handler
)

streaming_time = time.time() - start_time
print(f"\nStreaming detection completed in {streaming_time:.2f} seconds")
print(f"Total anomalies found: {sum(len(al.anomalies) for al in anomalies_streaming.values())}")

# Show summary using the progress handler
progress_handler.summary()
print()

# Example 2: Parallel Processing
print("=== Example 2: Parallel Processing ===")

async def parallel_example():
    """Demonstrate parallel processing capabilities."""
    
    def parallel_progress_handler(column, event, data):
        """Handle progress updates for parallel processing."""
        timestamp = time.strftime("%H:%M:%S")
        if event == "start":
            print(f"[{timestamp}] 🚀 Started '{column}'")
        elif event == "column_complete":
            count = data['anomaly_count']
            time_taken = data['processing_time']
            print(f"[{timestamp}] ✅ '{column}' complete: {count} anomalies ({time_taken:.2f}s)")
        elif event == "error":
            print(f"[{timestamp}] ❌ '{column}' error: {data['error']}")

    print("Running parallel detection (max 3 concurrent tasks)...")
    start_time = time.time()
    
    # Run parallel detection
    anomalies_parallel = await agent.detect_anomalies_parallel(
        df,
        max_concurrent=3,  # Process up to 3 columns simultaneously
        progress_callback=parallel_progress_handler,
        n_verify_steps=1   # Use single verification for speed
    )
    
    parallel_time = time.time() - start_time
    print(f"\nParallel detection completed in {parallel_time:.2f} seconds")
    print(f"Total anomalies found: {sum(len(al.anomalies) for al in anomalies_parallel.values())}")
    
    # Performance comparison
    speedup = streaming_time / parallel_time
    print(f"Speedup: {speedup:.2f}x faster than streaming")
    return anomalies_parallel

# Run parallel example
anomalies_parallel = asyncio.run(parallel_example())
print()

# Example 3: Async Streaming Generator
print("=== Example 3: Async Streaming Generator ===")

async def async_streaming_example():
    """Demonstrate async streaming generator."""
    print("Running async streaming detection...")
    
    start_time = time.time()
    results = {}
    
    async for event in agent.detect_anomalies_streaming_async(df, n_verify_steps=1):
        if event["event"] == "start":
            total = event["data"]["total_columns"]
            columns = event["data"]["columns"]
            print(f"🎬 Starting async stream for {total} columns: {columns}")
        
        elif event["event"] == "progress":
            column = event["column"]
            status = event["data"]["status"]
            print(f"   📊 {column}: {status}")
        
        elif event["event"] == "result":
            column = event["column"]
            count = event["data"]["anomaly_count"]
            time_taken = event["data"]["processing_time"]
            print(f"   🎯 {column}: {count} anomalies ({time_taken:.2f}s)")
        
        elif event["event"] == "complete":
            results = event["data"]["results"]
            total_time = time.time() - start_time
            total_anomalies = sum(len(al.anomalies) for al in results.values())
            print(f"🏁 Async streaming complete: {total_anomalies} anomalies in {total_time:.2f}s")
        
        elif event["event"] == "error":
            print(f"❌ Error: {event['data']['error']}")
    
    return results

# Run async streaming example
asyncio.run(async_streaming_example())
print()

# Example 4: Compare Results
print("=== Example 4: Results Summary ===")

def summarize_results(anomalies_dict, method_name):
    """Summarize anomaly detection results."""
    print(f"\n{method_name} Results:")
    total_anomalies = 0
    for column, anomaly_list in anomalies_dict.items():
        count = len(anomaly_list.anomalies)
        total_anomalies += count
        if count > 0:
            timestamps = [a.timestamp for a in anomaly_list.anomalies]
            print(f"  {column}: {count} anomalies at {timestamps}")
        else:
            print(f"  {column}: No anomalies detected")
    print(f"  Total: {total_anomalies} anomalies")
    return total_anomalies

# Compare all methods
streaming_total = summarize_results(anomalies_streaming, "Streaming Detection")
parallel_total = summarize_results(anomalies_parallel, "Parallel Processing")

print("\n=== Performance & Feature Comparison ===")
print("┌─────────────────────┬─────────────────┬───────────────────┐")
print("│ Method              │ Processing Time │ Key Benefits      │")
print("├─────────────────────┼─────────────────┼───────────────────┤")
print(f"│ Streaming           │ {streaming_time:>13.2f}s │ Real-time updates │")
print(f"│ Parallel            │ {parallel_time:>13.2f}s │ Faster execution  │")
print("└─────────────────────┴─────────────────┴───────────────────┘")

print("\n🚀 New Capabilities Added:")
print("✓ Real-time progress updates with streaming")
print("✓ Parallel processing for multiple time series")
print("✓ Configurable concurrency limits")
print("✓ Async streaming generators for responsive UIs")
print("✓ Error handling and recovery in parallel execution")
print("✓ Performance monitoring with timing metrics")

print("\n💡 Use Cases:")
print("• Streaming: Interactive dashboards, real-time monitoring")
print("• Parallel: Batch processing, large datasets, performance-critical applications")
print("• Async Streaming: Web applications, reactive UIs, progressive data loading")

if __name__ == "__main__":
    print("\n🎉 Streaming and parallel processing example complete!")
    print("Set OPENAI_API_KEY environment variable to test with real data.")