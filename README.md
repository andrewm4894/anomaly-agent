# Anomaly Agent

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/andrewm4894/anomaly-agent)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Tests](https://github.com/andrewm4894/anomaly-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/andrewm4894/anomaly-agent/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/andrewm4894/anomaly-agent/branch/main/graph/badge.svg)](https://codecov.io/gh/andrewm4894/anomaly-agent)

<a target="_blank" href="https://pypi.org/project/anomaly-agent">
  <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/anomaly-agent">
</a>
<a target="_blank" href="https://pypi.org/project/anomaly-agent">
  <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/anomaly-agent">
</a>
<a target="_blank" href="https://pypi.org/project/anomaly-agent">
  <img alt="PyPI - License" src="https://img.shields.io/pypi/l/anomaly-agent">
</a>
<a target="_blank" href="https://pypi.org/project/anomaly-agent">
  <img alt="PyPI - Status" src="https://img.shields.io/pypi/status/anomaly-agent">
</a>
<a target="_blank" href="https://colab.research.google.com/github/andrewm4894/anomaly-agent/blob/main/examples/examples.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

🤖 A powerful Python library for detecting anomalies in time series data using Large Language Models (LLMs). Built with modern LangGraph architecture for robust, scalable anomaly detection across multiple variables and domains.

## ✨ Key Features

- 🧠 **LLM-Powered Detection**: Leverages advanced language models for intelligent anomaly identification
- 🔄 **Two-Stage Pipeline**: Detection and optional verification phases to reduce false positives  
- 📊 **Multi-Variable Support**: Analyze multiple time series variables simultaneously
- 🎯 **Domain Awareness**: Contextual understanding of different data types and domains
- ⚡ **Modern Architecture**: Built on LangGraph with Pydantic validation and robust error handling
- 🛠️ **Customizable**: Custom prompts, configurable verification, and flexible model selection
- 📈 **Rich Output**: Structured anomaly descriptions with timestamps and confidence indicators

## 🚀 Installation

```bash
pip install anomaly-agent
```

## 🏗️ How It Works

The Anomaly Agent uses a sophisticated two-stage pipeline powered by LangGraph state machines:

```mermaid
graph TD
    A[📊 Input Time Series Data] --> B[🔍 Detection Stage]
    B --> C{📋 Verification Enabled?}
    C -->|Yes| D[✅ Verification Stage]
    C -->|No| E[📤 Output Anomalies]
    D --> F{🎯 Anomalies Confirmed?}
    F -->|Yes| E
    F -->|No| G[❌ Filtered Out]
    G --> E
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style D fill:#fff3e0
    style E fill:#e8f5e8
```

### 🔧 Architecture Components

1. **🔍 Detection Node**: Uses LLM to identify potential anomalies with statistical and contextual analysis
2. **✅ Verification Node** (Optional): Secondary LLM review to reduce false positives with stricter criteria
3. **🎯 State Management**: Pydantic-based validation and error handling throughout the pipeline
4. **📊 Multi-Variable Processing**: Parallel analysis of multiple time series columns

## ⚡ Quick Start

### Basic Usage

```python
import pandas as pd
from anomaly_agent import AnomalyAgent

# Your time series data
df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=100, freq='D'),
    'temperature': [20 + i*0.1 + (10 if i==50 else 0) for i in range(100)],
    'pressure': [1013 + i*0.2 + (50 if i==75 else 0) for i in range(100)]
})

# Create agent and detect anomalies
agent = AnomalyAgent()
anomalies = agent.detect_anomalies(df)

# Convert to DataFrame for analysis
df_anomalies = agent.get_anomalies_df(anomalies)
print(df_anomalies)
```

### Advanced Configuration

```python
from anomaly_agent import AnomalyAgent

# Customize model and verification behavior
agent = AnomalyAgent(
    model_name="gpt-4o-mini",           # Choose your preferred model
    verify_anomalies=True,              # Enable verification stage
    timestamp_col="date"                # Custom timestamp column name
)

# Custom prompts for domain-specific detection
financial_detection_prompt = """
You are a financial analyst detecting market anomalies.
Focus on: unusual price movements, volume spikes, trend reversals.
Consider market hours and economic events in your analysis.
"""

agent = AnomalyAgent(detection_prompt=financial_detection_prompt)
anomalies = agent.detect_anomalies(financial_data)
```

## 📚 Examples and Notebooks

### 📁 Examples Directory

Explore comprehensive examples in the `examples/` folder:

- **[`examples.py`](examples/examples.py)**: Complete CLI examples with different scenarios
- **[`custom_prompts_example.py`](examples/custom_prompts_example.py)**: Domain-specific prompt customization
- **[`examples.ipynb`](examples/examples.ipynb)**: Interactive Jupyter notebook with visualizations

### 🎮 Interactive Examples

```bash
# Run basic example
python examples/examples.py --example basic --plot

# Try real-world sensor data scenario  
python examples/examples.py --example real-world --plot

# Custom model and plotting
python examples/examples.py --model gpt-4o-mini --example multiple --plot
```

### 📓 Jupyter Notebooks

Launch the interactive notebook:
- **Local**: Open `examples/examples.ipynb`
- **Colab**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andrewm4894/anomaly-agent/blob/main/examples/examples.ipynb)

## 📊 Output Formats

### Long Format (Default)
```python
df_anomalies = agent.get_anomalies_df(anomalies)
```
| timestamp | variable_name | value | anomaly_description |
|-----------|---------------|-------|-------------------|
| 2024-01-15 | temperature | 35.2 | Significant temperature spike... |
| 2024-01-20 | pressure | 1089.3 | Unusual pressure reading... |

### Wide Format
```python
df_anomalies = agent.get_anomalies_df(anomalies, format="wide")
```
| timestamp | temperature | temperature_description | pressure | pressure_description |
|-----------|-------------|------------------------|----------|---------------------|
| 2024-01-15 | 35.2 | Significant spike... | NaN | NaN |
| 2024-01-20 | NaN | NaN | 1089.3 | Unusual reading... |

## 🎛️ Model Configuration

Choose the right model for your needs and budget:

| Model | Cost (Input/Output per 1M tokens) | Best For | Performance |
|-------|-----------------------------------|----------|-------------|
| `gpt-5-nano` | $0.05 / $0.40 | Cost-effective anomaly detection | ⭐⭐⭐ |
| `gpt-5-mini` | $0.25 / $2.00 | Enhanced reasoning for complex patterns | ⭐⭐⭐⭐ |
| `gpt-5` | $1.25 / $10.00 | Sophisticated domain-specific analysis | ⭐⭐⭐⭐⭐ |
| `gpt-4o-mini` | $0.60 / $2.40 | Legacy support with good performance | ⭐⭐⭐⭐ |

```python
# Cost-optimized (default)
agent = AnomalyAgent(model_name="gpt-5-nano")

# Enhanced reasoning
agent = AnomalyAgent(model_name="gpt-5-mini") 

# Premium analysis
agent = AnomalyAgent(model_name="gpt-5")
```

## 🎯 Use Cases

### 🏢 Business & Operations
- **📈 Sales Analytics**: Detect unusual sales patterns, seasonal anomalies
- **🏭 Manufacturing**: Monitor equipment performance, quality metrics
- **💰 Financial Services**: Fraud detection, market anomaly identification
- **🌐 Web Analytics**: Traffic spikes, user behavior anomalies

### 🔬 Science & Engineering  
- **🌡️ IoT Sensors**: Temperature, humidity, pressure monitoring
- **⚡ Energy Systems**: Power consumption, grid stability analysis
- **🩺 Healthcare**: Patient monitoring, medical device readings
- **🌍 Environmental**: Weather patterns, pollution levels

### 📊 Data Quality
- **🔍 Data Validation**: Identify measurement errors, sensor failures
- **📋 ETL Monitoring**: Pipeline anomalies, data drift detection
- **🎯 Quality Assurance**: Automated anomaly flagging in data workflows

## 🛠️ Development

This project uses **uv** for fast, reliable dependency management. All commands automatically handle virtual environment management.

### 🏗️ Setup

```bash
# Clone the repository
git clone https://github.com/andrewm4894/anomaly-agent.git
cd anomaly-agent

# Install dependencies (creates .venv automatically)
make sync-dev
```

### 🧪 Testing

```bash
# Run all tests with coverage
make test

# Run specific test categories
uv run pytest tests/test_agent.py -v                    # Core functionality
uv run pytest tests/test_prompts.py -v                  # Prompt system
uv run pytest tests/test_graph_architecture.py -v       # Advanced architecture

# Integration tests (requires OPENAI_API_KEY in .env)
uv run pytest tests/ -m integration -v
```

### 📋 Code Quality

```bash
# Install pre-commit hooks
make pre-commit-install

# Run all quality checks
make pre-commit

# Individual tools
uv run black anomaly_agent/    # Formatting
uv run isort anomaly_agent/    # Import sorting  
uv run flake8 anomaly_agent/   # Linting
uv run mypy anomaly_agent/     # Type checking
```

### 📦 Dependencies

```bash
# Add new dependencies
make add PACKAGE=pandas              # Runtime dependency
make add-dev PACKAGE=pytest          # Development dependency

# Update all dependencies
make update

# Remove dependencies
make remove PACKAGE=old-package
```

## ⚙️ Environment Setup

Create a `.env` file in your project root:

```bash
# Required for anomaly detection
OPENAI_API_KEY=your-openai-api-key-here

# Optional: Custom model defaults
DEFAULT_MODEL_NAME=gpt-5-nano
```

The agent automatically loads environment variables via python-dotenv.

## 🏗️ Architecture Deep Dive

For detailed technical information about the internal architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).

Key architectural features:
- **🔧 LangGraph State Machines**: Robust workflow management with proper error handling
- **✅ Pydantic Validation**: Type-safe data models throughout the pipeline  
- **🎯 GraphManager Caching**: Optimized performance with reusable compiled graphs
- **📊 Class-based Nodes**: Modular, maintainable node architecture
- **🔄 Async Support**: Streaming and parallel processing capabilities

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details.

1. **🍴 Fork** the repository
2. **🌿 Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **✅ Test** your changes (`make test`)
4. **📝 Commit** your changes (`git commit -m 'Add amazing feature'`)
5. **🚀 Push** to the branch (`git push origin feature/amazing-feature`)
6. **🎯 Open** a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by OpenAI's language models
- Inspired by the need for intelligent, contextual anomaly detection
