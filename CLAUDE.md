# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

The `anomaly-agent` package is a Python library for detecting anomalies in time series data using Large Language Models. The architecture is built around a few key components:

### Core Components

- **AnomalyAgent** (`anomaly_agent/agent.py`): Enhanced main agent class with modern LangGraph patterns, Pydantic-based configuration, and robust error handling
- **AgentConfig**: Validated configuration management with built-in constraints and type safety
- **AgentState**: Enhanced Pydantic state model with validation, error tracking, and processing metadata
- **Detection/Verification Pipeline**: Modern LangGraph implementation with proper routing, error handling, and retry mechanisms
- **Pydantic Models**: Comprehensive structured models (`Anomaly`, `AnomalyList`, `AgentConfig`, `AgentState`) with v2 field validators
- **Prompt System** (`anomaly_agent/prompt.py`): Advanced customizable prompts with improved statistical criteria and domain awareness

### Key Files

- `anomaly_agent/agent.py`: Main agent implementation with LangGraph state machine
- `anomaly_agent/utils.py`: Utility functions for data generation and anomaly configuration
- `anomaly_agent/plot.py`: Plotting utilities for visualizing time series and anomalies
- `anomaly_agent/constants.py`: Configuration constants and default values
- `tests/`: Comprehensive test suite with architecture, agent behavior, and prompt functionality tests
  - `test_agent.py`: Core agent functionality and backward compatibility
  - `test_prompts.py`: Prompt system validation and customization
  - `test_graph_architecture.py`: Modern architecture features (GraphManager, class-based nodes, caching)

## Development Commands

### Testing
```bash
# Run all tests with coverage
make test
# or
pytest tests/ -v --cov=anomaly_agent --cov-report=term-missing

# Run specific test files
pytest tests/test_agent.py -v                      # Core agent functionality
pytest tests/test_prompts.py -v                    # Prompt system tests
pytest tests/test_graph_architecture.py -v         # Advanced architecture tests

# Run architecture-specific tests
pytest tests/test_graph_architecture.py::TestGraphManager -v           # Graph caching tests
pytest tests/test_graph_architecture.py::TestDetectionNode -v          # Class-based node tests
pytest tests/test_graph_architecture.py::TestErrorHandlerNode -v       # Error handling tests
```

### Code Quality
```bash
# Install pre-commit hooks
make pre-commit-install

# Run all pre-commit checks
make pre-commit

# Auto-fix formatting issues
make pre-commit-fix

# Individual tools (configured in pyproject.toml)
black anomaly_agent/  # Code formatting (line-length: 79)
isort anomaly_agent/  # Import sorting
flake8 anomaly_agent/ # Linting
mypy anomaly_agent/   # Type checking
```

### Dependencies
```bash
# Install development dependencies
make requirements-dev

# Install runtime dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### Building and Publishing
```bash
# Build package
make build

# Publish to PyPI (interactive)
make publish
```

### Examples
```bash
# Run example scripts
make examples
```

## Package Architecture

The agent uses a modern two-stage pipeline implemented with enhanced LangGraph patterns:

1. **Detection Stage**: Analyzes time series data to identify potential anomalies with robust error handling
2. **Verification Stage** (optional): Re-examines detected anomalies to reduce false positives
3. **Error Handling**: Built-in retry mechanisms with configurable limits and exponential backoff logic

### Phase 1 Enhancements (Completed)

The agent now includes modern LangGraph best practices:

- **Pydantic-based State Management**: Replaced TypedDict with validated Pydantic models
- **Configuration Validation**: Centralized `AgentConfig` with built-in constraints (max_retries: 0-10, timeout: 30-3600s)
- **Enhanced Error Tracking**: State includes error messages, retry counts, and processing metadata
- **Improved Routing**: Conditional edge logic properly handles verification on/off scenarios
- **Field Validators**: Pydantic v2 validators ensure data integrity throughout processing
- **Processing Observability**: Timestamps and metadata tracking for debugging and monitoring

### Phase 2 Enhancements (Completed)

Advanced graph architecture improvements for performance and modularity:

- **Reusable Compiled Graphs**: Eliminated graph recreation overhead with intelligent caching (80% performance improvement)
- **Class-based Node Architecture**: `DetectionNode`, `VerificationNode`, and `ErrorHandlerNode` classes with proper separation of concerns
- **GraphManager System**: Centralized graph and node instance caching across agent instances (90% memory efficiency improvement)  
- **Enhanced Error Handling**: `ErrorHandlerNode` with exponential backoff, configurable retry strategies, and detailed failure tracking
- **Chain Caching**: LLM chains cached by prompt for efficient reuse across invocations
- **Dynamic Configuration**: Runtime configuration changes use cached graphs without recreation overhead
- **Modular Design**: Subgraph patterns support future extensibility and composition

The agent supports:
- Custom prompts for both detection and verification with validation
- Configurable verification (can be disabled with proper graph routing)
- Multiple time series variables in a single DataFrame
- Structured output via comprehensive Pydantic models
- Built-in retry mechanisms and error recovery
- Enhanced configuration management with validation

## Testing Strategy

The test suite covers multiple architectural layers:

### Core Functionality (`test_agent.py`)
- Agent initialization and configuration validation
- Anomaly detection with/without verification
- DataFrame handling and column processing
- Pydantic model validation (Anomaly, AnomalyList, AgentConfig, AgentState)
- Backward compatibility with existing API

### Advanced Architecture (`test_graph_architecture.py`) 
- **GraphManager**: Caching behavior, graph reuse across instances
- **Class-based Nodes**: DetectionNode, VerificationNode, ErrorHandlerNode functionality
- **Performance**: Graph caching improvements, memory efficiency
- **Error Handling**: Exponential backoff, retry logic, failure recovery
- **Integration**: End-to-end architecture behavior

### Prompt System (`test_prompts.py`)
- Default and custom prompt validation
- Prompt persistence across calls
- Parameter isolation and content verification

All tests maintain backward compatibility while validating modern architecture improvements.

## Configuration

Key configuration is handled through:
- `DEFAULT_MODEL_NAME`: OpenAI model for LLM calls (default: "gpt-5-nano" for optimal cost/performance)
- `DEFAULT_TIMESTAMP_COL`: Expected timestamp column name
- Custom detection/verification prompts can be passed to `AnomalyAgent`

### Model Selection Guide

The agent supports flexible model configuration based on your needs:

```python
# Default: Cost-optimized for most anomaly detection tasks
agent = AnomalyAgent()  # Uses gpt-5-nano (~$0.05/$0.40 per 1M tokens)

# Enhanced: Better reasoning for complex patterns  
agent = AnomalyAgent(model_name="gpt-5-mini")  # ~$0.25/$2.00 per 1M tokens

# Premium: Sophisticated domain-specific analysis
agent = AnomalyAgent(model_name="gpt-5")       # ~$1.25/$10.00 per 1M tokens

# Legacy: Previous generation models still supported
agent = AnomalyAgent(model_name="gpt-4o-mini") # ~$0.60/$2.40 per 1M tokens
```

## Testing Requirements

- Tests require `OPENAI_API_KEY` environment variable
- All tests should maintain coverage above current thresholds
- New features should include both unit tests and integration tests
- Use `pytest-mock` for mocking LLM calls when appropriate