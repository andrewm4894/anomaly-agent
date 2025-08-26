import pytest
from langchain_core.runnables import RunnableLambda

from anomaly_agent import Anomaly, AnomalyList
from anomaly_agent.prompt import DEFAULT_SYSTEM_PROMPT, DEFAULT_VERIFY_SYSTEM_PROMPT


@pytest.fixture(autouse=True)
def mock_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch LLM calls to avoid external API usage in tests."""

    class DummyLLM:
        """Minimal stand-in for ChatOpenAI."""

    def _make_anomaly(time_series: str) -> AnomalyList:
        lines = time_series.strip().splitlines()
        first_data = lines[1] if len(lines) > 1 else ""
        parts = first_data.split()
        if len(parts) >= 3 and ":" in parts[1]:
            timestamp = f"{parts[0]} {parts[1]}"
            value_token = parts[2]
        elif len(parts) >= 2:
            timestamp = parts[0]
            value_token = parts[1]
        else:
            timestamp = "1970-01-01 00:00:00"
            value_token = "0"
        try:
            value = float(value_token)
        except ValueError:
            value = 0.0
        return AnomalyList(
            anomalies=[
                Anomaly(
                    timestamp=timestamp,
                    variable_value=value,
                    anomaly_description="dummy",
                )
            ]
        )

    def fake_detection_chain(llm, detection_prompt: str = DEFAULT_SYSTEM_PROMPT) -> RunnableLambda:
        return RunnableLambda(lambda inputs: _make_anomaly(inputs["time_series"]))

    def fake_verification_chain(
        llm, verification_prompt: str = DEFAULT_VERIFY_SYSTEM_PROMPT
    ) -> RunnableLambda:
        return RunnableLambda(lambda inputs: _make_anomaly(inputs["time_series"]))

    monkeypatch.setattr("anomaly_agent.agent.ChatOpenAI", lambda *args, **kwargs: DummyLLM())
    monkeypatch.setattr("anomaly_agent.tools.create_detection_chain", fake_detection_chain)
    monkeypatch.setattr("anomaly_agent.tools.create_verification_chain", fake_verification_chain)
    monkeypatch.setattr("anomaly_agent.nodes.create_detection_chain", fake_detection_chain)
    monkeypatch.setattr("anomaly_agent.nodes.create_verification_chain", fake_verification_chain)
