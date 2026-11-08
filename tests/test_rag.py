from html import escape
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest

from routing.context import format_incident_context
from routing.llm import LLMRoutingResult, get_routing_response
from routing.prompt import build_routing_prompt
from routing.rag import recommend_with_rag


@pytest.fixture(autouse=True)
def prevent_external_calls(monkeypatch):
    """Unexpected use of real model loading or the SDK fails immediately."""
    monkeypatch.setattr("routing.rag.IncidentSearch", Mock(side_effect=AssertionError("No real model loading")))
    monkeypatch.setattr("routing.llm.OpenAI", Mock(side_effect=AssertionError("No real API calls")))


@pytest.fixture
def incidents():
    return pd.DataFrame([{
        "incident_id": "RAG-1",
        "short_description": "Client cannot start",
        "description": "The installed client fails before authentication.",
        "assignment_group": "End User Computing",
        "resolution_summary": "Excluded repair details.",
        "similarity_score": 0.87654321,
    }])


@pytest.mark.parametrize("result", [
    LLMRoutingResult("recommended", "End User Computing", "The evidence indicates a local client issue."),
    LLMRoutingResult("manual_review", None, "The evidence does not establish ownership."),
])
def test_retrieves_formats_builds_prompt_and_returns_client_result(monkeypatch, incidents, result):
    search = Mock()
    search.search.return_value = incidents
    client = Mock(return_value=result)
    formatter = Mock(wraps=format_incident_context)
    prompt_builder = Mock(wraps=build_routing_prompt)
    monkeypatch.setattr("routing.prompt.format_incident_context", formatter)
    monkeypatch.setattr("routing.rag.build_routing_prompt", prompt_builder)
    original = incidents.copy(deep=True)
    query = "The desktop client closes before sign-in."

    actual = recommend_with_rag(f"  {query}\n", search=search, llm_client=client)

    assert actual is result
    search.search.assert_called_once_with(query, top_k=5)
    assert formatter.call_count == prompt_builder.call_count == client.call_count == 1
    assert formatter.call_args.args[0] is incidents
    assert prompt_builder.call_args.args == (query, incidents)
    prompt, supplied_evidence = client.call_args.args
    assert supplied_evidence is incidents
    assert query in prompt
    assert escape(format_incident_context(incidents), quote=False) in prompt
    assert "Excluded repair details." not in prompt
    assert "0.87654321" not in prompt
    pd.testing.assert_frame_equal(incidents, original)


@pytest.mark.parametrize("query", ["", " \n\t "])
def test_blank_input_fails_before_any_component_is_used(query):
    search, client = Mock(), Mock()

    with pytest.raises(ValueError, match="must not be blank"):
        recommend_with_rag(query, search=search, llm_client=client)
    search.search.assert_not_called()
    client.assert_not_called()
    # Also check the default path does not initialize a model for blank input.
    with pytest.raises(ValueError, match="must not be blank"):
        recommend_with_rag(query)


def test_empty_retrieval_returns_manual_review_without_prompt_or_llm(monkeypatch):
    search, client = Mock(), Mock()
    search.search.return_value = pd.DataFrame()
    builder = Mock(side_effect=AssertionError("No prompt needed"))
    monkeypatch.setattr("routing.rag.build_routing_prompt", builder)

    result = recommend_with_rag("A service is unavailable.", search=search, llm_client=client)

    assert result.status == "manual_review"
    assert result.assignment_group is None
    assert "No historical incidents" in result.reason
    builder.assert_not_called()
    client.assert_not_called()

    # The default client also remains unused, without requiring API configuration.
    assert recommend_with_rag("A service is unavailable.", search=search) == result


def test_default_components_are_connected_without_loading_a_real_model(monkeypatch, incidents):
    search = Mock()
    search.search.return_value = incidents
    constructor = Mock(return_value=search)
    result = LLMRoutingResult("manual_review", None, "Further investigation is needed.")
    client = Mock(return_value=result)
    monkeypatch.setattr("routing.rag.IncidentSearch", constructor)
    monkeypatch.setattr("routing.rag.get_routing_response", client)

    assert recommend_with_rag("The desktop client cannot start.") is result
    constructor.assert_called_once_with()
    search.search.assert_called_once_with("The desktop client cannot start.", top_k=5)
    assert client.call_args.args[1] is incidents


@pytest.mark.parametrize("group, valid", [("End User Computing", True), ("Invented Team", False)])
def test_complete_workflow_uses_real_llm_validation_with_mocked_sdk(monkeypatch, incidents, group, valid):
    import json

    for name, value in {
        "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com",
        "AZURE_OPENAI_API_KEY": "fake-unit-test-key",
        "AZURE_OPENAI_DEPLOYMENT": "routing-deployment",
    }.items():
        monkeypatch.setenv(name, value)
    constructor = MagicMock()
    sdk_client = constructor.return_value.__enter__.return_value
    sdk_client.chat.completions.create.return_value = SimpleNamespace(choices=[
        SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "status": "recommended", "assignment_group": group,
            "reason": "The historical incident describes a local client issue.",
        }))),
    ])
    monkeypatch.setattr("routing.llm.OpenAI", constructor)
    search = Mock()
    search.search.return_value = incidents

    if valid:
        result = recommend_with_rag("The desktop client cannot start.", search=search, llm_client=get_routing_response)
        assert isinstance(result, LLMRoutingResult)
        assert result.assignment_group == "End User Computing"
    else:
        with pytest.raises(ValueError, match="must appear in the supplied historical evidence"):
            recommend_with_rag("The desktop client cannot start.", search=search, llm_client=get_routing_response)
    sdk_client.chat.completions.create.assert_called_once()


def test_retrieval_failure_propagates_without_calling_llm():
    search, client = Mock(), Mock()
    search.search.side_effect = RuntimeError("Retrieval unavailable")

    with pytest.raises(RuntimeError, match="Retrieval unavailable"):
        recommend_with_rag("A service is unavailable.", search=search, llm_client=client)
    client.assert_not_called()


def test_llm_failure_propagates(incidents):
    search = Mock()
    search.search.return_value = incidents
    client = Mock(side_effect=RuntimeError("LLM unavailable"))

    with pytest.raises(RuntimeError, match="LLM unavailable"):
        recommend_with_rag("A client cannot start.", search=search, llm_client=client)
