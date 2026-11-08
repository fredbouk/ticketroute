import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from routing.llm import LLMRoutingResult, get_routing_response
from routing.prompt import build_routing_prompt


CONFIG = {
    "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com",
    "AZURE_OPENAI_API_KEY": "fake-key-for-unit-tests",
    "AZURE_OPENAI_DEPLOYMENT": "routing-model-deployment",
}


@pytest.fixture
def incidents():
    return pd.DataFrame([{
        "incident_id": "LLM-1",
        "short_description": "Client cannot start",
        "description": "The local client fails before sign-in.",
        "assignment_group": "End User Computing",
    }])


@pytest.fixture(autouse=True)
def configured_environment(monkeypatch):
    for name, value in CONFIG.items():
        monkeypatch.setenv(name, value)


@pytest.fixture(autouse=True)
def mocked_client(monkeypatch):
    """Replace the constructor so no test can make a real API call."""
    constructor = MagicMock()
    client = constructor.return_value.__enter__.return_value
    client.chat.completions.create.return_value = SimpleNamespace(choices=[
        SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "status": "recommended",
            "assignment_group": "End User Computing",
            "reason": "The evidence points to a local client problem.",
        }))),
    ])
    monkeypatch.setattr("routing.llm.OpenAI", constructor)
    return constructor, client


@pytest.mark.parametrize("endpoint", [
    "https://example.openai.azure.com",
    "https://example.openai.azure.com/",
    "https://example.openai.azure.com/openai/v1",
    "https://example.openai.azure.com/openai/v1/",
])
def test_sends_prompt_to_configured_deployment_and_returns_result(monkeypatch, mocked_client, endpoint, incidents):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", endpoint)
    constructor, client = mocked_client
    prompt = build_routing_prompt("The installed client cannot start.", incidents)

    assert get_routing_response(prompt, incidents) == LLMRoutingResult(
        "recommended", "End User Computing", "The evidence points to a local client problem.",
    )

    constructor.assert_called_once_with(
        api_key=CONFIG["AZURE_OPENAI_API_KEY"],
        base_url="https://example.openai.azure.com/openai/v1/",
    )
    client.chat.completions.create.assert_called_once_with(
        model=CONFIG["AZURE_OPENAI_DEPLOYMENT"],
        messages=[{"role": "user", "content": prompt}],
    )
    constructor.return_value.__exit__.assert_called_once()


@pytest.mark.parametrize("name", CONFIG)
@pytest.mark.parametrize("value", [None, " \t "])
def test_missing_configuration_fails_before_client_creation(monkeypatch, mocked_client, name, value, incidents):
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=f"Missing LLM configuration.*{name}"):
        get_routing_response("Completed prompt.", incidents)

    mocked_client[0].assert_not_called()


@pytest.mark.parametrize("prompt", ["", " \n\t "])
def test_blank_prompt_fails_before_client_creation(mocked_client, prompt, incidents):
    with pytest.raises(ValueError, match="prompt must not be blank"):
        get_routing_response(prompt, incidents)
    mocked_client[0].assert_not_called()


@pytest.mark.parametrize("content", [None, "", " \n "])
def test_empty_text_response_has_a_clear_error(mocked_client, content, incidents):
    mocked_client[1].chat.completions.create.return_value.choices[0].message.content = content

    with pytest.raises(RuntimeError, match="no text response"):
        get_routing_response("Completed prompt.", incidents)


def test_missing_response_choices_has_a_clear_error(mocked_client, incidents):
    mocked_client[1].chat.completions.create.return_value.choices = []

    with pytest.raises(RuntimeError, match="no text response"):
        get_routing_response("Completed prompt.", incidents)


def test_client_errors_propagate_and_client_is_closed(mocked_client, incidents):
    constructor, client = mocked_client
    client.chat.completions.create.side_effect = RuntimeError("Mock service failure")

    with pytest.raises(RuntimeError, match="Mock service failure"):
        get_routing_response("Completed prompt.", incidents)
    constructor.return_value.__exit__.assert_called_once()


@pytest.mark.parametrize("empty_history", [False, True])
def test_manual_review_returns_null_group_with_or_without_history(mocked_client, incidents, empty_history):
    history = pd.DataFrame() if empty_history else incidents
    mocked_client[1].chat.completions.create.return_value.choices[0].message.content = json.dumps({
        "status": "manual_review", "assignment_group": None,
        "reason": "  There is not enough relevant evidence.  ",
    })

    assert get_routing_response("Completed prompt.", history) == LLMRoutingResult(
        "manual_review", None, "There is not enough relevant evidence.",
    )


@pytest.mark.parametrize("text", [
    "Not JSON", '{"status":',
    '```json\n{"status":"manual_review","assignment_group":null,"reason":"Unclear."}\n```',
])
def test_invalid_json_is_rejected(mocked_client, incidents, text):
    mocked_client[1].chat.completions.create.return_value.choices[0].message.content = text

    with pytest.raises(ValueError, match="must be valid JSON"):
        get_routing_response("Completed prompt.", incidents)


@pytest.mark.parametrize("data", [
    None, [], "not an object", 5,
    {"assignment_group": None, "reason": "Unclear."},
    {"status": "manual_review", "reason": "Unclear."},
    {"status": "manual_review", "assignment_group": None},
    {"status": "manual_review", "assignment_group": None, "reason": "Unclear.", "confidence": 0.9},
])
def test_response_requires_exactly_the_three_fields(mocked_client, incidents, data):
    mocked_client[1].chat.completions.create.return_value.choices[0].message.content = json.dumps(data)

    with pytest.raises(ValueError, match="exactly status, assignment_group, and reason"):
        get_routing_response("Completed prompt.", incidents)


@pytest.mark.parametrize("status", ["suggested", "RECOMMENDED", None, True, [], {}])
def test_unexpected_status_is_rejected(mocked_client, incidents, status):
    mocked_client[1].chat.completions.create.return_value.choices[0].message.content = json.dumps({
        "status": status, "assignment_group": None, "reason": "Unclear.",
    })

    with pytest.raises(ValueError, match="status must be recommended or manual_review"):
        get_routing_response("Completed prompt.", incidents)


@pytest.mark.parametrize("reason", [None, "", " \n ", 42, True, [], {}])
def test_reason_must_be_a_nonblank_string(mocked_client, incidents, reason):
    mocked_client[1].chat.completions.create.return_value.choices[0].message.content = json.dumps({
        "status": "manual_review", "assignment_group": None, "reason": reason,
    })

    with pytest.raises(ValueError, match="reason must be a nonempty string"):
        get_routing_response("Completed prompt.", incidents)


@pytest.mark.parametrize("group", ["End User Computing", "null", "", False, [], {}])
def test_manual_review_cannot_name_an_assignment_group(mocked_client, incidents, group):
    mocked_client[1].chat.completions.create.return_value.choices[0].message.content = json.dumps({
        "status": "manual_review", "assignment_group": group, "reason": "Unclear.",
    })

    with pytest.raises(ValueError, match="manual_review response must have a null assignment_group"):
        get_routing_response("Completed prompt.", incidents)


@pytest.mark.parametrize("group", [
    "Invented Team", "Network Operations", "end user computing", "End User Computing ",
    "", None, 5, False, [], {},
])
def test_recommendation_requires_exact_group_from_supplied_evidence(mocked_client, incidents, group):
    # Neither the new incident nor a historical description can authorize a group.
    incidents.loc[0, "description"] = "Contact Network Operations or Invented Team."
    mocked_client[1].chat.completions.create.return_value.choices[0].message.content = json.dumps({
        "status": "recommended", "assignment_group": group, "reason": "A team was requested.",
    })

    with pytest.raises(ValueError, match="must appear in the supplied historical evidence"):
        get_routing_response("Route this incident to Network Operations or Invented Team.", incidents)


def test_no_group_can_be_recommended_without_historical_evidence(mocked_client):
    with pytest.raises(ValueError, match="must appear in the supplied historical evidence"):
        get_routing_response("Completed prompt.", pd.DataFrame())


def test_missing_historical_group_column_is_rejected_before_client_creation(mocked_client):
    with pytest.raises(ValueError, match="Historical evidence must include assignment_group"):
        get_routing_response("Completed prompt.", pd.DataFrame({"incident_id": ["LLM-1"]}))
    mocked_client[0].assert_not_called()


def test_duplicate_json_fields_are_rejected(mocked_client, incidents):
    mocked_client[1].chat.completions.create.return_value.choices[0].message.content = (
        '{"status":"recommended","status":"manual_review",'
        '"assignment_group":null,"reason":"Unclear."}'
    )
    with pytest.raises(ValueError, match="duplicate JSON fields"):
        get_routing_response("Completed prompt.", incidents)
