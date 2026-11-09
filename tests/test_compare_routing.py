"""Exercise the real baseline/RAG wiring with fake retrieval and a mocked SDK."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

from openai import OpenAIError
import pandas as pd
import pytest

from scripts import compare_routing


@pytest.fixture(autouse=True)
def offline_components(monkeypatch):
    for name, value in {
        "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com",
        "AZURE_OPENAI_API_KEY": "secret-never-print",
        "AZURE_OPENAI_DEPLOYMENT": "gpt-4.1-mini",
    }.items():
        monkeypatch.setenv(name, value)
    search = Mock()
    search.search.return_value = pd.DataFrame([{
        "incident_id": "FAKE-1", "short_description": "Local hardware problem",
        "description": "A device driver crashes.",
        "assignment_group": "End User Computing", "similarity_score": 0.8,
    }])
    factory = Mock(return_value=search)
    monkeypatch.setattr(compare_routing, "IncidentSearch", factory)
    monkeypatch.setattr("routing.rag.IncidentSearch", Mock(side_effect=AssertionError("No model loading")))
    constructor = MagicMock()
    client = constructor.return_value.__enter__.return_value
    client.chat.completions.create.return_value = SimpleNamespace(choices=[
        SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "status": "recommended", "assignment_group": "End User Computing",
            "reason": "The retrieved evidence describes a device driver failure.",
        }))),
    ])
    monkeypatch.setattr("routing.llm.OpenAI", constructor)
    return search, factory, constructor, client


def test_six_cases_use_real_components_and_disable_retries(offline_components, capsys):
    search, factory, constructor, client = offline_components

    assert compare_routing.main([]) == 0

    factory.assert_called_once_with()
    assert constructor.call_count == client.chat.completions.create.call_count == 6
    assert all(call.kwargs["max_retries"] == 0 for call in constructor.call_args_list)
    for number, (description, _) in enumerate(compare_routing.INCIDENTS):
        # Baseline and RAG use the same query, search instance, and retrieval settings.
        calls = search.search.call_args_list[2 * number:2 * number + 2]
        assert len(calls) == 2
        assert all(call.args == (description,) and call.kwargs == {"top_k": 5} for call in calls)
        request = client.chat.completions.create.call_args_list[number].kwargs
        assert request["model"] == "gpt-4.1-mini"
        assert description in request["messages"][0]["content"]
        assert "Expected group" not in request["messages"][0]["content"]
    captured = capsys.readouterr()
    output = " ".join(captured.out.split())
    for label in ("Incident summary", "Expected group", "Baseline", "LLM recommendation",
                  "LLM explanation", "Unclear; manual review", "driver failure."):
        assert label in output
    assert not captured.err


def test_suggestions_and_manual_review_are_displayed(offline_components, capsys):
    search, _, _, client = offline_components
    evidence = search.search.return_value
    search.search.return_value = pd.concat([
        evidence, evidence.assign(incident_id="FAKE-2", assignment_group="Network Operations"),
    ], ignore_index=True)
    client.chat.completions.create.return_value.choices[0].message.content = json.dumps({
        "status": "manual_review", "assignment_group": None,
        "reason": "Please clarify which screen disappears and provide the error message.",
    })

    assert compare_routing.main([]) == 0

    output = " ".join(capsys.readouterr().out.split())
    assert "Suggested:" in output
    assert "Network Operations" in output
    assert "Manual review" in output
    assert "Please clarify" in output


@pytest.mark.parametrize("error, label", [
    (OpenAIError("secret-never-print"), "API error"),
    (ValueError("secret-never-print"), "Validation error"),
    (RuntimeError("secret-never-print"), "Pipeline error"),
])
def test_errors_are_safe_and_do_not_add_requests(offline_components, capsys, error, label):
    _, _, _, client = offline_components
    client.chat.completions.create.side_effect = error

    assert compare_routing.main([]) == 1

    assert client.chat.completions.create.call_count == 6
    output = capsys.readouterr()
    assert label in output.out
    assert "Case 6:" in output.err
    assert "secret-never-print" not in output.out + output.err
    assert "Traceback" not in output.err


@pytest.mark.parametrize("name", [
    "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT",
])
def test_missing_configuration_prevents_search_and_requests(monkeypatch, offline_components, capsys, name):
    monkeypatch.setenv(name, "   ")

    assert compare_routing.main([]) == 1

    offline_components[1].assert_not_called()
    offline_components[2].assert_not_called()
    output = capsys.readouterr()
    assert name in output.err
    assert "secret-never-print" not in output.out + output.err


def test_help_does_not_initialize_search_or_call_api(offline_components, capsys):
    with pytest.raises(SystemExit) as exc:
        compare_routing.main(["--help"])
    assert exc.value.code == 0
    offline_components[1].assert_not_called()
    offline_components[2].assert_not_called()
    assert "GPT-4.1 mini" in capsys.readouterr().out


@pytest.mark.parametrize("during_initialization", [False, True])
def test_local_failures_do_not_expose_details_or_call_api(offline_components, capsys, during_initialization):
    search, factory, constructor, _ = offline_components
    component = factory if during_initialization else search.search
    component.side_effect = RuntimeError("secret-never-print")

    assert compare_routing.main([]) == 1

    constructor.assert_not_called()
    output = capsys.readouterr()
    assert output.err
    assert "secret-never-print" not in output.out + output.err


def test_no_evidence_uses_manual_review_without_requests(offline_components, capsys):
    offline_components[0].search.return_value = pd.DataFrame()

    assert compare_routing.main([]) == 0

    offline_components[2].assert_not_called()
    assert "Manual review" in capsys.readouterr().out


def test_request_budget_still_holds_if_more_cases_are_added(monkeypatch, offline_components):
    monkeypatch.setattr(compare_routing, "INCIDENTS", compare_routing.INCIDENTS * 2)

    assert compare_routing.main([]) == 0

    assert offline_components[3].chat.completions.create.call_count == 6


def test_six_new_fictional_descriptions_are_separate_from_datasets():
    cases = compare_routing.INCIDENTS
    assert len(cases) == len({description for description, _ in cases}) == 6
    assert any(expected is None for _, expected in cases)
    root = Path(__file__).resolve().parents[1]
    historical_texts = set()
    for filename in ("incidents.csv", "evaluation_incidents.csv"):
        with (root / "data" / filename).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                historical_texts.add(row["description"].strip().casefold())
    assert all(description.strip().casefold() not in historical_texts for description, _ in cases)
