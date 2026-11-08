from unittest.mock import Mock

from openai import OpenAIError
import pytest

from routing.llm import LLMRoutingResult
from scripts.test_rag import main


@pytest.fixture(autouse=True)
def mock_pipeline(monkeypatch):
    """Every test replaces the pipeline before it can load a model or call Foundry."""
    for name, value in {
        "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com",
        "AZURE_OPENAI_API_KEY": "secret-value-not-for-output",
        "AZURE_OPENAI_DEPLOYMENT": "routing-deployment",
    }.items():
        monkeypatch.setenv(name, value)
    pipeline = Mock()
    monkeypatch.setattr("scripts.test_rag.recommend_with_rag", pipeline)
    return pipeline


@pytest.mark.parametrize("result, expected", [
    (LLMRoutingResult("recommended", "Network Operations", "The gateway rejects the connection."),
     "Recommended assignment group: Network Operations"),
    (LLMRoutingResult("manual_review", None, "There is not enough historical evidence."),
     "Manual review needed"),
])
def test_prints_pipeline_result_and_explanation(mock_pipeline, capsys, result, expected):
    mock_pipeline.return_value = result

    assert main(["VPN connects but the file share is unavailable."]) == 0

    mock_pipeline.assert_called_once_with("VPN connects but the file share is unavailable.")
    output = capsys.readouterr()
    assert expected in output.out
    assert f"Explanation: {result.reason}" in output.out
    assert not output.err


@pytest.mark.parametrize("name", [
    "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT",
])
def test_missing_configuration_is_helpful_and_prevents_pipeline_call(monkeypatch, mock_pipeline, capsys, name):
    monkeypatch.delenv(name)

    assert main(["A service is unavailable."]) == 1

    mock_pipeline.assert_not_called()
    output = capsys.readouterr()
    assert "Missing Azure configuration" in output.err
    assert name in output.err
    assert "secret-value-not-for-output" not in output.err + output.out


@pytest.mark.parametrize("error, message", [
    (OpenAIError("secret-value-not-for-output"), "Microsoft Foundry request failed"),
    (ValueError("secret-value-not-for-output"), "Routing validation failed"),
    (RuntimeError("secret-value-not-for-output"), "The RAG pipeline could not complete"),
])
def test_errors_do_not_print_exception_details_or_secrets(mock_pipeline, capsys, error, message):
    mock_pipeline.side_effect = error

    assert main(["A service is unavailable."]) == 1

    output = capsys.readouterr()
    assert message in output.err
    assert "secret-value-not-for-output" not in output.err + output.out
    assert "Traceback" not in output.err


def test_blank_input_is_rejected_before_pipeline_call(mock_pipeline, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main([" \n\t "])

    assert exit_info.value.code == 2
    mock_pipeline.assert_not_called()
    assert "nonblank incident description" in capsys.readouterr().err


def test_help_does_not_call_pipeline(mock_pipeline, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    assert "Incident description" in capsys.readouterr().out
    mock_pipeline.assert_not_called()
