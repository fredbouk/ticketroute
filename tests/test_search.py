import numpy as np
import pandas as pd
import pytest

from routing.search import IncidentSearch


class FakeEmbeddingModel:
    """Provide predictable vectors without loading a pretrained model."""

    def __init__(self):
        self.calls = []

    def encode(self, texts, *, convert_to_numpy, normalize_embeddings):
        assert convert_to_numpy and normalize_embeddings
        self.calls.append(list(texts))
        vectors = {
            "Desktop unavailable\nDesktop session will not start.": [0.0, 2.0],
            "VPN unavailable\nVPN connection fails.": [3.0, 0.0],
            "Slow connection\nRemote session is slow.": [1.0, 1.0],
            "VPN query": [2.0, 0.0],
        }
        embeddings = np.array([vectors[text] for text in texts])
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


@pytest.fixture
def search_component(monkeypatch):
    incidents = pd.DataFrame([
        {
            "incident_id": "TEST-1",
            "short_description": "Desktop unavailable",
            "description": "Desktop session will not start.",
            "assignment_group": "Cloud Platform",
            "resolution_summary": "Repaired hosts.",
        },
        {
            "incident_id": "TEST-2",
            "short_description": "VPN unavailable",
            "description": "VPN connection fails.",
            "assignment_group": "Network Operations",
            "resolution_summary": "Restored route.",
        },
        {
            "incident_id": "TEST-3",
            "short_description": "Slow connection",
            "description": "Remote session is slow.",
            "assignment_group": "End User Computing",
            "resolution_summary": "Repaired client.",
        },
    ])
    monkeypatch.setattr("routing.search.load_incidents", lambda: incidents)
    return IncidentSearch(model=FakeEmbeddingModel())


def test_results_are_sorted_by_descending_cosine_similarity(search_component):
    results = search_component.search("VPN query")

    assert results["incident_id"].tolist() == ["TEST-2", "TEST-3", "TEST-1"]
    assert results["similarity_score"].tolist() == pytest.approx(
        [1.0, 1 / np.sqrt(2), 0.0]
    )
    assert results["similarity_score"].is_monotonic_decreasing
    assert results.columns.tolist() == [
        "incident_id", "short_description", "description", "assignment_group",
        "resolution_summary", "similarity_score",
    ]
    assert results.iloc[0]["assignment_group"] == "Network Operations"
    assert results.iloc[0]["resolution_summary"] == "Restored route."


@pytest.mark.parametrize("top_k, expected_count", [(1, 1), (2, 2), (3, 3), (10, 3)])
def test_requested_result_count(search_component, top_k, expected_count):
    assert len(search_component.search("VPN query", top_k=top_k)) == expected_count


def test_default_returns_five_results(search_component, monkeypatch):
    incidents = pd.concat([search_component.incidents] * 2, ignore_index=True)
    incidents["incident_id"] = [f"TEST-{i}" for i in range(6)]
    monkeypatch.setattr("routing.search.load_incidents", lambda: incidents)
    search = IncidentSearch(model=FakeEmbeddingModel())

    assert len(search.search("VPN query")) == 5


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_blank_queries_are_rejected(search_component, query):
    with pytest.raises(ValueError, match="non-empty incident description"):
        search_component.search(query)


def test_embeddings_use_only_incident_descriptions(search_component):
    assert search_component.model.calls == [[
        "Desktop unavailable\nDesktop session will not start.",
        "VPN unavailable\nVPN connection fails.",
        "Slow connection\nRemote session is slow.",
    ]]


def test_historical_embeddings_are_reused(search_component):
    search_component.search(" VPN query ")
    search_component.search("VPN query")

    assert len(search_component.model.calls) == 3
    assert search_component.model.calls[1:] == [["VPN query"], ["VPN query"]]


def test_empty_history_returns_empty_results(monkeypatch):
    columns = [
        "incident_id", "short_description", "description",
        "assignment_group", "resolution_summary",
    ]
    monkeypatch.setattr(
        "routing.search.load_incidents", lambda: pd.DataFrame(columns=columns)
    )
    search = IncidentSearch()

    results = search.search("VPN query")

    assert results.empty
    assert results.columns.tolist() == columns + ["similarity_score"]
    assert search.model is None
    with pytest.raises(ValueError, match="non-empty"):
        search.search(" ")


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_invalid_result_counts_are_rejected(search_component, top_k):
    with pytest.raises(ValueError, match="positive integer"):
        search_component.search("VPN query", top_k=top_k)
