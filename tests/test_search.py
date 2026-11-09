import sys
from types import SimpleNamespace

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


@pytest.fixture
def embedding_cache(search_component, monkeypatch, tmp_path):
    """Exercise the default model path without importing or downloading ML models."""
    settings = {"revision": "model-commit-one"}
    models = []

    class CacheModel(FakeEmbeddingModel):
        @property
        def model_card_data(self):
            return SimpleNamespace(base_model_revision=settings.get("card_revision"))

        def __getitem__(self, index):
            assert index == 0
            return SimpleNamespace(auto_model=SimpleNamespace(
                config=SimpleNamespace(_commit_hash=settings["revision"]),
            ))

        def get_sentence_embedding_dimension(self):
            return 2

    def create_model(name, *, device):
        assert device == "cpu"
        model = CacheModel()
        models.append(model)
        return model

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(
        SentenceTransformer=create_model,
    ))
    return SimpleNamespace(
        path=tmp_path / "cache" / "historical_embeddings.npz",
        incidents=search_component.incidents, models=models, settings=settings,
    )


def test_disk_cache_is_created(embedding_cache):
    cache = embedding_cache
    search = IncidentSearch(cache_path=cache.path)

    assert len(cache.models[0].calls) == 1
    with np.load(cache.path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["embeddings"], search.embeddings)
        assert len(saved["fingerprint"].item()) == 64
    assert list(cache.path.parent.iterdir()) == [cache.path]


def test_new_search_reuses_disk_cache_and_preserves_results(embedding_cache):
    cache = embedding_cache
    original = IncidentSearch(cache_path=cache.path)
    reused = IncidentSearch(cache_path=cache.path)

    assert cache.models[1].calls == []
    np.testing.assert_array_equal(original.embeddings, reused.embeddings)
    pd.testing.assert_frame_equal(original.search("VPN query"), reused.search("VPN query"))
    assert cache.models[1].calls == [["VPN query"]]


@pytest.mark.parametrize("change", ["text", "metadata", "order", "row_count"])
def test_disk_cache_invalidates_when_incidents_change(embedding_cache, monkeypatch, change):
    cache = embedding_cache
    IncidentSearch(cache_path=cache.path)
    incidents = cache.incidents.copy()
    if change == "text":
        incidents.loc[0, ["short_description", "description"]] = incidents.loc[
            1, ["short_description", "description"]
        ].to_numpy()
    elif change == "metadata":
        incidents.loc[0, "resolution_summary"] = "Updated synthetic resolution."
    elif change == "order":
        incidents = incidents.iloc[::-1].reset_index(drop=True)
    else:
        incidents = incidents.iloc[:2].copy()
    monkeypatch.setattr("routing.search.load_incidents", lambda: incidents)

    rebuilt = IncidentSearch(cache_path=cache.path)
    assert len(cache.models[1].calls) == 1
    uncached = IncidentSearch(model=FakeEmbeddingModel())
    pd.testing.assert_frame_equal(rebuilt.search("VPN query"), uncached.search("VPN query"))
    IncidentSearch(cache_path=cache.path)
    assert cache.models[2].calls == []


@pytest.mark.parametrize("change", ["name", "revision", "library"])
def test_disk_cache_invalidates_when_model_changes(embedding_cache, monkeypatch, change):
    cache = embedding_cache
    IncidentSearch(cache_path=cache.path)
    if change == "name":
        monkeypatch.setattr("routing.search.MODEL_NAME", "different-embedding-model")
    elif change == "revision":
        cache.settings["revision"] = "model-commit-two"
    else:
        monkeypatch.setattr("routing.search.version", lambda name: "new-library-version")

    IncidentSearch(cache_path=cache.path)
    assert len(cache.models[1].calls) == 1
    IncidentSearch(cache_path=cache.path)
    assert cache.models[2].calls == []


@pytest.mark.parametrize("damage", ["truncated", "missing_key", "shape", "nan", "dtype"])
def test_invalid_disk_cache_is_regenerated(embedding_cache, damage):
    cache = embedding_cache
    original = IncidentSearch(cache_path=cache.path)
    with np.load(cache.path, allow_pickle=False) as saved:
        fingerprint = saved["fingerprint"].item()
    if damage == "truncated":
        cache.path.write_bytes(b"PK\x03\x04broken")
    elif damage == "missing_key":
        np.savez(cache.path, embeddings=original.embeddings)
    else:
        vectors = {
            "shape": np.ones((3, 4)),
            "nan": np.full((3, 2), np.nan),
            "dtype": np.ones((3, 2), dtype=int),
        }[damage]
        np.savez(cache.path, embeddings=vectors, fingerprint=fingerprint)

    rebuilt = IncidentSearch(cache_path=cache.path)
    assert len(cache.models[1].calls) == 1
    np.testing.assert_array_equal(rebuilt.embeddings, original.embeddings)


def test_unwritable_cache_does_not_prevent_search(embedding_cache, monkeypatch, caplog):
    cache = embedding_cache

    def fail_replace(*args):
        raise PermissionError("Read-only cache")

    monkeypatch.setattr("routing.search.os.replace", fail_replace)
    search = IncidentSearch(cache_path=cache.path)

    assert search.search("VPN query").iloc[0]["incident_id"] == "TEST-2"
    assert "using in-memory embeddings" in caplog.text
    assert list(cache.path.parent.iterdir()) == []


def test_unknown_model_revision_bypasses_disk_cache(embedding_cache):
    cache = embedding_cache
    IncidentSearch(cache_path=cache.path)
    cache.settings["revision"] = None

    IncidentSearch(cache_path=cache.path)
    assert len(cache.models[1].calls) == 1


def test_model_card_revision_supports_newer_transformers(embedding_cache):
    cache = embedding_cache
    cache.settings["revision"] = None
    cache.settings["card_revision"] = "resolved-snapshot-one"
    IncidentSearch(cache_path=cache.path)
    assert cache.path.is_file()
    IncidentSearch(cache_path=cache.path)
    assert cache.models[1].calls == []

    cache.settings["card_revision"] = "resolved-snapshot-two"
    IncidentSearch(cache_path=cache.path)
    assert len(cache.models[2].calls) == 1


def test_injected_model_bypasses_disk_cache(embedding_cache):
    cache = embedding_cache
    IncidentSearch(cache_path=cache.path)
    model = FakeEmbeddingModel()

    IncidentSearch(model=model, cache_path=cache.path)
    assert len(model.calls) == 1


def test_disk_cache_can_be_disabled(embedding_cache):
    IncidentSearch(cache_path=None)
    assert len(embedding_cache.models[0].calls) == 1
    assert not embedding_cache.path.exists()
