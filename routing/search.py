from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from routing.data import load_incidents

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent.parent / ".cache" / "historical_embeddings.npz"


class IncidentSearch:
    """Retrieve similar historical incidents using locally generated embeddings."""

    def __init__(
        self, model: SentenceTransformer | None = None, *,
        cache_path: Path | None = DEFAULT_CACHE_PATH,
    ) -> None:
        self.incidents = load_incidents()
        self.model = model
        self.embeddings = np.empty((0, 0))
        # Injected models have no guaranteed identity; never reuse their vectors.
        use_cache = model is None and cache_path is not None

        if self.incidents.empty:
            return

        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(MODEL_NAME, device="cpu")

        fingerprint = self._cache_fingerprint() if use_cache else None
        if fingerprint is not None and self._load_embeddings(cache_path, fingerprint):
            return

        searchable_text = (
            self.incidents["short_description"] + "\n" + self.incidents["description"]
        ).tolist()
        self.embeddings = self.model.encode(
            searchable_text, convert_to_numpy=True, normalize_embeddings=True
        )
        if fingerprint is not None:
            self._save_embeddings(cache_path, fingerprint)

    def _cache_fingerprint(self) -> str | None:
        # The resolved Hub commit detects updates even when MODEL_NAME is unchanged.
        revision = (
            getattr(self.model[0].auto_model.config, "_commit_hash", None)
            or getattr(getattr(self.model, "model_card_data", None), "base_model_revision", None)
        )
        if not revision:
            return None  # Unknown model revision: computing is safer than stale reuse.
        identity = {
            "format": 1,  # Increment if text construction or encoding options change.
            "model": MODEL_NAME,
            "revision": revision,
            "sentence_transformers": version("sentence-transformers"),
            "transformers": version("transformers"),
            "incidents": self.incidents.to_json(orient="split", force_ascii=False),
        }
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()

    def _load_embeddings(self, cache_path: Path, fingerprint: str) -> bool:
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                if cached["fingerprint"].item() != fingerprint:
                    return False
                embeddings = cached["embeddings"]
            expected_shape = (len(self.incidents), self.model.get_sentence_embedding_dimension())
            if (embeddings.shape != expected_shape
                    or not np.issubdtype(embeddings.dtype, np.floating)
                    or not np.isfinite(embeddings).all()):
                return False
        except (OSError, ValueError, KeyError, EOFError, BadZipFile, TypeError):
            return False
        self.embeddings = embeddings
        return True

    def _save_embeddings(self, cache_path: Path, fingerprint: str) -> None:
        temporary_path = None
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            # A single atomic replacement keeps concurrent readers from partial files.
            with tempfile.NamedTemporaryFile(dir=cache_path.parent, delete=False) as temporary:
                temporary_path = Path(temporary.name)
                np.savez(temporary, embeddings=self.embeddings, fingerprint=fingerprint)
            os.replace(temporary_path, cache_path)
        except OSError:
            logging.getLogger(__name__).warning(
                "Historical embeddings could not be saved; using in-memory embeddings."
            )
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """Return up to top_k incidents, ordered by descending cosine similarity."""
        if not query.strip():
            raise ValueError("Please enter a non-empty incident description.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        if self.incidents.empty:
            return self.incidents.assign(similarity_score=pd.Series(dtype=float))

        query_embedding = self.model.encode(
            [query.strip()], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        # For unit-length embeddings, the dot product equals cosine similarity.
        scores = np.clip(self.embeddings @ query_embedding, -1.0, 1.0)
        top_indices = np.argsort(-scores, kind="stable")[:top_k]

        results = self.incidents.iloc[top_indices].copy()
        results["similarity_score"] = scores[top_indices]
        return results.reset_index(drop=True)
