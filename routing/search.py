from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from routing.data import load_incidents

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class IncidentSearch:
    """Retrieve similar historical incidents using locally generated embeddings."""

    def __init__(self, model: SentenceTransformer | None = None) -> None:
        self.incidents = load_incidents()
        self.model = model
        self.embeddings = np.empty((0, 0))

        if self.incidents.empty:
            return

        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(MODEL_NAME, device="cpu")

        searchable_text = (
            self.incidents["short_description"] + "\n" + self.incidents["description"]
        ).tolist()
        self.embeddings = self.model.encode(
            searchable_text, convert_to_numpy=True, normalize_embeddings=True
        )

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
