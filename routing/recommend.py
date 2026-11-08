"""Deterministic routing suggestions grounded in retrieved historical incidents."""

from dataclasses import dataclass
from math import isclose
from typing import Literal

import pandas as pd


@dataclass
class RecommendationResult:
    status: Literal["recommended", "manual_review"]
    assignment_group: str | None
    reason: str
    supporting_incidents: pd.DataFrame
    group_scores: dict[str, float]


def recommend_assignment_group(
    incidents: pd.DataFrame,
    *,
    min_similarity: float = 0.50,
    min_support: int = 1,
    min_weight_share: float = 0.65,
) -> RecommendationResult:
    """Sum threshold-eligible similarities and abstain when evidence is unclear.

    A winner needs ``min_support`` incidents at or above ``min_similarity``
    and ``min_weight_share`` of voting weight from eligible incidents only.
    Incidents below ``min_similarity`` do not vote. These defaults
    are provisional heuristics, not calibrated probabilities; see README.

    Recommended results include the winner's eligible incidents.
    Manual-review results include all valid retrieved incidents so a reviewer
    can inspect weak or conflicting evidence. All original fields are preserved;
    inputs are never modified.
    """
    if not 0 < min_similarity <= 1:
        raise ValueError("min_similarity must be in (0, 1].")
    if isinstance(min_support, bool) or not isinstance(min_support, int) or min_support < 1:
        raise ValueError("min_support must be a positive integer.")
    if not 0.5 < min_weight_share <= 1:
        raise ValueError("min_weight_share must be in (0.5, 1].")

    evidence = incidents.copy()
    if not evidence.empty:
        required = {"incident_id", "assignment_group", "similarity_score"}
        missing = required - set(evidence.columns)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

        # Invalid scores and missing identifiers/groups cannot support a route.
        scores = pd.to_numeric(evidence["similarity_score"], errors="coerce")
        valid_ids = evidence["incident_id"].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        )
        valid_groups = evidence["assignment_group"].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        )
        evidence = evidence.loc[
            scores.between(-1, 1) & valid_ids & valid_groups
        ].copy()
        evidence["similarity_score"] = scores.loc[evidence.index]
        if evidence["incident_id"].duplicated().any():
            raise ValueError("Each supporting incident_id must be unique.")

    if evidence.empty:
        return RecommendationResult(
            "manual_review", None, "Insufficient evidence: no valid retrieved incidents.",
            evidence, {},
        )

    eligible = evidence.loc[evidence["similarity_score"] >= min_similarity]
    if eligible.empty:
        return RecommendationResult(
            "manual_review", None,
            f"Insufficient evidence: no matches meet min_similarity >= {min_similarity:.2f}.",
            evidence, {},
        )

    totals = eligible.groupby("assignment_group")["similarity_score"].sum()
    # Alphabetical order makes equal scores deterministic without selecting a tied winner.
    group_scores = dict(sorted(totals.items(), key=lambda item: (-item[1], item[0])))
    winner = next(iter(group_scores))
    winner_score = group_scores[winner]
    winner_evidence = eligible.loc[eligible["assignment_group"] == winner].copy()
    strong_count = len(winner_evidence)
    share = winner_score / sum(group_scores.values())

    if strong_count < min_support:
        reason = (
            f"Insufficient evidence: leading group {winner} has {strong_count} matches "
            f"at similarity >= {min_similarity:.2f}; requires {min_support}."
        )
    elif any(isclose(winner_score, score) for score in list(group_scores.values())[1:]):
        reason = "Conflicting evidence: leading groups have tied similarity-weighted scores."
    elif share < min_weight_share:
        reason = (
            f"Conflicting evidence: leading group {winner} has {share:.1%} of voting "
            f"weight; requires {min_weight_share:.1%}."
        )
    else:
        return RecommendationResult(
            "recommended", winner,
            f"{winner} has {strong_count} strong matches and {share:.1%} of voting weight.",
            winner_evidence, group_scores,
        )

    return RecommendationResult("manual_review", None, reason, evidence, group_scores)
