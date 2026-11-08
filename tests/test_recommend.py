from dataclasses import fields

import pandas as pd
import pytest

from routing.recommend import RecommendationResult, recommend_assignment_group


def matches(groups, scores):
    return pd.DataFrame({
        "incident_id": [f"TEST-{i}" for i in range(len(groups))],
        "assignment_group": groups,
        "similarity_score": scores,
        "short_description": ["Synthetic symptom"] * len(groups),
        "assigned_to": ["Synthetic person"] * len(groups),
    })


def test_clear_recommendation_has_weighted_scores_and_traceable_evidence():
    incidents = matches(["Cloud", "Cloud", "Network"], [0.9, 0.8, 0.3])
    original = incidents.copy(deep=True)

    result = recommend_assignment_group(incidents)

    assert result.status == "recommended"
    assert result.assignment_group == "Cloud"
    assert result.group_scores == pytest.approx({"Cloud": 1.7})
    assert result.supporting_incidents["incident_id"].tolist() == ["TEST-0", "TEST-1"]
    pd.testing.assert_frame_equal(
        result.supporting_incidents, incidents.iloc[:2]
    )
    pd.testing.assert_frame_equal(incidents, original)


def test_conflicting_groups_require_manual_review():
    incidents = matches(
        ["Cloud", "Cloud", "Network", "Network", "Other"], [0.9, 0.8, 0.8, 0.7, 0.2]
    )
    result = recommend_assignment_group(incidents)

    assert result.status == "manual_review"
    assert result.assignment_group is None
    assert "Conflicting evidence" in result.reason
    assert result.group_scores == pytest.approx({"Cloud": 1.7, "Network": 1.5})
    pd.testing.assert_frame_equal(result.supporting_incidents, incidents)


def test_weak_matches_do_not_influence_totals_or_vote_share():
    incidents = matches(
        ["Cloud", "Cloud", "Cloud", "Network", "Network", "Network"],
        [0.6, 0.5, 0.4, 0.49, 0.49, 0.49],
    )

    result = recommend_assignment_group(incidents, min_weight_share=1.0)

    assert result.status == "recommended"
    assert result.assignment_group == "Cloud"
    assert result.group_scores == pytest.approx({"Cloud": 1.1})
    pd.testing.assert_frame_equal(result.supporting_incidents, incidents.iloc[:2])


def test_no_eligible_matches_preserves_valid_evidence_for_review():
    incidents = matches(["Cloud", "Network", "Other", "Other"], [0.49, 0.2, 0, -0.8])

    result = recommend_assignment_group(incidents)

    assert result.status == "manual_review"
    assert result.assignment_group is None
    assert "no matches meet min_similarity" in result.reason
    assert result.group_scores == {}
    pd.testing.assert_frame_equal(result.supporting_incidents, incidents)


@pytest.mark.parametrize("scores", [[0.4, 0.3], [0.9, 0.2], [0.9]])
def test_explicit_two_match_requirement_rejects_weak_or_single_match(scores):
    result = recommend_assignment_group(
        matches(["Cloud"] * len(scores), scores), min_support=2,
    )

    assert result.status == "manual_review"
    assert result.assignment_group is None
    assert "Insufficient evidence" in result.reason


@pytest.mark.parametrize("scores", [[0.5], [0.9], [0.9, 0.2]])
def test_default_support_allows_one_eligible_match(scores):
    incidents = matches(["Cloud"] * len(scores), scores)

    result = recommend_assignment_group(incidents)

    assert result.status == "recommended"
    assert result.assignment_group == "Cloud"
    assert result.group_scores == pytest.approx({"Cloud": scores[0]})
    pd.testing.assert_frame_equal(result.supporting_incidents, incidents.iloc[:1])


@pytest.mark.parametrize("incidents", [pd.DataFrame(), matches([], [])])
def test_empty_search_results(incidents):
    result = recommend_assignment_group(incidents)

    assert result.status == "manual_review"
    assert result.assignment_group is None
    assert result.supporting_incidents.empty
    assert result.group_scores == {}


def test_ties_abstain_and_order_groups_deterministically():
    incidents = matches(["Network", "Cloud", "Cloud", "Network"], [0.8] * 4)

    for frame in [incidents, incidents.iloc[::-1]]:
        result = recommend_assignment_group(frame)
        assert result.status == "manual_review"
        assert result.assignment_group is None
        assert "tied" in result.reason
        assert list(result.group_scores) == ["Cloud", "Network"]


def test_result_structure():
    result = recommend_assignment_group(matches(["Cloud", "Cloud"], [0.8, 0.7]))

    assert isinstance(result, RecommendationResult)
    assert {field.name for field in fields(result)} == {
        "status", "assignment_group", "reason", "supporting_incidents", "group_scores"
    }
    assert isinstance(result.reason, str) and result.reason
    assert isinstance(result.supporting_incidents, pd.DataFrame)
    assert isinstance(result.group_scores, dict)


@pytest.mark.parametrize("scores", [[0.8, 0.7, 0, -0.9], [0.4, 0.3, 0.2, 0.1]])
def test_supporting_ids_always_come_from_input(scores):
    incidents = matches(["Cloud", "Cloud", "Network", "Network"], scores)
    result = recommend_assignment_group(incidents)

    assert set(result.supporting_incidents["incident_id"]) <= set(incidents["incident_id"])


def test_zero_and_negative_scores_do_not_vote():
    result = recommend_assignment_group(
        matches(["Cloud", "Cloud", "Network", "Network"], [0.8, 0.7, 0, -0.9])
    )

    assert result.assignment_group == "Cloud"
    assert result.group_scores == pytest.approx({"Cloud": 1.5})
    assert len(result.supporting_incidents) == 2


@pytest.mark.parametrize("scores", [
    [float("nan"), float("inf")], [1.1, "bad"], [-1.1, float("-inf")],
])
def test_no_usable_scores_requires_manual_review(scores):
    result = recommend_assignment_group(matches(["Cloud", "Cloud"], scores))

    assert result.status == "manual_review"
    assert result.group_scores == {}
    assert result.supporting_incidents.empty


def test_missing_identifiers_or_groups_cannot_support_recommendation():
    incidents = matches(["Cloud", None, " "], [0.8, 0.9, 0.9])
    incidents.loc[0, "incident_id"] = None

    result = recommend_assignment_group(incidents)

    assert result.status == "manual_review"
    assert result.group_scores == {}


def test_thresholds_are_configurable_and_inclusive():
    incidents = matches(["Cloud", "Cloud", "Network"], [0.6, 0.6, 0.8])

    assert recommend_assignment_group(incidents).status == "manual_review"
    assert recommend_assignment_group(
        incidents, min_similarity=0.6, min_support=2, min_weight_share=0.6
    ).assignment_group == "Cloud"
    assert recommend_assignment_group(
        incidents, min_similarity=0.7, min_weight_share=0.6
    ).assignment_group == "Network"
    assert recommend_assignment_group(
        incidents, min_similarity=0.7, min_support=2, min_weight_share=0.6
    ).status == "manual_review"


@pytest.mark.parametrize("kwargs", [
    {"min_similarity": 0}, {"min_similarity": 1.1},
    {"min_support": 0}, {"min_support": True}, {"min_support": 1.5},
    {"min_weight_share": 0.5}, {"min_weight_share": 1.1},
])
def test_invalid_thresholds_are_rejected(kwargs):
    with pytest.raises(ValueError):
        recommend_assignment_group(pd.DataFrame(), **kwargs)


def test_missing_required_columns_are_rejected():
    with pytest.raises(ValueError, match="Missing required columns: incident_id"):
        recommend_assignment_group(matches(["Cloud"], [0.9]).drop(columns="incident_id"))


def test_duplicate_ids_cannot_inflate_evidence():
    incidents = matches(["Cloud", "Cloud"], [0.9, 0.9])
    incidents["incident_id"] = "TEST-0"

    with pytest.raises(ValueError, match="must be unique"):
        recommend_assignment_group(incidents)
