import pandas as pd
import pytest

from routing.context import format_incident_context


def test_context_includes_only_requested_fields_in_retrieval_order():
    incidents = pd.DataFrame([
        {
            "incident_id": "CONTEXT-2",
            "short_description": "Tunnel cannot start",
            "description": "The local client reports a driver error before sign-in.",
            "assignment_group": "End User Computing",
            "resolution_summary": "Excluded repair details.",
            "similarity_score": 0.87654321,
            "extra_field": "Excluded metadata.",
        },
        {
            "incident_id": "CONTEXT-1",
            "short_description": "Internal connection rejected",
            "description": "The gateway rejects traffic from a remote subnet.",
            "assignment_group": "Network Operations",
            "resolution_summary": "Excluded configuration details.",
            "similarity_score": 0.7654321,
            "extra_field": "Excluded metadata.",
        },
    ], index=[7, 3])
    original = incidents.copy(deep=True)

    context = format_incident_context(incidents)

    assert context == (
        "Incident ID: CONTEXT-2\n"
        "Short description: Tunnel cannot start\n"
        "Description: The local client reports a driver error before sign-in.\n"
        "Historical assignment group: End User Computing\n\n"
        "Incident ID: CONTEXT-1\n"
        "Short description: Internal connection rejected\n"
        "Description: The gateway rejects traffic from a remote subnet.\n"
        "Historical assignment group: Network Operations"
    )
    pd.testing.assert_frame_equal(incidents, original)


@pytest.mark.parametrize("incidents", [
    pd.DataFrame(),
    pd.DataFrame(columns=[
        "incident_id", "short_description", "description", "assignment_group",
        "resolution_summary", "similarity_score",
    ]),
])
def test_empty_results_produce_empty_context(incidents):
    assert format_incident_context(incidents) == ""


def test_nonempty_results_require_all_context_fields():
    incidents = pd.DataFrame({"incident_id": ["CONTEXT-1"]})

    with pytest.raises(
        ValueError,
        match="Missing context columns: assignment_group, description, short_description",
    ):
        format_incident_context(incidents)
