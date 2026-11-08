from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from routing.recommend import recommend_assignment_group


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def app_backend(monkeypatch):
    """Use the real decision rules with fake retrieval; never load a model."""
    st.cache_resource.clear()
    incidents = pd.DataFrame({
        "incident_id": ["TEST-1", "TEST-2", "TEST-3"],
        "short_description": ["Desktop unavailable", "Session launch failed", "VPN failed"],
        "assignment_group": ["Cloud Platform", "Cloud Platform", "Network Operations"],
        "similarity_score": [0.9, 0.8, 0.2],
        "description": ["Extra historical details"] * 3,
    })
    search = Mock()
    search.search.return_value = incidents
    constructor = Mock(return_value=search)
    recommend = Mock(wraps=recommend_assignment_group)
    monkeypatch.setattr("routing.search.IncidentSearch", constructor)
    monkeypatch.setattr("routing.recommend.recommend_assignment_group", recommend)
    yield constructor, search, recommend
    st.cache_resource.clear()


def analyze(app, description="Synthetic desktop session fails to launch."):
    app.text_area[0].input(description)
    app.button[0].click().run()
    assert not app.exception
    return app


def test_recommendation_and_only_its_supporting_incidents_are_displayed(app_backend):
    constructor, search, recommend = app_backend
    app = AppTest.from_file(APP_PATH).run()
    constructor.assert_not_called()

    analyze(app, "  Synthetic desktop session fails to launch.  ")

    assert app.success[0].value == "Recommended assignment group: Cloud Platform"
    assert any("Requires human review" in item.value for item in app.caption)
    assert any("strong matches" in item.value for item in app.markdown)
    constructor.assert_called_once_with()
    search.search.assert_called_once_with("Synthetic desktop session fails to launch.", top_k=5)
    assert recommend.call_args.args[0] is search.search.return_value
    recommend.assert_called_once()
    displayed = app.dataframe[0].value
    assert displayed["incident_id"].tolist() == ["TEST-1", "TEST-2"]
    assert displayed.columns.tolist() == [
        "incident_id", "short_description", "assignment_group", "similarity_score"
    ]


def test_conflicting_evidence_displays_manual_review(app_backend):
    _, search, _ = app_backend
    search.search.return_value.loc[2, "similarity_score"] = 1.0
    app = analyze(AppTest.from_file(APP_PATH).run())

    assert app.warning[0].value == "Manual review recommended"
    assert not app.success
    assert any("Conflicting evidence" in item.value for item in app.markdown)
    pd.testing.assert_frame_equal(
        app.dataframe[0].value,
        search.search.return_value[
            ["incident_id", "short_description", "assignment_group", "similarity_score"]
        ],
    )


def test_empty_evidence_displays_manual_review_without_table(app_backend):
    _, search, _ = app_backend
    search.search.return_value = search.search.return_value.iloc[:0]
    app = analyze(AppTest.from_file(APP_PATH).run())

    assert app.warning[0].value == "Manual review recommended"
    assert "No valid supporting" in app.info[0].value
    assert not app.dataframe


@pytest.mark.parametrize("description", ["", " \n\t "])
def test_blank_input_does_not_invoke_backend(app_backend, description):
    constructor, search, recommend = app_backend
    app = analyze(AppTest.from_file(APP_PATH).run(), description)

    assert app.warning[0].value == "Please enter an incident description."
    constructor.assert_not_called()
    search.search.assert_not_called()
    recommend.assert_not_called()


def test_search_resource_is_reused_across_submissions(app_backend):
    constructor, search, recommend = app_backend
    app = AppTest.from_file(APP_PATH).run()
    analyze(app)
    analyze(app, "Another synthetic incident.")

    constructor.assert_called_once_with()
    assert search.search.call_count == 2
    assert recommend.call_count == 2


@pytest.mark.parametrize("stage", ["load", "search", "recommend"])
def test_backend_errors_are_handled_without_exposing_details(app_backend, stage):
    constructor, search, recommend = app_backend
    failing_call = {"load": constructor, "search": search.search, "recommend": recommend}[stage]
    failing_call.side_effect = RuntimeError("Private incident description and internal path")

    app = analyze(AppTest.from_file(APP_PATH).run())

    assert len(app.error) == 1
    assert "Private" not in app.error[0].value
    assert not app.success
    assert not app.dataframe
    if stage == "load":
        assert "could not be loaded" in app.error[0].value
        search.search.assert_not_called()
        recommend.assert_not_called()
    else:
        assert "could not be analyzed" in app.error[0].value
