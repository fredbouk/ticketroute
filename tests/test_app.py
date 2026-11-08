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
        "resolution_summary": ["Resolved synthetic issue."] * 3,
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
    next(button for button in app.button if button.label == "Find assignment group").click().run()
    assert not app.exception
    return app


def test_recommendation_and_only_its_supporting_incidents_are_displayed(app_backend):
    constructor, search, recommend = app_backend
    app = AppTest.from_file(APP_PATH).run()
    constructor.assert_not_called()

    analyze(app, "  Synthetic desktop session fails to launch.  ")

    assert any(item.value == "Recommended assignment group" for item in app.caption)
    assert not app.success
    assert not app.warning
    assert any(item.value == "Cloud Platform" for item in app.subheader)
    assert not any("Confirm the team before routing" in item.value for item in app.caption)
    assert not any(item.label == "Why this result?" for item in app.expander)
    constructor.assert_called_once_with()
    search.search.assert_called_once_with("Synthetic desktop session fails to launch.", top_k=5)
    assert recommend.call_args.args[0] is search.search.return_value
    recommend.assert_called_once()
    assert [item.label for item in app.expander if item.label.startswith("TEST-")] == [
        "TEST-1 · Desktop unavailable  \nHistorical assignment group: **Cloud Platform**",
        "TEST-2 · Session launch failed  \nHistorical assignment group: **Cloud Platform**",
    ]
    assert not app.tabs
    assert not app.dataframe
    assert any(item.value == "Extra historical details" for item in app.markdown)
    assert any(item.value == "Resolved synthetic issue." for item in app.markdown)
    assert not any("similarity" in item.value.lower() for item in app.caption)
    assert search.search.return_value["similarity_score"].tolist() == [0.9, 0.8, 0.2]


def test_conflicting_evidence_displays_ranked_suggestions_and_support(app_backend):
    _, search, _ = app_backend
    search.search.return_value.loc[2, "similarity_score"] = 1.0
    app = analyze(AppTest.from_file(APP_PATH).run())

    assert any(item.value == "Suggested assignment groups" for item in app.caption)
    assert not app.warning
    assert not app.success
    assert any(item.value == "1. Cloud Platform" for item in app.subheader)
    assert any(item.value == "2. Network Operations" for item in app.subheader)
    assert not any("similarity" in item.value.lower() for item in app.caption)
    assert [item.label for item in app.expander if item.label.startswith("TEST-")] == [
        f"{row.incident_id} · {row.short_description}  \nHistorical assignment group: **{row.assignment_group}**"
        for row in search.search.return_value.itertuples(index=False)
    ]
    assert not app.tabs
    assert not app.dataframe


def test_empty_evidence_displays_manual_review_without_table(app_backend):
    _, search, _ = app_backend
    search.search.return_value = search.search.return_value.iloc[:0]
    app = analyze(AppTest.from_file(APP_PATH).run())

    assert any(item.value == "Manual review needed" for item in app.subheader)
    assert any(item.value == "Routing result" for item in app.caption)
    assert any(item.value == (
        "No closely related incidents were found. Please review this incident "
        "and determine the appropriate assignment group."
    ) for item in app.markdown)
    assert not app.expander
    assert not app.dataframe


def test_weak_evidence_displays_no_group_suggestions(app_backend):
    _, search, _ = app_backend
    search.search.return_value["similarity_score"] = [0.49, 0.4, 0.2]
    app = analyze(AppTest.from_file(APP_PATH).run())

    assert any(item.value == "Manual review needed" for item in app.subheader)
    assert not app.success
    assert not any("1. Cloud Platform" in item.value for item in app.subheader)
    assert not any("min_similarity" in item.value for item in app.markdown)
    assert not app.expander
    assert not any("historical incidents" in item.value.lower() for item in app.subheader)
    assert search.search.return_value["incident_id"].tolist() == ["TEST-1", "TEST-2", "TEST-3"]


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


@pytest.mark.parametrize("group, technology", [
    ("Cloud Platform", "Azure Container Apps"),
    ("Identity and Access Management", "Entra ID"),
    ("Collaboration Services", "Exchange Online"),
    ("Backup hardware failure", "LTO tape library"),
])
def test_example_populates_editable_text_without_automatic_analysis(app_backend, group, technology):
    constructor, search, recommend = app_backend
    app = AppTest.from_file(APP_PATH).run()

    assert app.selectbox[0].value == "Choose an example..."
    assert app.text_area[0].value == ""
    assert app.selectbox[0].options == [
        "Choose an example...",
        "Cloud Platform", "Identity and Access Management", "Collaboration Services",
        "Backup hardware failure",
    ]
    app.selectbox[0].select(group).run()

    assert not app.exception
    example = app.text_area[0].value
    assert technology in example
    assert "Short description:" in example and "Description:" in example
    constructor.assert_not_called()
    search.search.assert_not_called()
    recommend.assert_not_called()
    assert not app.expander

    next(button for button in app.button if button.label == "Find assignment group").click().run()
    assert not app.exception
    search.search.assert_called_once_with(example, top_k=5)

    analyze(app, "Edited synthetic description after trying the example.")
    assert search.search.call_args.args[0] == "Edited synthetic description after trying the example."


def test_switching_examples_replaces_text_without_automatic_analysis(app_backend):
    constructor, search, recommend = app_backend
    app = AppTest.from_file(APP_PATH).run()
    app.selectbox[0].select("Cloud Platform").run()
    cloud_example = app.text_area[0].value

    app.selectbox[0].select("Collaboration Services").run()

    assert not app.exception
    assert app.text_area[0].value != cloud_example
    assert "Exchange Online" in app.text_area[0].value
    constructor.assert_not_called()
    search.search.assert_not_called()
    recommend.assert_not_called()


def test_example_placeholder_restores_no_selection_without_erasing_text(app_backend):
    constructor, search, recommend = app_backend
    app = AppTest.from_file(APP_PATH).run()
    app.selectbox[0].select("Cloud Platform").run()
    example = app.text_area[0].value

    app.selectbox[0].select("Choose an example...").run()

    assert not app.exception
    assert app.selectbox[0].value == "Choose an example..."
    assert app.text_area[0].value == example
    constructor.assert_not_called()
    search.search.assert_not_called()
    recommend.assert_not_called()


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
