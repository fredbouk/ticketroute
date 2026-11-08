import pandas as pd
import pytest

from routing.data import load_incidents


REQUIRED_COLUMNS = [
    "incident_id",
    "short_description",
    "description",
    "assignment_group",
    "resolution_summary",
]

ASSIGNMENT_GROUPS = {
    "Cloud Platform",
    "Identity and Access Management",
    "Network Operations",
    "End User Computing",
    "Collaboration Services",
    "Business Applications",
}


@pytest.fixture
def incidents():
    return load_incidents()


def test_csv_loads_successfully(incidents):
    assert isinstance(incidents, pd.DataFrame)


def test_dataset_contains_120_incidents(incidents):
    assert len(incidents) == 120


def test_required_columns_exist(incidents):
    assert incidents.columns.tolist() == REQUIRED_COLUMNS


def test_incident_ids_are_unique(incidents):
    assert incidents["incident_id"].is_unique


def test_incident_ids_are_sequential(incidents):
    assert incidents["incident_id"].tolist() == [f"INC-{i}" for i in range(1001, 1121)]


def test_required_fields_have_no_missing_or_empty_values(incidents):
    for column in REQUIRED_COLUMNS:
        assert incidents[column].notna().all(), f"{column} contains missing values"
        assert incidents[column].str.strip().ne("").all(), (
            f"{column} contains empty or whitespace-only values"
        )


def test_dataset_contains_20_incidents_per_assignment_group(incidents):
    assert incidents["assignment_group"].value_counts().to_dict() == dict.fromkeys(
        ASSIGNMENT_GROUPS, 20,
    )


def test_loading_from_another_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert len(load_incidents()) == 120


def test_missing_csv_has_a_clear_error(monkeypatch):
    monkeypatch.setattr("routing.data.Path.is_file", lambda path: False)

    with pytest.raises(FileNotFoundError, match="Historical incident CSV not found"):
        load_incidents()
