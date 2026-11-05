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


@pytest.fixture
def incidents():
    return load_incidents()


def test_csv_loads_successfully(incidents):
    assert isinstance(incidents, pd.DataFrame)


def test_dataset_contains_36_incidents(incidents):
    assert len(incidents) == 36


def test_required_columns_exist(incidents):
    assert set(REQUIRED_COLUMNS).issubset(incidents.columns)


def test_incident_ids_are_unique(incidents):
    assert incidents["incident_id"].is_unique


def test_required_fields_have_no_missing_or_empty_values(incidents):
    for column in REQUIRED_COLUMNS:
        assert incidents[column].notna().all(), f"{column} contains missing values"
        assert incidents[column].str.strip().ne("").all(), (
            f"{column} contains empty or whitespace-only values"
        )


def test_dataset_contains_six_assignment_groups(incidents):
    assert incidents["assignment_group"].nunique() == 6


def test_loading_from_another_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert len(load_incidents()) == 36


def test_missing_csv_has_a_clear_error(monkeypatch):
    monkeypatch.setattr("routing.data.Path.is_file", lambda path: False)

    with pytest.raises(FileNotFoundError, match="Historical incident CSV not found"):
        load_incidents()
