"""Exercise evaluation with fake retrieval and the real recommendation rules."""

import pandas as pd
import pytest

from evaluation.evaluate import (
    DEFAULT_DATASET, Metric, evaluate, load_evaluation_incidents, print_summary,
)
from routing.data import load_incidents


CLOUD = "Cloud Platform"
NETWORK = "Network Operations"
COLUMNS = ["evaluation_id", "description", "scenario_type", "expected_assignment_group"]


class FakeSearch:
    def __init__(self, predictions):
        self.predictions = predictions
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        group = self.predictions[query]
        if group is None:
            return pd.DataFrame()
        return pd.DataFrame({
            "incident_id": ["SYNTHETIC-1", "SYNTHETIC-2"],
            "assignment_group": [group, group],
            "similarity_score": [0.9, 0.8],
        })


def write_cases(tmp_path, scenarios):
    cases = pd.DataFrame([
        [f"CASE-{i}", f"Synthetic query {i}", scenario, expected]
        for i, (scenario, expected) in enumerate(scenarios)
    ], columns=COLUMNS)
    path = tmp_path / "evaluation.csv"
    cases.to_csv(path, index=False)
    return path, cases


def test_predictions_and_metric_denominators(tmp_path, capsys):
    path, cases = write_cases(tmp_path, [
        ("clear", CLOUD), ("clear", CLOUD), ("clear", NETWORK),
        ("ambiguous", ""), ("unsupported", ""),
        ("ambiguous", ""), ("unsupported", ""),
    ])
    search = FakeSearch(dict(zip(
        cases["description"], [CLOUD, NETWORK, None, None, None, CLOUD, NETWORK],
    )))

    result = evaluate(path, search=search)

    assert search.queries == cases["description"].tolist()
    assert result.outcomes["correct"].tolist() == [True, False, False, True, True, False, False]
    assert result.outcomes["status"].tolist() == [
        "recommended", "recommended", "manual_review", "manual_review",
        "manual_review", "recommended", "recommended",
    ]
    assert result.outcomes["assignment_group"].dropna().tolist() == [
        CLOUD, NETWORK, CLOUD, NETWORK,
    ]
    assert result.outcomes.loc[2:4, "assignment_group"].isna().all()
    assert result.metrics["coverage"] == Metric(4, 7)
    assert result.metrics["routing_accuracy"] == Metric(1, 2)
    assert result.metrics["manual_review_rate"] == Metric(2, 4)
    assert result.metrics["coverage"].value == pytest.approx(4 / 7)
    assert result.metrics["routing_accuracy"].value == 0.5
    assert result.outcomes["reason"].str.len().gt(0).all()
    print_summary(result)
    output = capsys.readouterr().out
    assert "Evaluation incidents: 7" in output
    assert "Recommendations: 4" in output
    assert "Candidate guidance outcomes: 0" in output
    assert "Manual-review outcomes without suggestions: 3" in output
    assert "50.0% (1/2)" in output
    assert "CASE-2" in output
    assert "Reasons for unexpected outcomes:" in output


def test_candidates_are_reported_separately_and_never_counted_as_assignments(tmp_path, capsys):
    path, _ = write_cases(tmp_path, [("clear", CLOUD), ("ambiguous", "")])

    class ConflictingSearch:
        def search(self, query):
            return pd.DataFrame({
                "incident_id": ["SYNTHETIC-1", "SYNTHETIC-2"],
                "assignment_group": [CLOUD, NETWORK],
                "similarity_score": [0.8, 0.7],
            })

    result = evaluate(path, search=ConflictingSearch())

    assert result.outcomes["status"].tolist() == ["suggested", "suggested"]
    assert result.outcomes["assignment_group"].isna().all()
    assert result.outcomes["candidate_groups"].tolist() == [f"{CLOUD}, {NETWORK}"] * 2
    assert result.outcomes["correct"].tolist() == [False, True]
    assert result.metrics["coverage"] == Metric(0, 2)
    assert result.metrics["routing_accuracy"] == Metric(0, 0)
    assert result.metrics["manual_review_rate"] == Metric(1, 1)
    print_summary(result)
    output = capsys.readouterr().out
    assert "Recommendations: 0" in output
    assert "Candidate guidance outcomes: 2" in output
    assert "Manual-review outcomes without suggestions: 0" in output


@pytest.mark.parametrize("scenarios,predictions,expected", [
    ([], [], {"coverage": Metric(0, 0), "routing_accuracy": Metric(0, 0),
              "manual_review_rate": Metric(0, 0)}),
    ([("clear", CLOUD)], [None], {
        "coverage": Metric(0, 1), "routing_accuracy": Metric(0, 0),
        "manual_review_rate": Metric(0, 0)}),
    ([("clear", CLOUD)], [CLOUD], {
        "coverage": Metric(1, 1), "routing_accuracy": Metric(1, 1),
        "manual_review_rate": Metric(0, 0)}),
    ([("unsupported", "")], [None], {
        "coverage": Metric(0, 1), "routing_accuracy": Metric(0, 0),
        "manual_review_rate": Metric(1, 1)}),
])
def test_zero_denominators(tmp_path, capsys, scenarios, predictions, expected):
    path, cases = write_cases(tmp_path, scenarios)
    result = evaluate(path, search=FakeSearch(dict(zip(cases["description"], predictions))))

    assert result.metrics == expected
    for metric in result.metrics.values():
        if metric.denominator == 0:
            assert metric.value is None
    print_summary(result)
    assert "N/A (0/0)" in capsys.readouterr().out


def test_default_runner_constructs_search_once(tmp_path, monkeypatch):
    path, cases = write_cases(tmp_path, [("clear", CLOUD), ("clear", CLOUD)])
    fake = FakeSearch(dict.fromkeys(cases["description"], CLOUD))
    constructions = []

    def create_search():
        constructions.append(True)
        return fake

    monkeypatch.setattr("evaluation.evaluate.IncidentSearch", create_search)
    result = evaluate(path)

    assert constructions == [True]
    assert result.metrics["routing_accuracy"] == Metric(2, 2)


def test_dataset_composition_and_separation():
    cases = load_evaluation_incidents(DEFAULT_DATASET)
    history = load_incidents()

    assert cases.columns.tolist() == COLUMNS
    assert len(cases) == 30
    assert cases["scenario_type"].value_counts().to_dict() == {
        "clear": 18, "ambiguous": 8, "unsupported": 4,
    }
    clear = cases.loc[cases["scenario_type"].eq("clear")]
    assert clear["expected_assignment_group"].value_counts().to_dict() == dict.fromkeys(
        history["assignment_group"].unique(), 3,
    )
    assert set(cases["description"]).isdisjoint(history["description"])
    for case in clear.itertuples(index=False):
        assert case.expected_assignment_group.lower() not in case.description.lower()


@pytest.mark.parametrize("column,value,message", [
    ("evaluation_id", "", "IDs must be nonblank"),
    ("description", " ", "descriptions must be nonblank"),
    ("scenario_type", "unknown", "Unknown evaluation scenario_type"),
    ("expected_assignment_group", "Unknown group", "existing assignment group"),
])
def test_invalid_evaluation_data_is_rejected(tmp_path, column, value, message):
    path, cases = write_cases(tmp_path, [("clear", CLOUD)])
    cases.loc[0, column] = value
    cases.to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        load_evaluation_incidents(path)


def test_manual_review_expectations_cannot_contain_group_labels(tmp_path):
    path, _ = write_cases(tmp_path, [("ambiguous", CLOUD)])
    with pytest.raises(ValueError, match="empty expected groups"):
        load_evaluation_incidents(path)


def test_search_errors_are_not_silently_counted_as_manual_review(tmp_path):
    path, _ = write_cases(tmp_path, [("clear", CLOUD)])

    class BrokenSearch:
        def search(self, query):
            raise RuntimeError("Synthetic retrieval failure")

    with pytest.raises(RuntimeError, match="Synthetic retrieval failure"):
        evaluate(path, search=BrokenSearch())
