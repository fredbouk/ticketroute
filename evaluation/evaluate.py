"""Run with ``python -m evaluation.evaluate`` from the project directory."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from routing.data import load_incidents
from routing.recommend import recommend_assignment_group
from routing.search import IncidentSearch


DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "data" / "evaluation_incidents.csv"
OUTCOME_COLUMNS = [
    "evaluation_id", "scenario_type", "expected_assignment_group", "status",
    "assignment_group", "candidate_groups", "correct", "reason",
]


@dataclass(frozen=True)
class Metric:
    numerator: int
    denominator: int

    @property
    def value(self) -> float | None:
        """An undefined rate is None, rather than a misleading zero percent."""
        return self.numerator / self.denominator if self.denominator else None


@dataclass
class EvaluationResult:
    outcomes: pd.DataFrame
    metrics: dict[str, Metric]


def load_evaluation_incidents(path: str | Path = DEFAULT_DATASET) -> pd.DataFrame:
    """Load expectations separately from the historical retrieval corpus."""
    cases = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"evaluation_id", "description", "scenario_type", "expected_assignment_group"}
    missing = required - set(cases.columns)
    if missing:
        raise ValueError(f"Missing evaluation columns: {', '.join(sorted(missing))}")
    cases = cases.copy()
    for column in required:
        cases[column] = cases[column].str.strip()
    if cases["evaluation_id"].eq("").any() or cases["evaluation_id"].duplicated().any():
        raise ValueError("Evaluation IDs must be nonblank and unique.")
    if cases["description"].eq("").any():
        raise ValueError("Evaluation descriptions must be nonblank.")
    if not cases["scenario_type"].isin(["clear", "ambiguous", "unsupported"]).all():
        raise ValueError("Unknown evaluation scenario_type.")
    clear = cases["scenario_type"].eq("clear")
    groups = set(load_incidents()["assignment_group"])
    if not cases.loc[clear, "expected_assignment_group"].isin(groups).all():
        raise ValueError("Clear cases must name an existing assignment group.")
    if cases.loc[~clear, "expected_assignment_group"].ne("").any():
        raise ValueError("Ambiguous and unsupported cases must have empty expected groups.")
    return cases


def evaluate(
    path: str | Path = DEFAULT_DATASET, *, search: IncidentSearch | None = None,
) -> EvaluationResult:
    """Evaluate unchanged search/recommendation defaults; inject search for offline tests.

    Only descriptions enter retrieval. Labels are used after prediction to score
    outcomes. Errors propagate so a partial run cannot look like a complete one.
    Candidates are never scored as correct assignments on clear cases. Suggested
    and manual_review outcomes both require human judgment on non-clear cases.
    """
    cases = load_evaluation_incidents(path)
    if search is None and not cases.empty:
        search = IncidentSearch()
    rows = []
    for case in cases.itertuples(index=False):
        result = recommend_assignment_group(search.search(case.description))
        correct = (
            result.status == "recommended"
            and result.assignment_group == case.expected_assignment_group
            if case.scenario_type == "clear"
            else result.status in {"manual_review", "suggested"}
        )
        rows.append({
            "evaluation_id": case.evaluation_id,
            "scenario_type": case.scenario_type,
            "expected_assignment_group": case.expected_assignment_group,
            "status": result.status,
            "assignment_group": result.assignment_group,
            "candidate_groups": ", ".join(result.candidate_groups),
            "correct": correct,
            "reason": result.reason,
        })
    outcomes = pd.DataFrame(rows, columns=OUTCOME_COLUMNS)
    recommended = outcomes["status"].eq("recommended")
    recommended_clear = recommended & outcomes["scenario_type"].eq("clear")
    review_expected = outcomes["scenario_type"].isin(["ambiguous", "unsupported"])
    correct = outcomes["correct"].eq(True)
    metrics = {
        "coverage": Metric(int(recommended.sum()), len(outcomes)),
        "routing_accuracy": Metric(
            int((correct & recommended_clear).sum()), int(recommended_clear.sum()),
        ),
        "manual_review_rate": Metric(
            int((correct & review_expected).sum()), int(review_expected.sum()),
        ),
    }
    return EvaluationResult(outcomes, metrics)


def print_summary(result: EvaluationResult) -> None:
    """Show all 30 outcomes, with full reasons separately for unexpected outcomes."""
    total = len(result.outcomes)
    recommendations = result.metrics["coverage"].numerator
    print(f"Evaluation incidents: {total}")
    print(f"Recommendations: {recommendations}")
    candidates = int(result.outcomes["status"].eq("suggested").sum())
    print(f"Candidate guidance outcomes: {candidates}")
    print(f"Manual-review outcomes without suggestions: {total - recommendations - candidates}")
    labels = {
        "coverage": "Recommendation coverage",
        "routing_accuracy": "Routing accuracy on recommended clear cases",
        "manual_review_rate": "Correct manual-review rate (ambiguous + unsupported)",
    }
    for name, label in labels.items():
        metric = result.metrics[name]
        rate = f"{metric.value:.1%}" if metric.value is not None else "N/A"
        print(f"{label}: {rate} ({metric.numerator}/{metric.denominator})")
    print("Manual-review rate includes candidate guidance requiring technician judgment.")
    table = result.outcomes.drop(columns="reason").fillna("")
    print("\nIndividual outcomes (blank expected group means manual review):")
    print(table.to_string(index=False))
    unexpected = result.outcomes.loc[result.outcomes["correct"].eq(False)]
    if not unexpected.empty:
        print("\nReasons for unexpected outcomes:")
        for row in unexpected.itertuples(index=False):
            print(f"{row.evaluation_id}: {row.reason}")


if __name__ == "__main__":
    print_summary(evaluate())
