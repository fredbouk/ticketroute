# TicketRoute

**AI-Assisted IT Incident Routing**

## Project overview

TicketRoute is an AI-assisted IT incident routing application designed to help
first- and second-line IT support personnel identify the appropriate technical
team for an incident.

Incorrect incident assignments can lead to repeated ticket transfers, increased
administrative overhead, and delayed resolution. TicketRoute aims to streamline
incident routing by leveraging historical resolved incidents to recommend the
appropriate technical team, supported by relevant evidence.

## Key objectives

- Reduce the time support personnel spend identifying assignment groups.
- Minimize unnecessary incident transfers.
- Help technical teams focus on resolving problems rather than administrative routing work.
- Help users receive faster assistance by directing incidents to the appropriate team earlier.

## Development status

TicketRoute is under active development. The Streamlit interface, local semantic
incident retrieval, and deterministic assignment group recommendation component
are implemented. Recommendations are not yet connected to the interface.

## Planned capabilities

- Semantic search across historical resolved incidents.
- AI-assisted assignment group recommendations.
- Explanations grounded in supporting historical evidence.
- Handling of ambiguous incidents and insufficient evidence.
- Human review of routing recommendations.
- Evaluation of recommendation quality.

## Technology stack

- Python 3.12.
- Streamlit for the application interface.

## Local development and setup

From the project folder, activate your existing Python 3.12 virtual environment.
For the existing `.venv` environment in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Confirm the Python version, install the dependencies, and start the application:

```powershell
python --version
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL displayed in the terminal (usually `http://localhost:8501`).
Press `Ctrl+C` in the terminal to stop the application.

## Data and privacy

TicketRoute currently uses synthetic incident data for development and
demonstration purposes. The public demo is not intended for processing real
incidents or sensitive information.

## Evidence-based recommendation baseline

`routing/recommend.py` accepts the DataFrame from `IncidentSearch.search()`:

```python
from routing.recommend import recommend_assignment_group
from routing.search import IncidentSearch

search = IncidentSearch()
matches = search.search(
    "Users can authenticate to Azure Virtual Desktop, but their desktop session fails to launch."
)
result = recommend_assignment_group(matches)
print(result.status, result.assignment_group, result.reason)
print(result.supporting_incidents["incident_id"].tolist())
```

Each group receives the sum of cosine similarity scores at or above
`min_similarity`. Weaker, zero, negative, nonnumeric, nonfinite, and out-of-range
scores do not vote. Rows without
a nonblank historical incident ID or assignment group are excluded. Duplicate
valid incident IDs and missing required columns in nonempty input raise
`ValueError`, preventing malformed input from silently inflating evidence.

Three keyword arguments control the initial decision rules:

| Argument | Default | Rationale |
| --- | --- | --- |
| `min_similarity` | `0.50` | A moderate similarity floor for voting eligibility. |
| `min_support` | `2` | Require repeated evidence instead of relying on a single historical case. |
| `min_weight_share` | `0.65` | Require roughly two thirds of eligible voting weight to favor one group. |

Only valid matches at or above `min_similarity` contribute to group scores,
support counts, and voting-weight shares. A leading group must meet all three
criteria. No eligible evidence, too few strong matches,
a tied leading score (including numerical near-ties), or insufficient weight
share produces `manual_review` with no recommended group. Score ties are ordered
alphabetically for repeatable output but never resolve to an automatic winner.

The result dataclass contains `status`, `assignment_group`, `reason`,
`supporting_incidents`, and `group_scores`. A recommendation includes the winning
group's eligible historical rows. Manual review includes all valid retrieved
rows, including weaker matches and zero or negative similarities, so reviewers
can inspect the available evidence. Valid scores are finite cosine similarities
in [-1, 1]. Original incident IDs and all other historical fields are preserved.
The component never recommends individuals or assigns or updates incidents.

These thresholds are provisional heuristics, not statistically calibrated
confidence levels. Cosine similarity and voting-weight share are not routing
probabilities. Similar symptoms can have different causes and responsible teams.
Results also depend on the retrieval model and `top_k` (currently five by
default), so a different retrieval setup requires reevaluation.

Evaluate and tune the rules on held-out synthetic cases before using them as a
decision policy. Keep evaluation cases separate from the historical retrieval
corpus and retain a separate test set after tuning. Include clear, ambiguous,
weak, and missing-match cases. Measure routing accuracy among recommendations,
recommendation coverage, manual-review rate, and errors by group.

Later, compare a Microsoft Foundry agent against this baseline using the same
held-out queries, historical evidence, and evaluation criteria. Also compare
evidence traceability, appropriate abstention, latency, and cost. No Foundry or
LLM dependency is required for this baseline.

Run the automated suite with `python -m pytest`. Recommendation tests use small
synthetic DataFrames and do not initialize or download embedding models.
