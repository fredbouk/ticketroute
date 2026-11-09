# TicketRoute

**AI-Assisted IT Incident Routing**

## Project overview

TicketRoute is an AI-assisted IT incident routing application designed to help
first- and second-line IT support personnel identify the appropriate technical
team for an incident.

Incorrect incident assignments usually lead to repeated ticket transfers, increased
administrative overhead, and delayed resolution. TicketRoute aims to streamline
incident routing by leveraging historical resolved incidents to recommend the
appropriate technical team, supported by relevant evidence.

## Key objectives

- Reduce the time support personnel spend identifying assignment groups.
- Minimize unnecessary incident transfers.
- Help technical teams focus on resolving problems rather than administrative routing work.
- Help users receive faster assistance by directing incidents to the appropriate team earlier.

## Development status

TicketRoute is under active development. The Streamlit interface connects local
semantic incident retrieval to deterministic assignment group recommendations.
Submit a synthetic incident description with **Analyze Incident** to retrieve
five similar historical incidents and review the resulting suggestion or
manual-review outcome. Expand **Supporting historical incidents** to inspect
the evidence returned by the recommendation component.

The search instance is initialized on the first valid submission and reused
across reruns with `st.cache_resource`, avoiding repeated model loading and
historical embedding generation. The cached resource is shared across sessions;
incident queries and recommendation results are not cached. Loading or analysis
failures display a generic error without exposing exception details or logging
incident descriptions. No incident is automatically assigned or updated.

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

## Docker

With Docker running in Linux container mode, build and run from the project root:

```powershell
docker build -t ticketroute:local .
docker run --rm --name ticketroute -p 8501:8501 ticketroute:local
```

Open `http://localhost:8501`. Stop with `Ctrl+C`. The image uses Python 3.12,
runs as a non-root user, and includes the application, current Streamlit theme,
and synthetic CSV datasets. Its health check uses `/_stcore/health` on port 8501.
For a future Azure Container Apps deployment, set the ingress target port to
**8501**; no Azure resources are created by these commands.

The build installs `requirements.txt` with CPU-only PyTorch and downloads the
existing `sentence-transformers/all-MiniLM-L6-v2` model into the image. Building
requires internet access and takes longer; PyTorch and the model also make the
image larger. Runtime model loading is offline, so new replicas and the first
search do not depend on Hugging Face availability. The build also precomputes
historical embeddings, so the first search loads the model and saved vectors
without embedding the history again. Rebuild the image to refresh dependencies,
data, or the bundled model.

For Azure AI routing, set `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and
`AZURE_OPENAI_DEPLOYMENT` in your host environment, then forward them at runtime:

```powershell
docker run --rm --name ticketroute -p 8501:8501 --env AZURE_OPENAI_ENDPOINT --env AZURE_OPENAI_API_KEY --env AZURE_OPENAI_DEPLOYMENT ticketroute:local
```

The existing app reads these variables at runtime and uses local routing when
Azure AI configuration is absent or unavailable. For Azure Container Apps, supply
the key through a secret-backed environment variable. Never put credentials in
the Dockerfile or build arguments. The build context allows only the listed
runtime files, excluding local secrets, virtual environments, tests, and caches.

## Historical embedding cache

`IncidentSearch` saves historical vectors and their fingerprint together in
`.cache/historical_embeddings.npz`, a local NumPy archive loaded without pickle.
New Python processes reuse these vectors. The SHA-256 fingerprint covers the
loaded historical data (including row order), model name, resolved Hugging Face
model commit, and sentence-transformers/transformers versions. A data or model
change invalidates the cache. Query embeddings are still computed for each search.

Missing, outdated, or corrupt caches are regenerated automatically. Writes use
an atomic file replacement; an unwritable cache falls back to in-memory embeddings.
The cache is ignored by Git and excluded from the Docker build context; Docker
generates its own copy using the bundled model and data. Locally, the first run
still needs the model available or internet access to download it. Injected custom
models bypass disk caching because their identity cannot be verified; models
without a resolved commit also bypass it. Pass `cache_path=None` to disable disk
caching, or a `pathlib.Path` to use another cache location.

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
synthetic DataFrames. Streamlit app tests use its built-in `AppTest` harness with
fake retrieval and the real recommendation rules. Automated tests do not
initialize or download embedding models.

## Initial routing evaluation

With the virtual environment activated, run:

```powershell
python -m evaluation.evaluate
```

The runner reuses `IncidentSearch` (default top five matches) and
`recommend_assignment_group` with unchanged thresholds. It loads the local
embedding model once; the first run may download the model if it is not cached.
Only incident descriptions enter retrieval, never scenario types or expected
groups. The output includes counts, rates with numerator/denominator pairs,
individual predictions, and reasons for unexpected outcomes. An undefined rate
is displayed as `N/A (0/0)`. Retrieval errors stop the run instead of being
counted as manual-review decisions.

`data/evaluation_incidents.csv` contains 30 entirely synthetic cases: 18 clear
cases (three per existing group), eight ambiguous cases missing diagnostic
details, and four unsupported cases involving laboratory instruments, industrial
safety controls, mainframe batch processing, and FPGA logic. The clear cases
use different failure scenarios from the historical incidents. Blank expected
groups mean manual review is expected. These are initial expectations for a
learning exercise, not labels confirmed by production support teams.

Keeping these cases separate from the 120 historical retrieval examples tests
generalization to new reports rather than retrieval of the same reports used as
evidence. Evaluation labels are not added to the search corpus.

| Metric | Numerator | Denominator |
| --- | --- | --- |
| Recommendation coverage | Cases receiving a group recommendation | All evaluation cases |
| Routing accuracy on recommended clear cases | Correct group recommendations on clear cases | Clear cases receiving a recommendation |
| Correct manual-review rate | Ambiguous or unsupported cases sent to manual review | All ambiguous and unsupported cases |

Accuracy asks whether the group is right when a clear case receives a
recommendation. Coverage asks how often the system recommends any group.
An abstention on a clear case reduces coverage but is excluded from routing
accuracy; a recommendation on an ambiguous or unsupported case counts toward
coverage but fails the manual-review expectation. Read all three metrics
together and inspect the individual outcomes to avoid hiding these tradeoffs.

Initial measured run on November 8, 2026, using the cached
`sentence-transformers/all-MiniLM-L6-v2` model and unchanged defaults:

| Result | Measured value |
| --- | --- |
| Evaluated cases | 30 |
| Recommendations / manual-review outcomes | 3 / 27 |
| Recommendation coverage | 10.0% (3/30) |
| Routing accuracy on recommended clear cases | 100.0% (2/2) |
| Correct manual-review rate | 91.7% (11/12) |

The two recommended clear cases were EVAL-002 (Azure compute quota) and EVAL-005
(directory account lockout). The remaining 16 clear cases were sent to review:
15 had no retrieved match at similarity 0.50 or higher, and EVAL-013 had only one
eligible match instead of the required two. EVAL-020, an ambiguous shared-site
editing problem, received a Collaboration Services recommendation despite missing
information needed to distinguish site restrictions from identity membership.
Seven of eight ambiguous cases and all four unsupported cases went to review.
These results show limited coverage of new failure scenarios by the current
history and thresholds. Similarity-weighted agreement also does not establish
that an ambiguous report contains enough diagnostic information. The 100%
accuracy comes from just two clear predictions and is weak evidence of routing
quality; no thresholds were tuned on these results.

The historical dataset was subsequently expanded from 36 to 120 entirely
synthetic resolved incidents, retaining the original 36 and adding INC-1037
through INC-1120. Each of the six groups now has 20 examples. The additions
cover recurring enterprise support categories with different symptoms, scopes,
and resolutions, including virtual desktops, identity access, VPN connectivity,
managed clients, Azure infrastructure, collaboration tools, and business workflows.
The evaluation dataset was left unchanged and was not inspected to select the
new historical scenarios.

With the expanded history and the same cached model, retrieval defaults,
recommendation thresholds, and evaluation calculations, the measured results were:

| Metric | Original 36 incidents | Expanded 120 incidents |
| --- | --- | --- |
| Recommendation coverage | 10.0% (3/30) | 26.7% (8/30) |
| Routing accuracy on recommended clear cases | 100.0% (2/2) | 100.0% (6/6) |
| Correct manual-review rate | 91.7% (11/12) | 83.3% (10/12) |

The expanded run produced eight recommendations and 22 manual-review outcomes.
Six clear cases received correct group recommendations, while 12 clear cases
still went to review. Two ambiguous cases received recommendations, including
an additional mailbox problem without enough diagnostic information. All four
unsupported cases still went to review. More history increased coverage, but
also made the system more willing to recommend a group for incomplete reports.
Six correct clear predictions remain too few to establish reliable accuracy.
No historical incidents or routing settings were adjusted in response to this run.

This small dataset cannot establish production quality or calibrated confidence.
Its wording, diagnostic detail, group balance, and chosen unsupported technologies
affect the results. Ambiguous cases can resemble well-supported historical cases
without supplying enough information for a justified route. Review expectations
with domain experts and eventually evaluate on representative, appropriately
handled operational data. If these cases are used to tune thresholds, retain a
new untouched test set for the final assessment.

`tests/test_evaluate.py` uses fake retrieval with the real recommendation rules,
so metric and outcome tests never download the embedding model.
