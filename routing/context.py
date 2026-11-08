"""Format retrieved historical incidents for future prompt context."""

import pandas as pd


def format_incident_context(incidents: pd.DataFrame) -> str:
    """Return labeled incident blocks in retrieval order, or an empty string.

    Only the four context fields are included. Resolution summaries, similarity
    scores, and any other columns are excluded. The input is never modified.
    """
    if incidents.empty:
        return ""

    columns = ["incident_id", "short_description", "description", "assignment_group"]
    missing = set(columns) - set(incidents.columns)
    if missing:
        raise ValueError(f"Missing context columns: {', '.join(sorted(missing))}")

    blocks = []
    for incident_id, short_description, description, group in incidents[columns].itertuples(
        index=False, name=None,
    ):
        blocks.append(
            f"Incident ID: {incident_id}\n"
            f"Short description: {short_description}\n"
            f"Description: {description}\n"
            f"Historical assignment group: {group}"
        )
    return "\n\n".join(blocks)
