"""Connect retrieval, prompt preparation, and validated LLM routing responses."""

from collections.abc import Callable

import pandas as pd

from routing.llm import LLMRoutingResult, get_routing_response
from routing.prompt import build_routing_prompt
from routing.search import IncidentSearch


def recommend_with_rag(
    incident_description: str,
    *,
    search: IncidentSearch | None = None,
    llm_client: Callable[[str, pd.DataFrame], LLMRoutingResult] | None = None,
) -> LLMRoutingResult:
    """Retrieve five matches and return guidance, without assigning an incident.

    build_routing_prompt() formats the evidence with format_incident_context().
    The same retrieved rows are passed to the LLM client for group validation.
    Supply reusable search/client components for testing or repeated calls.
    Retrieval, configuration, and response-validation errors propagate to callers.
    """
    description = incident_description.strip()
    if not description:
        raise ValueError("Incident description must not be blank.")

    if search is None:
        search = IncidentSearch()
    incidents = search.search(description, top_k=5)
    if incidents.empty:
        return LLMRoutingResult(
            "manual_review", None,
            "No historical incidents were found. A technician needs to review this incident.",
        )

    prompt = build_routing_prompt(description, incidents)
    if llm_client is None:
        llm_client = get_routing_response
    return llm_client(prompt, incidents)
