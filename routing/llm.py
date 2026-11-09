"""Send completed prompts to Microsoft Foundry's OpenAI-compatible API.

Set AZURE_OPENAI_ENDPOINT to the resource URL or its full /openai/v1/ URL,
AZURE_OPENAI_API_KEY to the resource key, and AZURE_OPENAI_DEPLOYMENT to the
model deployment name. Configuration is read only when the function is called.
"""

from dataclasses import dataclass
import json
import os
from typing import Literal

import pandas as pd
from openai import OpenAI


@dataclass(frozen=True)
class LLMRoutingResult:
    status: Literal["recommended", "manual_review"]
    assignment_group: str | None
    reason: str


def _unique_json_fields(pairs: list[tuple[str, object]]) -> dict:
    fields = {}
    for key, value in pairs:
        if key in fields:
            raise ValueError("LLM response must not contain duplicate JSON fields.")
        fields[key] = value
    return fields


def _parse_routing_response(text: str, allowed_groups: set[str]) -> LLMRoutingResult:
    """Validate model output independently of the prompt's instructions."""
    try:
        data = json.loads(text, object_pairs_hook=_unique_json_fields)
    except json.JSONDecodeError:
        raise ValueError("LLM response must be valid JSON.") from None

    if not isinstance(data, dict) or set(data) != {"status", "assignment_group", "reason"}:
        raise ValueError("LLM response must contain exactly status, assignment_group, and reason.")

    status, group, reason = data["status"], data["assignment_group"], data["reason"]
    if status not in ("recommended", "manual_review"):
        raise ValueError("LLM response status must be recommended or manual_review.")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("LLM response reason must be a nonempty string.")
    if status == "manual_review":
        if group is not None:
            raise ValueError("A manual_review response must have a null assignment_group.")
    elif not isinstance(group, str) or group not in allowed_groups:
        raise ValueError("A recommended assignment_group must appear in the supplied historical evidence.")

    return LLMRoutingResult(status, group, reason.strip())


def get_routing_response(
    prompt: str, incidents: pd.DataFrame, *, max_retries: int | None = None,
) -> LLMRoutingResult:
    """Return a validated result; propagate SDK errors to the caller.

    Pass the same historical rows used by build_routing_prompt(). Group names
    are checked against those rows, never extracted from model output or the
    technician's text. A valid result does not automatically assign an incident.
    Set max_retries=0 for callers with a strict request budget; None keeps SDK defaults.
    """
    if not prompt.strip():
        raise ValueError("Routing prompt must not be blank.")

    allowed_groups = set()
    if not incidents.empty:
        if "assignment_group" not in incidents.columns:
            raise ValueError("Historical evidence must include assignment_group.")
        allowed_groups = {
            group for group in incidents["assignment_group"]
            if isinstance(group, str) and group.strip()
        }

    names = ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT")
    config = {name: os.environ.get(name, "").strip() for name in names}
    missing = [name for name in names if not config[name]]
    if missing:
        raise ValueError(f"Missing LLM configuration. Set environment variables: {', '.join(missing)}")

    endpoint = config["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    base_url = endpoint + "/" if endpoint.endswith("/openai/v1") else endpoint + "/openai/v1/"
    retry_options = {} if max_retries is None else {"max_retries": max_retries}
    with OpenAI(api_key=config["AZURE_OPENAI_API_KEY"], base_url=base_url, **retry_options) as client:
        response = client.chat.completions.create(
            model=config["AZURE_OPENAI_DEPLOYMENT"],
            messages=[{"role": "user", "content": prompt}],
        )

    text = response.choices[0].message.content if response.choices else None
    if text is None or not text.strip():
        raise RuntimeError("The LLM returned no text response.")
    return _parse_routing_response(text, allowed_groups)
