"""Compare six new fictional reports with the baseline and GPT-4.1 mini RAG.

Manual run from the project root:
    python -m scripts.compare_routing

Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT.
The deployment MUST host GPT-4.1 mini (its deployment name may be customized).
Running main makes up to six real requests; importing this module makes none.
Expected groups are illustrative judgments, not production-validated labels.
"""

import argparse
from functools import partial
import os
import sys
from textwrap import wrap

from openai import OpenAIError

from routing.llm import get_routing_response
from routing.rag import recommend_with_rag
from routing.recommend import recommend_assignment_group
from routing.search import IncidentSearch


# Written for this comparison, independently of historical/evaluation reports.
# Expected groups are display-only: neither retrieval nor the LLM sees them.
INCIDENTS = (
    (
        "VPN: remote staff can establish the tunnel and reach internal services by IP, "
        "but internal hostnames fail to resolve. The VPN adapter receives a public DNS "
        "resolver instead of the corporate resolvers; office connections resolve them normally.",
        "Network Operations",
    ),
    (
        "RDP: a published remote desktop disconnects only when the user's local USB "
        "label printer is redirected. Disabling printer redirection allows the session "
        "to remain open. The session host logs a Remote Desktop print redirection error; "
        "ordinary network access and authentication succeed.",
        "End User Computing",
    ),
    (
        "Microsoft 365: a shared Exchange Online mailbox rejects new mail as full, "
        "although its visible folders are small. The administrator sees the mailbox's "
        "Recoverable Items storage at its quota. Other mailboxes receive messages and "
        "the same failure occurs when senders use Outlook on the web.",
        "Collaboration Services",
    ),
    (
        "Identity: a contractor's Entra account was restored after an accidental "
        "deletion, but the enterprise application still rejects access. The sign-in "
        "log states that user assignment is required, and the restored account is "
        "absent from the application's assigned users. Colleagues can access it.",
        "Identity and Access Management",
    ),
    (
        "Workstation: a managed laptop restarts whenever its camera is enabled in "
        "either Teams or the Windows Camera app. The crash report names the installed "
        "camera driver. An external webcam works and the same user can join video "
        "meetings on another laptop.",
        "End User Computing",
    ),
    (
        "Insufficient information: a user says 'my work screen keeps disappearing "
        "this afternoon.' They have not said whether this is a physical monitor, "
        "an RDP session, or an application window. No error, affected device, "
        "connection type, or impact on other users is available.",
        None,
    ),
)


def print_row(values: tuple[str, ...]) -> None:
    """Wrap all columns so descriptions and explanations remain readable."""
    widths = (38, 26, 28, 28, 42)
    columns = [wrap(" ".join(value.split()), width=width) or [""]
               for value, width in zip(values, widths)]
    for line in range(max(map(len, columns))):
        print(" | ".join(
            (column[line] if line < len(column) else "").ljust(width)
            for column, width in zip(columns, widths)
        ))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    names = ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT")
    missing = [name for name in names if not os.environ.get(name, "").strip()]
    if missing:
        print(f"Missing Azure configuration. Set: {', '.join(missing)}", file=sys.stderr)
        return 1

    try:
        search = IncidentSearch()
    except Exception:
        print("Local search could not initialize. Check the dataset and embedding model availability.",
              file=sys.stderr)
        return 1

    # One pipeline call per case, with SDK retries disabled: at most six requests.
    llm_client = partial(get_routing_response, max_retries=0)
    print("Baseline vs GPT-4.1 mini RAG (configured deployment must host GPT-4.1 mini)")
    print("Six fictional cases; expected groups are illustrative. No automatic assignments.")
    print_row(("Incident summary", "Expected group", "Baseline", "LLM recommendation", "LLM explanation"))
    failed = False
    for number, (description, expected) in enumerate(INCIDENTS[:6], start=1):
        baseline_text, llm_text, explanation = "Error", "Not run", "Local retrieval/recommendation failed."
        try:
            baseline = recommend_assignment_group(search.search(description, top_k=5))
            if baseline.status == "recommended":
                baseline_text = baseline.assignment_group
            elif baseline.status == "suggested":
                baseline_text = "Suggested: " + ", ".join(baseline.candidate_groups)
            else:
                baseline_text = "Manual review"
        except Exception:
            failed = True
            print(f"Case {number}: local retrieval/recommendation failed. Check local data and model.",
                  file=sys.stderr)
        else:
            try:
                result = recommend_with_rag(description, search=search, llm_client=llm_client)
                llm_text = result.assignment_group if result.status == "recommended" else "Manual review"
                explanation = result.reason
            except OpenAIError:
                llm_text = "API error"
                explanation = "Foundry request failed. Check credentials, deployment, network, and quota."
                failed = True
            except ValueError:
                llm_text = "Validation error"
                explanation = "RAG configuration or response validation failed. Check configuration and response format."
                failed = True
            except Exception:
                llm_text = "Pipeline error"
                explanation = "RAG could not complete. Check local model and Foundry configuration."
                failed = True
            # Never print exception messages: they can contain credentials/request details.
            if llm_text in ("API error", "Validation error", "Pipeline error"):
                print(f"Case {number}: {explanation}", file=sys.stderr)
        print_row((f"{number}. {description}", expected or "Unclear; manual review", baseline_text,
                   llm_text, explanation))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
