"""Manually test Foundry from the project root after configuring Azure credentials.

Example:
    python -m scripts.test_rag "VPN connects, but internal websites time out."

Required environment variables: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
and AZURE_OPENAI_DEPLOYMENT. A normal run uses the local embedding model and
makes a real Foundry request when historical incidents are retrieved.
"""

import argparse
import os
import sys

from openai import OpenAIError

from routing.rag import recommend_with_rag


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test TicketRoute's RAG pipeline with Microsoft Foundry.")
    parser.add_argument("description", help="Incident description, enclosed in quotes.")
    args = parser.parse_args(argv)
    if not args.description.strip():
        parser.error("Please provide a nonblank incident description.")

    # Fail before loading the embedding model; display variable names, never values.
    names = ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT")
    missing = [name for name in names if not os.environ.get(name, "").strip()]
    if missing:
        print(f"Missing Azure configuration. Set: {', '.join(missing)}", file=sys.stderr)
        return 1

    try:
        result = recommend_with_rag(args.description)
    except OpenAIError:
        print(
            "Microsoft Foundry request failed. Check the endpoint, API key, deployment name, "
            "network access, and available quota.", file=sys.stderr,
        )
        return 1
    except ValueError:
        print(
            "Routing validation failed. Check the incident input and ensure the model returns "
            "the required JSON fields and an assignment group supported by the evidence.",
            file=sys.stderr,
        )
        return 1
    except Exception:
        # Exception messages may contain request details or credentials. Do not print them.
        print(
            "The RAG pipeline could not complete. Check local embedding model availability "
            "and the Azure endpoint and deployment configuration, then retry.", file=sys.stderr,
        )
        return 1

    if result.status == "recommended":
        print(f"Recommended assignment group: {result.assignment_group}")
    else:
        print("Manual review needed")
    print(f"Explanation: {result.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
