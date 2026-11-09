import os

import pandas as pd
import streamlit as st

from routing.llm import LLMRoutingResult
from routing.rag import recommend_with_rag
from routing.recommend import RecommendationResult, recommend_assignment_group
from routing.search import IncidentSearch


EXAMPLE_PLACEHOLDER = "Choose an example..."
# Entirely fictional examples, with no real company or personal information.
EXAMPLE_INCIDENTS = {
    "Cloud Platform": (
        "Short description: Azure Container Apps service unavailable after revision rollout\n\n"
        "Description: The document preview service returns HTTP 503 after a new "
        "Azure Container Apps revision is activated. The revision dashboard shows no "
        "ready replicas, and platform events report failed startup probes. The previous "
        "revision still responds through its revision-specific URL. The failure is reproducible "
        "from both office and remote connections."
    ),
    "Identity and Access Management": (
        "Short description: Entra ID sign-in requests security keys users do not have\n\n"
        "Description: Several users can enter their passwords, but sign-in "
        "to several enterprise applications stops with an authentication strength "
        "requirement. Their registered Authenticator method is not offered, and the "
        "page requests a security key they have not been issued. The same result occurs "
        "in a fresh browser on a second managed device. Other users can still sign in."
    ),
    "Collaboration Services": (
        "Short description: Exchange Online room declines bookings beyond the next two weeks\n\n"
        "Description: A meeting room accepts reservations for next week but "
        "automatically declines available slots later in the month. The rejection says "
        "the requested date is outside the permitted booking period. Several organizers "
        "reproduce the behavior in Outlook on the web, while another room accepts the "
        "same dates. Existing reservations and ordinary email delivery are unaffected."
    ),
    "Backup hardware failure": (
        "Short description: LTO tape library cannot mount backup cartridges\n\n"
        "Description: The on-premises backup system cannot load tapes "
        "into either drive of its LTO tape library. The library's front panel reports "
        "a robotic cartridge picker fault, and its built-in inventory self-test stops "
        "when moving a cartridge from a magazine. Several cartridges produce "
        "the same fault. The backup server and tape drives remain online, with no "
        "network connection errors recorded."
    ),
}


# Presentation only: native components handle forms, status, and evidence.
PAGE_STYLE = """
<style>
    .stMainBlockContainer { padding-top: 1.75rem; padding-bottom: 2rem; }
    .stApp h1 { font-size: 2.25rem; letter-spacing: -0.035em; padding: 0 0 0.25rem; }
    .st-key-app_header h1 [data-testid="stIconMaterial"] {
        color: #315E9B;
        font-size: 1.5rem;
        vertical-align: middle;
    }
    .stApp h3 { font-size: 1.15rem; padding-top: 0; padding-bottom: 0.25rem; }
    .st-key-app_header {
        border-left: 3px solid #315E9B;
        padding-left: 1rem;
        margin-bottom: 0.5rem;
    }
    .st-key-incident_input {
        background: #FFFFFF;
        border-color: #D9E1EC;
        border-radius: 0.75rem;
        padding: 1rem 1.25rem;
    }
    .st-key-incident_input [data-testid="stCaptionContainer"] p {
        font-size: 1rem;
        color: #526176;
    }
    .st-key-example_selector [data-testid="stCaptionContainer"] p { font-size: 15px; }
    .st-key-example_selector [data-baseweb="select"] {
        font-size: 15px;
        color: #526176;
    }
    .st-key-example_selector [data-baseweb="select"] div,
    .st-key-example_selector [data-baseweb="select"] input {
        font-size: 15px;
    }
    [data-testid="stExpander"] summary { align-items: flex-start; }
    [data-testid="stExpander"] summary [data-testid="stMarkdownContainer"] {
        min-width: 0;
        overflow-wrap: anywhere;
    }
    [data-testid="stExpander"] summary p { white-space: normal; line-height: 1.5; }
    [data-testid="stTextArea"] textarea {
        background: #FBFCFE;
        font-size: 1rem;
        line-height: 1.55;
    }
    .st-key-find_assignment_group button {
        min-height: 2.9rem;
        padding: 0.55rem 1.3rem;
        font-weight: 600;
    }
    .st-key-find_assignment_group button p { font-size: 16px; }
    .st-key-routing_result {
        background: #F8FAFE;
        border-left: 3px solid #315E9B;
        border-radius: 0.75rem;
    }
    .st-key-routing_result h3 {
        color: #254E82;
        font-size: 2rem;
        line-height: 1.3;
        overflow-wrap: anywhere;
    }
    @media (max-width: 640px) {
        .stMainBlockContainer { padding: 1rem; }
        .stApp h1 { font-size: 1.8rem; }
        .st-key-incident_input { padding: 0.85rem; }
        .st-key-routing_result h3 { font-size: 1.5rem; }
    }
</style>
"""


@st.cache_resource(show_spinner=False)
def get_incident_search() -> IncidentSearch:
    """Reuse the local model and historical embeddings across Streamlit reruns."""
    return IncidentSearch()


def populate_example_incident() -> None:
    """Populate the input without retrieving incidents or producing a result."""
    selected = st.session_state["example_incident"]
    if selected in EXAMPLE_INCIDENTS:
        st.session_state["incident_description"] = EXAMPLE_INCIDENTS[selected]


def display_recommendation(
    result: RecommendationResult | LLMRoutingResult,
    supporting_incidents: pd.DataFrame | None = None,
) -> None:
    with st.container(border=True, key="routing_result"):
        if result.status == "recommended":
            st.caption("Recommended assignment group")
            st.subheader(result.assignment_group)
        elif result.status == "suggested":
            st.caption("Suggested assignment groups")
            for rank, group in enumerate(result.candidate_groups, start=1):
                st.subheader(f"{rank}. {group}")
        else:
            st.caption("Routing result")
            st.subheader("Manual review needed")
            if isinstance(result, RecommendationResult):
                st.write(
                    "No closely related incidents were found. Please review this incident "
                    "and determine the appropriate assignment group."
                )

        if isinstance(result, LLMRoutingResult):
            st.write(result.reason)

    if result.status == "manual_review":
        return

    if supporting_incidents is None:
        supporting_incidents = result.supporting_incidents
    count = len(supporting_incidents)
    st.subheader(f"Similar historical incidents ({count})")
    if supporting_incidents.empty:
        st.caption("No similar historical incidents are available.")
    else:
        for _, incident in supporting_incidents.iterrows():
            label = (
                f"{incident['incident_id']} · {incident['short_description']}  \n"
                f"Historical assignment group: **{incident['assignment_group']}**"
            )
            with st.expander(label):
                description, resolution = st.columns([3, 2], gap="medium")
                with description:
                    if "description" in incident:
                        st.caption("Incident description")
                        st.write(incident["description"])
                with resolution:
                    if "resolution_summary" in incident:
                        st.caption("Resolution")
                        st.write(incident["resolution_summary"])


def display_page() -> None:
    with st.container(key="app_header", gap="small"):
        st.caption("IT SERVICE DESK / ASSIGNMENT GUIDANCE")
        st.title(":material/alt_route: TicketRoute")
        st.write("Find the right support team using similar resolved incidents.")

    with st.container(border=True, key="incident_input"):
        st.subheader("Incident information")
        st.caption("Paste the short description and/or description from the ServiceNow incident.")
        incident_description = st.text_area(
            "IT incident description",
            key="incident_description",
            label_visibility="collapsed",
            height=140,
            placeholder="Paste incident text here…",
        )
        # The selector stays outside a form so its callback can fill the text box
        # immediately. Only the primary button starts retrieval and analysis.
        with st.container(width=440, key="example_selector", gap="small"):
            st.caption("Try an example incident")
            st.selectbox(
                "Try an example incident",
                options=[EXAMPLE_PLACEHOLDER, *EXAMPLE_INCIDENTS],
                index=0,
                label_visibility="collapsed",
                key="example_incident",
                on_change=populate_example_incident,
            )
        submitted = st.button(
            "Find assignment group", type="primary", width=280, key="find_assignment_group",
        )

    if not submitted:
        return
    if not incident_description.strip():
        st.warning("Please enter an incident description.")
        return

    with st.spinner("Analyzing incident..."):
        try:
            search = get_incident_search()
        except Exception:
            # Keep model errors, paths, and potentially sensitive details out of the UI.
            st.error(
                "Incident search could not be loaded. Please check that the historical "
                "dataset and local embedding model are available, then try again."
            )
            return

        try:
            matches = search.search(incident_description.strip(), top_k=5)
            names = ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT")
            result = None
            if all(os.environ.get(name, "").strip() for name in names):
                try:
                    # The configured Foundry deployment hosts GPT-4.1 mini.
                    # Reuse the cached model; RAG validates the response against its retrieval.
                    result = recommend_with_rag(incident_description.strip(), search=search)
                except Exception:
                    # Do not expose SDK errors, credentials, or invalid response content.
                    pass
            if result is None:
                result = recommend_assignment_group(matches)
                st.caption("Local routing used: AI routing is unavailable.")
        except Exception:
            st.error("The incident could not be analyzed. Please try again or request manual review.")
            return

    display_recommendation(
        result, matches if isinstance(result, LLMRoutingResult) else None,
    )


def main() -> None:
    st.set_page_config(page_title="TicketRoute | Incident Routing", layout="wide")
    st.markdown(PAGE_STYLE, unsafe_allow_html=True)
    with st.container(horizontal_alignment="center"):
        with st.container(width=1200, gap="medium"):
            display_page()


if __name__ == "__main__":
    main()
