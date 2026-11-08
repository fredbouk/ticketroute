import streamlit as st

from routing.recommend import RecommendationResult, recommend_assignment_group
from routing.search import IncidentSearch


@st.cache_resource
def get_incident_search() -> IncidentSearch:
    """Reuse the local model and historical embeddings across Streamlit reruns."""
    return IncidentSearch()


def display_recommendation(result: RecommendationResult) -> None:
    if result.status == "recommended":
        st.success(f"Recommended assignment group: {result.assignment_group}")
    elif result.status == "suggested":
        st.warning("Suggested assignment groups — technician judgment required")
        for rank, group in enumerate(result.candidate_groups, start=1):
            st.write(f"{rank}. {group}")
        st.caption(
            "Ranked by similarity-weighted historical evidence. "
            "Equal scores are listed alphabetically."
        )
    else:
        st.warning("Insufficient evidence — manual review required")
    st.caption("Requires human review. No incident has been assigned or updated.")
    st.write(result.reason)

    with st.expander("Supporting historical incidents"):
        if result.supporting_incidents.empty:
            st.info("No valid supporting historical incidents are available.")
        else:
            st.caption("Cosine similarity measures textual similarity, not routing probability.")
            st.dataframe(
                result.supporting_incidents[
                    ["incident_id", "short_description", "assignment_group", "similarity_score"]
                ],
                hide_index=True,
                column_config={
                    "incident_id": "Incident ID",
                    "short_description": "Short description",
                    "assignment_group": "Historical assignment group",
                    "similarity_score": st.column_config.NumberColumn(
                        "Similarity score", format="%.3f"
                    ),
                },
            )


def main() -> None:
    st.title("TicketRoute")
    st.subheader("AI-Assisted IT Incident Routing")
    st.write(
        "Find a suggested technical team using similar resolved incidents. "
        "Review the historical evidence before deciding where to route the incident."
    )

    with st.form("incident_analysis"):
        incident_description = st.text_area(
            "IT incident description",
            placeholder="Describe the incident using synthetic demonstration data only.",
        )
        submitted = st.form_submit_button("Analyze Incident")

    if not submitted:
        return
    if not incident_description.strip():
        st.warning("Please enter an incident description.")
        return

    try:
        with st.spinner("Loading incident search..."):
            search = get_incident_search()
    except Exception:
        # Keep model errors, paths, and potentially sensitive details out of the UI.
        st.error(
            "Incident search could not be loaded. Please check that the historical "
            "dataset and local embedding model are available, then try again."
        )
        return

    try:
        with st.spinner("Analyzing historical evidence..."):
            matches = search.search(incident_description.strip(), top_k=5)
            result = recommend_assignment_group(matches)
    except Exception:
        st.error("The incident could not be analyzed. Please try again or request manual review.")
        return

    display_recommendation(result)


if __name__ == "__main__":
    main()
