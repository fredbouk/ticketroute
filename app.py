import streamlit as st


st.title("TicketRoute")
st.subheader("AI-Assisted IT Incident Routing")

st.write(
    "TicketRoute aims to help IT support personnel identify the appropriate "
    "technical team for an incident, reducing time spent searching for "
    "assignment groups and transferring incorrectly routed incidents. "
    "Future recommendations will include supporting evidence for human review."
)

incident_description = st.text_area(
    "IT incident description",
    placeholder="Describe the incident using synthetic demonstration data only.",
)

if st.button("Analyze Incident"):
    if not incident_description.strip():
        st.warning("Please enter an incident description.")
    else:
        st.info(
            "Incident routing functionality has not been implemented yet. "
            "This version provides the user interface only."
        )
