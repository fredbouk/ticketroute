# TicketRoute

**AI-Assisted IT Incident Routing**

## Project overview

TicketRoute is an AI-assisted IT incident routing application designed to help
first- and second-line IT support personnel identify the appropriate technical
team for an incident.

Incorrect incident assignments can lead to unnecessary investigation, repeated
ticket transfers, and delayed resolution. TicketRoute aims to reduce this
administrative overhead by using historical resolved incidents to support
evidence-based routing recommendations.

## Key objectives

- Reduce the time support personnel spend identifying assignment groups.
- Minimize unnecessary incident transfers.
- Help technical teams focus on resolving problems rather than administrative routing work.
- Help users receive faster assistance by directing incidents to the appropriate team earlier.

## Current functionality

The application provides a Streamlit interface for entering incident descriptions.
The **Analyze Incident** button validates that the description is not empty or
whitespace-only. Valid input displays a status message explaining that the
routing engine and AI integration have not yet been implemented.

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

## Demo environment and data privacy

The public demonstration will use entirely synthetic incident data. Real customer
incidents, confidential information, and personal data should not be submitted
to the public demo.
