# TicketRoute

**AI-Assisted IT Incident Routing**

TicketRoute is a Python portfolio project that aims to help first- and
second-line IT support personnel identify the appropriate technical team for
an incident. Its goal is to reduce time spent searching for assignment groups
and transferring incorrectly routed incidents, helping support teams focus on
resolving problems and assisting users.

This initial version is a simple Streamlit interface with an incident description
field and an **Analyze Incident** button. Clicking the button displays a message
explaining that routing functionality has not been implemented yet.

Future versions may use historical resolved incidents, semantic search, and AI
to suggest assignment groups with supporting evidence and human review.
AI, semantic search, databases, and Azure integration are not included in this version.

Use synthetic demonstration data only. Do not enter real incidents or personal
or confidential information.

## Run locally

Use Python 3.12. From the project folder, activate your existing virtual
environment if you have one. In Windows PowerShell, the project's existing
environment can be activated with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Check that the selected interpreter uses Python 3.12:

```powershell
python --version
```

Install the dependency and start the application:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL displayed in the terminal (usually `http://localhost:8501`).
Press `Ctrl+C` in the terminal to stop the application.

## Project files

- `app.py`: the Streamlit user interface and button response.
- `requirements.txt`: the application's Python dependency, Streamlit.
- `README.md`: project overview and local setup instructions.

The application uses one Python file because this first version only needs a
small user interface. Streamlit runs the script from top to bottom and reruns
it when a user interacts with a widget. No separate backend is needed yet.
