from pathlib import Path

import pandas as pd


def load_incidents() -> pd.DataFrame:
    """Load historical resolved incidents from the project's CSV dataset."""
    project_directory = Path(__file__).resolve().parent.parent
    csv_path = project_directory / "data" / "incidents.csv"

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Historical incident CSV not found at {csv_path}. "
            "Ensure data/incidents.csv exists in the project directory."
        )

    return pd.read_csv(csv_path)
